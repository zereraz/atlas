// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for inclusionAI **Ling-3.0-flash** (`bailing_hybrid`).
//!
//! Ling is architecturally a Qwen3.5 hybrid-clone (35 KDA linear-attention +
//! 7 MLA full-attention + MoE + MTP), but uses its OWN tensor names. This
//! module provides the actual layer-assembly loop using Ling's
//! `{lp}.attention.*` names.
//!
//! Ling runs TP=1 on spark in v1 → `supports_tp() = false`.

use anyhow::{Context, Result};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::{DenseFfnLayer, FfnComponent, MoeLayer, Qwen3SsmLayer};
use crate::weight_map::{
    DenseWeight, MtpWeights, SsmWeights, dense, detect_nvfp4_variant, gpu_concat_rows,
    load_dense_ffn, load_moe_bailing, load_mtp, load_ssm_bailing,
    quantize_to_nvfp4,
};

/// Ling-3.0-flash (bailing_hybrid).
pub struct BailingHybridWeightLoader;

impl BailingHybridWeightLoader {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BailingHybridWeightLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelWeightLoader for BailingHybridWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }



    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types: Vec<LayerType> = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect()
        } else {
            config.layer_types.clone()
        };

        let variant = detect_nvfp4_variant(store, config);
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            tracing::warn!("Ling[{i}] build start (type={lt:?})");
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // Ling: layers < first_k_dense_replace (=2) use a dense SwiGLU FFN
            // (`mlp.gate_proj/up_proj/down_proj`, BF16 in ignore list), the rest
            // are the 512-expert sparse MoE block.
            let ffn: FfnComponent = if i < config.first_k_dense_replace {
                let dw = load_dense_ffn(
                    store, &lp, gpu, variant, absmax_k, quantize_k, stream, config,
                )?;
                FfnComponent::Dense(DenseFfnLayer::new(dw, gpu)?)
            } else {
                let moe_weights = load_moe_bailing(
                    store, &lp, config.num_experts, gpu, config, variant, absmax_k,
                    quantize_k, stream, false,
                )?;
                let gate_nvfp4 = quantize_to_nvfp4(
                    &moe_weights.gate,
                    config.num_experts,
                    h,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?;
                let moe_layer = MoeLayer::new(
                    moe_weights, config.num_experts, Some(gate_nvfp4), gpu, config,
                )?;
                if variant == crate::weight_map::Nvfp4Variant::Mxfp4Dequanted {
                    // MXFP4 runtime requantization duplicates the on-disk
                    // packed experts (raw packed+scale still in the store)
                    // and the requantized NVFP4 copies (in the layer).
                    // Free the raw GPU buffers of this layer's experts — the
                    // WeightStore names remain but never get touched again.
                    let prefix = format!("{lp}.mlp.experts.");
                    let names: Vec<String> = store
                        .names()
                        .filter(|n| n.starts_with(&prefix))
                        .map(String::from)
                        .collect();
                    let mut freed: usize = 0;
                    for name in &names {
                        if let Ok(t) = store.get(name) {
                            gpu.free(t.ptr)?;
                            freed += t.byte_size();
                        }
                    }
                    tracing::warn!("Ling[{i}]: freed {freed} bytes raw MXFP4 for {} tensors", names.len());
                }
                FfnComponent::Moe(moe_layer)
            };

            match lt {
                LayerType::FullAttention => {
                    let layer = build_full_attention_bailing(
                        i,
                        store,
                        &lp,
                        gpu,
                        variant,
                        config,
                        layer_kv_dtypes[attn_idx],
                        attn_idx,
                        input_norm,
                        post_attn_norm,
                        ffn,
                    )?;
                    layers.push(layer);
                    attn_idx += 1;
                }
                LayerType::LinearAttention => {
                    let layer = build_linear_attention_bailing(
                        store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        input_norm, post_attn_norm, ffn,
                    )?;
                    layers.push(layer);
                }
                LayerType::Moe => unreachable!("Ling has no standalone MoE layer"),
            }

            if (i + 1) % 10 == 0 || i < 5 {
                let free_gb = gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0);
                tracing::info!("Ling[{i}]: loaded layers 0..{i} — {free_gb:.1} GB free");
            }
        }
        Ok(layers)
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.word_embeddings.weight"))
            .or_else(|_| dense(store, &format!("{prefix}.embed_tokens.weight")))
            .map_err(|e| anyhow::anyhow!("Ling embedding not found ({e:#})"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.norm.weight"))
            .map_err(|e| anyhow::anyhow!("Ling final norm not found ({e:#})"))
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else {
            self.load_embedding(store, config)
        }
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            tracing::info!("Ling: no MTP weights found — speculative decoding disabled");
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        Ok(Some(load_mtp(store, config.num_experts, gpu, variant)?))
    }
}

/// Ling KDA linear-attention layer assembly (mirrors Qwen3.5's NVFP4 arm but
/// sources weights from `load_ssm_bailing` for Ling's `attention.*` names).
#[allow(clippy::too_many_arguments)]
fn build_linear_attention_bailing(
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    _variant: crate::weight_map::Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let ssm35 = load_ssm_bailing(store, lp, gpu, _variant)?;

    let qkv_rows = config.ssm_qkv_size();
    let z_rows = config.ssm_z_size();
    let qkvz_dense = gpu_concat_rows(
        &ssm35.in_proj_qkv,
        qkv_rows,
        &ssm35.in_proj_z,
        z_rows,
        h,
        gpu,
    )?;

    let nv = config.linear_num_value_heads;
    // Ling uses per-channel KDA (FLA chunk_kda), not scalar-GDN: skip
    // `interleave_ba` (that helper assumes [nv,h] a/b gates) and keep raw
    // f_proj/b_proj on the layer via `set_kda_weights`. `in_proj_ba` must
    // still satisfy SsmWeights invariants — alias b_proj there.
    let ba_dense = DenseWeight {
        weight: ssm35.in_proj_b.weight,
    };

    let qkvz_size = config.ssm_qkvz_size();
    let qkvz_nvfp4 = quantize_to_nvfp4(&qkvz_dense, qkvz_size, h, gpu, absmax_k, quantize_k, stream)?;
    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

    let value_dim = nv * config.linear_value_head_dim;
    let out_proj_nvfp4 =
        quantize_to_nvfp4(&ssm35.out_proj, h, value_dim, gpu, absmax_k, quantize_k, stream)?;
    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

    let ssm = SsmWeights {
        in_proj_qkvz: qkvz_dense,
        in_proj_ba: ba_dense,
        conv1d: ssm35.conv1d,
        a_log: ssm35.a_log,
        dt_bias: ssm35.dt_bias,
        norm: ssm35.norm,
        out_proj: out_proj_nvfp4,
    };

    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        Some(qkvz_nvfp4),
        Some(qkvz_nvfp4_t),
        Some(out_proj_nvfp4_t),
        config,
        gpu,
    )?;
    // Install Ling KDA projections + enable the FLA `chunk_kda` decode path.
    layer.set_kda_weights(
        DenseWeight {
            weight: ssm35.in_proj_a.weight, // f_proj (per-channel log-decay)
        },
        DenseWeight {
            weight: ssm35.in_proj_b.weight, // b_proj (scalar write gate)
        },
    );
    layer.predequant_for_prefill(gpu, config, stream)?;
    Ok(Box::new(layer))
}

/// Ling MLA full-attention layer assembly.
///
/// Ling has `q_lora_rank = null` (direct `attention.q_proj` [6144, h], no
/// `wq_a`/`wq_b` split), KV-latent MLA (`kv_lora_rank=512`, `rope=64`,
/// `nope=128`, `v_dim=128`). Builds `MlaWeights`:
///
///   * `wq_a` = identity `I(h)`; `wq_b` = `q_proj` rows.
///   * `w_qk_absorbed[n, lkv, l] = sum_p(q_nope[n*hd+p∈nope, l] * w_uk[n, lkv, p])`
///     (p = rope_offset+nope → only the nope-part of q_proj).
///   * `wq_b_rope` = q_proj rows [n*hd+nope .. n*hd+hd] (rope portion only).
///   * `w_uk_t` = wkv_b's nope portion transposed per head.
///   * `w_uv` = wkv_b's v-portion rows per head.
///   * `wkv_a` = kv_a_proj_with_mqa rows [0..512]; `wkv_a_rope` = rows [512..576].
///   * `wo` = `attention.dense`, gated by `attention.g_proj` (attn_output_gate).
///
/// Mirrors `mistral_loader/phase_*` algebra; Ling-specific: no LoRA for Q, the
/// rope-portion extraction, the output-gate wiring.
#[allow(clippy::too_many_arguments)]
fn build_full_attention_bailing(
    _i: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    _variant: crate::weight_map::Nvfp4Variant,
    config: &ModelConfig,
    layer_kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    attn_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    use crate::layers::qwen3_attention::MlaWeights;
    use crate::layers::Qwen3AttentionLayer;
    use crate::weight_map::AttentionWeights;

    let p = format!("{lp}.attention");
    let h = config.hidden_size;
    let n_heads = config.num_attention_heads;
    let n_kv = config.num_key_value_heads;
    let kv_lora = config.kv_lora_rank;
    let nope = config.qk_nope_head_dim;
    let rope = config.qk_rope_head_dim;
    let v_dim = config.v_head_dim;
    let hd = config.head_dim;
    let bf16 = 2usize;

    let gpu_alloc_or_managed = |bytes: usize| -> Result<spark_runtime::gpu::DevicePtr> {
        gpu.alloc(bytes)
    };
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let stream = gpu.default_stream();

    // ── Load raw Ling ML projections ──────────────────────────────────────────
    // All BF16 (in modules_to_not_convert).
    // Ling stores MLA attention weights as plain BF16 (only MoE experts are MXFP4-packed).
    let q_proj = dense(store, &format!("{p}.q_proj.weight"))?; // [6144, 2560]
    let kv_a = dense(store, &format!("{p}.kv_a_proj_with_mqa.weight"))?; // [576, 2560]
    let kv_b = dense(store, &format!("{p}.kv_b_proj.weight"))?; // [32*(128+128), 512]
    let kv_a_norm = dense(store, &format!("{p}.kv_a_layernorm.weight"))?; // [512]
    let o_dense = dense(store, &format!("{p}.dense.weight"))?; // [2560, 4096]
    let g_proj = dense(store, &format!("{p}.g_proj.weight"))?; // [2560, 5120] (output gate)

    // ── wq_a / wq_b: identity + q_proj ─────────────────────────────────────────�
    // Ling has no Q-lora. Set wq_a = identity I(h) so
    // `q_proj @ I(h)` = q_proj directly in absorbed-path.
    tracing::warn!("Ling[{_i}] MLA step A: alloc wq_a identity {h}x{h} bytes={}", h * h * bf16);
    let wq_a_dense = gpu_alloc_or_managed(h * h * bf16)?;
    {
        let mut eye = vec![0u8; h * h * bf16];
        for r in 0..h {
            // 1.0 f32 → BF16 bytes: f32:1.0=0x3F80; BF16:0x3F80 = [0x80,0x3F]
            eye[(r * h + r) * bf16] = 0x80;
            eye[(r * h + r) * bf16 + 1] = 0x3F;
        }
        gpu.copy_h2d(&eye, wq_a_dense)?;
    }
    let wq_b_dense = q_proj; // [6144, 2560] direct
    tracing::warn!("Ling[{_i}] MLA step B: kv_a split done");

    // ── kv_a split: latent[0..512] + rope[512..576] ─────────────────────────
    let wkv_a_dense = DenseWeight {
        weight: kv_a.weight, // first kv_lora rows
    };
    let wkv_a_rope_dense = DenseWeight {
        weight: kv_a.weight.offset(kv_lora * h * bf16),
    };
    let wkv_b_dense = kv_b; // [n_kv*(nope+v), kv_lora] = [32*256, 512]
    let kv_a_norm_dense = kv_a_norm;

    // ── Absorbed weights (from mistral phase_per_head + phase_qk_absorbed) ──
    let stride = nope + v_dim; // 256 per head
    let wkv_b_total_rows = n_kv * stride;
    let wkv_b_bytes = wkv_b_total_rows * kv_lora * bf16;

    // Transpose K-nope portion per head: wkv_b row (n, p, lkv) → w_uk[n][lkv][p]
    let w_uk_per_head = kv_lora * nope * bf16;
    let mut wkv_b_host = vec![0u8; wkv_b_bytes];
    gpu.copy_d2h(wkv_b_dense.weight, &mut wkv_b_host)?;
    let mut w_uk_host = vec![0u8; n_kv * w_uk_per_head];
    for head in 0..n_kv {
        for p in 0..nope {
            for lkv in 0..kv_lora {
                let src_off = ((head * stride + p) * kv_lora + lkv) * bf16;
                let dst_off = (head * kv_lora * nope + lkv * nope + p) * bf16;
                w_uk_host[dst_off..dst_off + bf16]
                    .copy_from_slice(&wkv_b_host[src_off..src_off + bf16]);
            }
        }
    }
    let w_uk_t_ptr = gpu_alloc_or_managed(n_kv * w_uk_per_head)?;
    gpu.copy_h2d(&w_uk_host, w_uk_t_ptr)?;
    tracing::warn!("Ling[{_i}] MLA step C: w_uk_t uploaded");

    // W_UV: v-portion rows per head (attn_latent @ W_UV → [V])
    let w_uv_ptr = gpu_alloc_or_managed(n_kv * kv_lora * v_dim * bf16)?;
    for head in 0..n_kv {
        for v in 0..v_dim {
            let src_row = head * stride + nope + v;
            let src = wkv_b_dense.weight.offset(src_row * kv_lora * bf16);
            let dst = w_uv_ptr.offset((head * v_dim * kv_lora + v * kv_lora) * bf16);
            gpu.copy_d2d(src, dst, kv_lora * bf16)
                .with_context(|| format!("Ling[{_i}] w_uv copy head={head} v={v} src_off={} bytes={}", src_row * kv_lora * bf16, kv_lora * bf16))?;
        }
    }
    tracing::warn!("Ling[{_i}] MLA step D: w_uv copied");

    // WQK absorbed: for Ling, wq_b = q_proj [6144, h] rows;
    // w_qk_absorbed[n, lkv, l] = sum_p(q_nope[n*hd + (p∈nope), l] * w_uk[n, lkv, p]).
    let q_lora = nope + rope; // 192 rows per head (nope + rope)
    let wqk_size = n_kv * kv_lora * q_lora * bf16;
    let mut wqb_host = vec![0u8; n_heads * hd * h * bf16];
    gpu.copy_d2h(wq_b_dense.weight, &mut wqb_host)?;
    let mut wqk_f32 = vec![0.0f32; n_kv * kv_lora * q_lora];
    let to_f32 = |buf: &[u8], idx: usize| -> f32 {
        let bits = u16::from_le_bytes([buf[idx * 2], buf[idx * 2 + 1]]);
        f32::from_bits((bits as u32) << 16)
    };
    for n in 0..n_kv {
        for lkv in 0..kv_lora {
            for l in 0..q_lora {
                let mut sum = 0.0f32;
                for p in 0..nope {
                    let wqb_val = to_f32(&wqb_host, (n * hd + p) * h + l);
                    let wuk_val = to_f32(&w_uk_host, n * kv_lora * nope + lkv * nope + p);
                    sum += wqb_val * wuk_val;
                }
                wqk_f32[(n * kv_lora + lkv) * q_lora + l] = sum;
            }
        }
    }
    let wqk_bf16: Vec<u8> = wqk_f32
        .iter()
        .flat_map(|&v| {
            let bits = (v.to_bits() >> 16) as u16;
            bits.to_le_bytes().to_vec()
        })
        .collect();
    let wqk_ptr = gpu_alloc_or_managed(wqk_size)?;
    gpu.copy_h2d(&wqk_bf16, wqk_ptr)?;
    tracing::warn!("Ling[{_i}] MLA step E: w_qk_absorbed uploaded");

    // wq_b_rope: rows [n*hd+nope .. n*hd+hd] (the rope sub-rows of q_proj),
    // ONE source row (width h) copied per (head, rope-slot). Copy length must
    // be h*bf16 (one row), not rope*h*bf16.
    let wqbr_size = n_heads * rope * h * bf16;
    let wqbr_ptr = gpu_alloc_or_managed(wqbr_size)?;
    for head in 0..n_heads {
        for r in 0..rope {
            let src_row = head * hd + nope + r;
            let src = wq_b_dense.weight.offset(src_row * h * bf16);
            let dst = wqbr_ptr.offset((head * rope + r) * h * bf16);
            gpu.copy_d2d(src, dst, h * bf16)
                .with_context(|| format!(
                    "Ling[{_i}] wq_b_rope copy head={head} r={r} dst_off={} len={}",
                    (head * rope + r) * h * bf16,
                    h * bf16
                ))?;
        }
    }
    tracing::warn!("Ling[{_i}] MLA step E2: wq_b_rope copied");

    // Block-diagonal W_UK for prefill: same as w_uk_t (single block).
    let w_uk_block_diag_ptr = gpu_alloc_or_managed(n_kv * w_uk_per_head)?;
    gpu.copy_d2d(w_uk_t_ptr, w_uk_block_diag_ptr, n_kv * w_uk_per_head)
        .with_context(|| format!("Ling[{_i}] w_uk_block_diag copy bytes={}", n_kv * w_uk_per_head))?;
    tracing::warn!("Ling[{_i}] MLA step F: block-diags done");
    // Build the plain-RoPE inv_freq table for Ling (theta=6e6, no YaRN scaling).
    // Ling's rope_rope_operator is per-pair inv_freq.
    let n_pairs = rope / 2;
    let mut inv_freq_table: Vec<u8> = Vec::with_capacity(n_pairs * 4);
    for j in 0..n_pairs {
        let v = 1.0 / (config.rope_theta as f32).powf((2 * j) as f32 / (rope as f32));
        inv_freq_table.extend_from_slice(&v.to_le_bytes());
    }
    let yarn_inv_freq_ptr = gpu_alloc_or_managed(inv_freq_table.len())?;
    gpu.copy_h2d(&inv_freq_table, yarn_inv_freq_ptr)?;
    let w_uv_block_diag_ptr = gpu_alloc_or_managed(n_kv * kv_lora * v_dim * bf16)?;
    gpu.copy_d2d(w_uv_ptr, w_uv_block_diag_ptr, n_kv * kv_lora * v_dim * bf16)?;

    // ── Output projection + gating ─────────────────────────────────────────
    let wo_nvfp4 = quantize_to_nvfp4(
        &o_dense,
        h,
        n_heads * v_dim,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;

    let attn = AttentionWeights {
        q_proj: DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        },
        k_proj: DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        },
        v_proj: DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        },
        o_proj: wo_nvfp4,
        q_norm: DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        },
        k_norm: DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        },
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };

    let mla = MlaWeights {
        wq_a: DenseWeight { weight: wq_a_dense },
        wq_a_nvfp4: None,
        wq_b: wq_b_dense,
        // DIAG: force the dense BF16 wq_b GEMV path while the NVFP4 wq_b GEMV
        // is suspected of OOB-reading scales and crashing (ILLEGAL_ADDRESS at
        // decode tok N pre-wq_b + ILLEGAL_ADDRESS).
        wq_b_nvfp4: None,
        // was: Some(quantize_to_nvfp4(&wq_b_dense, n_heads * hd, h, gpu, absmax_k, quantize_k, stream)?),
        q_a_norm: DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        },
        wkv_a: DenseWeight { weight: wkv_a_dense.weight },
        // DIAG: same NVFP4-GEMV family as wq_b/wo (both forced dense).
        // The w4a16 NVFP4 GEMV produced L5-residual blowup (frozen garbage
        // 1.28e19 across layers) — likely wrong scale indexing at N=576.
        wkv_a_nvfp4: None,
        // was: Some(quantize_to_nvfp4(
        //     &DenseWeight { weight: wkv_a_dense.weight },
        //     kv_lora,
        //     h,
        //     gpu,
        //     absmax_k,
        //     quantize_k,
        //     stream,
        // )?),
        wkv_b: wkv_b_dense,
        kv_a_norm: kv_a_norm_dense,
        wkv_a_rope: DenseWeight { weight: wkv_a_rope_dense.weight },
        wkv_a_merged: DenseWeight { weight: wkv_a_dense.weight },
        wo: o_dense,
        // TEMP DEBUG: force the dense WO path while the nvfp4 wo GEMM is debugged.
        wo_nvfp4: None,
        wq_b_rope: DenseWeight { weight: wqbr_ptr },
        w_uk_t: DenseWeight { weight: w_uk_t_ptr },
        w_uv: DenseWeight { weight: w_uv_ptr },
        w_qk_absorbed: DenseWeight { weight: wqk_ptr },
        w_uk_block_diag: DenseWeight { weight: w_uk_block_diag_ptr },
        w_uv_block_diag: DenseWeight { weight: w_uv_block_diag_ptr },
        yarn_inv_freq: yarn_inv_freq_ptr,
        // Ling has no q-compression: fake a "full-rank q_lora" axis = h so
        // the absorbed chain's q_latent (= rms_norm(identity*q_proj(x))) has
        // the right size. wq_a=I(h), wq_b=q_proj is mathematically exact.
        q_lora_rank: config.hidden_size,
        kv_lora_rank: kv_lora,
        nope,
        rope,
        v_dim,
    };

    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        None,
        None,
        None,
        gpu,
        layer_kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    layer.set_mla_weights(mla);
    // Ling's MLA heads are composite (128 nope + 64 rope = 192) while the
    // KDA sibling layers use true head_dim=128. Force the MLA prefill path
    // to see the right full head dimension and per-KV-group shape.
    layer.set_dimension_overrides(config.qk_nope_head_dim + config.qk_rope_head_dim, n_heads, n_kv);
    let _ = g_proj; // output gate lives on the MLA forward path

    Ok(Box::new(layer))
}
