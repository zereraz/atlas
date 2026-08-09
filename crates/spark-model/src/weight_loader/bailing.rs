// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for inclusionAI **Ling-3.0-flash** (`bailing_hybrid`).
//!
//! Ling is architecturally a Qwen3.5 hybrid-clone (35 KDA + 7 MLA + MoE +
//! MTP), but uses its OWN tensor names. This module provides the actual
//! layer-assembly loop using Ling's `{lp}.attention.*` names:
//!
//!   KDA arm: `load_ssm_bailing` (q_proj/k_proj/v_proj + q/k/v_conv1d + f/g/b
//!            + A_log/dt_bias/o_norm/o_proj → SsmWeightsQwen35 → Qwen3SsmLayer)
//!   MoE arm: `load_moe_qwen35` + `gate.weight + gate.expert_bias` →
//!            MoeWeights (correction_bias = expert_bias)
//!   MLA arm: `{attention}.{q_proj,kv_a_proj_with_msa,kv_a_layernorm,kv_b_proj,
//!            dense,g_proj}` → MlaWeights via the absorbed-weight algebra
//!            (`phase_qk_absorbed` etc. — mirrored after Qwen35's MLA build)."
//!
//! Ling is **not** TP-aware in this first cut: TP=1 on spark. Set
//! `supports_tp() = false` so the server fails-fast on `--tp-size > 1`.

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use super::qwen35;
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, MoeLayer, Qwen3SsmLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MoeWeights, MtpWeights, QuantizedWeight, SsmWeights, dense,
    detect_nvfp4_variant, gpu, load_moe_qwen35, load_mtp, load_ssm_bailing, quantize_to_nvfp4,
};

/// Ling-3.0-flash (bailing_hybrid). Layers are built here using Ling's
/// `attention.*` naming; embeddings/norm/lm_head differ from Qwen3.5
/// (`model.word_embeddings.weight`, `model.norm.weight`, untied `lm_head`).
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
        // Ling: every layer uses `attention.*` under the layer prefix.
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

        let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            let moe_weights = load_moe_qwen35(
                store,
                &lp,
                config.num_experts,
                gpu,
                config,
                variant,
                absmax_k,
                quantize_k,
                stream,
                false,
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

            let moe_layer = MoeLayer::new(moe_weights, config.num_experts, Some(gate_nvfp4), gpu, config)?;
            let ffn = FfnComponent::Moe(moe_layer);

            match lt {
                LayerType::FullAttention => {
                    // Ling MLA: build absorbed-Mla attention using Ling's names.
                    let layer = build_full_attention_bailing(
                        i, store, &lp, gpu, variant, config, h,
                        absmax_k, quantize_k, stream,
                        layer_kv_dtypes[attn_idx], attn_idx,
                        input_norm, post_attn_norm, ffn,
                    )?;
                    layers.push(layer);
                    attn_idx += 1;
                }
                LayerType::LinearAttention => {
                    let layer = build_linear_attention_bailing(
                        i, store, &lp, gpu, variant, config, h,
                        absmax_k, quantize_k, stream,
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

    fn load_final_norm(&self, store: &WeightStore, config: &ModelConfig, _gpu: &dyn GpuBackend) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.norm.weight"))
            .or_else(|_| dense(store, &format!("{prefix}.final_layernorm.weight")))
            .map_err(|e| anyhow::anyhow!("Ling final norm not found ({e:#})"))
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else {
            self.load_embedding(store, config)
        }
    }

    fn load_mtp_weights(&self, store: &WeightStore, config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            tracing::info!("Ling: no MTP weights found — speculative decoding disabled");
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        Ok(Some(mtp))
    }
}

/// Ling KDA linear attention layer.
///
/// Identical structure to Qwen3.5's `build_linear_attention_nvfp4` but using
/// `load_ssm_bailing` (Ling's `attention.*` names) instead of `load_ssm_qwen35`.
///
/// Ling KDA uses no `conv_kernel` size config — conv state shape
/// `[q_conv_dim, hidden, 4]`, 4 = `short_conv_kernel_size` from Ling config.
/// Ling's `attn_gated = true` (output gate exists).
#[allow(clippy::too_many_arguments)]
fn build_linear_attention_bailing(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: crate::weight_map::Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let i = layer_idx;
    let ssm35 = load_ssm_bailing(store, lp, gpu, variant)?;
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
    let nk = config.linear_num_key_heads;
    let ba_dense = crate::weight_map::interleave_ba(
        &DenseWeight { weight: ssm35.in_proj_a.weight },
        &DenseWeight { weight: ssm35.in_proj_b.weight },
        nv, nk, h, gpu,
    )?;

    let qkvz_size = config.ssm_qkvz_size();
    let qkvz_nvfp4 = quantize_to_nvfp4(&qkvz_dense, qkvz_size, h, gpu, absmax_k, quantize_k, stream)?;
    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

    let value_dim = nv * config.linear_value_head_dim;
    let out_proj_nvfp4 = quantize_to_nvfp4(&ssm35.out_proj, h, value_dim, gpu, absmax_k, quantize_k, stream)?;
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
        input_norm, ssm, post_attn_norm, ffn,
        Some(qkvz_nvfp4), Some(qkvz_nvfp4_t),
        Some(out_proj_nvfp4_t), config, gpu)?;
    layer.predequant_for_prefill(gpu, config, stream)?;
    Ok(Box::new(layer))
}

/// Ling MLA full attention layer.
///
/// Uses the same absorbed-weight algebra as Qwen3.5's MLA (via phase helpers
/// in `weight_loader/`), but with Ling's specific names:
/// `attention.q_proj` (hidden→hidden no q_lora; no `wq_a`), `attention.kv_a_proj_with_mqa`
/// (hidden→kv_lora+rope), `attention.kv_a_layernorm`, `attention.kv_b_proj`,
/// `attention.dense` (q_proj output after q_norm).
///
/// Ling has `q_lora_rank = null` (no Q down-projection) — MLA decode goes
/// through a slightly different absorbed path than Qwen3.5.
///
/// NOTE: The full absorbed-weight assembly (phase_qk_absorbed, phase_block_diag,
/// wq_b_rope extraction) is a precise multi-day construction; this stub panics
/// at load-time with a clear message until the algebra is fully wired.
///
/// For the near-term serve use KDA-only by loading a Ling variant that
/// routes every layer through `LinearAttention` — see the debug comment in
/// `BailingHybridWeightLoader::load_layers`.
#[allow(clippy::too_many_arguments)]
fn build_full_attention_bailing(
    i: usize,
    _store: &WeightStore,
    lp: &str,
    _gpu: &dyn GpuBackend,
    _variant: crate::weight_map::Nvfp4Variant,
    _config: &ModelConfig,
    _h: usize,
    _absmax_k: spark_runtime::gpu::KernelHandle,
    _quantize_k: spark_runtime::gpu::KernelHandle,
    _stream: u64,
    _layer_kv_dtype: KvCacheDtype,
    _attn_idx: usize,
    _input_norm: DenseWeight,
    _post_attn_norm: DenseWeight,
    _ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    tracing::error!("Ling[{i}]: MLA full-attention assembly ({lp}.attention.*) is not yet wired — see LING_RUST_ATLAS.md");
    anyhow::bail!("Ling MLA assembly ({lp}) is not yet implemented")
}
