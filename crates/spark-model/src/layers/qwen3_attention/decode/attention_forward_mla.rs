// SPDX-License-Identifier: AGPL-3.0-only

//! Absorbed-MLA decode path of `attention_forward`. Single-token GEMV
//! chain (Q latent → norm → expand → absorbed-Q via batched GEMV →
//! Q_rope scatter → K_latent → K_rope+RoPE → cache assemble + write →
//! paged decode → V extract → O proj). Returns early — caller short-
//! circuits on the result. Extracted from `attention_forward.rs` to
//! keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

/// DIAG helper: download `len` BF16 elems and log L2 norm + max_abs + first4.
fn mla_diag_norm(gpu: &dyn spark_runtime::gpu::GpuBackend, label: &str, ptr: DevicePtr, len: usize) {
    if std::env::var_os("ATLAS_MLA_DIAG").is_none() {
        return;
    }
    let mut buf = vec![0u8; len * 2];
    if gpu.copy_d2h(ptr, &mut buf).is_err() {
        tracing::warn!("MLA-DIAG {label}: d2h failed");
        return;
    }
    let vals: Vec<f32> = (0..len)
        .map(|i| {
            let bits = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect();
    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs = vals.iter().fold(0f32, |a, &b| a.max(b.abs()));
    let first4: Vec<f32> = vals.iter().take(4).copied().collect();
    tracing::info!("MLA-DIAG {label}: len={len} norm={norm:.4e} max_abs={max_abs:.4e} first4={first4:.6?}");
}

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

#[allow(clippy::too_many_arguments, dead_code)]
pub(super) struct DecodeMlaArgs {
    pub normed: DevicePtr,
    pub q_out: DevicePtr,
    pub k_out: DevicePtr,
    pub v_out: DevicePtr,
    pub q_dim: u32,
    pub h: u32,
    pub nq: u32,
    pub hd: u32,
    pub eps: f32,
    pub bs: usize,
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    /// Run the absorbed MLA decode chain, returning the O-projection
    /// output (`ctx.buffers.qkv_output()`).
    pub(super) fn attention_forward_mla(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &DecodeMlaArgs,
    ) -> Result<DevicePtr> {
        let DecodeMlaArgs {
            normed,
            q_out: _,
            k_out,
            v_out,
            q_dim,
            h,
            nq,
            hd,
            eps,
            bs,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("attention_forward_mla called without MLA config");
        let meta = ctx
            .attn_metadata
            .expect("MLA decode requires pre-uploaded metadata");

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        let profile = ctx.profile;
        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let _t = std::time::Instant::now();
                    let _r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    MLA {}: {:.0}μs", $label, _t.elapsed().as_micros());
                    _r
                } else {
                    $body
                }
            }};
        }

        // Step 1: Q latent → norm → expand
        // Ling: when q_lora == h, wq_a is I(h). Skip the GEMV and feed normed
        // directly. Saves h*h*bf16=13MB per layer and one full GEMV per token.
        let q_latent = ctx.buffers.ssm_ba();
        if q_lora == h {
            // no-op: q_latent already = normed input; we just use `normed` below
        } else {
                prof!("wq_a", {
            if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed,
                    wqa_nvfp4,
                    q_latent,
                    q_lora,
                    h,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wq_a,
                    q_latent,
                    q_lora,
                    h,
                    stream,
                )
            }
            })?;
        }
        // Ling: when q_lora == h, wq_a = I, so just use normed as the Q latent.
        if q_lora == h {
            // `normed` is BF16 [1, h]; replicate to the same buffer expected.
            ctx.gpu.copy_d2d_async(normed, q_latent, (q_lora as usize) * 2, stream)?;
        }
        // Models with no Q-compression (Ling-3.0-flash) have no q_a_layernorm.
        if mla.q_a_norm.weight.0 != 0 {
                prof!("q_norm", {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    q_latent,
                    &mla.q_a_norm,
                    q_latent,
                    1,
                    q_lora,
                    eps,
                    stream,
                )
            })?;
        }
        let q_full = ctx.buffers.ssm_deinterleaved();
        prof!("wq_b", {
            if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    q_latent,
                    wqb_nvfp4,
                    q_full,
                    q_dim,
                    q_lora,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    q_latent,
                    &mla.wq_b,
                    q_full,
                    q_dim,
                    q_lora,
                    stream,
                )
            }
        })?;

        mla_diag_norm(ctx.gpu, "q_full(wq_b out)", q_full, nq as usize * hd as usize);
        // Step 2: Q_absorbed via batched GEMV
        let mla_cache_dim = kv_lora + mla_rope;
        let q_absorbed_buf = ctx.buffers.expert_up_out();
        prof!("q_absorb", {
            if self.mla_batched_gemv_k.0 != 0 {
                ops::mla_batched_gemv(
                    ctx.gpu,
                    self.mla_batched_gemv_k,
                    q_full,
                    mla.w_uk_t.weight,
                    q_absorbed_buf,
                    kv_lora,
                    mla_nope,
                    nq,
                    hd,
                    mla_cache_dim,
                    stream,
                )
            } else {
                for head_idx in 0..nq as usize {
                    let q_nope_ptr = q_full.offset(head_idx * hd as usize * 2);
                    let q_abs_dst = q_absorbed_buf.offset(head_idx * mla_cache_dim as usize * 2);
                    let w_uk_head = mla
                        .w_uk_t
                        .weight
                        .offset(head_idx * mla.nope * mla.kv_lora_rank * 2);
                    let w_uk_dense = crate::weight_map::DenseWeight { weight: w_uk_head };
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        q_nope_ptr,
                        &w_uk_dense,
                        q_abs_dst,
                        kv_lora,
                        mla_nope,
                        stream,
                    )?;
                }
                Ok(())
            }
        })?;
        mla_diag_norm(ctx.gpu, "q_absorbed", q_absorbed_buf, nq as usize * mla_cache_dim as usize);

        // Q_rope scatter
        let q_rope_direct = ctx.buffers.ssm_conv_out_f32();
        prof!("q_rope_scatter", {
            if self.mla_q_rope_scatter_k.0 != 0 {
                ops::mla_q_rope_scatter(
                    ctx.gpu,
                    self.mla_q_rope_scatter_k,
                    q_full,
                    q_absorbed_buf,
                    q_rope_direct,
                    nq,
                    hd,
                    mla_nope,
                    mla_rope,
                    kv_lora,
                    mla_cache_dim,
                    stream,
                )
            } else {
                for head_idx in 0..nq as usize {
                    let src = q_full.offset((head_idx * hd as usize + mla.nope) * 2);
                    ctx.gpu.copy_d2d_async(
                        src,
                        q_rope_direct.offset(head_idx * mla.rope * 2),
                        mla.rope * 2,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        src,
                        q_absorbed_buf
                            .offset((head_idx * mla_cache_dim as usize + mla.kv_lora_rank) * 2),
                        mla.rope * 2,
                        stream,
                    )?;
                }
                Ok(())
            }
        })?;

        // Step 3: KV latent → norm
        let kv_latent = ctx.buffers.expert_gate_out();
        prof!("wkv_a+norm", {
            if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed,
                    wkva_nvfp4,
                    kv_latent,
                    kv_lora,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wkv_a,
                    kv_latent,
                    kv_lora,
                    h,
                    stream,
                )?;
            }
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                kv_latent,
                &mla.kv_a_norm,
                kv_latent,
                1,
                kv_lora,
                eps,
                stream,
            )
        })?;

        // Step 4: K_rope + RoPE + writeback.
        // ⚠ Buffer aliasing fix: k_rope_single MUST NOT alias kv_latent.
        // Previously both used expert_gate_out, causing the k_rope GEMV
        // (64 elems) to overwrite the first 64 elements of kv_latent
        // (512 elems). The cache assemble then copied the corrupted
        // kv_latent into the K/V cache → garbage MLA decode output.
        // ssm_qkvz is free during MLA decode (only used by SSM/KDA layers).
        let k_rope_single = ctx.buffers.ssm_qkvz();
        prof!("k_rope+RoPE+wb", {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed,
                &mla.wkv_a_rope,
                k_rope_single,
                mla_rope,
                h,
                stream,
            )?;
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_k,
                q_rope_direct,
                k_rope_single,
                meta.positions,
                1,
                nq,
                1,
                mla_rope,
                mla_rope,
                mla.yarn_inv_freq,
                ctx.config.rope_theta as f32,
                stream,
            )?;
            if self.mla_q_rope_writeback_k.0 != 0 {
                ops::mla_q_rope_writeback(
                    ctx.gpu,
                    self.mla_q_rope_writeback_k,
                    q_rope_direct,
                    q_absorbed_buf,
                    nq,
                    mla_rope,
                    kv_lora,
                    mla_cache_dim,
                    stream,
                )
            } else {
                for head_idx in 0..nq as usize {
                    let src = q_rope_direct.offset(head_idx * mla.rope * 2);
                    let dst = q_absorbed_buf
                        .offset((head_idx * mla_cache_dim as usize + mla.kv_lora_rank) * 2);
                    ctx.gpu.copy_d2d_async(src, dst, mla.rope * 2, stream)?;
                }
                Ok(())
            }
        })?;
        mla_diag_norm(ctx.gpu, "kv_latent(normed)", kv_latent, kv_lora as usize);
        mla_diag_norm(ctx.gpu, "k_rope(post-rope)", k_rope_single, mla_rope as usize);

        // DIAG: print RoPE position
        if std::env::var_os("ATLAS_MLA_DIAG").is_some() && !ctx.graph_capture {
            let mut pos_b = [0u8; 4];
            ctx.gpu.copy_d2h(meta.positions, &mut pos_b)?;
            let pos0 = i32::from_le_bytes([pos_b[0], pos_b[1], pos_b[2], pos_b[3]]);
            tracing::info!("MLA-DIAG position={pos0}");
        }

        // Step 6: Cache assemble + write
        let k_cache_entry = k_out;
        let v_cache_entry = v_out;
        prof!("cache_asm+write", {
            if self.mla_cache_assemble_k.0 != 0 {
                ops::mla_cache_assemble(
                    ctx.gpu,
                    self.mla_cache_assemble_k,
                    kv_latent,
                    k_rope_single,
                    k_cache_entry,
                    v_cache_entry,
                    kv_lora,
                    mla_rope,
                    mla_cache_dim,
                    stream,
                )?;
            } else {
                ctx.gpu
                    .copy_d2d_async(kv_latent, k_cache_entry, mla.kv_lora_rank * 2, stream)?;
                ctx.gpu.copy_d2d_async(
                    k_rope_single,
                    k_cache_entry.offset(mla.kv_lora_rank * 2),
                    mla.rope * 2,
                    stream,
                )?;
                ctx.gpu
                    .copy_d2d_async(kv_latent, v_cache_entry, mla.kv_lora_rank * 2, stream)?;
                ctx.gpu.memset_async(
                    v_cache_entry.offset(mla.kv_lora_rank * 2),
                    0,
                    mla.rope * 2,
                    stream,
                )?;
            }
            self.write_kv_cache(
                ctx.gpu,
                k_cache_entry,
                v_cache_entry,
                kv_cache,
                meta.slot,
                1,
                1,
                mla_cache_dim,
                bs as u32,
                mla_cache_dim,
                mla_cache_dim,
                stream,
                ctx.graph_capture,
            )
        })?;

        // Step 8: Paged decode attention
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);

        if std::env::var_os("ATLAS_MLA_DIAG").is_some() && !ctx.graph_capture {
            ctx.gpu.synchronize(stream)?;
            // Dump the block table + seq_len so we can resolve LOGICAL block 0.
            let mut bt_b = [0u8; 32];
            ctx.gpu.copy_d2h(meta.block_table, &mut bt_b)?;
            let bt: Vec<i32> = bt_b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let mut sl_b = [0u8; 8];
            ctx.gpu.copy_d2h(meta.seq_len, &mut sl_b)?;
            let sl0 = i32::from_le_bytes([sl_b[0], sl_b[1], sl_b[2], sl_b[3]]);
            tracing::info!("MLA-DIAG block_table[0..8]={bt:?} seq_len[0]={sl0}");
            let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
            let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
            // Read logical position 0 = block_table[0], pos 0.
            let phys0 = bt[0].max(0) as u64;
            let block_stride = kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
            let k0 = k_pool.offset((phys0 * block_stride) as usize);
            let v0 = v_pool.offset((phys0 * block_stride) as usize);
            mla_diag_norm(ctx.gpu, "cache K@logical0", k0, mla_cache_dim as usize);
            mla_diag_norm(ctx.gpu, "cache V@logical0", v0, mla_cache_dim as usize);
        }

        prof!("paged_attn", {
            ops::paged_decode_attn_bf16(
                ctx.gpu,
                self.paged_decode_mla_k,
                q_absorbed_buf,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                1,
                nq,
                1,
                mla_cache_dim,
                bs as u32,
                inv_sqrt_d,
                nq * mla_cache_dim,
                0,
                stream,
            )
        })?;
        mla_diag_norm(ctx.gpu, "attn_out(paged)", attn_out, nq as usize * mla_cache_dim as usize);

        // Step 9: V extraction (batched GEMV)
        let v_extracted = ctx.buffers.norm_output();
        prof!("v_extract", {
            if self.mla_batched_gemv_k.0 != 0 {
                ops::mla_batched_gemv(
                    ctx.gpu,
                    self.mla_batched_gemv_k,
                    attn_out,
                    mla.w_uv.weight,
                    v_extracted,
                    mla_v_dim,
                    kv_lora,
                    nq,
                    mla_cache_dim,
                    mla_v_dim,
                    stream,
                )
            } else {
                for head_idx in 0..nq as usize {
                    let attn_head = attn_out.offset(head_idx * mla_cache_dim as usize * 2);
                    let w_uv_head = mla
                        .w_uv
                        .weight
                        .offset(head_idx * mla.v_dim * mla.kv_lora_rank * 2);
                    let v_dst = v_extracted.offset(head_idx * mla.v_dim * 2);
                    let w_uv_dense = crate::weight_map::DenseWeight { weight: w_uv_head };
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        attn_head,
                        &w_uv_dense,
                        v_dst,
                        mla_v_dim,
                        kv_lora,
                        stream,
                    )?;
                }
                Ok(())
            }
        })?;

        mla_diag_norm(ctx.gpu, "v_extracted", v_extracted, nq as usize * mla_v_dim as usize);

        // Step 9.5: Headwise sigmoid gate (Ling MLA)
        // vLLM bailing_moe_v3: attn_out.view(N, n_heads, v_dim) *=
        //   sigmoid(g_proj(normed)).unsqueeze(-1), then o_proj.
        // The prefill path (cache_skip_mla.rs) applies this gate but the
        // decode path was missing it, causing a 14x norm blow-up at MLA layers.
        if mla.g_proj.weight.0 != 0 {
            let gate_raw = ctx.buffers.gate_logits();
            prof!("gate_gemv", {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.g_proj,
                    gate_raw,
                    nq, // output dim = n_heads
                    h,  // input dim = hidden
                    stream,
                )
            })?;
            let gate_k = crate::layers::try_kernel(
                ctx.gpu,
                "ling_mla_attn",
                "ling_mla_headwise_gate",
            );
            if gate_k.0 != 0 {
                prof!("gate_apply", {
                    spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, gate_k)
                        .grid([1, 1, 1])
                        .block([128, 1, 1])
                        .arg_ptr(v_extracted)
                        .arg_ptr(gate_raw)
                        .arg_u32(1u32)
                        .launch(stream)
                        .map_err(|e| anyhow::anyhow!("MLA decode headwise-gate launch: {e}"))
                })?;
            } else {
                tracing::warn!(
                    "MLA decode: headwise-gate kernel missing; gate skipped (will diverge from vLLM)"
                );
            }
            mla_diag_norm(ctx.gpu, "v_extracted(gated)", v_extracted, nq as usize * mla_v_dim as usize);
        }

        // Step 10: O projection
        let o_out = ctx.buffers.qkv_output();
        prof!("wo", {
            if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    v_extracted,
                    wo_nvfp4,
                    o_out,
                    h,
                    nq * mla_v_dim,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    v_extracted,
                    &mla.wo,
                    o_out,
                    h,
                    nq * mla_v_dim,
                    stream,
                )
            }
        })?;
        mla_diag_norm(ctx.gpu, "o_out(wo)", o_out, h as usize);

        Ok(o_out)
    }
}
