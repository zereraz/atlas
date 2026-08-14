// SPDX-License-Identifier: AGPL-3.0-only

//! MLA branch of `prefill_attention_paged`. Mistral4-style 2-step
//! prefill: latent-rank Q/K/V projections, RoPE on the rope half,
//! assembled K/V for direct flash attention, compressed cache write.
//! Extracted from `paged.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

#[allow(clippy::too_many_arguments)]
pub(super) struct MlaPrefillArgs {
    pub normed: DevicePtr,
    pub num_tokens: usize,
    pub n: u32,
    pub h: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub kv_dim: usize,
    pub eps: f32,
    pub bf16: usize,
    pub bs: u32,
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    /// Run the MLA prefill kernel chain (Q latent → expand → RoPE → cache
    /// write → flash attn → O proj). Returns the O-projection output
    /// pointer (`ctx.buffers.norm_output()`).
    pub(super) fn prefill_attention_paged_mla(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
    ) -> Result<DevicePtr> {
        let MlaPrefillArgs {
            normed,
            num_tokens,
            n,
            h,
            nq,
            nkv,
            hd: _,
            kv_dim,
            eps,
            bf16,
            bs,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("prefill_attention_paged_mla called without MLA config");

        // Temporary sync checkpoints for diagnosing which MLA kernel writes
        // out-of-bounds under Ling dims. Set ATLAS_MLA_SYNC=1 to activate.
        let sync_dbg = std::env::var("ATLAS_MLA_SYNC").ok().as_deref() == Some("1");
        eprintln!("[MLA] sync_dbg={sync_dbg} layer={} n={n}", self.attn_layer_idx);
        macro_rules! sync {
            ($label:expr) => {
                if sync_dbg {
                    if let Err(e) = ctx.gpu.synchronize(stream) {
                        eprintln!("[MLA_SYNC] CORRUPTED at post-{}", $label);
                        return Err(e);
                    }
                    eprintln!("[MLA_SYNC] post-{} OK", $label);
                }
            };
        }

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        // CRITICAL: MLA per-head q/k dim = qk_nope + qk_rope (Ling: 128+64=192).
        // `ctx.config.head_dim` is 128 (only qk_nope) — using it for YOffs
        // makes every stride wrong on the expanded q/k/v buffers.
        let hd = mla_nope + mla_rope;

        // Q: latent → norm → expand → [N, nq*hd] in [nope|rope] per head
        let q_latent = ctx.buffers.ssm_ba();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wq_a,
            q_latent,
            n,
            q_lora,
            h,
            stream,
        )?;
        sync!("dense_gemm");
        // Models with no Q-compression (Ling-3.0-flash) have no q_a_layernorm.
        // Skip the rms_norm if q_a_norm was never allocated (NULL pointer).
        if mla.q_a_norm.weight.0 != 0 {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                q_latent,
                &mla.q_a_norm,
                q_latent,
                n,
                q_lora,
                eps,
                stream,
            )?;
        }
        sync!("rms_norm");
        // For MLA-expanded Q in Ling (nq*(nope+rope)=32*192=6144 per token),
        // qkv_output (sized as nq*hd(gated)*2 + 2*kv*hd) is too small when
        // attn_gated=false. norm_output is sized max_dim(=h)=2560; we need
        // 6144 per token. Allocate from attn_output: it's sized for the
        // absorbed-MLA path (nq*(kv_lora+rope) = 32*576=18432) and is
        // otherwise unused until after w_o — safe scratch here.
        let qg_out = ctx.buffers.attn_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            q_latent,
            &mla.wq_b,
            qg_out,
            n,
            nq * hd,
            q_lora,
            stream,
        )?;
        sync!("dense_gemm");

        // KV: latent → norm → expand
        let kv_latent = ctx.buffers.expert_gate_out();
        eprintln!("[MLA] layer={} PRE-KV-exp\n", self.attn_layer_idx);
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wkv_a,
            kv_latent,
            n,
            kv_lora,
            h,
            stream,
        )?;
        sync!("dense_gemm");
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            n,
            kv_lora,
            eps,
            stream,
        )?;
        sync!("rms_norm");
        let kv_expanded_dim = nkv * (mla_nope + mla_v_dim);
        let kv_expanded = ctx.buffers.ssm_deinterleaved();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            kv_latent,
            &mla.wkv_b,
            kv_expanded,
            n,
            kv_expanded_dim,
            kv_lora,
            stream,
        )?;
        sync!("dense_gemm");

        // K_rope: single shared head [N, rope=64] (MQA-style)
        let k_rope_buf = ctx.buffers.ssm_ba();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wkv_a_rope,
            k_rope_buf,
            n,
            mla_rope,
            h,
            stream,
        )?;
        sync!("dense_gemm");

        // Apply RoPE to Q rope portions and K_rope BEFORE assembly
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            qg_out,
            q_rope_tmp,
            n,
            nq,
            hd,
            mla_nope,
            mla_rope,
            nq * hd,
            stream,
        )?;
        sync!("mla_q_rope_extract_batched");
        let rope_meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_k,
            q_rope_tmp,
            k_rope_buf,
            rope_meta.positions,
            n,
            nq,
            1,
            mla_rope,
            mla_rope,
            mla.yarn_inv_freq,
            ctx.config.rope_theta as f32,
            stream,
        )?;
        sync!("rope_yarn");
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            q_rope_tmp,
            qg_out,
            n,
            nq,
            hd,
            mla_nope,
            mla_rope,
            nq * hd,
            stream,
        )?;
        sync!("mla_q_rope_writeback_batched");

        // Assemble K=[nope|rope] and extract V (1 kernel vs N*nkv*3 copies)
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        ops::mla_kv_assemble_batched(
            ctx.gpu,
            self.mla_kv_assemble_batched_k,
            kv_expanded,
            k_rope_buf,
            k_contiguous,
            v_contiguous,
            n,
            nkv,
            mla_nope,
            mla_v_dim,
            mla_rope,
            hd,
            nkv * (mla_nope + mla_v_dim),
            stream,
        )?;
        sync!("mla_kv_assemble_batched");

        // Write compressed MLA cache
        let mla_cache_dim = kv_lora + mla_rope;
        let mla_k_cache = ctx.buffers.expert_down_out();
        let mla_v_cache = mla_k_cache.offset(num_tokens * mla_cache_dim as usize * bf16);
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            k_rope_buf,
            mla_k_cache,
            mla_v_cache,
            n,
            kv_lora,
            mla_rope,
            mla_cache_dim,
            stream,
        )?;
        sync!("mla_cache_assemble_batched");

        if std::env::var_os("ATLAS_MLA_DIAG").is_some()
            && std::env::var_os("ATLAS_KDA_DIAG").is_some()
        {
            ctx.gpu.synchronize(stream)?;
            let nr = |gpu: &dyn spark_runtime::gpu::GpuBackend, tag: &str, ptr: spark_runtime::gpu::DevicePtr, len: usize| -> anyhow::Result<()> {
                let mut h = vec![0u8; len * 2];
                gpu.copy_d2h(ptr, &mut h)?;
                let mut s2 = 0.0f64;
                let mut mx = 0.0f32;
                for c in h.chunks_exact(2) {
                    let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                    s2 += (v as f64) * (v as f64);
                    if v.abs() > mx {
                        mx = v.abs();
                    }
                }
                tracing::info!("PREFILL-MLA {tag}: len={len} norm={:.4e} max_abs={:.4e}", s2.sqrt(), mx);
                Ok(())
            };
            if self.attn_layer_idx == 5 {
                let n_l = (n as usize) * (kv_lora as usize);
                nr(ctx.gpu, "kv_latent(normed)", kv_latent, n_l)?;
                let n_r = (n as usize) * (mla_rope as usize);
                nr(ctx.gpu, "k_rope(rope'd)", k_rope_buf, n_r)?;
            }
        }
        let meta = ctx.attn_metadata.expect("MLA prefill requires slot info");
        self.write_kv_cache(
            ctx.gpu,
            mla_k_cache,
            mla_v_cache,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            bs,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )?;

        // Direct flash attention with expanded Q/K/V (not from paged cache).
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let prefill_k = if hd > 256 && self.prefill_attn_512_k.0 != 0 {
            self.prefill_attn_512_k
        } else {
            self.prefill_attn_k
        };
        ops::prefill_attention(
            ctx.gpu,
            prefill_k,
            qg_out,
            k_contiguous,
            v_contiguous,
            attn_out,
            n,
            1,
            nq,
            nkv,
            hd,
            inv_sqrt_d,
            true,
            self.sliding_window.unwrap_or(0),
            stream,
        )?;
        sync!("prefill_attention");

        // O projection: [N, nq*v_dim] → [N, H]. Ling: attention output
        // buffer carries V-heads only (v_dim=128), so input K dim is
        // nq*v_dim=4096, not nq*hd=6144 (composite nope+rope head).
        let o_out = ctx.buffers.norm_output();
        let wo_k = nq * mla_v_dim;
        if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                attn_out,
                wo_nvfp4,
                o_out,
                n,
                h,
                wo_k,
                stream,
            )?;
        sync!("w4a16_gemm");
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                attn_out,
                &mla.wo,
                o_out,
                n,
                h,
                wo_k,
                stream,
            )?;
        sync!("dense_gemm");
        }
        Ok(o_out)
    }
}
