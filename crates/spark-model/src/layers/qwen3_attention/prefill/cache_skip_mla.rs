// SPDX-License-Identifier: AGPL-3.0-only

//! MLA branch of `prefill_attention_with_cache_skip`. Mistral4-style
//! 2-step prefill with the unabsorbed/MHA fused fallback path that
//! expands K/V via `wkv_b` and runs HDIM=128 FlashAttention. Extracted
//! from `cache_skip.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

#[allow(clippy::too_many_arguments)]
pub(super) struct CacheSkipMlaArgs {
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
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    /// Run the cache-skip MLA prefill chain. Always returns the output
    /// pointer — caller short-circuits with `return Ok(out)`.
    pub(super) fn prefill_attention_cache_skip_mla(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &CacheSkipMlaArgs,
    ) -> Result<DevicePtr> {
        let CacheSkipMlaArgs {
            normed,
            num_tokens,
            n,
            h,
            nq,
            nkv,
            hd,
            kv_dim,
            eps,
            bf16,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("prefill_attention_cache_skip_mla called without MLA config");

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        let use_tc = self.dense_gemm_tc_k.0 != 0;

        // Q: latent → norm → expand
        let q_latent = ctx.buffers.ssm_ba();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wq_a,
                q_latent,
                n,
                q_lora,
                h,
                stream,
            )?;
        eprintln!("[MLA-CS] post-dense_gemm_tc"); ctx.gpu.synchronize(stream)?;
        } else {
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
        eprintln!("[MLA-CS] post-dense_gemm"); ctx.gpu.synchronize(stream)?;
        }
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
        eprintln!("[MLA-CS] post-rms_norm"); ctx.gpu.synchronize(stream)?;
        }
        let qg_out = ctx.buffers.qkv_output();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                q_latent,
                &mla.wq_b,
                qg_out,
                n,
                nq * hd,
                q_lora,
                stream,
            )?;
        eprintln!("[MLA-CS] post-dense_gemm_tc"); ctx.gpu.synchronize(stream)?;
        } else {
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
        eprintln!("[MLA-CS] post-dense_gemm"); ctx.gpu.synchronize(stream)?;
        }

        // KV latent + K_rope
        let kv_latent = ctx.buffers.expert_gate_out();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wkv_a,
                kv_latent,
                n,
                kv_lora,
                h,
                stream,
            )?;
        eprintln!("[MLA-CS] post-dense_gemm_tc"); ctx.gpu.synchronize(stream)?;
        } else {
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
        eprintln!("[MLA-CS] post-dense_gemm"); ctx.gpu.synchronize(stream)?;
        }
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
        eprintln!("[MLA-CS] post-rms_norm"); ctx.gpu.synchronize(stream)?;
        let k_rope_buf = ctx.buffers.ssm_ba();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wkv_a_rope,
                k_rope_buf,
                n,
                mla_rope,
                h,
                stream,
            )?;
        eprintln!("[MLA-CS] post-dense_gemm_tc"); ctx.gpu.synchronize(stream)?;
        } else {
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
        eprintln!("[MLA-CS] post-dense_gemm"); ctx.gpu.synchronize(stream)?;
        }

        // Q rope extract → RoPE
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
        eprintln!("[MLA-CS] post-mla_q_rope_extract_batched"); ctx.gpu.synchronize(stream)?;
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
        eprintln!("[MLA-CS] post-rope_yarn"); ctx.gpu.synchronize(stream)?;

        let mla_cache_dim = kv_lora + mla_rope;
        // Cache assembly (needed for decode regardless of path)
        let meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
        let bs = kv_cache.block_size();
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
        // Ling (no Q compression): store EXPANDED K/V in cache so decode
        // can read it directly without absorption. K cache: per-head [k_nope=128,
        // k_rope=64]=192; V cache: per-head [v=128, 0-pad=64]=192.
        if (mla.q_lora_rank as u32) == h {
            // Compute k_contiguous/v_contiguous BEFORE writing to cache so we reuse
            // them below after kv expansion. Move the expand+assemble block up here.
            // (Arrives later via restructure below.)
        } else {
            ops::mla_cache_assemble_batched(
                ctx.gpu,
                self.mla_cache_assemble_batched_k,
                kv_latent,
                k_rope_buf,
                k_cache_assembled,
                v_cache_assembled,
                n,
                kv_lora,
                mla_rope,
                mla_cache_dim,
                stream,
            )?;
            self.write_kv_cache(
                ctx.gpu,
                k_cache_assembled,
                v_cache_assembled,
                kv_cache,
                meta.slot,
                n,
                1,
                mla_cache_dim,
                bs as u32,
                mla_cache_dim,
                mla_cache_dim,
                stream,
                ctx.graph_capture,
            )?;
        }

        // Unabsorbed (MHA) prefill: expand K/V via wkv_b, use HDIM=128 FlashAttention
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
        eprintln!("[MLA-CS] post-dense_gemm"); ctx.gpu.synchronize(stream)?;
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
        eprintln!("[MLA-CS] post-mla_kv_assemble_batched"); ctx.gpu.synchronize(stream)?;
        // Ling: write expanded K/V to cache (192/head).
        if (mla.q_lora_rank as u32) == h {
            // k_contiguous is [n, nq*192] with per-head [k_nope=128, k_rope=64];
            // for cache correctly we need V padded to 192. Instead of a new pad,
            // write K with hd=192 and reuse the same buffer location for V rows,
            // then pad the tail in place before writing.
            // For V, assebmle mla_kv_assemble_batched already wrote 128/head --
            // do an quick in-place pad: shift each head's 128 visitng rows to their
            // new 192-per-head locs within v_contiguous and zero the tails.
            let hd_c: usize = (mla_nope + mla_rope) as usize;
            let vd_bytes = (mla_v_dim as usize) * bf16;
            let n_tokens = n as usize;
            // If two-half columns: for each (token, head), V-row is at
            // token*nq*128 + head*128, target is at token*nq*192 + head*192
            let src_stride = (nkv as usize) * (mla_v_dim as usize);
            let dst_stride = (nkv as usize) * hd_c;
            let scratch_v = ctx.buffers.expert_down_out();
            for t in 0..n_tokens {
                for head in 0..nkv as usize {
                    let src = v_contiguous.offset(t * src_stride * bf16 + head * vd_bytes);
                    let dst = scratch_v.offset(t * dst_stride * bf16 + head * hd_c * bf16);
                    ctx.gpu.copy_d2d_async(src, dst, vd_bytes, stream)?;
                    let pad: usize = (hd_c - mla_v_dim as usize) * bf16;
                    if pad > 0 {
                        ctx.gpu.memset_async(dst.offset(vd_bytes), 0, pad, stream)?;
                    }
                }
            }
            self.write_kv_cache(
                ctx.gpu,
                k_contiguous,
                scratch_v,
                kv_cache,
                meta.slot,
                n,
                nq,
                hd_c as u32,
                bs as u32,
                (nq as usize * hd_c) as u32,
                (nq as usize * hd_c) as u32,
                stream,
                ctx.graph_capture,
            )?;
        }
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
        eprintln!("[MLA-CS] post-mla_q_rope_writeback_batched"); ctx.gpu.synchronize(stream)?;
        let attn_out_fb = ctx.buffers.attn_output();
        ops::prefill_attention_64(
            ctx.gpu,
            self.prefill_attn_64_k,
            qg_out,
            k_contiguous,
            v_contiguous,
            attn_out_fb,
            n,
            1,
            nq,
            nkv,
            hd,
            1.0f32 / (hd as f32).sqrt(),
            true,
            0,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("MLA flash_attn_64 fallback: {e}"))?;
        eprintln!("[MLA-CS] post-prefill_attention_64"); ctx.gpu.synchronize(stream)?;
        if std::env::var_os("ATLAS_MLA_DIAG").is_some() && self.attn_layer_idx == 5 {
            ctx.gpu.synchronize(stream)?;
            let dump = |tag: &str, ptr: DevicePtr, len: usize| -> anyhow::Result<()> {
                let mut hh = vec![0u8; len * 2];
                ctx.gpu.copy_d2h(ptr, &mut hh)?;
                let mut s2 = 0.0f64;
                let mut mx = 0.0f32;
                let mut first = [0.0f32; 4];
                for (i, c) in hh.chunks_exact(2).enumerate() {
                    let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                    s2 += (v as f64) * (v as f64);
                    if v.abs() > mx { mx = v.abs(); }
                    if i < 4 { first[i] = v; }
                }
                tracing::info!("PREFILL-MLA {tag}: len={len} norm={:.4e} max_abs={:.4e} first4={:?}", s2.sqrt(), mx, first);
                Ok(())
            };
            dump("kv_latent(normed)", kv_latent, (n as usize) * (kv_lora as usize))?;
            dump("k_rope(rope'd)", k_rope_buf, (n as usize) * (mla_rope as usize))?;
            dump("k_contiguous", k_contiguous, (n as usize) * (nq as usize) * (hd as usize))?;
            dump("q_full(qg_out)", qg_out, (n as usize) * (nq as usize) * (hd as usize))?;
            dump("normed", normed, (n as usize) * (h as usize))?;
        }

        // wo projection — output to qkv_output (norm_output aliases downstream).
        // Ling: Q/K head_dim is the composite 192 (nope+rope) but V heads are
        // v_dim=128 wide, so the attention output buffer is [n, nq*v_dim] and
        // the wo GEMM's input K dim is nq*v_dim (=4096), NOT nq*hd (=6144).
        let o_out = ctx.buffers.qkv_output();
        let wo_k = nq * mla_v_dim;
        if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                attn_out_fb,
                wo_nvfp4,
                o_out,
                n,
                h,
                wo_k,
                stream,
            )?;
        eprintln!("[MLA-CS] post-w4a16_gemm"); ctx.gpu.synchronize(stream)?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                attn_out_fb,
                &mla.wo,
                o_out,
                n,
                h,
                wo_k,
                stream,
            )?;
        eprintln!("[MLA-CS] post-dense_gemm"); ctx.gpu.synchronize(stream)?;
        }
        Ok(o_out)
    }
}
