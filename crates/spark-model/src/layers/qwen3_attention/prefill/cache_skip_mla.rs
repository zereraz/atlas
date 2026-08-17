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
#[allow(dead_code)] // `hd` is shadowed by mla_nope+mla_rope inside for correctness
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
            hd: _,
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
        // CRITICAL: MLA per-head qk dim = nope + rope (Ling: 128+64=192).
        // ctx.config.head_dim=128 is only the nope part — using it for spans
        // across the expanded q/k/v buffers corrupts every stride.
        let hd = mla_nope + mla_rope;
        // Ling MLA GEMMs hit the dense_gemm_tc kernel with M=N_tokens (=15).
        // That kernel's MMA fragment mapping was verified for M=16 but skips
        // rows 8..15 when M<16 (first-8-write-then-stale pattern observed in
        // q_expanded probe at t8..t14 zero-ish rows). Use the plain bf16 GEMM
        // (dense_gemm — Titan-generation fallback, known-correct) whenever
        // num_tokens < 16; swap back to TC only for larger prefills.
        let use_tc = self.dense_gemm_tc_k.0 != 0 && ((n as usize) % 16 == 0);

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
        }
        // MLA-expanded q = nq*(nope+rope) = 6144 bf16 per token.
        // qkv_output for Ling = m*(nq+2*nkv)*hd(cfg=128) = m*(32+64)*128 =
        // m*12288 — comfortably larger than 6144. Use it.
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
        let k_rope_buf = ctx.buffers.ssm_ba();
        // DIAG: dense_gemm_tc is suspect for small N (=rope=64) GEMMs — its TC
        // tiling assumes N>=64 per warp-pair and may drop tiles. If
        // ATLAS_MLA_FORCE_DENSE_ROPE=1, use the plain bf16 GEMM for k_rope.
        let force_dense_rope = std::env::var("ATLAS_MLA_FORCE_DENSE_ROPE").ok().as_deref() == Some("1");
        if use_tc && !force_dense_rope {
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

        let mla_cache_dim = kv_lora + mla_rope;
        // Cache assembly (needed for decode regardless of path).
        //
        // ALL models (Ling AND Mistral MLA) write the ABSORBED latent format:
        //   K cache per token: [kv_latent(512) | k_rope(64)] = mla_cache_dim=576
        //   V cache per token: [kv_latent(512) | zeros(64)]  = mla_cache_dim=576
        // This matches the KV-pool geometry (kv_lora_rank>0 → 1 head × 576 dims)
        // and matches what the decode path writes each step. The prefill
        // attention itself still uses the EXPANDED [nkv,hd] form computed below
        // (k_contiguous/v_contiguous) — the cache and the prefill-attention
        // buffers are independent representations.
        //
        // (The previous Ling arm wrote EXPANDED [nkv=32, hd=192] = 6144 elems
        // per token into a pool sized for 1×576 — overflowing slots by 10.7×
        // and leaving decode to read garbage. Root cause of the L5 explosion.)
        let meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
        let bs = kv_cache.block_size();
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
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
        // NOTE: the KV cache was ALREADY written above in absorbed latent format
        // (1 head x mla_cache_dim). k_contiguous/v_contiguous here are the
        // EXPANDED per-head forms used ONLY by the prefill FlashAttention below
        // — they are NOT written to the cache.
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
        let attn_out_fb = ctx.buffers.attn_output();
        // Ling's MLA prefill attention: qk_dim=192 != v_dim=128 — none of the
        // template-compiled inferspark kernels support that split. Use the
        // dedicated scalar kernel which writes O as [T, nq*v_dim] contiguous
        // so downstream wo-GEMM (K-dim nq*v_dim=4096) reads the right stride.
        if hd == 192 && mla_v_dim == 128 {
            let k = crate::layers::try_kernel(ctx.gpu, "ling_mla_attn", "ling_mla_prefill_attn");
            if k.0 != 0 {
                let smem = ((n as usize) * (hd as usize + mla_v_dim as usize) * 2) as u32;
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, k)
                    .grid([nq, (n + 15) / 16, 1])
                    .block([256, 1, 1])
                    .shared_mem(smem)
                    .arg_ptr(qg_out)
                    .arg_ptr(k_contiguous)
                    .arg_ptr(v_contiguous)
                    .arg_ptr(attn_out_fb)
                    .arg_u32(n)
                    .arg_u32(nq)
                    .arg_u32(nkv)
                    .arg_f32(1.0f32 / (hd as f32).sqrt())
                    .arg_u32(1)
                    .launch(stream)
                    .map_err(|e| anyhow::anyhow!("ling_mla_prefill_attn launch: {e}"))?;
                ctx.gpu.synchronize(stream)?;
                // DIAG: dump attn_out norm for last token
                if std::env::var_os("ATLAS_MLA_DIAG").is_some() {
                    let v_per_tok = nq as usize * mla_v_dim as usize;
                    let mut buf = vec![0u8; v_per_tok * 2];
                    let last_off = (n as usize - 1) * v_per_tok * 2;
                    let _ = ctx.gpu.copy_d2h(attn_out_fb.offset(last_off), &mut buf);
                    let vals: Vec<f32> = (0..v_per_tok)
                        .map(|i| { let b = u16::from_le_bytes([buf[i*2], buf[i*2+1]]); f32::from_bits((b as u32) << 16) })
                        .collect();
                    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!("PREFILL-MLA-DIAG L{} attn_out(last_tok) norm={:.4} n={}", self.attn_layer_idx, norm, n);
                    // Also dump q_full norm for last token
                    let q_per_tok = nq as usize * hd as usize;
                    let mut qbuf = vec![0u8; q_per_tok * 2];
                    let qlast = (n as usize - 1) * q_per_tok * 2;
                    let _ = ctx.gpu.copy_d2h(qg_out.offset(qlast), &mut qbuf);
                    let qvals: Vec<f32> = (0..q_per_tok)
                        .map(|i| { let b = u16::from_le_bytes([qbuf[i*2], qbuf[i*2+1]]); f32::from_bits((b as u32) << 16) })
                        .collect();
                    let qnorm: f32 = qvals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!("PREFILL-MLA-DIAG L{} q_full(last_tok) norm={:.4}", self.attn_layer_idx, qnorm);
                    // Dump K expanded for last token (correct stride: nkv * (nope + v_dim))
                    let kv_per_tok = nkv as usize * (mla_nope as usize + mla_v_dim as usize);
                    let mut kbuf = vec![0u8; kv_per_tok * 2];
                    let klast = (n as usize - 1) * kv_per_tok * 2;
                    let _ = ctx.gpu.copy_d2h(k_contiguous.offset(klast), &mut kbuf);
                    let kvals: Vec<f32> = (0..kv_per_tok)
                        .map(|i| { let b = u16::from_le_bytes([kbuf[i*2], kbuf[i*2+1]]); f32::from_bits((b as u32) << 16) })
                        .collect();
                    let knorm: f32 = kvals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!("PREFILL-MLA-DIAG L{} k_expanded(last_tok) norm={:.4}", self.attn_layer_idx, knorm);
                    // Dump V expanded for last token (correct stride: nkv * mla_v_dim)
                    let v_stride_tok = nkv as usize * mla_v_dim as usize;
                    let mut vbuf = vec![0u8; v_stride_tok * 2];
                    let vlast = (n as usize - 1) * v_stride_tok * 2;
                    let _ = ctx.gpu.copy_d2h(v_contiguous.offset(vlast), &mut vbuf);
                    let vvals: Vec<f32> = (0..v_stride_tok)
                        .map(|i| { let b = u16::from_le_bytes([vbuf[i*2], vbuf[i*2+1]]); f32::from_bits((b as u32) << 16) })
                        .collect();
                    let vnorm: f32 = vvals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!("PREFILL-MLA-DIAG L{} v_expanded(last_tok) norm={:.4}", self.attn_layer_idx, vnorm);
                }
                // ── Ling MLA headwise sigmoid gate ─────────────────────────
                // vLLM bailing_moe_v3: attn_out.view(N, n_heads, v_dim) *
                //   sigmoid(g_proj(normed)).unsqueeze(-1), then o_proj.
                // Compute gate_raw [N, 32] then fused-apply to attn_out_fb [N, 32*128].
                {
                    let n_heads = nq; // 32 for Ling
                    let gate_raw = ctx.buffers.gate_logits(); // reuse (unused in MLA)
                    ops::dense_gemm(
                        ctx.gpu, self.dense_gemm_k, normed, &mla.g_proj, gate_raw,
                        n, n_heads, h, stream,
                    )?;
                    let k = crate::layers::try_kernel(ctx.gpu, "ling_mla_attn", "ling_mla_headwise_gate");
                    if k.0 != 0 {
                        spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, k)
                            .grid([n, 1, 1])
                            .block([128, 1, 1])
                            .arg_ptr(attn_out_fb)
                            .arg_ptr(gate_raw)
                            .arg_u32(n)
                            .launch(stream)
                            .map_err(|e| anyhow::anyhow!("ling_mla_headwise_gate launch: {e}"))?;
                        ctx.gpu.synchronize(stream)?;
                    } else {
                        tracing::warn!("MLA headwise-gate kernel missing; gate skipped (will diverge from vLLM)");
                    }
                }
                // DIAG: dump gate_raw for last token
                if std::env::var_os("ATLAS_MLA_DIAG").is_some() {
                    let n_heads = nq as usize;
                    let mut gb = vec![0u8; n_heads * 2];
                    let glast = (n as usize - 1) * n_heads * 2;
                    let _ = ctx.gpu.copy_d2h(gate_raw.offset(glast), &mut gb);
                    let gv: Vec<f32> = gb.chunks_exact(2).take(8)
                        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                        .collect();
                    let sigv: Vec<f32> = gv.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                    tracing::info!("PREFILL-MLA-DIAG L{} gate_raw[:8]={:?} sigmoid[:8]={:?}", self.attn_layer_idx, gv, sigv);
                }
                let o_out = ctx.buffers.qkv_output();
                let wo_k = nq * mla_v_dim;
                if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
                    ops::w4a16_gemm(
                        ctx.gpu, self.w4a16_gemm_k, attn_out_fb, wo_nvfp4, o_out, n, h, wo_k, stream,
                    )?;
                } else {
                    ops::dense_gemm(
                        ctx.gpu, self.dense_gemm_k, attn_out_fb, &mla.wo, o_out, n, h, wo_k, stream,
                    )?;
                }
                // DIAG: dump o_out (Wo output) norm for last token
                if std::env::var_os("ATLAS_MLA_DIAG").is_some() {
                    ctx.gpu.synchronize(stream)?;
                    let mut obuf = vec![0u8; h as usize * 2];
                    let last_off = (n as usize - 1) * h as usize * 2;
                    let _ = ctx.gpu.copy_d2h(o_out.offset(last_off), &mut obuf);
                    let ovals: Vec<f32> = (0..h as usize)
                        .map(|i| { let b = u16::from_le_bytes([obuf[i*2], obuf[i*2+1]]); f32::from_bits((b as u32) << 16) })
                        .collect();
                    let onorm: f32 = ovals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    // Also dump gated attn_out norm
                    let v_per_tok = nq as usize * mla_v_dim as usize;
                    let mut gbuf = vec![0u8; v_per_tok * 2];
                    let glast = (n as usize - 1) * v_per_tok * 2;
                    let _ = ctx.gpu.copy_d2h(attn_out_fb.offset(glast), &mut gbuf);
                    let gvals: Vec<f32> = (0..v_per_tok)
                        .map(|i| { let b = u16::from_le_bytes([gbuf[i*2], gbuf[i*2+1]]); f32::from_bits((b as u32) << 16) })
                        .collect();
                    let gnorm: f32 = gvals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!("PREFILL-MLA-DIAG L{} o_out(Wo) norm={:.4} gated_attn norm={:.4}", self.attn_layer_idx, onorm, gnorm);
                }
                return Ok(o_out);
            }
        }
        let prefill_k = if hd == 192 {
            let k = crate::layers::try_kernel(ctx.gpu, "prefill_h192", "inferspark_prefill_64_h192");
            if k.0 != 0 { k } else { self.prefill_attn_64_k }
        } else {
            self.prefill_attn_64_k
        };
        ops::prefill_attention_64(
            ctx.gpu,
            prefill_k,
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
        // ── Ling MLA headwise sigmoid gate (fallback path) ──────────────
        // See comment above. Gate attn_out_fb [N, 32*128] with sigmoid of
        // gate_raw [N, 32] = dense_gemm(normed, g_proj).
        {
            let n_heads = nq;
            let gate_raw = ctx.buffers.gate_logits();
            ops::dense_gemm(
                ctx.gpu, self.dense_gemm_k, normed, &mla.g_proj, gate_raw,
                n, n_heads, h, stream,
            )?;
            let k = crate::layers::try_kernel(ctx.gpu, "ling_mla_attn", "ling_mla_headwise_gate");
            if k.0 != 0 {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, k)
                    .grid([n, 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(attn_out_fb)
                    .arg_ptr(gate_raw)
                    .arg_u32(n)
                    .launch(stream)
                    .map_err(|e| anyhow::anyhow!("ling_mla_headwise_gate launch: {e}"))?;
                ctx.gpu.synchronize(stream)?;
            } else {
                tracing::warn!("MLA headwise-gate kernel missing; gate skipped (will diverge from vLLM)");
            }
        }
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

            // Per-token max_abs scan: find which token positions are NaN/huge.
            {
                let len = (n as usize) * (h as usize);
                let mut hh = vec![0u8; len * 2];
                ctx.gpu.copy_d2h(normed, &mut hh)?;
                let mut per_tok: Vec<f32> = vec![0.0; n as usize];
                for t in 0..n as usize {
                    let base = t * h as usize;
                    let row = &hh[base * 2..(base + h as usize) * 2];
                    let mut mx = 0.0f32;
                    for c in row.chunks_exact(2) {
                        let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                        if !v.is_finite() { mx = f32::INFINITY; break; }
                        if v.abs() > mx { mx = v.abs(); }
                    }
                    per_tok[t] = mx;
                }
                tracing::info!("PREFILL-MLA normed per-token max_abs: {:?}", per_tok);
            }

            // Per-token scan of attn_out_fb (prefill_attention_64 out) AND the
            // wo-projected residual add result in hidden.
            {
                let nn = (n as usize) * (nq as usize) * (mla_v_dim as usize);
                let mut hh = vec![0u8; nn * 2];
                ctx.gpu.copy_d2h(attn_out_fb, &mut hh)?;
                let mut per_tok: Vec<f32> = Vec::with_capacity(n as usize);
                for t in 0..n as usize {
                    let base = t * (nq as usize) * (mla_v_dim as usize);
                    let row = &hh[base * 2..(base + (nq as usize) * (mla_v_dim as usize)) * 2];
                    let mut mx = 0.0f32;
                    for c in row.chunks_exact(2) {
                        let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                        if !v.is_finite() { mx = f32::INFINITY; break; }
                        if v.abs() > mx { mx = v.abs(); }
                    }
                    per_tok.push(mx);
                }
                tracing::info!("PREFILL-MLA attn_out_fb per-token max_abs: {:?}", per_tok);
            }
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
        }
        Ok(o_out)
    }
}
