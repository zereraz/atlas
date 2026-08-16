// SPDX-License-Identifier: AGPL-3.0-only

//! prefill_phase3 + alloc_state.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase3_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;

        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let value_dim = nv * vd;

        // ── 9. Gated RMS norm (batched: all chunk tokens × heads) ──
        // Read GDN output and Z from full-sequence buffers at token_offset.
        let gdn_out_chunk = gdn_bufs.output.offset(token_offset * value_dim * bf16);
        let z_chunk = gdn_bufs.z.offset(token_offset * value_dim * bf16);

        // Output buffer: reuse ssm_qkvz (same as monolithic prefill)
        let normed_out_buf = ctx.buffers.ssm_qkvz();
        ops::gated_rms_norm_prefill(
            ctx.gpu,
            self.gated_rms_norm_prefill_k,
            gdn_out_chunk,
            z_chunk,
            &self.ssm.norm,
            normed_out_buf,
            nv as u32,
            vd as u32,
            eps,
            k,
            value_dim as u32, // input_token_stride: GDN output is [N, value_dim] contiguous
            value_dim as u32, // gate_token_stride: Z buffer is [N, value_dim] contiguous
            stream,
        )?;

        // ATLAS_GDN_DUMP: post gated_rms_norm (o_g) [N, value_dim] bf16
        {
            let dir = std::env::var("ATLAS_GDN_DUMP").unwrap_or_default();
            if !dir.is_empty() {
                let tl = super::trait_prefill_phase1::get_thread_layer_idx();
                let li = if tl != usize::MAX { tl } else { 0 };
                let mut buf = vec![0u8; num_tokens * value_dim * bf16];
                ctx.gpu.synchronize(stream)?;
                ctx.gpu.copy_d2h(normed_out_buf, &mut buf)?;
                std::fs::write(format!("{dir}/kda_o_g_L{li}.bin"), &buf).ok();
            }
        }

        // ── 10. Output projection GEMM: [N, 4096] × [4096, 2048] → [N, 2048] ──
        let out_proj_buf = ctx.buffers.moe_output();
        let use_dense_out = std::env::var("ATLAS_DENSE_QKVZ").ok().as_deref() == Some("1");
        if use_dense_out {
            if let Some(ref dense_out) = self.out_proj_dense {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    normed_out_buf,
                    dense_out,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            } else {
                // Env set but no dense fallback installed: fall through to NVFP4.
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed_out_buf,
                    self.out_proj_nvfp4_t.as_ref().unwrap(),
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            }
        } else if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_out_buf,
                dense_out,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if let Some(fp8) = self.out_proj_fp8 {
            if k > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            }
        } else if let Some(ref nvfp4_t) = self.out_proj_nvfp4_t {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t_k,
                normed_out_buf,
                nvfp4_t,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        }
        .map_err(|e| anyhow::anyhow!("ssm phase3: out_proj GEMM failed: {e}"))?;

        // ATLAS_GDN_DUMP: post o_proj (a_out) [N, h] bf16
        {
            let dir = std::env::var("ATLAS_GDN_DUMP").unwrap_or_default();
            if !dir.is_empty() {
                let tl = super::trait_prefill_phase1::get_thread_layer_idx();
                let li = if tl != usize::MAX { tl } else { 0 };
                let mut buf = vec![0u8; num_tokens * h * bf16];
                ctx.gpu.synchronize(stream)?;
                ctx.gpu.copy_d2h(out_proj_buf, &mut buf)?;
                std::fs::write(format!("{dir}/kda_a_out_L{li}.bin"), &buf).ok();
            }
        }

        if std::env::var_os("ATLAS_KDA_DIAG").is_some() && num_tokens > 1 {
            ctx.gpu.synchronize(stream)?;
            // Norm/Gate/GDN raw-input sanity: print per-token norms of GDN raw,
            // gated-normalized out, and o_proj for t=0..num_tokens.
            let bf16 = 2usize;
            let vdim = value_dim;
            let read_norms = |buf: DevicePtr, elems_per_tok: usize| -> Result<Vec<f32>> {
                let len = num_tokens * elems_per_tok;
                let mut hh = vec![0u8; len * bf16];
                ctx.gpu.copy_d2h(buf, &mut hh)?;
                let mut v = Vec::with_capacity(num_tokens);
                for t in 0..num_tokens {
                    let row = &hh[t * elems_per_tok * bf16..(t + 1) * elems_per_tok * bf16];
                    let mut ss = 0f64;
                    for c in row.chunks_exact(2) {
                        let x = half::bf16::from_le_bytes([c[0], c[1]]).to_f32() as f64;
                        ss += x * x;
                    }
                    v.push(ss.sqrt() as f32);
                }
                Ok(v)
            };
            if let Ok(dir) = std::env::var("ATLAS_GDN_DUMP") {
                let len = num_tokens * vdim;
                let mut gb = vec![0u8; len * bf16];
                ctx.gpu.copy_d2h(gdn_out_chunk, &mut gb)?;
                let mut floats: Vec<f32> = Vec::with_capacity(len);
                for c in gb.chunks_exact(2) {
                    floats.push(half::bf16::from_le_bytes([c[0], c[1]]).to_f32());
                }
                let bytes: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::create_dir_all(&dir).ok();
                use std::sync::atomic::{AtomicUsize, Ordering};
                static CTR: AtomicUsize = AtomicUsize::new(0);
                let n = CTR.fetch_add(1, Ordering::SeqCst);
                std::fs::write(
                    std::path::Path::new(&dir).join(format!("kda_gdn_{}", n)),
                    &bytes,
                ).ok();
            }
            let gdn_norms = read_norms(gdn_out_chunk, vdim)?;
            let z_norms = read_norms(z_chunk, vdim)?;
            let normed_norms = read_norms(normed_out_buf, vdim)?;
            tracing::info!(
                "KDA-PH3-NORMS gdn={gdn_norms:?} z={z_norms:?} normed={normed_norms:?}",
            );
            let bf16 = 2usize;
            let len = num_tokens * h;
            let mut hh = vec![0u8; len * bf16];
            ctx.gpu.copy_d2h(out_proj_buf, &mut hh)?;
            let mut per_tok = Vec::with_capacity(num_tokens);
            for t in 0..num_tokens {
                let row = &hh[t * h * bf16..(t + 1) * h * bf16];
                let mut mx = 0.0f32;
                for c in row.chunks_exact(2) {
                    let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                    if !v.is_finite() { mx = f32::INFINITY; break; }
                    if v.abs() > mx { mx = v.abs(); }
                }
                per_tok.push(mx);
            }
            let n_nan = per_tok.iter().filter(|v| !v.is_finite()).count();
            tracing::info!(
                "KDA-PHASE3 out_proj per_tok: total={num_tokens} n_nan={n_nan} per_tok_max={:?}",
                per_tok.iter().take(24).collect::<Vec<_>>()
            );
        }

        // ── 11. Batched residual + post-norm + MoE ──
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            out_proj_buf,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            num_tokens as u32,
            h as u32,
            eps,
            stream,
        )?;
        self.ffn
            .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
        // L5-MoE-localization: per-token scan of the FFN/MoE continuation
        // output BEFORE residual_add, so an exploded routed-expert position
        // (norm 1e14) is identifiable per-position without hidden muddying it.
        if std::env::var_os("ATLAS_KDA_DIAG").is_some() && std::env::var_os("ATLAS_MLA_DIAG").is_some() && num_tokens > 1 {
            ctx.gpu.synchronize(stream)?;
            let bf16 = 2usize;
            let len = num_tokens * h;
            let mut hh = vec![0u8; len * bf16];
            ctx.gpu.copy_d2h(ctx.buffers.moe_output(), &mut hh)?;
            let mut per_tok = Vec::with_capacity(num_tokens);
            for t in 0..num_tokens {
                let row = &hh[t * h * bf16..(t + 1) * h * bf16];
                let mut mx = 0.0f32;
                for c in row.chunks_exact(2) {
                    let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                    if !v.is_finite() { mx = f32::INFINITY; break; }
                    if v.abs() > mx { mx = v.abs(); }
                }
                per_tok.push(mx);
            }
            let n_nan = per_tok.iter().filter(|v| !v.is_finite()).count();
            tracing::info!(
                "KDA-PHASE3 moe_output per_tok: total={num_tokens} n_nan={n_nan} per_tok_max={:?}",
                per_tok.iter().take(24).collect::<Vec<_>>()
            );
        }
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            ctx.buffers.moe_output(),
            (num_tokens * h) as u32,
            stream,
        )?;

        if std::env::var_os("ATLAS_KDA_DIAG").is_some() && num_tokens > 1 {
            ctx.gpu.synchronize(stream)?;
            let bf16 = 2usize;
            let len = num_tokens * h;
            let mut hh = vec![0u8; len * bf16];
            ctx.gpu.copy_d2h(hidden, &mut hh)?;
            let mut per_tok = Vec::with_capacity(num_tokens);
            for t in 0..num_tokens {
                let row = &hh[t * h * bf16..(t + 1) * h * bf16];
                let mut mx = 0.0f32;
                for c in row.chunks_exact(2) {
                    let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                    if !v.is_finite() { mx = f32::INFINITY; break; }
                    if v.abs() > mx { mx = v.abs(); }
                }
                per_tok.push(mx);
            }
            let n_nan = per_tok.iter().filter(|v| !v.is_finite()).count();
            tracing::info!(
                "KDA-PHASE3 hidden-post-moe per_tok: total={num_tokens} n_nan={n_nan} per_tok_max={:?}",
                per_tok.iter().take(24).collect::<Vec<_>>()
            );
        }

        Ok(())
    }

    pub(super) fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_state = gpu.alloc(self.h_state_bytes)?;
        gpu.memset(h_state, 0, self.h_state_bytes)?;
        let conv_state = gpu.alloc(self.conv_state_bytes)?;
        gpu.memset(conv_state, 0, self.conv_state_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
        }))
    }
}
