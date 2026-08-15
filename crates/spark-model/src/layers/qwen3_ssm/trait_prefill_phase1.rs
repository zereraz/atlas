// SPDX-License-Identifier: AGPL-3.0-only

//! is_ssm_layer + prefill_phase1.

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn is_ssm_layer_inner(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase1_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();

        // ENTRY diagnostic sync removed: it stalled the GPU pipeline at every
        // SSM layer entry, killing async kernel pipelining. Errors will surface
        // at the natural end-of-prefill sync.

        // ── 1. RMS norm + residual for N tokens ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            k,
            h as u32,
            eps,
            stream,
        )?;

        // ── 2+3. QKVZ GEMM (+ deinterleave if needed) ──
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        if let Some(fp8) = self.qkvz_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                normed,
                fp8,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm phase1: QKVZ FP8 GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else if std::env::var("ATLAS_DENSE_QKVZ").ok().as_deref() == Some("1") {
            // Correctness probe: bypass NVFP4 (which HF never quantizes for
            // Ling) and run the KDA QKVZ GEMM in BF16 to isolate quant noise
            // from logic error during layer-diff vs FLA oracle.
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.ssm.in_proj_qkvz,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            if k > 128 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm phase1: QKVZ m128 GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm phase1: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm phase1: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.ssm.in_proj_qkvz,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        if !self.sequential_qkvz {
            ops::deinterleave_qkvz(
                ctx.gpu,
                self.deinterleave_k,
                proj_dst,
                deinterleaved,
                k,
                nk as u32,
                kd as u32,
                vpg as u32,
                vd as u32,
                stream,
            )?;
        }

        // ── 4+5. Gates: GDN fused-BA + gates, or KDA f/b GEMM + kda_gates ──
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2;
        if self.kda_mode {
            // KDA (Ling per-channel gated delta rule): f/b are two independent
            // dense GEMMs from `normed` (NOT the fused interleaved BA weight).
            // Stage raw BF16 GEMM outputs in the tail of ssm_conv_out_f32 (head
            // is free pre-conv1d — conv1d_update_prefill targets ssm_qkvz).
            //   log_decay → gdn_bufs.gate_beta  [total, nv*kd] FP32 (repurposed)
            //   beta      → gates_buf (ssm_gates) scratch [N, nv] FP32
            let fb_stage = ctx.buffers.ssm_conv_out_f32();
            let f_raw = fb_stage;
            let b_raw = f_raw.offset(num_tokens * nv * kd * bf16);
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.kda_f_proj,
                f_raw,
                k,
                (nv * kd) as u32,
                h as u32,
                stream,
            )?;
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.kda_b_proj,
                b_raw,
                k,
                nv as u32,
                h as u32,
                stream,
            )?;
            // Per-channel log-decay + sigmoid(beta) for all N tokens (grid.x=N).
            let log_decay_dst = gdn_bufs.gate_beta.offset(token_offset * nv * kd * fp32);
            ops::kda_gates(
                ctx.gpu,
                self.kda_gates_k,
                f_raw,
                b_raw,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                log_decay_dst,
                gates_buf, // beta scratch [N, nv] FP32
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                self.kda_lower_bound_f,
                stream,
            )?;
            // ATLAS_GDN_DUMP: diag — raw f/b GEMM outputs before kda_gates
            {
                let dir = std::env::var("ATLAS_GDN_DUMP").unwrap_or_default();
                if !dir.is_empty() {
                    ctx.gpu.synchronize(stream)?;
                    let mut buf = vec![0u8; num_tokens * nv * kd * bf16];
                    ctx.gpu.copy_d2h(f_raw, &mut buf)?;
                    std::fs::write(format!("{dir}/kda_f_raw_L.bin"), &buf).ok();
                    let mut bbuf = vec![0u8; num_tokens * nv * bf16];
                    ctx.gpu.copy_d2h(b_raw, &mut bbuf)?;
                    std::fs::write(format!("{dir}/kda_b_raw_L.bin"), &bbuf).ok();
                    let mut nbuf = vec![0u8; num_tokens * h * bf16];
                    ctx.gpu.copy_d2h(normed, &mut nbuf)?;
                    std::fs::write(format!("{dir}/kda_normed_L.bin"), &nbuf).ok();
                }
            }
            if std::env::var_os("ATLAS_MLA_DIAG").is_some() {
                ctx.gpu.synchronize(stream)?;
                let mut ld = vec![0u8; (nv * kd).min(8) * 4];
                let _ = ctx.gpu.copy_d2h(log_decay_dst, &mut ld);
                let v: Vec<f32> = ld.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
                let mut bb = vec![0u8; nv.min(8) * 4];
                let _ = ctx.gpu.copy_d2h(gates_buf, &mut bb);
                let bv: Vec<f32> = bb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
                let mut fb0 = vec![0u8; 16];
                let _ = ctx.gpu.copy_d2h(f_raw, &mut fb0);
                let fv: Vec<f32> = fb0.chunks_exact(2).map(|c|{let b=u16::from_le_bytes([c[0],c[1]]); f32::from_bits((b as u32)<<16)}).collect();
                tracing::info!("KDA-PH1-N{} log_decay[0..8]={:?} beta[0..8]={:?} f_raw[0..8]={:?}", num_tokens, v, bv, fv);
            }
        } else {
            let ba_size = ctx.config.ssm_ba_size();
            ops::dense_gemm_ba_gates_prefill(
                ctx.gpu,
                self.ba_gates_prefill_k,
                normed,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates_buf,
                k,
                ba_size as u32,
                h as u32,
                h as u32,
                gate_stride as u32,
                nv as u32,
                vpg as u32,
                stream,
            )?;
        }

        // ── 6. Batched conv1d for all N tokens ──
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            ssm_state.conv_state,
            deinterleaved,
            &self.ssm.conv1d,
            DevicePtr::NULL,
            conv_out_buf,
            conv_dim as u32,
            d_conv as u32,
            k,
            qkvz_size as u32,
            conv_dim as u32,
            stream,
        )?;

        // ATLAS_GDN_DUMP: post-conv1d+silu, pre-L2norm (oracle diff point)
        let ssm_layer_idx = {
            static SSM_CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            SSM_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        };
        {
            let dir = std::env::var("ATLAS_GDN_DUMP").unwrap_or_default();
            let layers = std::env::var("ATLAS_GDN_DUMP_LAYERS").unwrap_or_default();
            let li = ssm_layer_idx % 36;
            if !dir.is_empty() && layers.split(',').any(|s| s.trim() == li.to_string()) {
                let mut buf = vec![0u8; num_tokens * conv_dim * 2];
                ctx.gpu.synchronize(stream)?;
                ctx.gpu.copy_d2h(conv_out_buf, &mut buf)?;
                let p = format!("{}/post_conv_L{li}.bin", dir);
                std::fs::write(&p, &buf).ok();
                let mut buf2 = vec![0u8; num_tokens * qkvz_size * 2];
                ctx.gpu.copy_d2h(deinterleaved, &mut buf2)?;
                let p2 = format!("{}/pre_conv_L{li}.bin", dir);
                std::fs::write(&p2, &buf2).ok();
            }
        }
        // ── 7. Batched L2 norm on Q,K for all N tokens ──
        ops::l2_norm(
            ctx.gpu,
            self.l2_norm_k,
            conv_out_buf,
            (nk * 2) as u32,
            kd as u32,
            1e-6,
            k,
            conv_dim as u32,
            stream,
        )?;

        // ── 8. Copy GDN inputs to full-sequence buffers ──
        // QKV: conv_out_buf [num_tokens, conv_dim] BF16 → gdn_bufs.qkv at token_offset
        // This is a contiguous copy because both layouts are [N, conv_dim].
        let qkv_dst = gdn_bufs.qkv.offset(token_offset * conv_dim * bf16);
        ctx.gpu
            .copy_d2d_async(conv_out_buf, qkv_dst, num_tokens * conv_dim * bf16, stream)?;
        // ATLAS_GDN_DUMP: post-L2 packed qkv as FEED TO kda_delta_rule_prefill.
        {
            let dir = std::env::var("ATLAS_GDN_DUMP").unwrap_or_default();
            let layers = std::env::var("ATLAS_GDN_DUMP_LAYERS").unwrap_or_default();
            let idx8 = ssm_layer_idx % 36;
            if !dir.is_empty() && layers.split(',').any(|s| s.trim() == idx8.to_string()) {
                let mut buf8 = vec![0u8; num_tokens * conv_dim * bf16];
                ctx.gpu.synchronize(stream)?;
                ctx.gpu.copy_d2h(conv_out_buf, &mut buf8)?;
                std::fs::write(format!("{}/post_l2_qkv_L{idx8}.bin", dir), &buf8).ok();
            }
        }

        // Gate/beta: gates_buf [num_tokens, 2*nv] FP32 → gdn_bufs.gate_beta at token_offset
        // Contiguous copy: both layouts are [N, 2*nv] FP32.
        // (KDA wrote log_decay directly into gdn_bufs.gate_beta above — skip.)
        if !self.kda_mode {
            let gb_dst = gdn_bufs.gate_beta.offset(token_offset * gate_stride * fp32);
            ctx.gpu
                .copy_d2d_async(gates_buf, gb_dst, num_tokens * gate_stride * fp32, stream)?;
        }

        // Z gate: deinterleaved [num_tokens, qkvz_size] BF16, Z at offset (key_dim*2 + value_dim).
        // Z stride in source = qkvz_size, Z stride in dest = value_dim.
        // Strided copy: one per-token D2D async call.
        let z_src_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        let z_dst_base = gdn_bufs.z.offset(token_offset * value_dim * bf16);
        let z_elem_bytes = value_dim * bf16;
        for t in 0..num_tokens {
            let z_src = z_src_base.offset(t * qkvz_size * bf16);
            let z_dst = z_dst_base.offset(t * value_dim * bf16);
            ctx.gpu.copy_d2d_async(z_src, z_dst, z_elem_bytes, stream)?;
        }

        Ok(())
    }
}
