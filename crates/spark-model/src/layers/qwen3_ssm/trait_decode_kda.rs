// SPDX-License-Identifier: AGPL-3.0-only

//! KDA decode path (Ling / per-channel delta rule) for `Qwen3SsmLayer::decode`.
//!
//! Invoked from [`Qwen3SsmLayer::ssm_forward`] when `self.kda_mode`. Follows
//! the FLA `fused_recurrent_kda` semantics: Q/K conv+L2-norm, f/b gates via
//! the `kda_gates` kernel, then the persistent per-channel recurrence in
//! `kda_delta_rule_decode_f32`.

use super::*;

/// BF16 bits → f32 (host-side DIAG helper).
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

impl Qwen3SsmLayer {
    /// KDA decode: replaces the gate + GDN + gated-norm + out_proj steps of the
    /// scalar-GDN `ssm_forward`. The QKVZ projection, conv1d and L2-norm steps
    /// are identical to the GDN path.
    pub(super) fn ssm_forward_kda(
        &self,
        normed: DevicePtr,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
        trace: bool,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nk = ctx.config.linear_num_key_heads;
        let nv = ctx.config.linear_num_value_heads;
        let kd = ctx.config.linear_key_head_dim;
        let vd = ctx.config.linear_value_head_dim;
        let value_dim = nv * vd;
        let bf16 = 2usize;
        let fp32 = 4usize;

        // ── 1. QKVZ projection ──────────────────────────────────────────────
        // Ling uses separate BF16 {q,k,v,g}_proj fused at load time into
        // [Q|K|V|Z] → dense_bf16 GEMV; no w4a16 variant in this path.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let qkvz_size = ctx.config.ssm_qkvz_size() as u32;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &self.ssm.in_proj_qkvz,
            deinterleaved,
            qkvz_size,
            h,
            stream,
        )?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at kda qkvz_gemv");
            })?;
        }

        // ── 2. Split QKV layout and Z pointer (same as GDN) ────────────────
        let key_dim = nk * kd;
        let conv_dim = (key_dim * 2 + value_dim) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim as u32;
        let qk_channels = (key_dim * 2) as u32;
        let qkv_ptr = deinterleaved;
        let z_ptr = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);

        // ── Buffer layout for the KDA FP32 tail ─────────────────────────
        // All five buffers live in ssm_conv_out_f32, sized so the head is conv's
        // (qk+v)*FP32 output, then f_raw/b_raw/log_decay/beta/kda_out in sequence.
        // (Previous revision aliased f_raw/b_raw on conv's head → conv1d_l2norm
        // overwrote them before kda_decode ran.)
        let conv_f32_head = ctx.buffers.ssm_conv_out_f32();
        let f_raw_bytes = nv * kd * bf16;
        let b_raw_bytes = nv * bf16;
        let conv_bytes = ((key_dim * 2 + value_dim) * fp32) as usize;
        let log_decay_bytes = nv * kd * fp32;
        let beta_bytes = nv * fp32;
        let f_raw = conv_f32_head.offset(conv_bytes);
        let b_raw = f_raw.offset(f_raw_bytes);
        let log_decay = b_raw.offset(b_raw_bytes);
        let beta = log_decay.offset(log_decay_bytes);
        let kda_out_f32 = beta.offset(beta_bytes);

        // ── 3. KDA gates: f_proj + b_proj → log_decay[nv,kd] + sigmoid(beta) ──
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &self.kda_f_proj,
            f_raw,
            (nv * kd) as u32,
            h,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &self.kda_b_proj,
            b_raw,
            nv as u32,
            h,
            stream,
        )?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at kda f/b gemv");
            })?;
            let mut fb = vec![0u8; 32];
            ctx.gpu.copy_d2h(f_raw, &mut fb)?;
            let fv: Vec<f32> = fb.chunks_exact(2).take(4)
                .map(|c| { let b = u16::from_le_bytes([c[0], c[1]]); bf16_to_f32(b) }).collect();
            let mut bb = vec![0u8; 8];
            ctx.gpu.copy_d2h(b_raw, &mut bb)?;
            let bv: Vec<f32> = bb.chunks_exact(2).take(4)
                .map(|c| { let b = u16::from_le_bytes([c[0], c[1]]); bf16_to_f32(b) }).collect();
            tracing::info!("KDA-DIAG f_raw[:4]={:?} b_raw[:4]={:?}", fv, bv);
        }

        ops::kda_gates(
            ctx.gpu,
            self.kda_gates_k,
            f_raw,
            b_raw,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            log_decay,
            beta,
            1,
            nk as u32,
            nv as u32,
            kd as u32,
            self.kda_lower_bound_f,
            stream,
        )?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at kda_gates");
            })?;
            let mut ldb = vec![0u8; 32];
            ctx.gpu.copy_d2h(log_decay, &mut ldb)?;
            let ld: Vec<f32> = ldb.chunks_exact(4).take(4)
                .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
            let mut bb = vec![0u8; 16];
            ctx.gpu.copy_d2h(beta, &mut bb)?;
            let bv: Vec<f32> = bb.chunks_exact(4).take(4)
                .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
            tracing::info!("KDA-DIAG log_decay[:4]={:?} beta[:4]={:?}", ld, bv);
        }

        // ── 4. Conv1d + SiLU + L2-norm on Q/K ─────────────────────────────
        // Exact same op as GDN (Q,K get L2; V gets SiLU only). FP32 variant.
        let (conv_out, use_f32_conv) = if self.conv1d_l2norm_f32_k.0 != 0 {
            (ctx.buffers.ssm_conv_out_f32(), true)
        } else {
            (ctx.buffers.ssm_qkvz(), false)
        };
        let conv_kernel = if use_f32_conv {
            self.conv1d_l2norm_f32_k
        } else {
            self.conv1d_l2norm_k
        };
        ops::conv1d_update_l2norm(
            ctx.gpu,
            conv_kernel,
            state.conv_state,
            qkv_ptr,
            &self.ssm.conv1d,
            conv_out,
            conv_dim,
            d_conv,
            1,
            qk_channels,
            kd as u32,
            1e-6,
            stream,
        )?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at kda conv1d_l2norm");
            })?;
            let mut qb = vec![0u8; 32];
            ctx.gpu.copy_d2h(conv_out, &mut qb)?;
            let q: Vec<f32> = if use_f32_conv {
                qb.chunks_exact(4).take(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
            } else {
                qb.chunks_exact(2).take(4).map(|c| { let b = u16::from_le_bytes([c[0], c[1]]); bf16_to_f32(b) }).collect()
            };
            tracing::info!("KDA-DIAG conv_out[:4]={:?}", q);
        }

        // ── 5. Per-channel delta rule (v-major H state), FP32 output ──────
        let elem = if use_f32_conv { 4 } else { 2 };
        let q_conv = conv_out;
        let k_conv = conv_out.offset(key_dim * elem);
        let v_conv = conv_out.offset(key_dim * 2 * elem);
        let kda_kernel = if use_f32_conv {
            self.kda_decode_f32i_k
        } else {
            self.kda_decode_k
        };
        ops::kda_decode(
            ctx.gpu,
            kda_kernel,
            state.h_state,
            q_conv,
            k_conv,
            v_conv,
            log_decay,
            beta,
            kda_out_f32,
            1,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            stream,
        )?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at kda_decode");
            })?;
            let mut ob = vec![0u8; 32];
            ctx.gpu.copy_d2h(kda_out_f32, &mut ob)?;
            let o: Vec<f32> = ob.chunks_exact(4).take(4)
                .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
            tracing::info!("KDA-DIAG kda_out[:4]={:?}", o);

            // Dump ALL intermediates to files for oracle comparison
            let layer_idx = super::trait_prefill_phase1::get_thread_layer_idx();
            let dump_dir = "/tmp/kda_decode_dump";
            let _ = std::fs::create_dir_all(dump_dir);
            // h_state: [nv, vd, kd] FP32
            let hsz = nv * vd * kd * 4;
            let mut hb = vec![0u8; hsz];
            ctx.gpu.copy_d2h(state.h_state, &mut hb)?;
            std::fs::write(format!("{dump_dir}/h_state_pre_L{layer_idx}.bin"), &hb)?;
            // q_conv, k_conv, v_conv: [nk/nv, kd] each
            let qksz = nk * kd * elem;
            let vsz = nv * vd * elem;
            let mut qb = vec![0u8; qksz];
            ctx.gpu.copy_d2h(q_conv, &mut qb)?;
            std::fs::write(format!("{dump_dir}/q_conv_L{layer_idx}.bin"), &qb)?;
            let mut kb = vec![0u8; qksz];
            ctx.gpu.copy_d2h(k_conv, &mut kb)?;
            std::fs::write(format!("{dump_dir}/k_conv_L{layer_idx}.bin"), &kb)?;
            let mut vb = vec![0u8; vsz];
            ctx.gpu.copy_d2h(v_conv, &mut vb)?;
            std::fs::write(format!("{dump_dir}/v_conv_L{layer_idx}.bin"), &vb)?;
            // log_decay: [nv, kd] FP32, beta: [nv] FP32
            let mut ldb = vec![0u8; nv * kd * 4];
            ctx.gpu.copy_d2h(log_decay, &mut ldb)?;
            std::fs::write(format!("{dump_dir}/log_decay_L{layer_idx}.bin"), &ldb)?;
            let mut bb = vec![0u8; nv * 4];
            ctx.gpu.copy_d2h(beta, &mut bb)?;
            std::fs::write(format!("{dump_dir}/beta_L{layer_idx}.bin"), &bb)?;
            // kda_out: [nv, vd] FP32
            let mut ob2 = vec![0u8; nv * vd * 4];
            ctx.gpu.copy_d2h(kda_out_f32, &mut ob2)?;
            std::fs::write(format!("{dump_dir}/kda_out_L{layer_idx}.bin"), &ob2)?;
            // h_state AFTER (re-read to see updated state)
            let mut hb2 = vec![0u8; hsz];
            ctx.gpu.copy_d2h(state.h_state, &mut hb2)?;
            std::fs::write(format!("{dump_dir}/h_state_post_L{layer_idx}.bin"), &hb2)?;
            tracing::info!("KDA-DIAG dumped all intermediates to {dump_dir}/ (L{layer_idx})");
        }

        // FP32 GDN path needs the dedicated FP32 norm kernel.
        if self.gated_rms_norm_f32_k.0 == 0 {
            anyhow::bail!("kda_mode requires gated_rms_norm_f32 kernel");
        }

        // ── 6. Gated RMS norm with Z (sigmoid) gate — FP32 GDN output ─────
        let normed_out = ctx.buffers.ssm_qkvz();
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_f32_k,
            kda_out_f32,
            z_ptr,
            &self.ssm.norm,
            normed_out,
            nv as u32,
            vd as u32,
            vd as u32,
            ctx.config.rms_norm_eps as f32,
            vd as u32,
            stream,
        )?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at kda gated_rms_norm");
            })?;
        }

        // ── 7. Output projection [value_dim → hidden_size] ────────────────
        let out = ctx.buffers.moe_output();
        if let Some(ref fp8) = self.out_proj_fp8w {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed_out,
                fp8.weight,
                fp8.row_scale,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed_out,
                dense_out,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                normed_out,
                &self.ssm.out_proj,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        }
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at kda out_proj");
            })?;
        }
        Ok(out)
    }
}
