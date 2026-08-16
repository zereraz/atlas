// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode (single-token).

use super::*;

static SSM_DIAG_IDX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Qwen3SsmLayer {
    pub(super) fn decode_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let debug = tracing::enabled!(tracing::Level::DEBUG);
        let trace = std::env::var("ATLAS_KDA_DIAG").map(|v| v == "1").unwrap_or(false);

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;
        // DIAG: dump hidden after rms_norm_residual (should be unchanged = embed)
        if let Ok(dir) = std::env::var("ATLAS_DECODE_DUMP") && !dir.is_empty() {
            let li = SSM_DIAG_IDX.load(std::sync::atomic::Ordering::Relaxed);
            if li < 84 {  // only first 2 decode steps (42 layers × 2)
                ctx.gpu.synchronize(stream)?;
                let mut vals = vec![0u8; h * 2];
                let _ = ctx.gpu.copy_d2h(hidden, &mut vals);
                let allf: Vec<f32> = vals.chunks_exact(2)
                    .map(|c| { let b = u16::from_le_bytes([c[0],c[1]]); f32::from_bits((b as u32) << 16) })
                    .collect();
                let bytes: Vec<u8> = allf.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(format!("ssm_L{li}_post_rms.bin")), &bytes).ok();
            }
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "pre-norm", normed, 4);
        }

        let ssm_out = self.ssm_forward(normed, ssm_state, ctx, stream, trace)?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "ssm-out", ssm_out, 4);
        }

        // Profile: time SSM vs MoE separately
        if ctx.profile {
            use std::time::Instant;
            ctx.gpu.synchronize(stream)?;
            let t0 = Instant::now();

            let normed2 = ctx.buffers.norm_output();
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                ssm_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
            if trace {
                // post-norm output for MoE decode is BF16; hidden is FP32 residual.
                Self::norm_f32_dump(ctx.gpu, "decode.moe_input_normed_bf16", normed2, h, false)?;
                Self::norm_f32_dump(ctx.gpu, "decode.post_ssm_hidden", hidden, h, ctx.config.use_fp32_residual())?;
            }
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;
            ctx.gpu.synchronize(stream)?;
            let moe_us = t0.elapsed().as_micros();
            tracing::info!("  SSM-MoE: {:.1}ms", moe_us as f64 / 1000.0);
            if trace {
                // MoE decode output is BF16 (residual_add consumes BF16).
                Self::norm_f32_dump(ctx.gpu, "decode.moe_output_bf16", moe_out, h, false)?;
            }

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                h as u32,
                stream,
            )?;
            return Ok(());
        }

        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            ssm_out,
            &self.post_attn_norm,
            normed2,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;
        // DIAG: dump hidden after residual_add_rms_norm (should be embed + ssm_out)
        if let Ok(dir) = std::env::var("ATLAS_DECODE_DUMP") && !dir.is_empty() {
            let li = SSM_DIAG_IDX.load(std::sync::atomic::Ordering::Relaxed);
            if li < 84 {
                ctx.gpu.synchronize(stream)?;
                let mut vals = vec![0u8; h * 2];
                let _ = ctx.gpu.copy_d2h(hidden, &mut vals);
                let allf: Vec<f32> = vals.chunks_exact(2)
                    .map(|c| { let b = u16::from_le_bytes([c[0],c[1]]); f32::from_bits((b as u32) << 16) })
                    .collect();
                let bytes: Vec<u8> = allf.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(format!("ssm_L{li}_post_addrms.bin")), &bytes).ok();
            }
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "post-ssm-residual", residual, 4);
            Self::debug_bf16(ctx.gpu, "post-ssm-hidden", hidden, 4);
            Self::debug_bf16(ctx.gpu, "moe-input-normed", normed2, 4);
        }
        if trace {
            ctx.gpu.synchronize(stream)?;
            Self::norm_f32_dump(ctx.gpu, "decode.moe_input_normed", normed2, h, true)?;
            Self::norm_f32_dump(ctx.gpu, "decode.post_ssm_hidden", hidden, h, ctx.config.use_fp32_residual())?;
        }

        let moe_out = self.ffn.forward(normed2, ctx, stream)?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "moe-output", moe_out, 8);
        }
        if trace {
            ctx.gpu.synchronize(stream)?;
            Self::norm_f32_dump(ctx.gpu, "decode.moe_output", moe_out, h, true)?;
        }
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            moe_out,
            h as u32,
            stream,
        )?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "final-hidden", hidden, 4);
        }

        Ok(())
    }
}
