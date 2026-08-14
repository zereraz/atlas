// SPDX-License-Identifier: AGPL-3.0-only

//! prefill_gdn_full.

use super::*;

impl Qwen3SsmLayer {
    /// KDA full-sequence prefill recurrence (Ling per-channel gated delta rule).
    ///
    /// Phase 1 already packed Q/K/V (BF16, L2-normed Q/K + SiLU'd V) into
    /// `gdn_bufs.qkv` and wrote per-channel `log_decay` (FP32) into
    /// `gdn_bufs.gate_beta` (repurposed as [total, nv*kd]) and `beta` (FP32)
    /// into the shared `ssm_gates()` scratch ([total, nv]).
    ///
    /// This launches `kda_delta_rule_prefill` (H in v-major [nv, vd, kd], same
    /// layout as decode) so h_state is directly compatible with the decode
    /// kernel. If the prompt exceeds the per-launch grid limit, it is split
    /// into sequential sub-launches that carry h_state (mathematically exact).
    pub(super) fn prefill_kda_full_inner(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.kda_prefill_k.0 == 0 {
            anyhow::bail!("KDA prefill kernel unavailable on this target");
        }
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let total = gdn_bufs.total_len;

        // Packed QKV (BF16) from phase-1 conv1d_update_prefill + l2_norm.
        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);
        // Per-channel log-decay written into gate_beta (repurposed). Stride nv*kd.
        let log_decay = gdn_bufs.gate_beta;
        let ld_stride = (nv * kd) as u32;
        // Beta scratch (ssm_gates), [total, nv] FP32, stride nv.
        let beta = ctx.buffers.ssm_gates();

        // Grid: (nv, batch, 1) — grid.y = batch = 1 (single stream). Sub-launch
        // bound only by grid.y; use a conservative 65535 per launch.
        const MAX_SEQ_PER_LAUNCH: usize = 65535;
        let mut offset = 0usize;
        while offset < total {
            let chunk = (total - offset).min(MAX_SEQ_PER_LAUNCH);
            ops::kda_prefill(
                ctx.gpu,
                self.kda_prefill_k,
                ssm_state.h_state,
                q_ptr.offset(offset * conv_dim * bf16),
                k_ptr.offset(offset * conv_dim * bf16),
                v_ptr.offset(offset * conv_dim * bf16),
                log_decay.offset(offset * nv * kd * fp32),
                beta.offset(offset * nv * fp32),
                gdn_bufs.output.offset(offset * value_dim * bf16),
                1,
                chunk as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32, // qk_stride (elements between tokens in Q/K)
                conv_dim as u32, // v_stride
                ld_stride,       // g_stride (log_decay per token = nv*kd)
                nv as u32,       // b_stride (beta per token = nv)
                stream,
            )?;
            offset += chunk;
        }

        // ATLAS_GDN_DUMP: dump h_state post-recurrence (fp32 [nv, kd, vd]) for
        // direct cos-vs-torch diff of the recurrence core.
        {
            let dir = std::env::var("ATLAS_GDN_DUMP").unwrap_or_default();
            let layers = std::env::var("ATLAS_GDN_DUMP_LAYERS").unwrap_or_default();
            static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let idx = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if !dir.is_empty() && layers.split(',').any(|s| s.trim() == idx.to_string()) {
                let bytes = nv * kd * vd * fp32;
                let mut buf = vec![0u8; bytes];
                ctx.gpu.synchronize(stream)?;
                ctx.gpu.copy_d2h(ssm_state.h_state, &mut buf)?;
                std::fs::write(format!("{}/h_state_L{idx}.bin", dir), &buf).ok();
            }
        }
        if std::env::var_os("ATLAS_MLA_DIAG").is_some() {
            ctx.gpu.synchronize(stream)?;
            // Dump q/k/v/log_decay/beta INPUT norms for first tokens (find
            // whether NaN enters before the recurrence or is created inside).
            {
                let kd_ = kd; let nv_ = nv; let conv_dim_ = conv_dim;
                let dump_norm = |name: &str, ptr: DevicePtr, elems_per_tok: usize, is_bf16: bool, ntok: usize| -> Result<()> {
                    let len = elems_per_tok * ntok;
                    let nbytes = len * if is_bf16 { 2 } else { 4 };
                    let mut h = vec![0u8; nbytes];
                    ctx.gpu.copy_d2h(ptr, &mut h)?;
                    let mut tok_max = Vec::with_capacity(ntok);
                    for t in 0..ntok {
                        let row_off = t * elems_per_tok * if is_bf16 { 2 } else { 4 };
                        let row = &h[row_off..row_off + elems_per_tok * if is_bf16 { 2 } else { 4 }];
                        let mut mx = 0.0f32; let mut nan = false;
                        if is_bf16 {
                            for c in row.chunks_exact(2) {
                                let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                                if !v.is_finite() { nan = true; break; }
                                if v.abs() > mx { mx = v.abs(); }
                            }
                        } else {
                            for c in row.chunks_exact(4) {
                                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                                if !v.is_finite() { nan = true; break; }
                                if v.abs() > mx { mx = v.abs(); }
                            }
                        }
                        tok_max.push(if nan { f32::INFINITY } else { mx });
                    }
                    let n_nan = tok_max.iter().filter(|v| !v.is_finite()).count();
                    tracing::info!("KDA-IN {name}: ntok={ntok} ept={elems_per_tok} n_nan={n_nan} tok_max={:?}", &tok_max[..tok_max.len().min(12)]);
                    Ok(())
                };
                let ntok = total.min(8);
                // q/k/v are per-token conv_dim-strided, not contiguous per-token in this packed layout
                // so dump just the first head-dim section: first kd elems of qtok row t at offset t*conv_dim
                for t in 0..ntok {
                    dump_norm("q", q_ptr.offset(t * conv_dim_ * 2), kd_, true, 1)?;
                    dump_norm("k", k_ptr.offset(t * conv_dim_ * 2), kd_, true, 1)?;
                }
                dump_norm("v", v_ptr, vd, true, 1)?;
                dump_norm("log_decay", log_decay, nv_ * kd_, false, ntok)?;
                dump_norm("beta", beta, nv_, false, ntok)?;
            }
            let bf16 = 2usize;
            let len = total * value_dim;
            let mut hh = vec![0u8; len * bf16];
            ctx.gpu.copy_d2h(gdn_bufs.output, &mut hh)?;
            let mut per_tok = Vec::with_capacity(total);
            for t in 0..total {
                let row = &hh[t * value_dim * bf16..(t + 1) * value_dim * bf16];
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
                "KDA-PREFILL-L{} recurrence out: total={total} nv={nv} vd={vd} n_nan={n_nan} per_tok_max={:?}",
                super::debug::SSM_LAYER_CALL_COUNTER.load(std::sync::atomic::Ordering::Relaxed),
                per_tok.iter().take(24).collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    pub(super) fn prefill_gdn_full_inner(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let total = gdn_bufs.total_len as u32;

        // Packed QKV layout: Q at offset 0, K at key_dim, V at key_dim*2
        // Strides: qk_stride = conv_dim, v_stride = conv_dim (elements, not bytes)
        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);

        // Gate/beta: interleaved [total_len, 2*nv] FP32.
        // (KDA mode never reaches here — it uses prefill_kda_full_inner, which
        // repurposes `gate_beta` as per-channel log_decay [total, nv*kd].)
        if ctx.config.ssm_per_channel_gates {
            anyhow::bail!(
                "GDN prefill called with ssm_per_channel_gates: gate_beta holds KDA log_decay, not interleaved gate/beta"
            );
        }
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;

        // WY32 persistent: processes 32 tokens per WY iteration with H in
        // shared memory (~84KB). ~30× faster than per-token for 14k+ sequences.
        // Falls through to WY4 or sub-chunked persistent for shorter sequences.
        tracing::info!(
            "GDN prefill: total={total} wy32_k={} wy4_k={} persistent_k={} split4_k={}",
            self.gdn_prefill_wy32_k.0 != 0,
            self.gdn_prefill_persistent_wy4_k.0 != 0,
            self.gdn_prefill_persistent_k.0 != 0,
            self.gdn_prefill_split4_k.0 != 0
        );
        if self.gdn_prefill_wy32_k.0 != 0 && total > 32 {
            let smem = (kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 32 * 32 * 4 + 256) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_wy32_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if total > 4096 {
            // Sub-chunk fallback for >4096 tokens when WY32 isn't available.
            let chunk_max = 4096u32;
            let mut offset = 0u32;
            while offset < total {
                let chunk = (total - offset).min(chunk_max);
                let q_chunk = q_ptr.offset(offset as usize * conv_dim * bf16);
                let k_chunk = k_ptr.offset(offset as usize * conv_dim * bf16);
                let v_chunk = v_ptr.offset(offset as usize * conv_dim * bf16);
                let gate_chunk = gate_ptr.offset(offset as usize * gb_stride as usize * fp32);
                let beta_chunk = beta_ptr.offset(offset as usize * gb_stride as usize * fp32);
                let out_chunk = gdn_bufs.output.offset(offset as usize * value_dim * bf16);

                if self.gdn_prefill_persistent_k.0 != 0 && chunk >= 256 {
                    ops::gdn_prefill_persistent(
                        ctx.gpu,
                        self.gdn_prefill_persistent_k,
                        ssm_state.h_state,
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        gate_chunk,
                        beta_chunk,
                        out_chunk,
                        1,
                        chunk,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        stream,
                    )?;
                } else {
                    ops::gdn_prefill_split4(
                        ctx.gpu,
                        self.gdn_prefill_split4_k,
                        ssm_state.h_state,
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        gate_chunk,
                        beta_chunk,
                        out_chunk,
                        1,
                        chunk,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        stream,
                    )?;
                }
                offset += chunk;
            }
        } else if self.gdn_prefill_persistent_wy4_k.0 != 0 {
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&total) && self.gdn_prefill_persistent_k.0 != 0 {
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        }

        Ok(())
    }

    /// Q12 Path B: batched GDN recurrence — mirrors prefill_gdn_full_inner
    /// dispatch ladder but routes to the `*_batched` kernel variants and
    /// passes `h_state_ptrs` (device array of N pointers) instead of a
    /// single h_state device pointer.
    ///
    /// Constraint: scheduler-enforced same-chunk-len across all N streams.
    /// `gdn_bufs.qkv` / `gate_beta` / `output` are stacked
    /// `[batch_size, chunk_len, *]` contiguous in memory. Each batch
    /// element's QKV starts at `b * chunk_len * conv_dim` (BF16).
    ///
    /// Validation status: kernels unvalidated against hardware.
    pub(super) fn prefill_gdn_full_batched_inner(
        &self,
        h_state_ptrs: spark_runtime::gpu::DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);
        // KDA repurposes gate_beta as per-channel log_decay — batched GDN
        // recurrence (gb_stride=nv*2) would read garbage. Not supported.
        if ctx.config.ssm_per_channel_gates {
            anyhow::bail!("batched GDN prefill incompatible with KDA per-channel gates");
        }
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;

        // Mirror the single-stream dispatch ladder. Total tokens per stream
        // is `chunk_len`; the kernel internally processes `batch_size` such
        // streams (grid dim Y).
        if self.gdn_prefill_wy32_batched_k.0 != 0 && chunk_len > 32 {
            let smem = (kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 32 * 32 * 4 + 256) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_wy32_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if self.gdn_prefill_persistent_wy4_batched_k.0 != 0 {
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&chunk_len) && self.gdn_prefill_persistent_batched_k.0 != 0
        {
            ops::gdn_prefill_persistent_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if self.gdn_prefill_split4_batched_k.0 != 0 {
            ops::gdn_prefill_split4_batched(
                ctx.gpu,
                self.gdn_prefill_split4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            anyhow::bail!(
                "Qwen3SsmLayer::prefill_gdn_full_batched_inner: no batched GDN \
                 kernel handle is loaded for this target — caller should fall \
                 back to per-stream prefill_gdn_full."
            );
        }

        Ok(())
    }
}
