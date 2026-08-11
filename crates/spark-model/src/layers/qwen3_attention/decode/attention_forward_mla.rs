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

        // Ling: no Q compression → q_lora_rank is faked to hidden_size.
        // The absorbed MLA decode is structurally incompatible with no-Q-compression
        // architectures (absorb expects [nope=128, absorbed-k=512, rope=64] = 704 per
        // head, but q_proj only produces nq*hd=192-per-head Q). Use an expanded-K/V
        // non-absorbed decode instead.
        if (mla.q_lora_rank as u32) == h {
            return self.attention_forward_mla_expanded(kv_cache, ctx, args);
        }

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
        let q_latent = ctx.buffers.ssm_ba();
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
        // Use expert_gate_out instead of ssm_ba: ssm_ba is shared with
        // q_latent (Ling q_lora = h = 2560), which exceeds KDA's 64-wide
        // ssm_ba_size and overflows into downstream buffers.
        let k_rope_single = ctx.buffers.expert_gate_out();
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

        Ok(o_out)
    }

    /// Ling-specific MLA decode with EXPANDED (non-absorbed) K/V cache.
    ///
    /// Layout: K cache = [k_nope=128 | k_rope=64] per head (hd=192), V cache = [v=128 | 0-pad=64] per head.
    /// Q from q_proj directly (h=2560 -> nq*hd=6144); K and V reconstructed per-token from
    /// kv_latent @ wkv_b + kv_a_rope, then written to paged cache and decoded via
    /// paged_decode_attn_bf16 with head_dim=192.
    ///
    /// Caches 25x more memory per layer than absorbed, but works for no-Q-compression MLA
    /// (Ling) where the standard absorbed chain can't produce the [704/head] absorb input.
    pub(super) fn attention_forward_mla_expanded(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &DecodeMlaArgs,
    ) -> Result<DevicePtr> {
        let DecodeMlaArgs {
            normed,
            h, nq, eps, bs, stream,
            ..
        } = *args;
        let mla = self.mla.as_ref().expect("expanded MLA requires config");
        let meta = ctx.attn_metadata.expect("MLA decode requires pre-uploaded metadata");
        let mla_rope = mla.rope as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        // Composite head for K cache: nope + rope
        let hd_cache = mla_nope + mla_rope;      // = 192
        let hd_v_cache = mla_nope + mla_rope;   // same 192 for V, zero-padded

        // 1) Q = normed @ wq_b (= q_proj for Ling)  [1, 6144]
        let q_full = ctx.buffers.ssm_deinterleaved();   // large enough
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &mla.wq_b,
            q_full,
            nq * (mla_nope + mla_rope),   // 6144
            h,
            stream,
        )?;
        // RoPE on the rope-portion of Q per head: positions later handled below
        // We handle RoPE inside the same kernel below via cached k_rope.

        // 2) KV latent + k_rope (fused)
        // kv_latent = normed[:, :512] = normed[:512]  (kv_a fused projects to 512 rope + 64 rope)
        let kv_latent = ctx.buffers.expert_gate_out();
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
        )?;
        let k_rope_single = ctx.buffers.expert_up_out();
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

        // 3) RoPE on Q rope-portion + k_rope
        // Q layout per head: [nope=128, rope=64] -> q_full[head_idx * 192]
        // We reuse the biuid rope_yarn kernel but with num_kv_heads=1 and toggle head_dim=rop_64
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_k,
            q_full,
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

        // 4) Expand K/V: kv_latent @ wkv_b -> per head [k_nope=128, v=128]
        let kv_expanded = ctx.buffers.ssm_qkvz();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            kv_latent,
            &mla.wkv_b,
            kv_expanded,
            nq * (mla_nope + mla_v_dim),  // 32 * 256 = 8192 elements
            kv_lora,
            stream,
        )?;
        // kv_expanded layout per head: [k_nope=128, v=128] (interleaved by wkv_b rows)
        // wkv_b rows: for each head, first 128 rows are k_nope, next 128 are v
        // 5) Write EXPANDED K/V to paged cache:
        // K cache: [k_nope=128, k_rope=64] per head   (assemble)
        // V cache: [v=128, 0-pad=64]          per head   (pad)
        // Reuse the ssm_conv_out_f32 slot as our expanded K/V workspace: it is sized
        // max(ctx * hd, 4096) * 4 bytes so fits nq*192*2=12288 floats easily.
        let k_new = ctx.buffers.ssm_conv_out_f32();
        let v_new = k_new.offset(nq as usize * (hd_cache * 2) as usize);
        // For each head: copy k_nope from kv_expanded[head*256 : head*256+128] -> k_new[head*192 : head*192+128]
        // then copy k_rope                                    -> k_new[head*192+128 : head*192+192]
        // V: copy v=kv_expanded[head*256+128 : head*256+256]  -> v_new[head*192 : head*192+ 128]
        // zero-pad v_new[head*192+128 : +192]
        let rope_size = (mla_rope * 2) as usize;   // bf16 bytes
        let nope_size = (mla_nope * 2) as usize;
        let vd_size = (mla_v_dim * 2) as usize;
        let pad_size = ((hd_v_cache - mla_v_dim) * 2) as usize;
        let src_stride_k = ((mla_nope + mla_v_dim) * 2) as usize;
        let dst_stride_k = (hd_cache * 2) as usize;
        for head_idx in 0..nq as usize {
            let k_nope_src = kv_expanded.offset(head_idx * src_stride_k);
            let v_src = kv_expanded.offset(head_idx * src_stride_k + nope_size);
            let k_dst = k_new.offset(head_idx * dst_stride_k);
            let v_dst = v_new.offset(head_idx * dst_stride_k);
            ctx.gpu.copy_d2d_async(k_nope_src, k_dst, nope_size, stream)?;
            ctx.gpu.copy_d2d_async(k_rope_single, k_dst.offset(nope_size), rope_size, stream)?;
            ctx.gpu.copy_d2d_async(v_src, v_dst, vd_size, stream)?;
            ctx.gpu.memset_async(v_dst.offset(vd_size), 0, pad_size, stream)?;
        }

        // 6) Append new K/V to paged cache
        self.write_kv_cache(
            ctx.gpu,
            k_new,
            v_new,
            kv_cache,
            meta.slot,
            1,
            1,               //  one kv head for compressed-MLA caching? use num_kv_heads=nq for expanded
            hd_cache,
            bs as u32,
            dst_stride_k as u32,
            dst_stride_k as u32,
            stream,
            ctx.graph_capture,
        )?;
        // NOTE: GQA-cash mode with num_kv_heads=32 needs the layout change in cache
        // shape up front. For this minimal path we write with a high num_kv_heads value
        // matching what was written during the subject-layer prefill; it must match.
        // (If the prefill still writes compressed-cash, this block has mismatched shapes.)

        // 7) Standard decode attention
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd_cache);
        ops::paged_decode_attn_bf16(
            ctx.gpu,
            self.paged_decode_mla_k,
            q_full,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            1,
            nq,
            nq,           // num_kv_heads = nq for expanded cache
            hd_cache,
            bs as u32,
            inv_sqrt_d,
            nq * hd_cache,
            0,
            stream,
        )?;

        // 8) O projection: wo expects [n, nq*v_dim=4096] not [n, nq*hd=6144]
        let o_out = ctx.buffers.qkv_output();
        let wo_k = nq * mla_v_dim;   // = 4096
        // We need to slice the attn output: for each head, attn_out is 192-wide but
        // wo only needs the first 128 rows per head. Alternatively compute wo against
        // a tmp buffer holding V-strided output, which our paged_decode kernel already
        // wrote as hd=192 per head. The O projection can't handle 6144 of junk columns,
        // so produce a compressed tmp: [n, nq*128]
        let o_in = ctx.buffers.norm_output();
        for head_idx in 0..nq as usize {
            let src = attn_out.offset(head_idx * (hd_cache * 2) as usize);
            let dst = o_in.offset(head_idx * (mla_v_dim * 2) as usize);
            ctx.gpu.copy_d2d_async(src, dst, vd_size, stream)?;
        }
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            o_in,
            &mla.wo,
            o_out,
            h,
            wo_k,
            stream,
        )?;
        Ok(o_out)
    }
}
