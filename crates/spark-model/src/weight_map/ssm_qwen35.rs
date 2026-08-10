// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Qwen3.5 SSM weights with separate projections.
pub struct SsmWeightsQwen35 {
    /// QKV projection: [qkv_size, hidden_size] BF16 (Q+K+V, no Z).
    pub in_proj_qkv: DenseWeight,
    /// Z gate projection: [z_size, hidden_size] BF16.
    pub in_proj_z: DenseWeight,
    /// Alpha projection: [num_value_heads, hidden_size] BF16.
    pub in_proj_a: DenseWeight,
    /// Beta projection: [num_value_heads, hidden_size] BF16.
    pub in_proj_b: DenseWeight,
    /// Conv1d weight: [d_inner, 1, d_conv] BF16.
    pub conv1d: DenseWeight,
    /// A_log parameter: `[num_v_heads]` FP32.
    pub a_log: DenseWeight,
    /// dt_bias parameter: `[num_v_heads]` FP32.
    pub dt_bias: DenseWeight,
    /// Gate norm weight: `[value_dim]` BF16.
    pub norm: DenseWeight,
    /// Output projection: [value_dim, hidden_size] BF16 (NOT NVFP4 — quantizer skipped these).
    pub out_proj: DenseWeight,
}

/// Load SSM weights for Qwen3.5 (separate projections, BF16 out_proj).
pub(crate) fn load_ssm_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<SsmWeightsQwen35> {
    let p = format!("{layer_prefix}.linear_attn");

    // For FP8 models: in_proj_qkv, in_proj_z, out_proj are FP8 block-scaled.
    // conv1d, in_proj_a, in_proj_b are BF16 (in modules_to_not_convert).
    let load_proj = |name: &str| -> Result<DenseWeight> {
        match variant {
            Nvfp4Variant::Fp8Dequanted => dense_auto(store, name, gpu),
            _ => dense(store, name),
        }
    };

    Ok(SsmWeightsQwen35 {
        in_proj_qkv: load_proj(&format!("{p}.in_proj_qkv.weight"))?,
        in_proj_z: load_proj(&format!("{p}.in_proj_z.weight"))?,
        in_proj_a: dense(store, &format!("{p}.in_proj_a.weight"))?,
        in_proj_b: dense(store, &format!("{p}.in_proj_b.weight"))?,
        conv1d: dense(store, &format!("{p}.conv1d.weight"))?,
        // A_log and dt_bias MUST be FP32 — BF16 precision causes exponential
        // error amplification in the GDR decay gate at 8k+ tokens.
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        // norm.weight is safe as BF16 (no recurrent amplification)
        norm: dense_f32_safe(store, &format!("{p}.norm.weight"), gpu)?,
        out_proj: load_proj(&format!("{p}.out_proj.weight"))?,
    })
}

/// Load KDA (KimiDeltaAttention) weights for Ling-3.0-flash.
///
/// Ling's tensor naming differs from Qwen3.5's GDN; this assembles the same
/// `SsmWeightsQwen35` structure from Ling's `{lp}.attention.*` names:
///
///   q_proj/k_proj/v_proj  → in_proj_qkv ([(qk+kv+vv), h] after concat)
///   f_proj (hidden→proj) → in_proj_b (gate; `no_kda_lora=true` → 1 matrix)
///   g_proj (hidden→proj) → in_proj_z (output gate; Qwen3.5 z)
///   b_proj (hidden→heads) → in_proj_a (beta)
///   q_conv1d/k_conv1d/v_conv1d → stacked conv1d [d, 1, kernel]
///   o_proj → out_proj
///
/// Ling's conv weights are 3 separate 1D kernels: fuse into Q|K|V order.
pub(crate) fn load_ssm_bailing(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    _variant: Nvfp4Variant,
) -> Result<SsmWeightsQwen35> {
    let p = format!("{layer_prefix}.attention");

    // Ling's in_proj tensors are all BF16 (in modules_to_not_convert).
    let qkv_combined = {
        let q_ptr = dense(store, &format!("{p}.q_proj.weight"))?;
        let k_ptr = dense(store, &format!("{p}.k_proj.weight"))?;
        let v_ptr = dense(store, &format!("{p}.v_proj.weight"))?;
        (q_ptr, k_ptr, v_ptr)
    };
    let k_conv = dense(store, &format!("{p}.k_conv1d.weight"))?;
    let q_conv = dense(store, &format!("{p}.q_conv1d.weight"))?;
    let v_conv = dense(store, &format!("{p}.v_conv1d.weight"))?;
    let f_proj = dense(store, &format!("{p}.f_proj.weight"))?;
    let g_proj = dense(store, &format!("{p}.g_proj.weight"))?;
    let b_proj = dense(store, &format!("{p}.b_proj.weight"))?;
    let o_proj = dense(store, &format!("{p}.o_proj.weight"))?;
    let o_norm = dense(store, &format!("{p}.o_norm.weight"))?;

    // Stack conv1d: q_conv + k_conv + v_conv → [3*d, 1, kernel] in Q|K|V order.
    let conv_shape = &store.get(&format!("{p}.q_conv1d.weight"))?.shape;
    let kernel = *conv_shape.get(2).unwrap_or(&1usize);
    let d_conv = conv_shape[0] * kernel * 2; // bytes per conv row
    let conv_buf = gpu.alloc(d_conv * 3)?;
    gpu.copy_d2d(q_conv.weight, conv_buf, d_conv)?;
    gpu.copy_d2d(k_conv.weight, conv_buf.offset(d_conv), d_conv)?;
    gpu.copy_d2d(v_conv.weight, conv_buf.offset(d_conv * 2), d_conv)?;
    let conv1d = DenseWeight { weight: conv_buf };

    // Fuse q/k/v into one [qkv, h] weight.
    let q_shape = &store.get(&format!("{p}.q_proj.weight"))?.shape;
    let k_shape = &store.get(&format!("{p}.k_proj.weight"))?.shape;
    let v_shape = &store.get(&format!("{p}.v_proj.weight"))?.shape;
    let h = q_shape[1];
    let q_rows = q_shape[0];
    let k_rows = k_shape[0];
    let v_rows = v_shape[0];
    // BF16 = 2 bytes per element: allocation and copy lengths must be in bytes.
    let bf16 = 2usize;
    let qkv_buf = gpu.alloc((q_rows + k_rows + v_rows) * h * bf16)?;
    gpu.copy_d2d(qkv_combined.0.weight, qkv_buf, q_rows * h * bf16)?;
    gpu.copy_d2d(
        qkv_combined.1.weight,
        qkv_buf.offset(q_rows * h * bf16),
        k_rows * h * bf16,
    )?;
    gpu.copy_d2d(
        qkv_combined.2.weight,
        qkv_buf.offset((q_rows + k_rows) * h * bf16),
        v_rows * h * bf16,
    )?;
    let in_proj_qkv = DenseWeight { weight: qkv_buf };

    Ok(SsmWeightsQwen35 {
        in_proj_qkv,
        in_proj_z: g_proj,
        in_proj_a: b_proj,
        in_proj_b: f_proj,
        conv1d,
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        norm: o_norm,
        out_proj: o_proj,
    })
}

/// Load MoE weights for Qwen3.5, auto-selecting NVFP4 naming convention.
///
/// Under EP (ep_world_size > 1), only local experts are loaded from the store.
/// Remote experts get NULL pointers — kernels detect NULL and write zero output.
/// `skip_routed_experts`: when true, routed experts get NULL weights (saves memory
/// when native FP8 MoE dispatch handles them). Shared expert is always loaded.
pub(crate) fn load_moe_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    skip_routed_experts: bool,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense(store, &format!("{p}.gate.weight"))?;
    let shared_expert_gate = dense(store, &format!("{p}.shared_expert_gate.weight"))?;

    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;

    let load_bf16_then_nvfp4 = |full_prefix: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        let bf16 = dense(store, &format!("{full_prefix}.weight"))?;
        quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)
    };

    // Qwen3.6-35B-A3B BF16 release ships a FUSED MoE layout: one
    // `experts.gate_up_proj: [num_experts, 2*inter, hidden]` and one
    // `experts.down_proj: [num_experts, hidden, inter]` per layer. Slice
    // each expert at load time and runtime-quantize to NVFP4.
    let fused_gate_up_key = format!("{p}.experts.gate_up_proj");
    let fused_down_key = format!("{p}.experts.down_proj");
    let is_fused_bf16 = variant == Nvfp4Variant::Bf16Raw
        && store.contains(&fused_gate_up_key)
        && store.contains(&fused_down_key);

    let load_expert_fused = |expert_idx: usize| -> Result<ExpertWeight> {
        // gate_up: [num_experts, 2*inter, hidden] BF16
        let fused_gu = store.get(&fused_gate_up_key)?;
        // down: [num_experts, hidden, inter] BF16
        let fused_d = store.get(&fused_down_key)?;
        let bf16 = 2usize;
        let gu_per_expert_bytes = 2 * inter * h * bf16;
        let d_per_expert_bytes = h * inter * bf16;
        let gate_off = expert_idx * gu_per_expert_bytes;
        let up_off = gate_off + inter * h * bf16;
        let down_off = expert_idx * d_per_expert_bytes;
        let gate_dw = DenseWeight {
            weight: fused_gu.ptr.offset(gate_off),
        };
        let up_dw = DenseWeight {
            weight: fused_gu.ptr.offset(up_off),
        };
        let down_dw = DenseWeight {
            weight: fused_d.ptr.offset(down_off),
        };
        Ok(ExpertWeight {
            gate_proj: quantize_to_nvfp4(&gate_dw, inter, h, gpu, absmax_k, quantize_k, stream)?,
            up_proj: quantize_to_nvfp4(&up_dw, inter, h, gpu, absmax_k, quantize_k, stream)?,
            down_proj: quantize_to_nvfp4(&down_dw, h, inter, gpu, absmax_k, quantize_k, stream)?,
        })
    };

    let load_expert = |prefix: &str| -> Result<ExpertWeight> {
        match variant {
            Nvfp4Variant::Bf16Raw => Ok(ExpertWeight {
                gate_proj: load_bf16_then_nvfp4(&format!("{prefix}.gate_proj"), inter, h)?,
                up_proj: load_bf16_then_nvfp4(&format!("{prefix}.up_proj"), inter, h)?,
                down_proj: load_bf16_then_nvfp4(&format!("{prefix}.down_proj"), h, inter)?,
            }),
            Nvfp4Variant::Mxfp4Dequanted => {
                let mx = |name: &str, n, k| {
                    let bf16 = DenseWeight {
                        weight: dequant_mxfp4_to_bf16(store, name, gpu)?,
                    };
                    let q = quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)?;
                    gpu.free(bf16.weight)?;
                    Ok::<QuantizedWeight, anyhow::Error>(q)
                };
                Ok(ExpertWeight {
                    gate_proj: mx(&format!("{prefix}.gate_proj"), inter, h)?,
                    up_proj: mx(&format!("{prefix}.up_proj"), inter, h)?,
                    down_proj: mx(&format!("{prefix}.down_proj"), h, inter)?,
                })
            }
            Nvfp4Variant::Fp8Dequanted => Ok(ExpertWeight {
                gate_proj: quantized_from_fp8(
                    store,
                    &format!("{prefix}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
                up_proj: quantized_from_fp8(
                    store,
                    &format!("{prefix}.up_proj"),
                    inter,
                    h,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
                down_proj: quantized_from_fp8(
                    store,
                    &format!("{prefix}.down_proj"),
                    h,
                    inter,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
            }),
            _ => Ok(ExpertWeight {
                gate_proj: quantized_auto(store, &format!("{prefix}.gate_proj"), gpu, variant)?,
                up_proj: quantized_auto(store, &format!("{prefix}.up_proj"), gpu, variant)?,
                down_proj: quantized_auto(store, &format!("{prefix}.down_proj"), gpu, variant)?,
            }),
        }
    };

    let shared_expert = load_expert(&format!("{p}.shared_expert"))?;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if skip_routed_experts || !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
        } else if is_fused_bf16 {
            experts.push(load_expert_fused(e)?);
        } else {
            experts.push(load_expert(&format!("{p}.experts.{e}"))?);
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: None,
    })
}

/// Load MoE experts as native FP8 weights (no NVFP4 conversion).
///
/// Returns the standard MoeWeights (with NVFP4 gate/shared for compatibility)
/// PLUS a Vec of Fp8ExpertWeight for native FP8 dispatch.
pub(crate) fn load_moe_qwen35_fp8_experts(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> Result<Vec<Fp8ExpertWeight>> {
    let p = format!("{layer_prefix}.mlp");
    let mut fp8_experts = Vec::with_capacity(num_experts);

    for e in 0..num_experts {
        if config.is_local_expert(e) {
            let ep = format!("{p}.experts.{e}");
            fp8_experts.push(Fp8ExpertWeight {
                gate_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{ep}.gate_proj"),
                    gpu,
                )?,
                up_proj: load_fp8_block_scaled_as_fp8weight(store, &format!("{ep}.up_proj"), gpu)?,
                down_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{ep}.down_proj"),
                    gpu,
                )?,
            });
        } else {
            fp8_experts.push(Fp8ExpertWeight {
                gate_proj: Fp8Weight {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                },
                up_proj: Fp8Weight {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                },
                down_proj: Fp8Weight {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                },
            });
        }
    }

    // Also load shared expert as FP8
    let shared_prefix = format!("{p}.shared_expert");
    let _shared_fp8 = Fp8ExpertWeight {
        gate_proj: load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{shared_prefix}.gate_proj"),
            gpu,
        )?,
        up_proj: load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{shared_prefix}.up_proj"),
            gpu,
        )?,
        down_proj: load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{shared_prefix}.down_proj"),
            gpu,
        )?,
    };

    Ok(fp8_experts)
}

/// Load MoE weights for models without shared experts (e.g. Qwen3-VL).
///
/// Creates zero-filled dummy shared expert weights so the fused MoE kernels
/// (which always launch top_k+1 blocks) produce zero contribution from the
/// shared expert slot. `weight_scale_2 = 0.0` ensures dequant → 0.
pub(crate) fn load_moe_no_shared(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense(store, &format!("{p}.gate.weight"))?;

    // Allocate correctly-sized zero-filled GPU buffers for dummy shared expert.
    // The fused kernel always runs a shared expert block (blockIdx.y == top_k),
    // which reads full expert-sized weight matrices. Buffers must match real
    // expert dimensions or the kernel will read out of bounds (CUDA error 900).
    // weight_scale_2 = 0.0 ensures dequant → 0 regardless of packed contents.
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let group_size = 16usize; // NVFP4 quantization group size (matches kernel GROUP_SIZE)

    // gate_proj/up_proj: [inter, h] → packed = inter * h / 2, scale = inter * (h / group_size)
    let gu_packed_bytes = inter * h / 2;
    let gu_scale_bytes = inter * (h / group_size);
    // down_proj: [h, inter] → packed = h * inter / 2, scale = h * (inter / group_size)
    let d_packed_bytes = h * inter / 2;
    let d_scale_bytes = h * (inter / group_size);

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let ptr = gpu.alloc(size)?;
        gpu.memset(ptr, 0, size)?;
        Ok(ptr)
    };

    let mk_zero_quant = |packed_sz: usize, scale_sz: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(packed_sz)?,
            weight_scale: alloc_zero(scale_sz)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
        })
    };

    let shared_expert = ExpertWeight {
        gate_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        up_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        down_proj: mk_zero_quant(d_packed_bytes, d_scale_bytes)?,
    };
    // Gate weight for shared expert: zero BF16 [hidden_size] → sigmoid(0)=0.5.
    // Doesn't matter since shared_out is all zeros (0.5 * 0 = 0).
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            experts.push(ExpertWeight {
                gate_proj: quantized_auto(
                    store,
                    &format!("{p}.experts.{e}.gate_proj"),
                    gpu,
                    variant,
                )?,
                up_proj: quantized_auto(store, &format!("{p}.experts.{e}.up_proj"), gpu, variant)?,
                down_proj: quantized_auto(
                    store,
                    &format!("{p}.experts.{e}.down_proj"),
                    gpu,
                    variant,
                )?,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: None,
    })
}

/// Load MoE weights for Ling-3.0-flash (bailing_hybrid).
///
/// Identical structure to `load_moe_qwen35` but for Ling's sparse+shared MoE:
///   - `gate.weight`: the routing scorer — read as BF16.
///   - NO `shared_expert_gate.weight` (Ling has no separate shared-expert
///     gating projection — the shared expert's contribution is unweighted).
///     Allocate zero-filled dummy so the fused MoE kernel produces zero from
///     the shared slot (`weight_scale_2 = 0.0` forces dequant → 0).
///   - `gate.expert_bias`: DeepSeek-V3-style loss-free routing bias; add to
///     `correction_bias` (`sigmoid(scores) + bias` for selection only).
pub(crate) fn load_moe_bailing(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    skip_routed_experts: bool,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense(store, &format!("{p}.gate.weight"))?;
    // Ling has no `shared_expert_gate` — pass a NULL DenseWeight so the MoE
    // forward's null-check short-circuits the shared expert's contribution
    // to zero (Ling's shared expert isn't gating-balanced).
    let shared_expert_gate = DenseWeight {
        weight: spark_runtime::gpu::DevicePtr::NULL,
    };

    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;
    let load_expert = |prefix: &str| -> Result<ExpertWeight> {
        match variant {
            Nvfp4Variant::Bf16Raw => {
                let bf16 = |name: &str, n, k| {
                    let d = dense(store, &format!("{prefix}.{name}.weight"))?;
                    quantize_to_nvfp4(&d, n, k, gpu, absmax_k, quantize_k, stream)
                };
                Ok(ExpertWeight {
                    gate_proj: bf16("gate_proj", inter, h)?,
                    up_proj: bf16("up_proj", inter, h)?,
                    down_proj: bf16("down_proj", h, inter)?,
                })
            }
            Nvfp4Variant::Mxfp4Dequanted => {
                let mx = |name: &str, n, k| {
                    let bf16 = DenseWeight {
                        weight: dequant_mxfp4_to_bf16(store, &format!("{prefix}.{name}"), gpu)?,
                    };
                    let q = quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)?;
                    gpu.free(bf16.weight)?;
                    Ok::<QuantizedWeight, anyhow::Error>(q)
                };
                Ok(ExpertWeight {
                    gate_proj: mx("gate_proj", inter, h)?,
                    up_proj: mx("up_proj", inter, h)?,
                    down_proj: mx("down_proj", h, inter)?,
                })
            }
            _ => Ok(ExpertWeight {
                gate_proj: quantized_auto(store, &format!("{prefix}.gate_proj"), gpu, variant)?,
                up_proj: quantized_auto(store, &format!("{prefix}.up_proj"), gpu, variant)?,
                down_proj: quantized_auto(store, &format!("{prefix}.down_proj"), gpu, variant)?,
            }),
        }
    };

    let shared_expert = load_expert(&format!("{p}.shared_experts"))?;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if skip_routed_experts || !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
        } else {
            experts.push(load_expert(&format!("{p}.experts.{e}"))?);
        }
    }

    // Ling uses a per-layer `gate.expert_bias` (loss-free routing).
    let correction_bias = if store.contains(&format!("{p}.gate.expert_bias")) {
        Some(dense(store, &format!("{p}.gate.expert_bias"))?)
    } else if config.use_routing_bias {
        // routed_bias config says bias exists — should be in the store; if
        // not, the shared gate trick ensures correctness either way.
        Some(dense(store, &format!("{p}.gate.expert_bias"))?)
    } else {
        None
    };

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias,
    })
}
