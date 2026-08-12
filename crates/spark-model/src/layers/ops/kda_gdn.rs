// SPDX-License-Identifier: AGPL-3.0-only

//! Ling KDA (per-channel gated delta rule) ops — FLA `chunk_kda` /
//! `fused_recurrent_kda` port. Distinct from GDN: decay is a full k-channel
//! vector per (token, head), state layout is [batch, nv, v_dim, k_dim].

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Transform raw f_proj/b_proj GEMV outputs into log-decay + sigmoid beta.
///
/// Kernel: `kda_gates(f_raw, b_raw, A_log, dt_bias, log_decay, beta,
///          nk, nv, kd, lower_bound)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn kda_gates(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    f_raw: DevicePtr,
    b_raw: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    log_decay: DevicePtr,
    beta: DevicePtr,
    num_tokens: u32,
    nk: u32,
    nv: u32,
    kd: u32,
    lower_bound: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(f_raw)
        .arg_ptr(b_raw)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(log_decay)
        .arg_ptr(beta)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(kd)
        .arg_f32(lower_bound)
        .launch(stream)
}

/// KDA single-token decode recurrence.
///
/// Kernel: `kda_delta_rule_decode_f32(h_state, q, k, v, log_decay, beta, out,
///          batch, nk, nv, kd, vd)`
/// Grid: (nv, batch, 1)  Block: (128, 1, 1)
/// h_state layout: [batch, nv, v_dim, k_dim] FP32 (v-major — differs from GDN).
#[allow(clippy::too_many_arguments)]
pub fn kda_decode(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    log_decay: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([nv, batch_size, 1])
        .block([vd, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(log_decay)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(kd)
        .arg_u32(vd)
        .launch(stream)
}

/// KDA multi-token prefill recurrence (sequential, H in shared memory).
///
/// Kernel: `kda_delta_rule_prefill(h_state, q, k, v, log_decay, beta, out,
///          batch, seq, nk, nv, kd, vd, qk_stride, v_stride, g_stride, b_stride)`
/// Grid: (nv, batch, 1)  Block: (vd, 1, 1)  smem = (vd*kd + 3*kd)*4
#[allow(clippy::too_many_arguments)]
pub fn kda_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    log_decay: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    qk_stride: u32,
    v_stride: u32,
    g_stride: u32,
    b_stride: u32,
    stream: u64,
) -> Result<()> {
    let smem = (vd * kd + 3 * kd) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([nv, batch_size, 1])
        .block([vd, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(log_decay)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(kd)
        .arg_u32(vd)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(g_stride)
        .arg_u32(b_stride)
        .launch(stream)
}
