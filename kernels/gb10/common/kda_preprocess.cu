// SPDX-License-Identifier: AGPL-3.0-only

// KDA gate preprocessing: turns raw f_proj/b_proj GEMV outputs into the
// log-space per-channel decay and sigmoid beta consumed by
// `kda_delta_rule_{decode,prefill}` (FLA `chunk_kda` safe_gate semantics):
//
//   log_decay[c] = -max( exp(A_log[kh(c)]) * softplus(f_raw[c] + dt_bias[c]), lower_bound )
//   beta[h]      = sigmoid(b_raw[h])
//
// Layouts (single-token decode: T=1 grid.x=1):
//   f_raw:     [T, nv*kd]  BF16  (f_proj GEMV output)
//   b_raw:     [T, nv]     BF16  (b_proj GEMV output)
//   A_log:     [nk]        FP32
//   dt_bias:   [nv*kd]     FP32
//   log_decay: [T, nv, kd] FP32  out
//   beta:      [T, nv]     FP32  out
//
// Grid: (T, 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void kda_gates(
    const __nv_bfloat16* __restrict__ f_raw,
    const __nv_bfloat16* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ log_decay,
    float* __restrict__ beta,
    const unsigned int nk,            // num key heads (A_log length)
    const unsigned int nv,            // num value heads
    const unsigned int kd,            // k head dim (128)
    const float lower_bound           // -5.0
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int nvk = nv * kd;

    // One element per thread per stride over the f-channel space.
    for (unsigned int c = tid; c < nvk; c += blockDim.x) {
        const unsigned int vh = c / kd;
        const unsigned int kh = vh / (nv / nk);
        const float f = (float)f_raw[(unsigned long long)t * nvk + c];
        const float A = __expf(fminf(a_log[kh], 20.0f));
        const float g_in = f + dt_bias[c];
        float ld;
        if (lower_bound == 0.0f) {
            // No lower bound: FLA non-bounded path. log-decay = -exp(A)*softplus(f+dt).
            float sp = __logf(1.0f + __expf(fminf(g_in, 20.0f)));
            ld = -(A * sp);
        } else {
            // FLA `USE_LOWER_BOUND` path (Ling / KDA with `kda_lower_bound` set):
            // log_decay = lower_bound * sigmoid(exp(A) * (f + dt_bias)).
            // With lower_bound negative (e.g. -5) this yields log_decay ∈
            // (lower_bound, 0] — gated via SIGMOID, NOT softplus. Matches
            // fla/ops/kda/gate.py kda_gate_fwd_kernel USE_LOWER_BOUND branch.
            float x = A * g_in;
            ld = lower_bound / (1.0f + __expf(-x));
        }
        log_decay[(unsigned long long)t * nvk + c] = ld;
    }
    if (tid < nv) {
        const float b_r = (float)b_raw[(unsigned long long)t * nv + tid];
        beta[(unsigned long long)t * nv + tid] = 1.0f / (1.0f + __expf(-b_r));
    }
}
