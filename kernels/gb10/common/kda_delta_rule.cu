// SPDX-License-Identifier: AGPL-3.0-only

// Atlas KDA (per-CHANNEL gated delta rule) — Ling / Bailing-hybrid linear
// attention. Ported 1:1 from vLLM's FLA kernels:
//   vllm/model_executor/layers/fla/ops/fused_recurrent.py
//       fused_recurrent_gated_delta_rule_fwd_kernel  (IS_KDA=true)
//   vllm/model_executor/layers/fla/ops/chunk_delta_h.py
//       chunk_gated_delta_rule_fwd_h                 (USE_GK=true)
//
// Recurrence, per token t (head vh; H stored as [v_dim, k_dim] row-major —
// v-major state to match the reference's [BV, BK] register tile):
//
//   h[v][k] *= exp(log_decay[k])                    // decay FIRST, then read
//   hk[v]    = Σ_k h[v][k] * k_t[k]
//   v'[v]    = β_t * (v_t[v] - hk[v])               // β pre-sigmoided upstream
//   h[v][k] += v'[v] * k_t[k]
//   o[v]     = Σ_k h[v][k] * q_t[k]                 // NO extra output scale
//
// log_decay[k] is the *already-transformed* log gate: the caller computes
//   log_decay[c] = -max(exp(A_log[kh(c)]) * softplus(f_raw[c] + dt_bias[c]), lower_bound)
// (FLA safe_gate semantics; lower_bound = -5 ⇒ decay ∈ [e^-5, 1]).
//
// Unlike GDN (scalar decay per head), the decay is a full k-channel vector —
// this is what makes Ling's KDA non-expressible in the scalar-GDN kernel.
//
// Grid: (num_v_heads, batch, 1); Block: (v_dim, 1, 1).

#include <cuda_bf16.h>

// ============================================================================
// DECODE (one token per launch). H persistent in global FP32.
//   log_decay: [batch, nv, k_dim] FP32
//   beta:      [batch, nv]      FP32 (already sigmoided)
// ============================================================================
extern "C" __global__ void kda_delta_rule_decode_f32(
    float* __restrict__ h_state,                  // [batch, nv, v_dim, k_dim]
    const __nv_bfloat16* __restrict__ query,      // [batch, nk, k_dim]
    const __nv_bfloat16* __restrict__ key,        // [batch, nk, k_dim]
    const __nv_bfloat16* __restrict__ value,      // [batch, nv, v_dim]
    const float* __restrict__ log_decay,          // [batch, nv, k_dim]
    const float* __restrict__ beta,               // [batch, nv]
    float* __restrict__ output,                   // [batch, nv, v_dim] FP32
    const unsigned int batch_size,
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int k_dim,
    const unsigned int v_dim
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // H[b][vh][v][k]: this thread owns row v=tid, iterates k.
    float* H = h_state + (((unsigned long long)b * num_v_heads + vh) * v_dim * k_dim)
                       + (unsigned long long)tid * k_dim;
    const __nv_bfloat16* q_ptr = query + (((unsigned long long)b * num_k_heads + kh) * k_dim);
    const __nv_bfloat16* k_ptr = key   + (((unsigned long long)b * num_k_heads + kh) * k_dim);
    const __nv_bfloat16* v_ptr = value + (((unsigned long long)b * num_v_heads + vh) * v_dim);
    const float* ld = log_decay + (((unsigned long long)b * num_v_heads + vh) * k_dim);
    const float beta_t = beta[(unsigned long long)b * num_v_heads + vh];

    if (tid >= v_dim) return;

    // Pass 1: decay H row, then hk = Σ_k H[v][k]·k[k]
    float hk = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        H[j + 0] *= __expf(ld[j + 0]);
        H[j + 1] *= __expf(ld[j + 1]);
        H[j + 2] *= __expf(ld[j + 2]);
        H[j + 3] *= __expf(ld[j + 3]);
        hk += H[j + 0] * (float)k_ptr[j + 0]
            + H[j + 1] * (float)k_ptr[j + 1]
            + H[j + 2] * (float)k_ptr[j + 2]
            + H[j + 3] * (float)k_ptr[j + 3];
    }

    // v' = β (v - hk)
    const float v_new = beta_t * ((float)v_ptr[tid] - hk);

    // Pass 2: write-back H += v' ⊗ k, accumulate output Σ_k H'·q
    float o = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        H[j + 0] += v_new * (float)k_ptr[j + 0];
        H[j + 1] += v_new * (float)k_ptr[j + 1];
        H[j + 2] += v_new * (float)k_ptr[j + 2];
        H[j + 3] += v_new * (float)k_ptr[j + 3];
        o += H[j + 0] * (float)q_ptr[j + 0]
           + H[j + 1] * (float)q_ptr[j + 1]
           + H[j + 2] * (float)q_ptr[j + 2]
           + H[j + 3] * (float)q_ptr[j + 3];
    }

    // FLA scale: q is multiplied by 1/sqrt(k_dim) before the * o step
    // (fla/ops/kda/naive.py; vLLM fused_recurrent applies it via `scale`).
    output[((unsigned long long)b * num_v_heads + vh) * v_dim + tid] =
        o * rsqrtf((float)k_dim);
}

// ============================================================================
// FP32-INPUT DECODE variant. Some pipelines produce FP32 conv outputs
// (conv1d_update_l2norm_f32) — read q/k/v as FP32 instead of BF16.
// Same H layout and numerics as the BF16-input decode above.
// ============================================================================
extern "C" __global__ void kda_delta_rule_decode_f32_inputs(
    float* __restrict__ h_state,                  // [batch, nv, v_dim, k_dim]
    const float* __restrict__ query,              // [batch, nk, k_dim] FP32
    const float* __restrict__ key,                // [batch, nk, k_dim] FP32
    const float* __restrict__ value,              // [batch, nv, v_dim] FP32
    const float* __restrict__ log_decay,          // [batch, nv, k_dim]
    const float* __restrict__ beta,               // [batch, nv]
    float* __restrict__ output,                   // [batch, nv, v_dim] FP32
    const unsigned int batch_size,
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int k_dim,
    const unsigned int v_dim
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + (((unsigned long long)b * num_v_heads + vh) * v_dim * k_dim)
                       + (unsigned long long)tid * k_dim;
    const float* q_ptr = query + (((unsigned long long)b * num_k_heads + kh) * k_dim);
    const float* k_ptr = key   + (((unsigned long long)b * num_k_heads + kh) * k_dim);
    const float* v_ptr = value + (((unsigned long long)b * num_v_heads + vh) * v_dim);
    const float* ld = log_decay + (((unsigned long long)b * num_v_heads + vh) * k_dim);
    const float beta_t = beta[(unsigned long long)b * num_v_heads + vh];

    if (tid >= v_dim) return;

    float hk = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        H[j + 0] *= __expf(ld[j + 0]);
        H[j + 1] *= __expf(ld[j + 1]);
        H[j + 2] *= __expf(ld[j + 2]);
        H[j + 3] *= __expf(ld[j + 3]);
        hk += H[j + 0] * k_ptr[j + 0]
            + H[j + 1] * k_ptr[j + 1]
            + H[j + 2] * k_ptr[j + 2]
            + H[j + 3] * k_ptr[j + 3];
    }

    const float v_new = beta_t * (v_ptr[tid] - hk);

    float o = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        H[j + 0] += v_new * k_ptr[j + 0];
        H[j + 1] += v_new * k_ptr[j + 1];
        H[j + 2] += v_new * k_ptr[j + 2];
        H[j + 3] += v_new * k_ptr[j + 3];
        o += H[j + 0] * q_ptr[j + 0]
           + H[j + 1] * q_ptr[j + 1]
           + H[j + 2] * q_ptr[j + 2]
           + H[j + 3] * q_ptr[j + 3];
    }

    output[((unsigned long long)b * num_v_heads + vh) * v_dim + tid] =
        o * rsqrtf((float)k_dim);
}

// ============================================================================
// PREFILL (sequential tokens, H resident in shared memory for the whole
// sequence). smem = (v_dim*k_dim + 2*k_dim + 1) * 4 bytes (64KB + 1KB for
// 128×128). Strides in elements, following the GDN prefill convention.
//   log_decay token t at: log_decay + t*g_stride + vh*k_dim   (g_stride=nv*k_dim)
//   beta      token t at: beta      + t*b_stride + vh         (b_stride=nv)
// ============================================================================
extern "C" __global__ void kda_delta_rule_prefill(
    float* __restrict__ h_state,                  // [batch, nv, v_dim, k_dim]
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ log_decay,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,           // [batch, seq, nv, v_dim]
    const unsigned int batch_size,
    const unsigned int seq_len,
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int k_dim,
    const unsigned int v_dim,
    const unsigned int qk_stride,
    const unsigned int v_stride,
    const unsigned int g_stride,
    const unsigned int b_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ float H_smem[];  // [v_dim, k_dim]
    float* smem_ld = H_smem + v_dim * k_dim;
    float* smem_k = smem_ld + k_dim;
    float* smem_q = smem_k + k_dim;

    float* H_global = h_state + (((unsigned long long)b * num_v_heads + vh) * v_dim * k_dim);
    for (unsigned int i = tid; i < v_dim * k_dim; i += blockDim.x) {
        H_smem[i] = H_global[i];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        const __nv_bfloat16* q_t = query + (unsigned long long)t * qk_stride + (unsigned long long)kh * k_dim;
        const __nv_bfloat16* k_t = key   + (unsigned long long)t * qk_stride + (unsigned long long)kh * k_dim;
        const __nv_bfloat16* v_t = value + (unsigned long long)t * v_stride  + (unsigned long long)vh * v_dim;
        const float* ld_t = log_decay + (unsigned long long)t * g_stride + (unsigned long long)vh * k_dim;
        const float beta_t = beta[(unsigned long long)t * b_stride + vh];

        // Stage this token's per-channel values; apply decay to this thread's
        // H row [k_dim] while staging.
        float* Hv = H_smem + (unsigned long long)tid * k_dim;
        float hk = 0.0f;
        if (tid < v_dim) {
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                Hv[j + 0] *= __expf(ld_t[j + 0]);
                Hv[j + 1] *= __expf(ld_t[j + 1]);
                Hv[j + 2] *= __expf(ld_t[j + 2]);
                Hv[j + 3] *= __expf(ld_t[j + 3]);
                const float k0 = (float)k_t[j + 0], k1 = (float)k_t[j + 1];
                const float k2 = (float)k_t[j + 2], k3 = (float)k_t[j + 3];
                smem_k[j + 0] = k0; smem_k[j + 1] = k1;
                smem_k[j + 2] = k2; smem_k[j + 3] = k3;
                hk += Hv[j + 0] * k0 + Hv[j + 1] * k1 + Hv[j + 2] * k2 + Hv[j + 3] * k3;
            }
        }
        // NOTE: every thread stages the same smem_k — redundant but correct and
        // keeps the code single-warp simple (128 threads × 128 k = full row each).
        __syncthreads();

        if (tid < v_dim) {
            const float v_new = beta_t * ((float)v_t[tid] - hk);
            float o = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                const float q0 = (float)q_t[j + 0], q1 = (float)q_t[j + 1];
                const float q2 = (float)q_t[j + 2], q3 = (float)q_t[j + 3];
                Hv[j + 0] += v_new * smem_k[j + 0];
                Hv[j + 1] += v_new * smem_k[j + 1];
                Hv[j + 2] += v_new * smem_k[j + 2];
                Hv[j + 3] += v_new * smem_k[j + 3];
                o += Hv[j + 0] * q0 + Hv[j + 1] * q1 + Hv[j + 2] * q2 + Hv[j + 3] * q3;
            }
            // FLA scale: q multiplied by 1/sqrt(k_dim).
            output[(((unsigned long long)b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(o * rsqrtf((float)k_dim));
        }
        __syncthreads();
    }

    for (unsigned int i = tid; i < v_dim * k_dim; i += blockDim.x) {
        H_global[i] = H_smem[i];
    }
}
