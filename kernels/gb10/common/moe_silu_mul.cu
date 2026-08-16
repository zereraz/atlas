// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE element-wise SiLU activation + multiply.
//
// output[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x)
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)
//
// Used after grouped gate+up GEMMs to fuse activation before down GEMM.

#include <cuda_bf16.h>

extern "C" __global__ void moe_silu_mul(
    const __nv_bfloat16* __restrict__ gate,   // [total_expanded, inter_size]
    const __nv_bfloat16* __restrict__ up,     // [total_expanded, inter_size]
    __nv_bfloat16* __restrict__ output,        // [total_expanded, inter_size]
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}

// Element-wise BF16 clamp: data[i] = clamp(data[i], -limit, +limit).
// When limit <= 0, the kernel is a no-op (guard in dispatch).
// Used for Ling-3.0-flash `expert_swiglu_limit_list` (layers 35–41, limit=4.0)
// to clamp `silu(gate)*up` before the down GEMM.
extern "C" __global__ void clamp_bf16(
    __nv_bfloat16* __restrict__ data,
    float clamp_limit,
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    if (clamp_limit > 0.0f) {
        float v = __bfloat162float(data[idx]);
        v = fminf(fmaxf(v, -clamp_limit), clamp_limit);
        data[idx] = __float2bfloat16(v);
    }
}
