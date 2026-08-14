// SPDX-License-Identifier: AGPL-3.0-only

// Ling-3.0-flash MLA prefill attention: QK head dim 192 (128 nope + 64 rope),
// V head dim 128, MQA (single KV head broadcast to all query heads).
//
// Scalar BF16 reads with FP32 accumulation — the tensor-core inferspark
// template assumes qk_dim == v_dim; Ling's 192/128 split breaks it.
// Prefill here is short (<= a few thousand tokens); memory-bound scalar is fine.
//
// Q: [seq_len, nq, 192]
// K: [seq_len, 1, 192]
// V: [seq_len, 1, 128]
// O: [seq_len, nq, 128]   (contiguous, feeds the wo GEMM with K-dim nq*128)
//
// Grid: (num_q_heads, ceil(seq_len/BR), batch)  Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <float.h>

#define LING_QKD 192
#define LING_VD 128
#define LING_BR 16

extern "C" __global__ void ling_mla_prefill_attn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    unsigned int seq_len,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,   // 1 for Ling; kept for generality
    float inv_sqrt_d,
    unsigned int causal
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * LING_BR;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + LING_BR, seq_len);

    const unsigned int gqa = max(num_q_heads / max(num_kv_heads, 1u), 1u);
    const unsigned int kv_head = q_head / gqa;

    const unsigned int q_stride = num_q_heads * LING_QKD;
    const unsigned int k_stride = num_kv_heads * LING_QKD;
    const unsigned int v_stride = num_kv_heads * LING_VD;
    const unsigned int o_stride = num_q_heads * LING_VD;

    const __nv_bfloat16* Qb = Q + (unsigned long long)batch * seq_len * q_stride;
    const __nv_bfloat16* Kb = K + (unsigned long long)batch * seq_len * k_stride;
    const __nv_bfloat16* Vb = V + (unsigned long long)batch * seq_len * v_stride;
    __nv_bfloat16* Ob = O + (unsigned long long)batch * seq_len * o_stride;

    // Stage the KV sequence in shared memory (short prefills).
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sK = smem;                    // [seq_len][192]
    __nv_bfloat16* sV = smem + seq_len * LING_QKD;  // [seq_len][128]
    for (unsigned int idx = tid; idx < seq_len * LING_QKD; idx += blockDim.x) {
        unsigned int t = idx / LING_QKD, d = idx % LING_QKD;
        sK[idx] = Kb[t * k_stride + kv_head * LING_QKD + d];
    }
    for (unsigned int idx = tid; idx < seq_len * LING_VD; idx += blockDim.x) {
        unsigned int t = idx / LING_VD, d = idx % LING_VD;
        sV[idx] = Vb[t * v_stride + kv_head * LING_VD + d];
    }
    __syncthreads();

    // Each thread group handles one query row; 256 threads / 8 groups of 32.
    const unsigned int rows_per_block = blockDim.x / 32;  // 8
    const unsigned int lane = tid % 32;
    const unsigned int grp = tid / 32;

    for (unsigned int r = grp; r < q_end - q_start; r += rows_per_block) {
        unsigned int t = q_start + r;
        const __nv_bfloat16* qrow = Qb + t * q_stride + q_head * LING_QKD;

        float m = -FLT_MAX, l = 0.0f;
        float acc[LING_VD / 32];  // 128/32 = 4 per lane
        #pragma unroll
        for (int i = 0; i < LING_VD / 32; i++) acc[i] = 0.0f;

        unsigned int kv_limit = causal ? t + 1 : seq_len;
        for (unsigned int kv = 0; kv < kv_limit; kv++) {
            // dot(Q_row[192], K_row[192]) with warp reduce
            float part = 0.0f;
            for (unsigned int d = lane; d < LING_QKD; d += 32) {
                part = fmaf(__bfloat162float(qrow[d]),
                            __bfloat162float(sK[kv * LING_QKD + d]), part);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                part += __shfl_down_sync(0xffffffffu, part, off);
            float score = __shfl_sync(0xffffffffu, part, 0) * inv_sqrt_d;
            float p = 0.0f;
            if (score > -FLT_MAX / 2) {
                float m_new = fmaxf(m, score);
                float rescale = __expf(m - m_new);
                float e = __expf(score - m_new);
                l = l * rescale + e;
                #pragma unroll
                for (int i = 0; i < LING_VD / 32; i++) acc[i] *= rescale;
                m = m_new;
                p = e;
            }
            // accumulate V: lane covers V dims lane, lane+32, lane+64, lane+96
            if (p != 0.0f) {
                #pragma unroll
                for (int i = 0; i < LING_VD / 32; i++) {
                    float v = __bfloat162float(sV[kv * LING_VD + lane + 32 * i]);
                    acc[i] = fmaf(p, v, acc[i]);
                }
            }
        }
        float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
        __nv_bfloat16* orow = Ob + t * o_stride + q_head * LING_VD;
        #pragma unroll
        for (int i = 0; i < LING_VD / 32; i++)
            orow[lane + 32 * i] = __float2bfloat16(acc[i] * inv_l);
    }
}
