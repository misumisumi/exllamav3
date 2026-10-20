#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define MOE_ACT_SILU 0
#define MOE_ACT_GELU 1
#define MOE_ACT_RELU2_NOGATE 2  // non-gated relu2 (NemotronH): gate GEMM and staging skipped

#define MOE_SMS_PER_EXPERT 8       // default/minimum group width, also sets max concurrency (buffer count)
#define MOE_MAX_SMS_PER_EXPERT 32  // widest expert group when few experts are active
#define MOE_TILESIZE_K 16
#define MOE_TILESIZE_M 16          // default row tile; the 32 / 64 row tiles are separate kernel instances
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3

#ifndef EXL3_GEMM_BASE_THREADS
#define EXL3_GEMM_BASE_THREADS 256
#endif

#ifndef SMEM_MAX
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6
#endif

// Exact dynamic shared-memory footprint of one mixed-K CTA, in bytes.
//
// This mirrors the layout in exl3_gemm_inner.cuh exactly:
//
//   extern __shared__ half shared[];
//   half*     sh_a = shared;                                     SH * sh_a_stage_size
//   uint16_t* sh_b = (uint16_t*) (sh_a + SH * sh_a_stage_size);   SH * sh_b_stage_size
//   float*    sh_c = (float*)    (sh_b + SH * sh_b_stage_size);   sh_c_size
//
// sh_a/sh_b are ring buffers indexed by (slice_iters % SH); sh_c is a single buffer
// (the MoE path passes shmem_out_had = false, and at MOE_TILESIZE_K == 16
// TILEBLOCKS_K == 1 so threadblock_reduce() compiles to nothing). The three extents
// below are therefore the entire reachable range, and the launcher requests exactly
// this many bytes instead of SMEM_MAX. MAXBITS (not the launch's K) sets the sh_b
// extent because one instance serves every K <= MAXBITS at run time.
#ifndef MOE_A_STAGE_WARPS
#define MOE_A_STAGE_WARPS 8
#endif

constexpr int moe_smem_bytes(int TILESIZE_N, int TILESIZE_M, int TILESIZE_K, int bits, int SH)
{
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);
    // The ring predicate lives here and only here, so every accessor, the launcher and the
    // in-kernel static_assert agree by construction (a separate predicate argument that a
    // call site forgets would request 29 184 B for a 47 104 B layout).
    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K
        * (((TILESIZE_M == 16) && (TILESIZE_K == 16) && (TILEBLOCKS_N * bits <= 128))
               ? MOE_A_STAGE_WARPS : 1);                                          // halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;    // uint16s
    const int sh_c_size = 4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP * (TILESIZE_M / 16); // floats
    return SH * sh_a_stage_size * 2 + SH * sh_b_stage_size * 2 + sh_c_size * 4;
}

#define MOE_SMEM_BYTES(N_TILE, M_TILE, MAXBITS, SH) \
    moe_smem_bytes(N_TILE, M_TILE, MOE_TILESIZE_K, MAXBITS, SH)

#define EXL3_MOE_KERNEL_ARGS                    \
    const half* __restrict__ hidden_state,      \
    half* __restrict__ temp_state_g,            \
    half* __restrict__ temp_state_u,            \
    half* __restrict__ temp_intermediate_g,     \
    half* __restrict__ temp_intermediate_u,     \
    float* __restrict__ output_state,           \
                                                \
    const uint16_t** __restrict__ gate_trellis, \
    const half** __restrict__ gate_suh,         \
    const half** __restrict__ gate_svh,         \
    const uint16_t** __restrict__ up_trellis,   \
    const half** __restrict__ up_suh,           \
    const half** __restrict__ up_svh,           \
    const uint16_t** __restrict__ down_trellis, \
    const half** __restrict__ down_suh,         \
    const half** __restrict__ down_svh,         \
                                                \
    const int64_t* __restrict__ expert_count,   \
    const int64_t* __restrict__ token_sorted,   \
    const half* __restrict__ weight_sorted,     \
                                                \
    const int hidden_dim,                       \
    const int intermediate_dim,                 \
    const int num_experts,                      \
    const int num_experts_per_tok,              \
    const int max_tokens_per_expert,            \
    const int concurrency,                      \
    const float act_limit,                      \
    const int act_function,                     \
    const int K_gate,                           \
    const int K_up,                             \
    const int K_down,                           \
                                                \
    int* __restrict__ locks,                    \
                                                \
    float* __restrict__ output_scratch,         \
    const int64_t* __restrict__ fused_base,     \
                                                \
    const int count_lo,                         \
    const int count_hi

#define EXL3_MOE_MIXEDK_KERNEL_ARGS             \
    const half* __restrict__ hidden_state,      \
    half* __restrict__ temp_state_g,            \
    half* __restrict__ temp_state_u,            \
    half* __restrict__ temp_intermediate_g,     \
    half* __restrict__ temp_intermediate_u,     \
    float* __restrict__ output_state,           \
                                                \
    const uint16_t** __restrict__ gate_trellis, \
    const half** __restrict__ gate_suh,         \
    const half** __restrict__ gate_svh,         \
    const uint16_t** __restrict__ up_trellis,   \
    const half** __restrict__ up_suh,           \
    const half** __restrict__ up_svh,           \
    const uint16_t** __restrict__ down_trellis, \
    const half** __restrict__ down_suh,         \
    const half** __restrict__ down_svh,         \
                                                \
    const int64_t* __restrict__ expert_count,   \
    const int64_t* __restrict__ token_sorted,   \
    const half* __restrict__ weight_sorted,     \
                                                \
    const int hidden_dim,                       \
    const int intermediate_dim,                 \
    const int num_experts,                      \
    const int num_experts_per_tok,              \
    const int max_tokens_per_expert,            \
    const int concurrency,                      \
    const float act_limit,                      \
    const int act_function,                     \
    const int* __restrict__ K_gate_arr,         \
    const int* __restrict__ K_up_arr,           \
    const int* __restrict__ K_down_arr,         \
                                                \
    int* __restrict__ locks,                    \
                                                \
    float* __restrict__ output_scratch,         \
    const int64_t* __restrict__ fused_base,     \
                                                \
    const int count_lo,                         \
    const int count_hi,                         \
                                                \
    const int* __restrict__ expert_order

