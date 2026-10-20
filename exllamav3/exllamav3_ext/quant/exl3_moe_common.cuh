#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include "exl3_kernel_map.cuh"   // EXL3_GEMM_BASE_THREADS, SMEM_MAX, exl3_gemm_smem_bytes()

#define MOE_ACT_SILU 0
#define MOE_ACT_GELU 1
#define MOE_ACT_RELU2_NOGATE 2  // non-gated relu2 (NemotronH): gate GEMM and staging skipped

#define MOE_SMS_PER_EXPERT 8       // default/minimum group width, also sets max concurrency (buffer count)
#define MOE_MAX_SMS_PER_EXPERT 32  // widest expert group when few experts are active
#define MOE_TILESIZE_K 16
#define MOE_TILESIZE_M 16          // default row tile; the 32 / 64 row tiles are separate kernel instances
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3

// Exact dynamic shared-memory footprint of one mixed-K CTA, in bytes.
//
// CUDA (MMA inner): a thin, named wrapper over the one layout definition in
// exl3_kernel_map.cuh (exl3_gemm_smem_bytes with shmem_out_had = false), which the inner
// kernel static_asserts against and which the shape filter's budget uses. Keeping a second
// copy of the arithmetic here is how the launcher and the kernel drift apart, so there is
// only this call.
//
// sh_a/sh_b are ring buffers indexed by (slice_iters % SH) and sh_a carries the per-warp
// ring (EXL3_GEMM_A_STAGE_WARPS) where the shape takes it; sh_c is a single buffer (the MoE
// path passes shmem_out_had = false, and at MOE_TILESIZE_K == 16 TILEBLOCKS_K == 1 so
// threadblock_reduce() compiles to nothing). MAXBITS (not the launch's K) sets the sh_b
// extent because one instance serves every K <= MAXBITS at run time.
#if defined(USE_ROCM)
// RDNA inner (rocm/quant/exl3_gemm_inner_rdna.cuh): the warp-private stages are CUDA-only,
// so the RDNA mixed-K launch extent is bit-for-bit what master ships -- no per-warp A ring,
// and sh_c is the 2x over-reservation that path has always used. Do not "unify" this with
// exl3_gemm_smem_bytes without re-measuring the RDNA inner.
constexpr int moe_smem_bytes(int TILESIZE_N, int TILESIZE_M, int TILESIZE_K, int bits, int SH)
{
    const int TILEBLOCKS_M = TILESIZE_M / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);
    const int sh_a = TILESIZE_M * TILESIZE_K * 2;                                   // halfs -> bytes
    const int sh_b = TILEBLOCKS_K * TILEBLOCKS_N * (256 / 16) * bits * 2;           // uint16s -> bytes
    const int sh_c_reduce = 4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP * TILEBLOCKS_M;
    const int sh_c_had = TILESIZE_N * TILESIZE_M;                                   // floats
    const int sh_c = (sh_c_reduce > sh_c_had) ? sh_c_reduce : sh_c_had;
    return SH * (sh_a + sh_b) + 2 * 4 * sh_c;
}
#else
constexpr int moe_smem_bytes(int TILESIZE_N, int TILESIZE_M, int TILESIZE_K, int bits, int SH)
{
    return exl3_gemm_smem_bytes(TILESIZE_M, TILESIZE_K, TILESIZE_N, SH, MOE_FRAG_STAGES,
                                bits, /* half_k = */ false, /* shmem_out_had = */ false);
}
#endif

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

