#pragma once

#include "../exl3_moe_common.cuh"

typedef void (*fp_exl3_moe_mixedk_kernel) (EXL3_MOE_MIXEDK_KERNEL_ARGS);

// Mixed-K instances: cb1/cb2 x n128/n256 for m16 (default), plus cb2 x n128 for m32/m64
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb1();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n256_cb1();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n256_cb2();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_m32();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_m64();

// Deeper-pipeline m16 / n128 / mul1 variants (see exl3_moe.cu: EXL3_MK_SHPIPE)
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_sh4fs3();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_sh6fs5();

// Low-K / deep-pipeline instances (MAXBITS, SH_STAGES): see exl3_moe_mixedk_inst_lowk.cu
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n256_cb2_k1();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_k2();
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n256_cb2_k4();

extern fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_instances[];      // [cb_idx * 2 + N_off]
extern fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_instances_m32[];  // [0]
extern fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_instances_m64[];  // [0]

// Dynamic shared memory each mixed-K instance actually indexes, in bytes.
// Mirrors the template arguments in exl3_moe_mixedk_inst_cb1.cu / _cb2.cu / _lowk.cu:
//   <N, cb>                      -> M=MOE_TILESIZE_M, MAXBITS=8, SH=MOE_SH_STAGES
//   <N, cb, M>                   -> MAXBITS=8, SH=MOE_SH_STAGES
//   <N, cb, M, MAXBITS, SH>      -> explicit
// MOE_TILESIZE_K is a #define in exl3_moe_common.cuh, so these stay constexpr and no
// kernel is instantiated here.
constexpr int exl3_moe_mixedk_smem_n128_cb1()      { return moe_smem_bytes(128, 16, MOE_TILESIZE_K, 8, MOE_SH_STAGES); }
constexpr int exl3_moe_mixedk_smem_n256_cb1()      { return moe_smem_bytes(256, 16, MOE_TILESIZE_K, 8, MOE_SH_STAGES); }
constexpr int exl3_moe_mixedk_smem_n128_cb2()      { return moe_smem_bytes(128, 16, MOE_TILESIZE_K, 8, MOE_SH_STAGES); }
constexpr int exl3_moe_mixedk_smem_n256_cb2()      { return moe_smem_bytes(256, 16, MOE_TILESIZE_K, 8, MOE_SH_STAGES); }
constexpr int exl3_moe_mixedk_smem_n128_cb2_m32()  { return moe_smem_bytes(128, 32, MOE_TILESIZE_K, 8, MOE_SH_STAGES); }
constexpr int exl3_moe_mixedk_smem_n128_cb2_m64()  { return moe_smem_bytes(128, 64, MOE_TILESIZE_K, 8, MOE_SH_STAGES); }
constexpr int exl3_moe_mixedk_smem_n256_cb2_k1()   { return moe_smem_bytes(256, 16, MOE_TILESIZE_K, 1, 6); }
constexpr int exl3_moe_mixedk_smem_n128_cb2_k2()   { return moe_smem_bytes(128, 16, MOE_TILESIZE_K, 2, 6); }
constexpr int exl3_moe_mixedk_smem_n256_cb2_k4()   { return moe_smem_bytes(256, 16, MOE_TILESIZE_K, 4, 5); }
constexpr int exl3_moe_mixedk_smem_n128_cb2_sh4fs3() { return moe_smem_bytes(128, 16, MOE_TILESIZE_K, 8, 4); }
constexpr int exl3_moe_mixedk_smem_n128_cb2_sh6fs5() { return moe_smem_bytes(128, 16, MOE_TILESIZE_K, 8, 6); }
