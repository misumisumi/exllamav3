#include "exl3_moe_mixedk_instances.cuh"
#include "../exl3_moe_kernel.cuh"

// Low-K specialised mixed-K instances.
//
// 55% of the experts in the DSV4.1-Flash v3 pack are K=1, and the page table of K values
// is a load-time property of the expert, so whenever the plugin dispatches per
// (Kg,Ku,Kd) tuple the largest K in the launch is a host-side integer. Specialising on it
// buys two things, both measured on clang-20 -> ptxas 12.9 sm_121a:
//
//   instance                        frame    spill stores   sh_b stages in flight
//   n256 cb2 m16 (baseline, K<=8)   1200 B   312 B          2 x 1 KiB  (K=1)
//   n256 cb2 m16 K<=1 SH=6           432 B    52 B          5 x 1 KiB
//   n128 cb2 m16 K<=1 SH=8            24 B     0 B          7 x 512 B
//   n128 cb2 m16 K<=2 SH=6           112 B    96 B          5 x 1 KiB
//
// SH_STAGES is the bytes-in-flight knob: (SH-1) stages of sh_b are outstanding per CTA.
// At K=1/n256 the baseline leaves 2 KiB in flight per SM, i.e. 96 KiB across the chip,
// against the ~164 KiB Little's law asks for (273 GB/s x 600 ns) -- that is the 82 GB/s
// ceiling, and it is why the K=1 experts (the majority) run at a third of peak while the
// K>=2 ones saturate.
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n256_cb2_k1() { return exl3_moe_mixedk_kernel<256, 2, 16, 1, 6>; }
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n128_cb2_k2() { return exl3_moe_mixedk_kernel<128, 2, 16, 2, 6>; }
// K<=4: the instance every layer of DSV4.1-Flash v3 actually selects (max K = 4).
// Four live bits bodies instead of eight, at the deepest pipeline SMEM_MAX allows.
fp_exl3_moe_mixedk_kernel exl3_moe_mixedk_kernel_n256_cb2_k4() { return exl3_moe_mixedk_kernel<256, 2, 16, 4, 5>; }
