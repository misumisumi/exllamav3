#pragma once

#include <ATen/Tensor.h>
#include <memory>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "mlp.h"
#include "../graph.cuh"

// Routed experts of a mixed-K layer (experts quantized at different K or codebooks) at bsz
// 1..MAX_BSZN, one CUDA graph per bsz. Each projection runs one mgemm per (K, codebook) group of its
// own: the group's pointer tables hold only its members, and routes are remapped global -> local
// with other groups' experts at -1, which the kernel's position-preserving range mask skips
// (num_tokens = -1 selects that mask for a single token). Gate and up groups fill disjoint slots of
// shared buffers; each down group reduces its weighted slots into its own buffer, and the group sums
// (plus the shared expert) accumulate in down group 0's buffer. Every per-call input is copied into
// static buffers before the graph launches, so replays patch nothing.
struct BC_MixedKExperts
{
    struct Group
    {
        at::Tensor ptrs_trellis;
        at::Tensor ptrs_suh;
        at::Tensor ptrs_svh;
        int K;
        bool mcg;
        bool mul1;
        int size;
    };

    std::vector<Group> gate_groups;     // empty for gateless experts
    std::vector<Group> up_groups;
    std::vector<Group> down_groups;
    at::Tensor remap;                   // (gate + up + down groups, num_experts) int64: local id or -1
    at::Tensor yh;                      // (MAX_BSZN * top_k, 1, H) fp16 hadamard scratch (gate/up)
    at::Tensor interm_g;                // (MAX_BSZN * top_k, 1, I)
    at::Tensor interm_u;                // (MAX_BSZN * top_k, 1, I)
    at::Tensor interm_a;                // (MAX_BSZN * top_k, 1, I) fp16
    std::vector<at::Tensor> out_d;      // per down group (MAX_BSZN * top_k, 1, H) fp32
    at::Tensor y_static;                // (1, MAX_BSZN, H) fp16
    std::shared_ptr<BC_GatedMLP> shared_experts;
    c10::optional<at::Tensor> out_sh;   // (1, MAX_BSZN, H) fp32
    int top_k;
    int act;                            // 0 silu, 1 gelu, 2 silu_oai, 3 relu2 (gateless)
    float act_limit;

    // Sized on first use per bsz (index bsz - 1) and fixed afterwards, so graph addresses stay valid
    std::vector<at::Tensor> local_idx;  // (groups, bsz * top_k) int64
    std::vector<at::Tensor> w_static;   // (1, bsz * top_k) fp16
    std::vector<at::Tensor> a_gather;   // (bsz * top_k, H) fp16, bsz > 1
    std::vector<at::Tensor> flat_token; // (bsz * top_k) int64, bsz > 1
    Graph graphs[MAX_BSZN];

    BC_MixedKExperts
    (
        at::Tensor _remap,
        at::Tensor _yh,
        at::Tensor _interm_g,
        at::Tensor _interm_u,
        at::Tensor _interm_a,
        std::vector<at::Tensor> _out_d,
        at::Tensor _y_static,
        std::shared_ptr<BC_GatedMLP> _shared_experts,
        c10::optional<at::Tensor> _out_sh,
        int _top_k,
        int _act,
        float _act_limit
    );

    // proj: 0 gate, 1 up, 2 down; groups must be added in remap row order
    void add_group(int proj, at::Tensor ptrs_trellis, at::Tensor ptrs_suh, at::Tensor ptrs_svh,
                   int K, bool mcg, bool mul1, int size);

    void run_gr(int num_tokens, Graph* graph);

    // y (bsz, H) fp16 contiguous; selected (bsz, top_k) int64; weights (bsz, top_k) fp16.
    // Returns (bsz, H) fp32, a view of a static buffer valid until the next call on any layer
    at::Tensor run(const at::Tensor& y, const at::Tensor& selected, const at::Tensor& weights);
};
