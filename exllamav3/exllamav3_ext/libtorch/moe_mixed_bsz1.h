#pragma once

#include <ATen/Tensor.h>
#include <memory>
#include <vector>

#include "linear.h"

// Batch-1 decode through a routed-expert layer whose experts differ in K or codebook. The fused,
// mgemm and BC_BlockSparseMLP kernels take one K per projection for the whole layer, so these
// layers otherwise run the Python per-expert loop: a host sync plus ~6 Python-driven launches per
// selected expert. This runs the same per-expert BC GEMVs, activation and weighted accumulation
// from one C++ call, in the Python loop's order (ascending expert id), so results are identical.
struct BC_MixedExpertsBsz1
{
    std::vector<std::shared_ptr<BC_LinearEXL3>> gates;
    std::vector<std::shared_ptr<BC_LinearEXL3>> ups;
    std::vector<std::shared_ptr<BC_LinearEXL3>> downs;
    at::Tensor interm_g;    // (1, I) gate output
    at::Tensor interm_u;    // (1, I) up output
    at::Tensor interm_a;    // (1, I) activation output (aliases interm_u when the up output is fp16)
    at::Tensor out_d;       // (1, H) down output, dtype of the down projections
    int act;                // 0 = silu, 1 = gelu, 2 = silu_oai (swiglu_oai)
    float act_limit;

    BC_MixedExpertsBsz1
    (
        std::vector<std::shared_ptr<BC_LinearEXL3>> _gates,
        std::vector<std::shared_ptr<BC_LinearEXL3>> _ups,
        std::vector<std::shared_ptr<BC_LinearEXL3>> _downs,
        at::Tensor _interm_g,
        at::Tensor _interm_u,
        at::Tensor _interm_a,
        at::Tensor _out_d,
        int _act,
        float _act_limit
    ) :
        gates(std::move(_gates)),
        ups(std::move(_ups)),
        downs(std::move(_downs)),
        interm_g(std::move(_interm_g)),
        interm_u(std::move(_interm_u)),
        interm_a(std::move(_interm_a)),
        out_d(std::move(_out_d)),
        act(_act),
        act_limit(_act_limit)
    {}

    // y (1, H) contiguous; selected (1, k) expert ids; weights (1, k); out (1, H) fp32, overwritten
    void run(const at::Tensor& y, const at::Tensor& selected, const at::Tensor& weights, at::Tensor& out);
};
