#include <Python.h>
#include "moe_mixed_bsz1.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <algorithm>
#include <numeric>
#include "../util.h"
#include "../activation.cuh"

void BC_MixedExpertsBsz1::run
(
    const at::Tensor& y,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    TORCH_CHECK(y.dim() == 2 && y.size(0) == 1 && y.is_contiguous(), "BC_MixedExpertsBsz1: y must be (1, H) contiguous");
    TORCH_CHECK(selected.numel() == weights.numel(), "BC_MixedExpertsBsz1: selected/weights size mismatch");

    // One host round trip for the routing result (the Python loop reads expert counts back too)
    at::Tensor sel = selected.reshape({-1}).to(at::kCPU, at::kLong);
    at::Tensor w = weights.reshape({-1}).to(at::kCPU, at::kFloat);
    const int64_t k = sel.numel();
    const int64_t* ids = sel.data_ptr<int64_t>();
    const float* ws = w.data_ptr<float>();
    const int64_t num_experts = (int64_t) ups.size();

    // Ascending expert id: the Python path groups assignments with argsort and accumulates with
    // index_add_ in expert order, so the fp32 sum has the same operand order
    std::vector<int64_t> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return ids[a] < ids[b]; });

    out.zero_();
    for (int64_t j = 0; j < k; j++)
    {
        const int64_t i = order[j];
        const int64_t e = ids[i];
        TORCH_CHECK(e >= 0 && e < num_experts, "BC_MixedExpertsBsz1: expert id out of range");

        ups[e]->run(y, interm_u);
        gates[e]->run(y, interm_g);
        switch (act)
        {
            case 0: silu_mul(interm_g, interm_u, interm_a, act_limit); break;
            case 1: gelu_mul(interm_g, interm_u, interm_a, act_limit); break;
            case 2: silu_oai_mul(interm_g, interm_u, interm_a, act_limit); break;
            default: TORCH_CHECK(false, "BC_MixedExpertsBsz1: unsupported activation");
        }
        downs[e]->run(interm_a, out_d);

        // fp16 routing weight as an fp32 scalar is exact, matching the Python mul_ by the fp16 tensor
        out_d.mul_((double) ws[i]);
        out.add_(out_d);
    }
}
