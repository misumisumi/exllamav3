#include <Python.h>
#include "moe_mixedk_graph.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../quant/exl3_gemm.cuh"
#include "../activation.cuh"
#include "../add.cuh"

BC_MixedKExperts::BC_MixedKExperts
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
) :
    remap           (std::move(_remap)),
    yh              (std::move(_yh)),
    interm_g        (std::move(_interm_g)),
    interm_u        (std::move(_interm_u)),
    interm_a        (std::move(_interm_a)),
    out_d           (std::move(_out_d)),
    y_static        (std::move(_y_static)),
    shared_experts  (_shared_experts),
    out_sh          (std::move(_out_sh)),
    top_k           (_top_k),
    act             (_act),
    act_limit       (_act_limit)
{
    TORCH_CHECK(act >= 0 && act <= 3, "BC_MixedKExperts: unsupported activation");
    TORCH_CHECK(!shared_experts || out_sh, "BC_MixedKExperts: shared experts need out_sh");
    TORCH_CHECK(!out_d.empty(), "BC_MixedKExperts: no down output buffers");
    TORCH_CHECK(y_static.dim() == 3 && y_static.size(0) == 1 && y_static.size(1) >= MAX_BSZN,
                "BC_MixedKExperts: y_static must be (1, MAX_BSZN, H)");
    TORCH_CHECK(yh.size(0) >= MAX_BSZN * top_k, "BC_MixedKExperts: scratch buffers too small");
    local_idx.resize(MAX_BSZN);
    w_static.resize(MAX_BSZN);
    a_gather.resize(MAX_BSZN);
    flat_token.resize(MAX_BSZN);
}

void BC_MixedKExperts::add_group(int proj, at::Tensor ptrs_trellis, at::Tensor ptrs_suh, at::Tensor ptrs_svh,
                                 int K, bool mcg, bool mul1, int size)
{
    TORCH_CHECK(proj >= 0 && proj <= 2, "BC_MixedKExperts: proj must be 0 (gate), 1 (up) or 2 (down)");
    TORCH_CHECK(ptrs_trellis.size(0) == size, "BC_MixedKExperts: pointer table size mismatch");
    Group g { std::move(ptrs_trellis), std::move(ptrs_suh), std::move(ptrs_svh), K, mcg, mul1, size };
    if (proj == 0) gate_groups.push_back(std::move(g));
    else if (proj == 1) up_groups.push_back(std::move(g));
    else down_groups.push_back(std::move(g));
}

void BC_MixedKExperts::run_gr(int num_tokens, Graph* graph)
{
    int gi = num_tokens - 1;
    int bszm = num_tokens * top_k;
    int mask_tokens = num_tokens == 1 ? -1 : num_tokens;   // position-preserving range mask at every bsz

    at::Tensor A_in = num_tokens == 1 ? y_static.narrow(1, 0, 1) : a_gather[gi].unsqueeze(1);
    at::Tensor yh_n       = yh.slice(0, 0, bszm);
    at::Tensor interm_g_n = interm_g.slice(0, 0, bszm);
    at::Tensor interm_u_n = interm_u.slice(0, 0, bszm);
    at::Tensor interm_a_n = interm_a.slice(0, 0, bszm);
    at::Tensor& li = local_idx[gi];
    int row = 0;

    for (auto& grp : gate_groups)
    {
        exl3_mgemm_gr(A_in, grp.ptrs_trellis, interm_g_n, grp.ptrs_suh, yh_n, grp.ptrs_svh,
                      li.select(0, row).unsqueeze(0), {}, grp.K, -1, grp.mcg, grp.mul1,
                      0, grp.size, 0, graph, mask_tokens);
        row++;
    }
    for (auto& grp : up_groups)
    {
        exl3_mgemm_gr(A_in, grp.ptrs_trellis, interm_u_n, grp.ptrs_suh, yh_n, grp.ptrs_svh,
                      li.select(0, row).unsqueeze(0), {}, grp.K, -1, grp.mcg, grp.mul1,
                      0, grp.size, 0, graph, mask_tokens);
        row++;
    }

    switch (act)
    {
        case 0: silu_mul_gr(interm_g_n, interm_u_n, interm_a_n, act_limit, graph); break;
        case 1: gelu_mul_gr(interm_g_n, interm_u_n, interm_a_n, act_limit, graph); break;
        case 2: silu_oai_mul_gr(interm_g_n, interm_u_n, interm_a_n, act_limit, graph); break;
        default: relu_mul_gr(interm_u_n, interm_u_n, interm_a_n, act_limit, graph); break;
    }

    // A_had must not alias A: interm_g_n is free once the activation has run
    for (size_t d = 0; d < down_groups.size(); ++d)
    {
        auto& grp = down_groups[d];
        at::Tensor out_n = out_d[d].slice(0, 0, bszm);
        exl3_mgemm_gr(interm_a_n, grp.ptrs_trellis, out_n, grp.ptrs_suh, interm_g_n, grp.ptrs_svh,
                      li.select(0, row).unsqueeze(0), w_static[gi], grp.K, -1, grp.mcg, grp.mul1,
                      0, grp.size, 0, graph, mask_tokens);
        row++;
    }

    at::Tensor acc = out_d[0].slice(0, 0, num_tokens).squeeze(1);
    for (size_t d = 1; d < down_groups.size(); ++d)
    {
        at::Tensor src = out_d[d].slice(0, 0, num_tokens).squeeze(1);
        add_gr(acc, src, acc, graph);
    }
    if (shared_experts)
    {
        at::Tensor x_dense = y_static.narrow(1, 0, num_tokens);
        at::Tensor sh_n = out_sh.value().slice(1, 0, num_tokens);
        shared_experts->run_bszN_gr(x_dense, sh_n, num_tokens, graph);
        add_gr(acc, sh_n, acc, graph);
    }
}

at::Tensor BC_MixedKExperts::run(const at::Tensor& y, const at::Tensor& selected, const at::Tensor& weights)
{
    int num_tokens = (int) y.size(0);
    TORCH_CHECK(y.dim() == 2 && num_tokens >= 1 && num_tokens <= MAX_BSZN,
                "BC_MixedKExperts: y must be (bsz, H) with bsz in 1..MAX_BSZN");
    TORCH_CHECK(y.dtype() == at::kHalf && y.is_contiguous(), "BC_MixedKExperts: y must be contiguous fp16");
    TORCH_CHECK(selected.dtype() == at::kLong && weights.dtype() == at::kHalf,
                "BC_MixedKExperts: selected must be int64 and weights fp16");
    TORCH_CHECK(selected.numel() == num_tokens * top_k && weights.numel() == num_tokens * top_k,
                "BC_MixedKExperts: routing shape mismatch");
    TORCH_CHECK((size_t) remap.size(0) == gate_groups.size() + up_groups.size() + down_groups.size() &&
                out_d.size() == down_groups.size(), "BC_MixedKExperts: groups do not match remap/out_d");

    int gi = num_tokens - 1;
    int bszm = num_tokens * top_k;
    c10::cuda::CUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (!local_idx[gi].defined())
    {
        auto opts_long = at::TensorOptions().dtype(at::kLong).device(y.device());
        local_idx[gi] = at::empty({remap.size(0), bszm}, opts_long);
        w_static[gi] = at::empty({1, bszm}, weights.options());
        if (num_tokens > 1)
        {
            a_gather[gi] = at::empty({bszm, y.size(1)}, y.options());
            flat_token[gi] = at::arange(num_tokens, opts_long).unsqueeze(1)
                .expand({num_tokens, top_k}).reshape({bszm}).contiguous();
        }
    }

    // Per-call inputs into the static buffers the graph reads
    at::Tensor y_n = y_static.select(0, 0).narrow(0, 0, num_tokens);
    y_n.copy_(y);
    local_idx[gi].copy_(at::index_select(remap, 1, selected.reshape({-1})));
    w_static[gi].copy_(weights.reshape({1, -1}));
    if (num_tokens > 1)
        a_gather[gi].copy_(at::index_select(y_n, 0, flat_token[gi]));

    Graph& g = graphs[gi];
    if (g.disabled || (!g.ready && !g.ready_to_record))
    {
        run_gr(num_tokens, nullptr);
        g.ready_to_record = true;
    }
    else
    {
        if (!g.ready)
        {
            g.capture_begin();
            run_gr(num_tokens, &g);
            g.capture_end();
        }
        // Every address is static: nothing to patch (the launcher needs at least one entry)
        std::vector<PPTR> args = { PPTR(GP_end, nullptr) };
        g.launch(args, stream);
    }

    return out_d[0].slice(0, 0, num_tokens).squeeze(1);
}
