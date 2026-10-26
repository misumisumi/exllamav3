"""4090 CUDA numerical parity tests for WSL LM-head low-VRAM Hessian refit.

Small seeded reference tensors are strictly implementation tests, not evidence
for Nanbeige model fidelity or any global teacher quality gates.
"""
from __future__ import annotations

import pytest
import torch

from exllamav3.modules.quant.exl3_lib.quantize import refit_scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the actual local CUDA RTX 4090")
@pytest.mark.parametrize("n,chunk", [(49, 8), (257, 19), (2048, 89)])
def test_cpu_backed_hessian_output_parity(n: int, chunk: int) -> None:
    torch.manual_seed(99817)
    device = "cuda:0"
    k = 24
    a = torch.randn((k, k), device=device, dtype=torch.float32)
    h = (a @ a.T / k + 0.1 * torch.eye(k, device=device))
    w = torch.randn((k, n), device=device, dtype=torch.float32) * 0.21
    q = (w * 0.94 + 0.02 * torch.randn_like(w)).detach()
    su = torch.full((k, 1), 1.02, device=device, dtype=torch.float32)
    sv = torch.full((1, n), 0.97, device=device, dtype=torch.float32)

    orig = refit_scales(w.clone(), q.clone(), h, su, sv, chunk=chunk,
                        rounds=2, stream_hw_to_cpu=False)
    stream = refit_scales(w.clone(), q.clone(), h, su, sv, chunk=chunk,
                          rounds=2, stream_hw_to_cpu=True)
    torch.cuda.synchronize(device)
    for index in range(3):
        assert orig[index].device.type == "cuda"
        assert stream[index].device.type == "cuda"
        torch.testing.assert_close(orig[index], stream[index], rtol=6e-4, atol=5e-5)
    assert orig[3] >= 0 and stream[3] >= 0
    assert orig[4] >= 0 and stream[4] >= 0
    assert abs(orig[3] - stream[3]) <= 2e-5
    assert abs(orig[4] - stream[4]) <= 2e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_real_default_small_matrix_uses_original_path() -> None:
    torch.manual_seed(18)
    k, n = 12, 55
    device = "cuda:0"
    w = torch.randn(k, n, device=device)
    q = w * 0.92
    h = torch.eye(k, device=device)
    su = torch.ones(k, 1, device=device)
    sv = torch.ones(1, n, device=device)
    default = refit_scales(w, q.clone(), h, su, sv)
    expected = refit_scales(w, q.clone(), h, su, sv, stream_hw_to_cpu=False)
    for index in range(3):
        torch.testing.assert_close(default[index], expected[index], rtol=0, atol=0)
    assert default[3:] == expected[3:]
