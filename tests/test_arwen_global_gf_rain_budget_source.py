"""The closing statement of Grell-Freitas's precipitation (GP-1), read off
the carried kernel source.  Kept apart from the device budget test
(test_arwen_global_gf_rain_budget_cuda.py) so a CPU-only run checks it."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_the_kernel_adds_the_remaining_evaporation_to_the_condensate():
    """The closing statement itself, checked where no card is."""
    source = (_ROOT / "src" / "arwen_global" / "core" / "kernels"
              / "gf.cu").read_text("utf-8")
    body = source[source.index("__device__ void gfd_cup_output_ens_3d("):]
    body = body[:body.index("\n}\n")]
    code = [line.strip() for line in body.splitlines()
            if line.strip() and not line.strip().startswith("//")]
    assert "pre = FSUB(pre, FMUL(xmb, dtpwd));" in code
    assert "pre = FADD(pre, FMUL(xmb, pwtot));" in code
    assert "pre = FADD(-pre, FMUL(xmb, pwtot));" not in code
