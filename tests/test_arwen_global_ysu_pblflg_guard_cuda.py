"""The device half of the YSU pblflg guard (GP-4): the carried kernel keeps
the engine's measured onset column local, as the float64 mirror does
(test_arwen_global_ysu_pblflg_guard.py, which holds the CPU checks so a
CPU-only run still exercises them)."""
from __future__ import annotations

import numpy as np
import pytest

from test_arwen_global_ysu_pblflg_guard import _onset_column


@pytest.mark.gpu
def test_the_kernel_keeps_the_same_column_local():
    cp = pytest.importorskip("cupy")
    from arwen_global.core.ysu import launch_ysu

    args = _onset_column()
    dt = args.pop("dt")
    topdown = args.pop("ysu_topdown_pblmix")
    f32 = np.float32
    dev = {}
    for name in ("u", "v", "theta", "qv", "qc", "qi", "p", "exner", "dz"):
        dev[name] = cp.asarray(np.asarray(args[name], f32).reshape(-1, 1, 1))
    dev["p_interface"] = cp.asarray(
        np.asarray(args["p_interface"], f32).reshape(-1, 1, 1))
    for name in ("psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim",
                 "psih", "xland", "u10", "v10"):
        dev[name] = cp.asarray(np.full((1, 1), args[name], f32))
    out = launch_ysu(dev.pop("u"), dev.pop("v"), dev.pop("theta"),
                     dev.pop("qv"), dev.pop("qc"), dev.pop("qi"),
                     dev.pop("p"), dev.pop("p_interface"), dev.pop("exner"),
                     dev.pop("dz"), dt=dt, ysu_topdown_pblmix=topdown, **dev)
    cp.cuda.Stream.null.synchronize()
    assert int(out["kpbl"].get()[0, 0]) == 1
    assert float(out["hpbl"].get()[0, 0]) == pytest.approx(95.4339, abs=2e-2)
    assert float(out["delta"].get()[0, 0]) == 0.0
    assert float(out["exch_h"].max().get()) == pytest.approx(0.01, abs=1e-6)
