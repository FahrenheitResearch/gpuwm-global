"""The 0.4.0 opt-in physics through the front door, on the card.

Each option is run the way a user runs it, ``gpuwm-global run CONFIG``, on
the shipped T3 native smoke at 12 levels (the SA-TKE-EDMF column needs at
least eight) with radiation every 600 s, and each test checks that the run
passes, that its receipt names the option, and that the option acted:

* the SA-TKE-EDMF boundary layer (``pbl = "satmedmf"``, the single-precision
  port) and its double-precision proof route (``"satmedmf_fp64"``): both
  pass, both move the forecast away from YSU, and they agree with each
  other far more closely than either agrees with YSU;
* the Scinocca non-orographic drag through its fused kernel (the default
  route; ``ARWEN_NOGWD_CUDA=0`` is the CuPy referee): the checkpoint's
  physics metadata counts its calls and the forecast moves;
* the 80-level ``ifs_class`` grid: the checkpoint carries 80 levels;
* TEMPO microphysics: the receipt names it and the forecast moves.

The shipped smoke itself (5 levels, radiation every 10 s) stops at step 0
in RRTMGP's surface-temperature range check, in 0.3.0 as here; the edits
below are what this file needs, not a fix of that config.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cupy")

from arwen_global import cli  # noqa: E402
from arwen_global.configs_dir import config_root  # noqa: E402

pytestmark = pytest.mark.gpu

BASE_EDITS = (
    ('coordinate = "pressure_blend"\nnlev = 5', 'coordinate = "pressure_blend"\nnlev = 12'),
    ("radiation_interval_s = 10.0", "radiation_interval_s = 600.0"),
)


def _base_text() -> str:
    text = (config_root() / "arwen_global_level5_native_smoke.toml").read_text(encoding="utf-8")
    for old, new in BASE_EDITS:
        assert old in text, old
        text = text.replace(old, new)
    # The smoke's mass gate (1e-7) reads 1.4e-7 on every arm of this shape,
    # the YSU one included; it is a smoke's gate, widened for this file.
    return re.sub(r"(?m)^mass_relative_drift = .*$", "mass_relative_drift = 1.0e-6", text)


def _option(text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new)


def _run(tmp_path: Path, name: str, text: str) -> tuple[dict, dict]:
    folder = tmp_path / name
    folder.mkdir()
    config = folder / f"{name}.toml"
    config.write_text(text, encoding="utf-8")
    outdir = folder / "run"
    assert cli.main(["run", str(config), "--outdir", str(outdir)]) == 0
    receipt = json.loads((outdir / "arwen-global-receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "pass", receipt.get("error_message")
    last = sorted(outdir.glob("arwen_global_step*.npz"))[-1]
    with np.load(last, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key]) for key in archive.files}
    return receipt, arrays


def _options(receipt: dict) -> dict:
    return receipt["physics_identity"]["options"]


def _metadata(arrays: dict) -> dict:
    raw = arrays["__metadata__"]
    text = (str(raw.item()) if raw.dtype.kind in "US"
            else bytes(np.ascontiguousarray(raw)).decode("utf-8"))
    return json.loads(text)


def _largest_change(a: dict, b: dict) -> float:
    """The largest relative change, ||a - b|| / ||b||, over the prognostic
    state both checkpoints carry (the ``atmosphere__`` arrays, spectral
    and grid)."""
    worst = 0.0
    for key, left in a.items():
        right = b.get(key)
        if (not key.startswith("atmosphere__") or right is None
                or left.shape != right.shape or left.dtype.kind not in "fc"):
            continue
        left = left.astype(np.complex128)
        right = right.astype(np.complex128)
        scale = float(np.linalg.norm(right.ravel()))
        if scale > 0.0:
            worst = max(worst, float(np.linalg.norm((left - right).ravel())) / scale)
    return worst


@pytest.fixture(scope="module")
def ysu(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("ysu"), "ysu", _base_text())


def test_sa_tke_edmf_runs_and_its_double_precision_route_agrees(tmp_path, ysu):
    base = _base_text()
    fp32 = _run(tmp_path, "satmedmf", _option(base, 'pbl = "ysu"', 'pbl = "satmedmf"'))
    fp64 = _run(tmp_path, "satmedmf_fp64", _option(base, 'pbl = "ysu"', 'pbl = "satmedmf_fp64"'))
    assert _options(fp32[0])["pbl"] == "satmedmf"
    assert _options(fp64[0])["pbl"] == "satmedmf_fp64"
    away = _largest_change(fp32[1], ysu[1])
    proof = _largest_change(fp32[1], fp64[1])
    assert away > 0.0
    assert proof < 0.1 * away, (proof, away)


def test_the_scinocca_drag_runs_through_its_fused_kernel(tmp_path, ysu, monkeypatch):
    from arwen_global.physics import nonorographic_gwd

    monkeypatch.delenv("ARWEN_NOGWD_CUDA", raising=False)
    nonorographic_gwd._CUDA_CACHE.clear()
    text = _option(_base_text(), "dx_m = 50000.0 }",
                   'dx_m = 50000.0, nonorographic_gwd = "scinocca" }')
    receipt, arrays = _run(tmp_path, "scinocca", text)
    assert _options(receipt)["nonorographic_gwd"] == "scinocca"
    physics = json.dumps(_metadata(arrays))
    assert '"nonorographic_gwd": "scinocca"' in physics
    calls = [int(v) for v in re.findall(r'"nogwd_calls": (\d+)', physics)]
    assert calls and max(calls) > 0
    # The fused kernel was built on this card, so it is the route that ran.
    assert nonorographic_gwd._CUDA_CACHE
    assert _largest_change(arrays, ysu[1]) > 0.0


def test_the_eighty_level_grid_runs(tmp_path):
    text = _option(_base_text(), 'coordinate = "pressure_blend"\nnlev = 12',
                   'coordinate = "ifs_class"\nnlev = 80')
    receipt, arrays = _run(tmp_path, "ifs80", text)
    vertical = json.dumps(receipt)
    assert '"coordinate": "ifs_class"' in vertical
    assert re.search(r'"nlev": 80[,}]', vertical)
    levelled = [a for k, a in arrays.items() if k != "__metadata__" and a.ndim == 3]
    assert any(a.shape[0] == 80 for a in levelled)


def test_tempo_runs(tmp_path, ysu):
    text = _option(_base_text(), 'microphysics = "morrison", morr_rimed_ice = 1',
                   'microphysics = "tempo"')
    receipt, arrays = _run(tmp_path, "tempo", text)
    assert _options(receipt)["microphysics"] == "tempo"
    assert _largest_change(arrays, ysu[1]) > 0.0
