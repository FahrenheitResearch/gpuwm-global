"""The latitude-band escape flag on the device (MG-1, the CUDA kernels).

BAND-SL-2 proves the escape on the numpy specification only: a read
outside the band's held rows raises HaloEscape there at once.  On the card
the band kernels raise a device flag instead (atomicOr, redirected to held
row 0, read back once per band), and the step widens the halo and recomputes
the band.  That flag is the only thing that stops a banded run on the
production backend from silently reading the wrong row, so it is run here
on the device: a one-row halo forces escapes, the step widens, and the
answer is the same band schedule's bit for bit.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

from arwen_global.semilag import SemiLagrangianOptions  # noqa: E402
from arwen_global.semilag import step as step_module  # noqa: E402
from arwen_global.spectral.transform import SphericalHarmonicTransform  # noqa: E402
from arwen_global.vertical import HybridCoordinate  # noqa: E402

from test_arwen_global_semilag_bands import _model  # noqa: E402
from test_arwen_global_semilag_step import _baroclinic  # noqa: E402


def _to_device(value):
    if isinstance(value, np.ndarray):
        return cp.asarray(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{
            field.name: _to_device(getattr(value, field.name))
            for field in dataclasses.fields(value) if field.init})
    if isinstance(value, dict):
        return {key: _to_device(item) for key, item in value.items()}
    return value


def _digest(value) -> str:
    array = np.ascontiguousarray(cp.asnumpy(value))
    return hashlib.sha256(
        array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _run(bands: int, steps: int):
    host = SphericalHarmonicTransform.create(21, backend="numpy",
                                             precision="float64")
    device = SphericalHarmonicTransform.create(21, backend="cupy",
                                               precision="float64")
    vertical = HybridCoordinate.pressure_blend(8, 100.0)
    bundle = _to_device(_baroclinic(host, vertical, wind_m_s=35.0))
    model = _model(device, vertical, bands=bands,
                   options=SemiLagrangianOptions())
    for _ in range(steps):
        bundle, _metrics = model.step(bundle, 900.0)
    atmosphere = bundle.atmosphere
    out = {name: _digest(getattr(atmosphere, name))
           for name in ("vorticity", "divergence", "theta",
                        "log_surface_pressure", "qv")}
    out.update({"tracer." + name: _digest(value)
                for name, value in atmosphere.grid_tracers().items()})
    return model, out


def test_the_device_escape_flag_widens_the_band_and_keeps_the_answer(
        monkeypatch):
    # The reference is the same four-band schedule behind its default
    # halo: the answer is a function of the band schedule (the waists'
    # contraction order on the card), and an escape must cost a
    # recomputation of that schedule's band, not a different answer.
    _wide, reference = _run(4, 3)
    monkeypatch.setattr(step_module, "default_halo_rows",
                        lambda nlat, dt, radius: 1)
    model, narrow = _run(4, 3)
    assert model.semilag_halo_rows[0] == 1
    assert model.semilag_halo_rows[1] > 1, "no band escaped a one-row halo"
    assert narrow == reference
