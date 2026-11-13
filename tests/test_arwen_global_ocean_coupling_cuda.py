"""The slab ocean's coupling on the card (arwen_global.ocean_coupler).

Its own module because the suite marks every test of a module that opens a
device (tests/conftest.py), and the CPU known answers of
test_arwen_global_ocean_coupling.py must stay in the CPU tiers.  The
stand-ins for the other packages' functions while they are stubs, and the
fixtures, are that module's.

* The coupler's march on cupy planes gives the numpy march's slab, ice and
  books to float64 rounding, and the same known answer.
* The T3 reference smoke on the cupy backend under the slab, straight and
  restarted inside an ice record across January's mid-month instant: every
  checkpoint array, the physics metadata and the trackers equal byte for
  byte.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global import ocean_coupler as oc
from arwen_global.boundary_options import OceanOptions, SeaIceOptions
from arwen_global.state import PhysicsState

from test_arwen_global_ocean_coupling import (  # noqa: F401 - the fixture
    RHO_CP, SMOKE, FakeQFlux, _climatology, _march, _provider, _same_checkpoints, _smoke,
    _smoke_climatology, parts,
)


@pytest.mark.gpu
def test_the_march_on_the_card_is_the_host_march(tmp_path, parts):
    cp = pytest.importorskip("cupy")
    host, host_state, _ = _provider(tmp_path, flux=-15.0,
                                    options=None)
    _march(host, host_state, 100.0, 12)
    card, card_state, _ = _provider(tmp_path, flux=-15.0)
    surface = card_state.surface
    for name in vars(surface):
        setattr(surface, name, cp.asarray(getattr(surface, name)))
    card_state.physics_state = PhysicsState(
        arrays={k: cp.asarray(v) for k, v in card_state.physics_state.arrays.items()},
        metadata=dict(card_state.physics_state.metadata))
    for _ in range(12):
        card_state.physics_state.arrays[oc.HELD_NET_FLUX] = cp.full(
            surface.temperature_k.shape, 100.0)
        card_state.time_s = float(card_state.time_s) + 300.0
        card.advance(card_state)
    for name in oc.SLAB_ARRAYS:
        value = card_state.physics_state.arrays[name]
        assert isinstance(value, cp.ndarray) and value.dtype == cp.float64, name
        np.testing.assert_allclose(cp.asnumpy(value), host_state.physics_state.arrays[name],
                                   rtol=1e-13, atol=1e-9, err_msg=name)
    expect = 290.0 + 12 * 300.0 * (100.0 + 15.0) / (RHO_CP * 50.0)
    np.testing.assert_allclose(
        cp.asnumpy(card_state.physics_state.arrays[oc.SLAB_TEMPERATURE]), expect,
        rtol=0, atol=1e-10)
    sst, defined = card.sst_at(3600.0, cp, cp.float32)
    assert isinstance(sst, cp.ndarray) and bool(defined.all())


@pytest.mark.gpu
def test_a_restarted_slab_run_on_the_card_reproduces_the_straight_run(
        tmp_path, monkeypatch, parts):
    pytest.importorskip("cupy")
    from arwen_global import ocean_qflux
    from arwen_global.config import load_config
    from arwen_global.runner import run

    start = datetime(2026, 1, 16, 11, 59, 20, tzinfo=timezone.utc)
    base = load_config(SMOKE)
    grid, sst, ice = _smoke_climatology(tmp_path, base.truncation, base.dealias_factor)
    clim = _climatology(tmp_path, grid, sst=sst, ice=ice)
    qpath = tmp_path / "q.npz"
    qpath.write_bytes(b"test q-flux")
    qsha = hashlib.sha256(b"test q-flux").hexdigest()
    shape = (int(grid.nlat), int(grid.nlon))
    monkeypatch.setattr(ocean_qflux, "load_qflux",
                        lambda path: FakeQFlux(shape, flux=-20.0, sha256=qsha))
    options = OceanOptions(
        declared=True, implicit=False, mode="slab", climatology=str(clim.path),
        climatology_sha256=clim.sha256, slab_qflux=str(qpath), slab_qflux_sha256=qsha,
        slab_depth=10.0, slab_relaxation_days=30.0, anomaly_reference="start",
        anomaly_truncation=None,
        sea_ice=SeaIceOptions(mode="match", record_interval_hours=30.0 / 3600.0,
                              new_ice_thickness_m=0.5))
    cfg = replace(_smoke(tmp_path, monkeypatch, start, options), backend="cupy")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(cfg, full)
    run(cfg, resumed, restart=full / "arwen_global_step00000004.npz")
    meta, arrays = _same_checkpoints(full / "arwen_global_step00000008.npz",
                                     resumed / "arwen_global_step00000008.npz")
    assert meta["physics_metadata"][oc.METADATA_KEY]["advances"] == 7
    assert arrays[f"physics__{oc.SLAB_TEMPERATURE}"].dtype == np.float64
