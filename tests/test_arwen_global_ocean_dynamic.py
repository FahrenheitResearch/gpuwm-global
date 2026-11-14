"""The dynamic ocean (arwen_global.ocean_dyn), stage 1, on known answers.

Design: docs/ARWEN_GLOBAL_DYNAMIC_OCEAN.md.  The card gates (rest, Kelvin,
gyre, the GODAS spin-up, the coupled runs) are tools/ocean_dyn_gates.py and
the run notes; these are the parts a CPU checks in seconds:

* the equation of state against EOS-80 one-atmosphere values, its analytic
  derivatives against differences;
* the tridiagonal solve against a dense solve;
* the GRIB 1 integer and IBM-float reads;
* the conservative remap, exactly;
* a small ocean's books (heat, salt, volume) under random forcing, and its
  restart bit for bit;
* OD-1: no shipped config moves and no rule payload gains a dynamic key; the
  parser refuses the dynamic keys without the mode;
* OD-2: the T3 smoke under the dynamic ocean, straight and restarted inside
  a coupling interval, equal byte for byte in both precisions.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from arwen_global.boundary_options import OceanOptions, SeaIceOptions, parse_ocean
from arwen_global.ocean_dyn import eos
from arwen_global.ocean_dyn.column import implicit_diffusion, thomas
from arwen_global.ocean_dyn.grid import OceanGrid, idealized, level_edges, write_npz
from arwen_global.ocean_dyn.model import BOOKS, OceanModel, OceanParams


def test_wright_eos_against_eos80_one_atmosphere_values():
    for s, t, ref in ((35.0, 5.0, 1027.67547), (35.0, 25.0, 1023.34306), (0.0, 5.0, 999.96675)):
        assert abs(eos.density(t, s, 0.0) - ref) < 0.02
    rho, dt, ds = eos.density_derivatives(10.0, 35.0, 2.0e7)
    e = 1e-4
    assert abs(dt - (eos.density(10 + e, 35, 2e7) - eos.density(10 - e, 35, 2e7)) / (2 * e)) < 1e-8
    assert abs(ds - (eos.density(10, 35 + e, 2e7) - eos.density(10, 35 - e, 2e7)) / (2 * e)) < 1e-8
    assert dt < 0 < ds


def test_tridiagonal_against_a_dense_solve():
    rng = np.random.default_rng(0)
    n, m = 30, 20
    a, c = rng.uniform(-1, 0, (n, m)), rng.uniform(-1, 0, (n, m))
    b = 2.5 + rng.uniform(0, 1, (n, m))
    d = rng.normal(size=(n, m))
    x = thomas(a, b, c, d, np)
    for j in range(m):
        A = np.diag(b[:, j]) + np.diag(a[1:, j], -1) + np.diag(c[:-1, j], 1)
        assert np.allclose(x[:, j], np.linalg.solve(A, d[:, j]), rtol=0, atol=1e-12)


def test_implicit_diffusion_conserves_and_leaves_the_floor_alone():
    h = np.array([10.0, 10.0, 20.0, 0.0])[:, None]
    phi = np.array([5.0, 1.0, 0.0, 0.0])[:, None]
    k = np.array([1e-2, 1e-2, 0.0, 0.0])[:, None]
    out = implicit_diffusion(phi, h, k, 3600.0, np)
    assert abs((out * h).sum() - (phi * h).sum()) < 1e-12
    assert out[3, 0] == 0.0 and out[0, 0] < 5.0 and out[2, 0] > 0.0


def test_grib1_signed_integers_and_ibm_floats():
    from arwen_global.ocean_dyn.grib1 import _ibm, _s

    assert _s(bytes([0x80, 0x00, 0x05])) == -5 and _s(bytes([0x00, 0x01])) == 1
    # 118.625 = 0x4276A000 in IBM single precision
    assert _ibm(bytes([0x42, 0x76, 0xA0, 0x00])) == 118.625
    assert _ibm(bytes([0xC2, 0x76, 0xA0, 0x00])) == -118.625


def _tiny_grid(periodic=True):
    lat = np.linspace(-60.0, 60.0, 13)
    lon = np.arange(0.0, 361.0, 20.0)
    z = np.array([0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 400.0, 1000.0])
    rng = np.random.default_rng(1)
    depth = rng.uniform(30.0, 1000.0, (12, 18))
    depth[:, 5] = 0.0  # a continent
    depth[0, :] = 0.0
    return idealized(lat_edges_deg=lat, lon_edges_deg=lon, z_edges_m=z, depth_m=depth,
                     periodic=periodic)


def test_the_conservative_remap_is_exact():
    from arwen_global.ocean_dyn.remap import Remap
    from arwen_global.spectral.grid import GaussianGrid

    og = _tiny_grid()
    gg = GaussianGrid.create(21)
    lat = np.rad2deg(gg.lat_rad)
    rng = np.random.default_rng(2)
    ocean = rng.uniform(size=(gg.nlat, gg.nlon)) < 0.8
    rm = Remap(og, atm_lat_deg=lat, atm_weights=gg.quadrature_weights, atm_nlon=gg.nlon,
               atm_ocean=ocean, radius_m=gg.radius_m)
    f = rng.normal(size=(gg.nlat, gg.nlon))
    g, _ = rm.to_ocean(f, np)
    total_a = (f * rm.area_a * rm.owned).sum()
    total_o = (g * rm.area_o * rm.wet_o).sum()
    assert abs(total_a - total_o) <= 1e-12 * np.abs(f * rm.area_a).sum()
    back = rm.to_atmosphere(np.full((og.ny, og.nx), 3.0), np)
    assert np.allclose(back[rm.has_a], 3.0, atol=1e-13)


def _stratified(g):
    z = g.z_t_nom[:, None, None]
    T = 2.0 + 20.0 * np.exp(-z / 300.0) + np.zeros((g.nz, g.ny, g.nx))
    S = 34.7 + 0.3 * np.exp(-z / 300.0) + np.zeros((g.nz, g.ny, g.nx))
    return T, S


def _forcings(g, n, seed=3):
    rng = np.random.default_rng(seed)
    shape = (g.ny, g.nx)
    return [{"q_ns": rng.normal(-30, 150, shape), "q_sw": np.clip(rng.normal(150, 100, shape), 0, None),
             "fw": rng.normal(0, 5e-5, shape), "taux": rng.normal(0, 0.1, shape),
             "tauy": rng.normal(0, 0.1, shape)} for _ in range(n)]


def test_a_small_ocean_closes_its_books_under_random_forcing():
    g = _tiny_grid()
    m = OceanModel(g, OceanParams(sss_piston_m_per_day=None, sponge=False), np)
    st = m.new_state(*_stratified(g))
    for fc in _forcings(g, 20):
        st, _ = m.step(st, fc)
    heat, salt, vol = m.contents(st)
    b = st["books"]
    ii = BOOKS.index
    assert abs(heat - b[ii("heat_content0_j")] - b[ii("heat_sources_j")]) <= 1e-12 * heat
    assert abs(salt - b[ii("salt_content0")] - b[ii("salt_sources")]) <= 1e-12 * salt
    assert abs(vol - b[ii("volume0_m3")]) <= 1e-12 * vol
    assert np.isfinite(st["u"]).all() and np.abs(st["u"]).max() < 5.0


def test_a_small_ocean_restarts_bit_for_bit():
    g = _tiny_grid()
    fcs = _forcings(g, 6)
    m = OceanModel(g, OceanParams(), np)
    a = m.new_state(*_stratified(g))
    for fc in fcs:
        a, _ = m.step(a, fc)
    b = m.new_state(*_stratified(g))
    for fc in fcs[:3]:
        b, _ = m.step(b, fc)
    b = {k: np.array(v, copy=True) for k, v in b.items()}
    m2 = OceanModel(g, OceanParams(), np)
    for fc in fcs[3:]:
        b, _ = m2.step(b, fc)
    for name in a:
        assert np.array_equal(a[name], b[name]), name


def test_the_grid_file_is_byte_reproducible(tmp_path):
    g = _tiny_grid()
    a = g.write(tmp_path / "a.npz")
    b = g.write(tmp_path / "b.npz")
    assert a == b
    back = OceanGrid.load(tmp_path / "a.npz")
    assert np.array_equal(back.kmt, g.kmt) and back.substeps == g.substeps


# -- OD-1 ----------------------------------------------------------------------

def test_no_rule_or_slab_payload_gains_a_dynamic_key():
    assert OceanOptions().identity is None
    for mode in ("anomaly", "climatology", "slab"):
        payload = OceanOptions(declared=True, implicit=False, mode=mode).identity
        assert not any(key.startswith("dynamic_") for key in payload), mode
        assert payload["mode"] == mode


#: The shipped configs whose hash 0.2.3 moves on purpose: the two forecast
#: days turn the orographic drag on, and the T383 day calls radiation every
#: 3000 s instead of 2100 s.  Every other config v0.2.1 shipped keeps its
#: hash, the dynamic ocean's parser included.
CHANGED_IN_0_2_3 = {
    "arwen_global_gdas_t255_native_sl_si_24h.toml",
    "arwen_global_gdas_t383_native_sl_si_24h.toml",
}


def test_every_shipped_config_hash_is_the_one_v0_2_1_wrote():
    """The hashes of every shipped config with the parser of this branch are
    pinned in tests/data/ocean_dynamic_config_hashes.json, computed with
    v0.2.1's own parser (55 of the 59 shipped TOML files; the other four
    are not Arwen Global run configs and load under neither).  The configs
    0.2.3 changes on purpose are named above and must have moved."""
    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root

    pinned = json.loads((Path(__file__).parent / "data" / "ocean_dynamic_config_hashes.json")
                        .read_text())
    assert CHANGED_IN_0_2_3 <= set(pinned)
    for name, digest in pinned.items():
        if name in CHANGED_IN_0_2_3:
            assert load_config(config_root() / name).config_hash != digest, name
        else:
            assert load_config(config_root() / name).config_hash == digest, name


def test_the_parser_refuses_dynamic_keys_without_the_mode(tmp_path):
    kw = dict(source_dir=tmp_path, surface_series=None, duration_s=86400.0,
              initial_mode="analysis", statics_source="real")
    (tmp_path / "c.npz").write_bytes(b"c")
    with pytest.raises(ValueError, match="dynamic"):
        parse_ocean({"mode": "anomaly", "climatology": "c.npz", "anomaly_reference": "start",
                     "dynamic_grid": "g.npz"}, None, **kw)
    (tmp_path / "q.npz").write_bytes(b"q")
    with pytest.raises(ValueError, match="dynamic_grid"):
        parse_ocean({"mode": "dynamic", "climatology": "c.npz", "slab_qflux": "q.npz"}, None, **kw)
    (tmp_path / "g.npz").write_bytes(b"g")
    (tmp_path / "i.npz").write_bytes(b"i")
    opts = parse_ocean({"mode": "dynamic", "climatology": "c.npz", "slab_qflux": "q.npz",
                        "dynamic_grid": "g.npz", "dynamic_initial": "i.npz"}, None, **kw)
    assert opts.dynamic and opts.slab and opts.ice_mode == "thermodynamic"
    payload = opts.identity
    assert payload["mode"] == "dynamic"
    assert payload["dynamic_grid_sha256"] == hashlib.sha256(b"g").hexdigest()
    with pytest.raises(ValueError):
        parse_ocean({"mode": "dynamic", "climatology": "c.npz", "slab_qflux": "q.npz",
                     "dynamic_grid": "g.npz", "dynamic_initial": "i.npz"},
                    {"mode": "hold"}, **kw)


# -- OD-2 on the T3 reference smoke ---------------------------------------------

def _dynamic_files(tmp_path):
    """A small global ocean grid and an initial file for the T3 smoke."""
    from arwen_global.ocean_dyn.init import SCHEMA

    grid = _tiny_grid()
    grid_path = tmp_path / "ocean-grid.npz"
    grid.write(grid_path)
    T, S = _stratified(grid)
    tm = grid.tmask
    rows = np.nonzero(grid.lat_t > 60.0)[0].astype(np.int32)
    arrays = {"schema": SCHEMA, "theta": np.where(tm, T, 0.0), "salt": np.where(tm, S, 0.0),
              "u": np.zeros((grid.nz, grid.ny, grid.nx), np.float32),
              "v": np.zeros((grid.nz, grid.ny, grid.nx), np.float32),
              "eta": np.zeros((grid.ny, grid.nx)), "valid_utc": "2026-01-16T11:59:20Z",
              "provenance_json": json.dumps({"test": True, "grid_sha256": grid.sha256}),
              "sss_month": np.full((12, grid.ny, grid.nx), 34.9, np.float32),
              "sponge_t": np.zeros((12, grid.nz, rows.size, grid.nx), np.float32),
              "sponge_s": np.zeros((12, grid.nz, rows.size, grid.nx), np.float32),
              "sponge_rows": rows}
    init_path = tmp_path / "ocean-init.npz"
    write_npz(init_path, arrays)
    return grid_path, init_path


@pytest.mark.parametrize("record_s", [30.0, 3600.0])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_restarted_dynamic_run_reproduces_the_straight_run_bit_for_bit(
        tmp_path, monkeypatch, precision, record_s):
    """OD-2: the ocean steps every 20 s of the 10 s smoke (two atmosphere
    steps), so the restart at step 3 sits inside a coupling interval with a
    partial sum in the accumulators.  With the hourly ice record the restart
    also sits inside record zero: the straight run must have applied it at
    the cold start (measured on the card at T255: it did not, and the restart
    re-applied the ice on 34,727 columns)."""
    import test_arwen_global_ocean_coupling as slabtest
    from arwen_global import ocean_qflux
    from arwen_global.checkpoint import read_checkpoint
    from arwen_global.config import load_config
    from arwen_global.ocean_dyn import coupler
    from arwen_global.ocean_dyn.grid import sha256_of
    from arwen_global.runner import run

    monkeypatch.setattr(coupler, "params_from_options",
                        lambda options: OceanParams(dt=20.0, substeps=4))
    start = datetime(2026, 1, 16, 11, 59, 20, tzinfo=timezone.utc)
    base = load_config(slabtest.SMOKE)
    grid, sst, ice = slabtest._smoke_climatology(tmp_path, base.truncation, base.dealias_factor)
    clim = slabtest._climatology(tmp_path, grid, sst=sst, ice=ice)
    qpath = tmp_path / "q.npz"
    qpath.write_bytes(b"test q-flux")
    qsha = hashlib.sha256(b"test q-flux").hexdigest()
    shape = (int(grid.nlat), int(grid.nlon))
    monkeypatch.setattr(ocean_qflux, "load_qflux",
                        lambda path: slabtest.FakeQFlux(shape, flux=-20.0, sha256=qsha))
    grid_path, init_path = _dynamic_files(tmp_path)
    options = OceanOptions(
        declared=True, implicit=False, mode="dynamic", climatology=str(clim.path),
        climatology_sha256=clim.sha256, slab_qflux=str(qpath), slab_qflux_sha256=qsha,
        slab_depth=10.0, slab_relaxation_days=None, anomaly_reference="start",
        anomaly_truncation=None, dynamic_grid=str(grid_path),
        dynamic_grid_sha256=sha256_of(grid_path), dynamic_initial=str(init_path),
        dynamic_initial_sha256=sha256_of(init_path),
        sea_ice=SeaIceOptions(mode="match", record_interval_hours=record_s / 3600.0,
                              new_ice_thickness_m=0.5))
    cfg = slabtest._smoke(tmp_path, monkeypatch, start, options, precision)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    result = run(cfg, full)
    assert result["surface_boundary"]["ocean_mode"] == "dynamic"
    run(cfg, resumed, restart=full / "arwen_global_step00000004.npz")
    meta, arrays = slabtest._same_checkpoints(full / "arwen_global_step00000008.npz",
                                              resumed / "arwen_global_step00000008.npz")
    dyn = meta["physics_metadata"][coupler.METADATA_KEY]
    # the restart identity names the ocean's files by hash, never by path (a
    # run moved to another host refused to continue when it did)
    stated = meta["physics_metadata"]["surface_boundary"]["parameters"]
    assert "dynamic_grid" not in stated and "dynamic_initial" not in stated
    assert stated["dynamic_grid_sha256"] == sha256_of(grid_path)
    # advances at t = 10 .. 70 s: couplings at 20, 40 and 60 s
    assert dyn["couplings"] == 3
    names = [n for n in arrays if n.startswith("physics__ocean_dyn_")]
    assert len(names) == len(coupler.state_names())
    assert arrays["physics__ocean_dyn_theta"].dtype == np.float64
    assert arrays["physics__ocean_dyn_u"].dtype == np.float32
    b = arrays["physics__ocean_dyn_books"]
    assert b[BOOKS.index("steps")] == 3
    _meta0, arrays0 = read_checkpoint(full / "arwen_global_step00000000.npz")
    assert not np.array_equal(arrays0["physics__ocean_dyn_theta"], arrays["physics__ocean_dyn_theta"])


def test_the_tape_carries_the_ocean_planes_only_when_the_state_has_them():
    from types import SimpleNamespace

    from arwen_global import wrfout_export
    from arwen_global.ocean_coupler import SLAB_TEMPERATURE
    from arwen_global.ocean_dyn.coupler import TAPE_FIELD_ATTRS, TAPE_FIELDS, TAPE_PREFIX

    shape = (4, 8)
    skin = np.full(shape, 290.0)
    slab_only = {SLAB_TEMPERATURE: np.full(shape, 291.0)}
    with_ocean = {**slab_only, **{TAPE_PREFIX + n: np.full(shape, 1.5, np.float32)
                                  for n in TAPE_FIELDS.values()}}
    ident = lambda a: np.asarray(a)
    for arrays, expect in ((slab_only, set()), (with_ocean, set(TAPE_FIELDS))):
        bundle = SimpleNamespace(physics_state=SimpleNamespace(arrays=arrays),
                                 surface=SimpleNamespace(temperature_k=skin))
        planes = wrfout_export._slab_ocean_planes(bundle, ident)
        assert {"SST", "HML", "ARWEN_QFLUX"} <= set(planes)
        assert set(planes) - {"SST", "HML", "ARWEN_QFLUX"} == expect
    assert set(TAPE_FIELD_ATTRS) == set(TAPE_FIELDS)
