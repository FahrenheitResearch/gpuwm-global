"""The render tape carries SST, as every WRF history file does.

THE BREAKAGE THIS GATE PREVENTS.  A reader of WRF files asks for ``SST`` by
name for a sea-surface map or anomaly, and a tape without it is refused or
drawn from another field the reader picks for itself.  The model has no
ocean model: its open water's skin is the water temperature it ran with
(held at the analysis's, or following a dated series), so the tape's SST
is the skin, ``TSK``.  On land and under sea ice that is also what WRF's
real.exe writes when its input carries no SST of its own (WPS's GFS table
carries none): it fills SST from TSK on every point.

So SST is pinned here to TSK on every point, land and water both present,
and to the model's own skin sampled by the reference oracle, not by the
exporter.  None of it needs a device.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_netcdf_writer, requires_render_kernels  # noqa: E402

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root

START = "2026-08-30_18:00:00"


@pytest.fixture(scope="module")
def tapes(tmp_path_factory):
    """The moist smoke configuration run once and exported on 18 x 36."""
    from arwen_global.runner import run
    from arwen_global.wrfout_export import export_wrfout

    root = tmp_path_factory.mktemp("tape-sst")
    cfg = load_config(config_root() / "arwen_global_moist_smoke.toml")
    run(cfg, root / "run")
    checkpoints = sorted((root / "run").glob("arwen_global_step*.npz"))
    assert len(checkpoints) >= 2
    written = export_wrfout(
        cfg, checkpoints, root / "tapes", nlat=18, nlon=36, start_date=START)
    return cfg, checkpoints, written, root / "tapes"


@requires_netcdf_writer
@requires_render_kernels
def test_every_tape_carries_sst_and_it_is_the_skin_on_every_point(tapes):
    from netCDF4 import Dataset

    from render_kernel_oracle import gaussian_to_regular

    from arwen_global.checkpoint import read_checkpoint, state_from_checkpoint
    from arwen_global.runner import (
        adopt_checkpoint_clock, build_model_and_cold_state, build_transform)
    from arwen_global.wrfout_export import EXPORT_RECEIPT_NAME, TAPE_SST_POLICY

    cfg, checkpoints, written, outdir = tapes
    transform = build_transform(cfg)
    model, _cold = build_model_and_cold_state(cfg, transform)
    assert len(written) == len(checkpoints)
    for checkpoint, tape in zip(checkpoints, written):
        with Dataset(tape) as ds:
            variable = ds.variables["SST"]
            assert variable.dimensions == ("Time", "south_north", "west_east")
            assert variable.dtype == np.float32
            assert variable.getncattr("units") == "K"
            sst = np.asarray(variable[0])
            tsk = np.asarray(ds["TSK"][0])
            xland = np.asarray(ds["XLAND"][0])
            xlat = np.asarray(ds["XLAT"][0, :, 0], dtype=np.float64)
            xlong = np.asarray(ds["XLONG"][0, 0, :], dtype=np.float64)
        # The skin on every point, bit for bit, land and open water alike.
        np.testing.assert_array_equal(sst, tsk)
        assert 0 < int((xland == 2.0).sum()) < xland.size   # both are on the tape
        assert np.all(np.isfinite(sst)) and np.all(sst > 150.0) and np.all(sst < 350.0)
        # And that skin is the model's own, sampled by the reference oracle
        # rather than by the exporter's kernels.
        metadata, arrays = read_checkpoint(
            checkpoint, expected_config_hash=cfg.config_hash,
            semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator)
        bundle = state_from_checkpoint(metadata, arrays, transform.backend)
        adopt_checkpoint_clock(model, bundle, f"checkpoint {checkpoint}")
        model.enforce(bundle)
        skin = np.asarray(transform.backend.to_numpy(bundle.surface.temperature_k))
        regrid = gaussian_to_regular(
            transform.grid.latitude_deg, transform.grid.longitude_deg,
            xlat, np.mod(xlong, 360.0))
        np.testing.assert_allclose(sst, regrid(skin).astype(np.float32), rtol=1.0e-6)
    receipt = json.loads((outdir / EXPORT_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["sea_surface_temperature"] == TAPE_SST_POLICY


def test_the_policy_names_land_ice_and_the_mask():
    from arwen_global.wrfout_export import TAPE_SST_POLICY

    for phrase in ("skin temperature (TSK) on every point", "under sea ice",
                   "real.exe fills SST from TSK", "XLAND or SEAICE"):
        assert phrase in TAPE_SST_POLICY, phrase


def test_the_page_says_what_the_tapes_sst_is():
    page = Path(__file__).resolve().parents[1] / "docs" / "ARWEN_GLOBAL.md"
    if not page.is_file():
        pytest.skip("the docs are not beside this test")
    text = " ".join(page.read_text(encoding="utf-8").split())
    for phrase in ("`SST` is the skin temperature", "`sea_surface_temperature`",
                   "fills `SST` from `TSK`"):
        assert phrase in text, phrase
