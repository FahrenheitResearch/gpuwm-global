"""GI-5 (audit 2026-10-05): the render tape writes WRF's moisture names in
WRF's convention.

The model carries every water field as specific humidity (kg per kg of
moist air); WRF's QVAPOR, QCLOUD, ... and Q2 are dry mixing ratios (kg per
kg of dry air), and every WRF-shaped reader (the renderer's dewpoint and
RH, the ABI reference columns) reads them so.  Writing q under those names
read 2 percent dry at q = 0.02, a 0.33 K dewpoint bias in the tropics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import requires_netcdf_writer  # noqa: E402

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs

SMOKE_CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _dewpoint_from_mixing_ratio(w, p_pa):
    """WRF's own reading of QVAPOR: e = w p / (eps + w), Bolton inversion."""
    e = w * p_pa / (0.622 + w)
    ln = np.log(e / 611.2)
    return 273.15 + 243.5 * ln / (17.67 - ln)


def test_tape_water_fields_are_dry_mixing_ratios_and_the_dewpoint_round_trips():
    from arwen_global.surface_energy import dewpoint_from_specific_humidity
    from arwen_global.wrfout_export import tape_water_fields

    q = np.asarray([0.002, 0.010, 0.020])
    qc = np.asarray([0.0, 1.0e-4, 5.0e-4])
    p = np.full(3, 100_000.0)
    species = {"qv": q, "qc": qc, "qr": 0 * q, "qi": 0 * q, "qs": 0 * q, "qg": 0 * q}
    tape, q2 = tape_water_fields(species, q)
    np.testing.assert_allclose(tape["qv"], q / (1.0 - q), rtol=1.0e-15)
    np.testing.assert_allclose(tape["qc"], qc / (1.0 - q), rtol=1.0e-15)
    np.testing.assert_allclose(q2, q / (1.0 - q), rtol=1.0e-15)
    # The dewpoint a WRF reader draws from the tape is the model's own.
    np.testing.assert_allclose(
        _dewpoint_from_mixing_ratio(tape["qv"], p),
        dewpoint_from_specific_humidity(q, p), atol=0.02)
    # And the defect it replaces: q read as w is 0.33 K dry at q = 0.02.
    dry = dewpoint_from_specific_humidity(q, p) - _dewpoint_from_mixing_ratio(q, p)
    assert dry[-1] > 0.3


@requires_netcdf_writer
def test_the_tape_writes_the_mixing_ratio_of_the_model_water(tmp_path, monkeypatch):
    from netCDF4 import Dataset

    import arwen_global.wrfout_export as wrfout_export
    from arwen_global.runner import run

    cfg = load_config(SMOKE_CONFIG)
    receipt = run(cfg, tmp_path / "run")
    checkpoint = Path(receipt["checkpoints"][-1])

    def export(where):
        (tape,) = wrfout_export.export_wrfout(
            cfg, [checkpoint], tmp_path / where, nlat=18, nlon=36,
            start_date="2026-08-30_18:00:00",
        )
        with Dataset(tape) as ds:
            return (
                {name: np.asarray(ds[name][0], dtype=np.float64)
                 for name in ("QVAPOR", "QCLOUD", "Q2")},
                ds.getncattr(wrfout_export.WATER_CONVENTION_ATTR),
            )

    tape, convention = export("tapes")
    assert "mixing ratio" in convention
    # The same export with the model's specific humidities written as they
    # are (the pre-fix tape): the real tape is w = q / (1 - q) of it.
    monkeypatch.setattr(
        wrfout_export, "tape_water_fields", lambda species, q2: (species, q2))
    raw, _ = export("raw")
    q = raw["QVAPOR"]
    assert float(q.max()) > 1.0e-3
    np.testing.assert_allclose(tape["QVAPOR"], q / (1.0 - q), rtol=2.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(
        tape["QCLOUD"], raw["QCLOUD"] / (1.0 - q), rtol=2.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(
        tape["Q2"], raw["Q2"] / (1.0 - raw["Q2"]), rtol=2.0e-6, atol=1.0e-12)
