"""A world's air on the card (worlds plan, package 3.3): the forcing the
model's own radiation gives a world's CO2, methane and nitrous oxide.

The radiation is the package's existing clear-sky check,
``arwen_global.core.rrtmgp.rfmip_clear_sky``: the 100 RFMIP sites, one
longwave angle, the solar spectrum scaled to each site's irradiance, night
columns zeroed.  Here its well-mixed gases are not the experiment's own but
the ones a ``[world.air]`` table hands the radiation through
``world_air_options`` and ``NativePhysicsOptions.greenhouse_gas_overrides``,
the path a world run takes.  The forcing is the change of the net downward
flux (longwave plus shortwave) at the top of the atmosphere, averaged with
the RFMIP profile weights.

What it holds:

* the quadrupling step of CO2 forcing is about twice the doubling step
  (CO2 forcing is close to logarithmic);
* at two, four and eight times the pre-industrial CO2 and for the
  pre-industrial-to-present methane and nitrous oxide, the forcing agrees
  with the published RTE-RRTMGP reference fluxes of the same RFMIP
  experiments (rrtmgp-data v1.9, fetched and SHA-256 checked by
  ``gpuwm.core.rfmip_upstream``; never shipped).

Seconds of card.  Skipped with no card or with the RFMIP files absent from
the local cache (``gpuwm.core.rfmip_upstream.fetch_rfmip``, download off).
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _card_present() -> bool:
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - no driver, no card
        return False


pytestmark = pytest.mark.skipif(
    not _card_present(), reason="the world air forcing check needs a card")

ACK = "device-pending-arwen-native-physics-v1"

#: RFMIP experiment index -> its label in the input file (expt_label).
PD, X4, X05, X2, X3, X8, PI_CO2, PI_CH4, PI_N2O = 0, 2, 4, 5, 6, 7, 8, 9, 10

#: The RFMIP input file's global means, ppm / ppb, as its _GM variables
#: hold them (read 2026-11-12 from the pinned file).
PI_CO2_PPM = 284.3169860839844
PD_CO2_PPM = 397.54696655273437
PD_CH4_PPB = 1831.4709472656252
PI_CH4_PPB = 808.2490234375
PD_N2O_PPB = 326.9880065917969
PI_N2O_PPB = 273.02105712890625


def _files():
    from gpuwm.core.rfmip_upstream import RfmipUnavailable, fetch_rfmip

    names = ("rfmip-clear-sky-inputs.nc",
             "rfmip-clear-sky-reference-lw-down.nc",
             "rfmip-clear-sky-reference-lw-up.nc",
             "rfmip-clear-sky-reference-sw-down.nc",
             "rfmip-clear-sky-reference-sw-up.nc")
    try:
        return {name: fetch_rfmip(name, download=False) for name in names}
    except RfmipUnavailable as exc:
        pytest.skip(f"RFMIP files not in the local cache: {exc}")


def _world_gases(co2_ppm, ch4_ppb, n2o_ppb):
    """What the radiation of a world with this air is handed."""
    import tomllib

    from arwen_global.physics.native_options import (
        NativePhysicsOptions, world_air_options)
    from arwen_global.world import parse_world

    world = parse_world(tomllib.loads(
        f'[world]\nlabel = "air"\n[world.air]\nco2_ppm = {co2_ppm!r}\n'
        f'ch4_ppb = {ch4_ppb!r}\nn2o_ppb = {n2o_ppb!r}\n')["world"])
    options = world_air_options(
        {"acknowledgement": ACK, "start_time_utc": "2026-11-12T00:00:00Z"},
        world)
    gases = NativePhysicsOptions.from_mapping(options).greenhouse_gas_overrides()
    assert set(gases) == {"co2", "ch4", "n2o"}
    return gases


def _toa_net(result, top: int) -> np.ndarray:
    lw = cp.asnumpy(result.lw_dn[:, top] - result.lw_up[:, top])
    sw = cp.asnumpy(result.sw_dn[:, top] - result.sw_up[:, top])
    return np.asarray(lw + sw, dtype=np.float64)


@pytest.fixture(scope="module")
def forcing():
    from netCDF4 import Dataset

    from arwen_global.core import rrtmgp

    files = _files()
    with Dataset(files["rfmip-clear-sky-inputs.nc"]) as nc:
        nc.set_auto_mask(False)
        weight = np.asarray(nc["profile_weight"][:], np.float64)
        plev = np.asarray(nc["pres_level"][0], np.float64)
    top = int(np.argmin(plev))
    weight = weight / weight.sum()

    def mean(values):
        return float(np.sum(weight * values))

    # The published reference: net downward flux at the top, per experiment.
    reference = {}
    parts = {}
    for key, var in (("lw-down", "rld"), ("lw-up", "rlu"),
                     ("sw-down", "rsd"), ("sw-up", "rsu")):
        with Dataset(files[f"rfmip-clear-sky-reference-{key}.nc"]) as nc:
            nc.set_auto_mask(False)
            parts[key] = np.asarray(nc[var][:], np.float64)[:, :, top]
    net = parts["lw-down"] - parts["lw-up"] + parts["sw-down"] - parts["sw-up"]
    for expt in (PD, X2, X4, X8, PI_CO2, PI_CH4, PI_N2O):
        reference[expt] = mean(net[expt])

    original = rrtmgp._rfmip_profiles

    def model(co2_ppm, ch4_ppb, n2o_ppb):
        gases = _world_gases(co2_ppm, ch4_ppb, n2o_ppb)

        def profiles(tables, sites, experiments, inputs=None):
            out = list(original(tables, sites, experiments, inputs))
            vmr = out[5].copy()
            for gas, value in gases.items():
                vmr[..., tables.gas_index[gas]] = value
            out[5] = vmr
            return tuple(out)

        rrtmgp._rfmip_profiles = profiles
        try:
            # The present-day atmosphere (temperature, water, ozone, the
            # minor gases); the three gases are the world's.
            result = rrtmgp.rfmip_clear_sky(
                experiments=[PD], inputs=files["rfmip-clear-sky-inputs.nc"])
        finally:
            rrtmgp._rfmip_profiles = original
        return mean(_toa_net(result, top))

    base = model(PI_CO2_PPM, PD_CH4_PPB, PD_N2O_PPB)
    out = {
        "model": {
            "2x": model(2 * PI_CO2_PPM, PD_CH4_PPB, PD_N2O_PPB) - base,
            "4x": model(4 * PI_CO2_PPM, PD_CH4_PPB, PD_N2O_PPB) - base,
            "8x": model(8 * PI_CO2_PPM, PD_CH4_PPB, PD_N2O_PPB) - base,
            "ch4": (model(PD_CO2_PPM, PD_CH4_PPB, PD_N2O_PPB)
                    - model(PD_CO2_PPM, PI_CH4_PPB, PD_N2O_PPB)),
            "n2o": (model(PD_CO2_PPM, PD_CH4_PPB, PD_N2O_PPB)
                    - model(PD_CO2_PPM, PD_CH4_PPB, PI_N2O_PPB)),
        },
        "reference": {
            "2x": reference[X2] - reference[PI_CO2],
            "4x": reference[X4] - reference[PI_CO2],
            "8x": reference[X8] - reference[PI_CO2],
            "ch4": reference[PD] - reference[PI_CH4],
            "n2o": reference[PD] - reference[PI_N2O],
        },
    }
    for key in out["model"]:
        model_value, reference_value = out["model"][key], out["reference"][key]
        print(f"world-air forcing {key:>4s}: model {model_value:.6f} W/m2, "
              f"reference {reference_value:.6f} W/m2, "
              f"difference {model_value - reference_value:+.2e}")
    print(f"world-air 4x/2x: model {out['model']['4x'] / out['model']['2x']:.4f}, "
          f"reference {out['reference']['4x'] / out['reference']['2x']:.4f}")
    print(f"world-air cupy pool: {cp.get_default_memory_pool().total_bytes() / 2**20:.1f} MiB held")
    return out


def test_the_quadrupling_step_is_about_twice_the_doubling_step(forcing):
    model = forcing["model"]
    assert model["2x"] > 0.0
    ratio = model["4x"] / model["2x"]
    assert 1.9 <= ratio <= 2.3, ratio
    # And the next doubling adds about as much again.
    assert 2.8 <= model["8x"] / model["2x"] <= 3.6, model


@pytest.mark.parametrize("key", ["2x", "4x", "8x", "ch4", "n2o"])
def test_the_forcing_agrees_with_the_published_reference(forcing, key):
    """Measured on an RTX 4090 2026-11-12: every forcing equal to the
    reference to four decimals (2.8571, 5.8313, 9.0295, 0.6154 and 0.1808
    W/m2), so the bar is 1e-3 W/m2, not a percentage."""
    model, reference = forcing["model"][key], forcing["reference"][key]
    assert reference > 0.0
    assert abs(model - reference) <= 1.0e-3, (key, model, reference)
