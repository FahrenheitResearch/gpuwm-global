"""Two opt-in cloud-radiation couplings: WRF's own liquid radius for Morrison
cloud, and the Tiedtke (1996) inhomogeneity factor.

``radiation_liquid_radius = "wrf_relcalc"`` radiates Morrison liquid at the
radius WRF v4.7.1's RRTMG gives it.  WRF leaves Morrison (mp_physics = 10)
out of ``has_reqc`` (``phys/module_physics_init.F:1000-1024``), so RRTMG
sizes Morrison liquid with ``relcalc`` (``phys/module_ra_rrtmg_lw.F:
14542-14601``): 8 um over warm snow-free land, 14 um over ocean and sea ice,
ramped to 14 um below freezing and under snow, on ``landm = 2 - XLAND``,
``icefrac = XICE`` and ``snowh = 0.001 * SNOW``
(``module_ra_rrtmg_sw.F:11020-11023``).

``radiation_cloud_inhomogeneity = "tiedtke1996"`` multiplies the in-cloud
liquid and ice paths both solvers radiate by 0.7 (Cahalan et al. 1994;
Tiedtke 1996).  The driver half is on the card,
``test_arwen_global_cloud_radiation_options_card.py``.

What this file holds, without a card:

* both options are off by default and, off, join no identity;
* they refuse values they cannot mean;
* relcalc's known answers, each one step of WRF's four-step ramp;
* the runtime hands the driver nothing new when the options are off, and
  under them hands the factor and a relcalc radius built on its own xland,
  leaving Morrison's persistent radius alone.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT
from arwen_global.physics import cloud_radiation_options as cro
from arwen_global.physics.native_options import NativePhysicsOptions
from arwen_global.physics.native_runtime import NativePhysicsRuntime


def _options(**extra):
    return NativePhysicsOptions.from_mapping({
        "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
        "start_time_utc": "2024-05-21T00:00:00Z",
        **extra,
    })


# ------------------------------------------------------------ the options

def test_both_default_to_the_shipped_coupling_and_join_no_identity():
    options = _options()
    assert options.radiation_liquid_radius == "scheme"
    assert options.radiation_cloud_inhomogeneity == "none"
    assert "radiation_liquid_radius" not in options.identity
    assert "radiation_cloud_inhomogeneity" not in options.identity
    spelled = _options(radiation_liquid_radius="scheme",
                       radiation_cloud_inhomogeneity="none")
    assert spelled.identity == options.identity


def test_each_option_on_joins_the_identity():
    base = _options().identity
    relcalc = _options(radiation_liquid_radius="wrf_relcalc").identity
    tiedtke = _options(radiation_cloud_inhomogeneity="tiedtke1996").identity
    assert relcalc["radiation_liquid_radius"] == "wrf_relcalc"
    assert tiedtke["radiation_cloud_inhomogeneity"] == "tiedtke1996"
    assert len({repr(sorted(x.items())) for x in (base, relcalc, tiedtke)}) == 3


@pytest.mark.parametrize("extra, message", [
    ({"radiation_liquid_radius": "kiehl"}, "radiation_liquid_radius must be"),
    ({"radiation_cloud_inhomogeneity": 0.7}, "radiation_cloud_inhomogeneity must be"),
    ({"radiation_cloud_inhomogeneity": "0.7"}, "radiation_cloud_inhomogeneity must be"),
])
def test_the_options_refuse_what_they_cannot_mean(extra, message):
    with pytest.raises(ValueError, match=message):
        _options(**extra)


def test_the_factor_is_the_published_one_and_nothing_else():
    assert cro.CLOUD_INHOMOGENEITY_FACTORS == {"none": None, "tiedtke1996": 0.7}


# ----------------------------------------------------- relcalc's answers

def _relcalc(temperature, landm, icefrac=0.0, snow=0.0):
    t = np.full((1, 1, 1), temperature, np.float32)
    plane = lambda v: np.full((1, 1), v, np.float32)  # noqa: E731
    return float(cro.wrf_relcalc(t, plane(landm), plane(icefrac),
                                 plane(snow))[0, 0, 0])


def test_relcalc_constants_are_wrfs():
    """module_ra_rrtmg_lw.F:14582-14585, WRF v4.7.1."""
    assert (cro.RELCALC_TMELT_K, cro.RELCALC_RLIQ_OCEAN_UM,
            cro.RELCALC_RLIQ_ICE_UM, cro.RELCALC_RLIQ_LAND_UM) == (
                273.16, 14.0, 14.0, 8.0)


@pytest.mark.parametrize("temperature, landm, icefrac, snow, expected", [
    (290.0, 0.0, 0.0, 0.0, 14.0),     # warm ocean
    (290.0, 1.0, 0.0, 0.0, 8.0),      # warm snow-free land
    (263.16, 1.0, 0.0, 0.0, 11.0),    # land, 10 K below tmelt: half way
    (253.16, 1.0, 0.0, 0.0, 14.0),    # land, 20 K below: the ocean value
    (230.0, 1.0, 0.0, 0.0, 14.0),     # colder still: held there
    (290.0, 1.0, 0.0, 50.0, 11.0),    # 0.05 m of snow water: half way
    (290.0, 1.0, 0.0, 200.0, 14.0),   # deep snow: the ocean value
    (290.0, 0.25, 0.0, 0.0, 12.5),    # three quarters water
    (290.0, 1.0, 0.6, 0.0, 11.6),     # a sea-ice column (xland 1)
    (290.0, 1.0, 1.0, 0.0, 14.0),     # full ice
])
def test_relcalc_known_answers(temperature, landm, icefrac, snow, expected):
    assert _relcalc(temperature, landm, icefrac, snow) == pytest.approx(
        expected, abs=1.0e-5)


def test_relcalc_is_float32_contiguous_and_layered():
    t = np.linspace(250.0, 290.0, 24, dtype=np.float32).reshape(4, 2, 3)
    landm = np.array([[1.0, 0.0, 0.5], [1.0, 1.0, 0.0]], np.float32)
    out = cro.wrf_relcalc(t, landm, np.zeros_like(landm), np.zeros_like(landm))
    assert out.dtype == np.float32 and out.flags.c_contiguous
    assert out.shape == t.shape
    assert float(out.min()) >= 8.0 and float(out.max()) <= 14.0
    # Ocean columns are 14 um at every level, whatever the temperature.
    assert np.all(out[:, 0, 1] == np.float32(14.0))


# -------------------------------------------------------------- the runtime

def _batch(land_fraction, sea_ice, temperature):
    land_fraction = np.asarray(land_fraction, np.float32).reshape(1, -1)
    sea_ice = np.asarray(sea_ice, np.float32).reshape(1, -1)
    nz = 3
    return SimpleNamespace(
        xp=np,
        arrays={"temperature": np.full((nz, *land_fraction.shape),
                                       temperature, np.float32)},
        surface=SimpleNamespace(land_fraction=land_fraction,
                                sea_ice_fraction=sea_ice),
        surface_shape=land_fraction.shape)


def _persistent(shape, nz=3):
    effc = np.full((nz, *shape), 4.0, np.float32)
    return SimpleNamespace(
        arrays={"noah_snow": np.zeros(shape, np.float32), "effc": effc},
        fake_radiation_state=lambda: SimpleNamespace(effc=effc, qc=None))


def _owner(**extra):
    return SimpleNamespace(options=_options(**extra),
                           _xland=NativePhysicsRuntime._xland)


def test_off_the_runtime_hands_the_driver_what_it_always_did():
    owner = _owner()
    assert NativePhysicsRuntime._cloud_inhomogeneity_kwargs(owner) == {}
    batch = _batch([0.0, 1.0], [0.0, 0.0], 290.0)
    persistent = _persistent(batch.surface_shape)
    state = NativePhysicsRuntime._radiation_state(owner, batch, persistent)
    assert state.effc is persistent.arrays["effc"]


def test_on_the_runtime_hands_the_factor():
    owner = _owner(radiation_cloud_inhomogeneity="tiedtke1996")
    assert NativePhysicsRuntime._cloud_inhomogeneity_kwargs(owner) == {
        "cloud_path_scale": 0.7}


def test_on_the_runtime_radiates_relcalc_on_its_own_xland():
    """Open water 14, warm land 8, a 0.3-land coast 12.2 (landm = 2 - xland
    = land_fraction), a sea-ice column 14 (xland 1, then the ice ramp);
    Morrison's persistent radius is left as it was."""
    owner = _owner(radiation_liquid_radius="wrf_relcalc")
    batch = _batch([0.0, 1.0, 0.3, 0.0], [0.0, 0.0, 0.0, 1.0], 290.0)
    persistent = _persistent(batch.surface_shape)
    before = persistent.arrays["effc"].copy()
    state = NativePhysicsRuntime._radiation_state(owner, batch, persistent)
    np.testing.assert_allclose(
        state.effc[0, 0], [14.0, 8.0, 12.2, 14.0], atol=1.0e-5)
    assert state.effc.shape == persistent.arrays["effc"].shape
    np.testing.assert_array_equal(persistent.arrays["effc"], before)
