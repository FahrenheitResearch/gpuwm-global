"""Morrison's fixed droplet number: WRF's one value, or a land and an ocean one.

WRF's Morrison (``module_mp_morr_two_moment.F``) runs its ``INUM = 1`` arm
outside WRF-Chem: every call resets the cloud droplet number to one constant,
``NDCNST = 250.`` cm-3 (F:271-276, F:1493-1495).  The carried kernel does the
same twice per call (``morrison.cu`` morr_bound on entry to the process
section, and after the final radii in ``morrison_finalize_levels``).

``morrison_droplet_number = "land_ocean"`` is a DECLARED DEPARTURE from that
kernel: the constant becomes one value per column, the land value where the
model calls the column land and the ocean value everywhere else.  The default
pair is MPAS-Thompson's landmask prescription, 300 cm-3 over land and
100 cm-3 over ocean (``init_thompson_clouddroplets_forMPAS``), the
maritime/continental guidance WRF's own Thompson prints beside its single
``Nt_c`` (``module_mp_thompson.F``, WRF 4.7.1, lines 80-88).

What this file holds, without a card:

* the option is off by default and, off, joins no identity, so every
  checkpoint hash stays what it was;
* the float64 mirror, handed WRF's own value, is the mirror it was;
* an ocean column leaves the scheme carrying the ocean number;
* the liquid effective radius moves with the number exactly as Morrison's
  gamma distribution says it must: the cube root of the number ratio where
  the shape parameter is clamped, and the cube root times the shape term
  where it is not (lower number, narrower spectrum, a smaller rise);
* autoconversion responds as Khairoutdinov-Kogan's N^-1.79 says, which is
  the second, real effect: fewer droplets drizzle sooner and the cloud
  keeps less water;
* the runtime builds the plane from the land fraction on the model's own
  split, with sea ice taking the ocean value.

The two card nodes (the plane build handed 250 cm-3 everywhere is the WRF
build bit for bit; on a land/ocean plane each column leaves with its own
number) are ``test_arwen_global_morrison_droplet_number_card.py``, kept apart
so this file stays in the CPU tier.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import constants as c

from arwen_global.core import npref
from arwen_global.physics.native_options import NativePhysicsOptions
from arwen_global.physics.native_runtime import NativePhysicsRuntime
from arwen_global.constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT

ROOT = Path(__file__).resolve().parents[1]
KERNEL_SRC = ROOT / "src" / "arwen_global" / "core" / "kernels" / "morrison.cu"

#: WRF's NDCNST (F:276) and the MPAS-Thompson pair, in m-3.
WRF_M3 = 250.0e6
LAND_M3 = 300.0e6
OCEAN_M3 = 100.0e6


def _options(**extra):
    return NativePhysicsOptions.from_mapping({
        "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
        "start_time_utc": "2024-05-21T00:00:00Z",
        **extra,
    })


# ------------------------------------------------------------ the option

def test_the_default_is_wrfs_single_number_and_joins_no_identity():
    options = _options()
    assert options.morrison_droplet_number == "fixed"
    identity = options.identity
    for key in ("morrison_droplet_number",
                "morrison_droplet_number_land_cm3",
                "morrison_droplet_number_ocean_cm3"):
        assert key not in identity
    # Off, the two values are not read, so moving them moves no hash.
    moved = _options(morrison_droplet_number_land_cm3=500.0)
    assert moved.identity == identity


def test_the_departure_joins_the_identity_with_its_two_values():
    options = _options(morrison_droplet_number="land_ocean")
    identity = options.identity
    assert identity["morrison_droplet_number"] == "land_ocean"
    assert identity["morrison_droplet_number_land_cm3"] == 300.0
    assert identity["morrison_droplet_number_ocean_cm3"] == 100.0
    assert identity != _options().identity
    other = _options(morrison_droplet_number="land_ocean",
                     morrison_droplet_number_ocean_cm3=50.0)
    assert other.identity != identity


@pytest.mark.parametrize("extra, message", [
    ({"morrison_droplet_number": "ocean"}, "morrison_droplet_number must be"),
    ({"morrison_droplet_number_land_cm3": 0.0}, "land_cm3"),
    ({"morrison_droplet_number_ocean_cm3": 5000.0}, "ocean_cm3"),
    ({"morrison_droplet_number_ocean_cm3": True}, "ocean_cm3"),
    ({"morrison_droplet_number_land_cm3": float("nan")}, "land_cm3"),
])
def test_the_option_refuses_what_it_cannot_mean(extra, message):
    with pytest.raises(ValueError, match=message):
        _options(**extra)


# ------------------------------------------------------- the kernel text

def test_the_wrf_build_reads_the_literal_ndcnst_in_both_kernels():
    """Without MORR_DROPLET_PLANE both resets read WRF's 250 cm-3."""

    text = KERNEL_SRC.read_text(encoding="utf-8")
    fallbacks = text.count(
        "#else\n    real nc_fixed = 250.0e6f;   // WRF NDCNST (F:276)")
    assert fallbacks == 2, fallbacks
    assert text.count("nc_plane[idx % (size_t)ncol]") == 2
    assert "*nc = nc_fixed / rhoa;" in text
    assert "nc[idx] = nc_fixed / rhoa;" in text
    assert "250.0e6f / rhoa" not in text


# ------------------------------------------------------- the mirror

def _warm_cloud_column(qc=5.0e-4):
    """Three saturated warm levels holding cloud water and nothing else."""

    pressure = np.array([90000.0, 85000.0, 80000.0])
    temperature = np.array([288.0, 285.0, 282.0])
    pii = (pressure / c.P0) ** c.RCP
    ew = np.minimum(0.99 * pressure,
                    npref._np_morrison_polysvp(temperature, False))
    qvs = c.EP2 * ew / (pressure - ew)
    zero = np.zeros(3)
    rho = pressure / (c.RD * temperature)
    return dict(
        theta=temperature / pii, qv=qvs.copy(), qc=np.full(3, qc),
        qr=zero.copy(), qi=zero.copy(), qs=zero.copy(), qg=zero.copy(),
        nc=np.full(3, WRF_M3) / rho, nr=zero.copy(), ni=zero.copy(),
        ns=zero.copy(), ng=zero.copy(), rho=rho, pii=pii,
        pressure=pressure, dz=np.full(3, 500.0))


def test_the_mirror_given_wrfs_value_is_the_mirror_it_was():
    column = _warm_cloud_column()
    before = npref.np_morrison_column(**column, dt=60.0)
    named = npref.np_morrison_column(**column, dt=60.0,
                                     droplet_number_m3=WRF_M3)
    assert before.keys() == named.keys()
    for key, value in before.items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(named[key]),
                                      err_msg=key)


def test_an_ocean_column_leaves_with_the_ocean_number():
    column = _warm_cloud_column()
    out = npref.np_morrison_column(**column, dt=60.0,
                                   droplet_number_m3=OCEAN_M3)
    t = out["theta"] * out["pii"]
    # The finalize reset divides by the density diagnosed on entry
    # (WRF :1325), which the mirror and the kernel both carry.
    entry_t = column["theta"] * column["pii"]
    rhoa = column["pressure"] / (c.RD * entry_t)
    np.testing.assert_array_equal(out["nc"], OCEAN_M3 / rhoa)
    assert np.all(np.isfinite(t))


def test_the_mirror_refuses_a_number_that_is_not_one():
    with pytest.raises(ValueError, match="droplet_number_m3"):
        npref.np_morrison_column(**_warm_cloud_column(), dt=60.0,
                                 droplet_number_m3=0.0)


def _effc_um(number_m3, qc=3.0e-4, temperature=283.0, pressure=85000.0):
    """Morrison's liquid effective radius (F:3920-3947) at fixed water."""

    rhoa = pressure / (c.RD * temperature)
    q = {"qc": qc, "qr": 0.0, "qi": 0.0, "qs": 0.0, "qg": 0.0}
    n = {"nc": 0.0, "nr": 0.0, "ni": 0.0, "ns": 0.0, "ng": 0.0}
    lam, pgam, nn = npref._np_morrison_level_moments(
        q, n, rhoa, temperature, pressure, reset_cloud=True,
        droplet_number_m3=number_m3)
    assert nn["nc"] == pytest.approx(number_m3 / rhoa, rel=1e-15)
    return (pgam + 3.0) / (2.0 * lam["c"]) * 1.0e6, pgam


def _shape_term(pgam):
    """effc / (q/N)^(1/3) up to constants: (mu+3) / ((mu+1)(mu+2)(mu+3))^(1/3)."""

    return (pgam + 3.0) / ((pgam + 1.0) * (pgam + 2.0) * (pgam + 3.0)) ** (1 / 3)


def test_effc_scales_as_the_cube_root_where_the_shape_is_clamped():
    """Both numbers above about 535 cm-3 hold the shape at 2 (F:1558-1560)."""

    low, mu_low = _effc_um(600.0e6)
    high, mu_high = _effc_um(1200.0e6)
    assert mu_low == mu_high == 2.0
    assert low / high == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-12)


def test_effc_between_the_land_and_ocean_numbers_carries_the_shape_term():
    """At 100 against 300 cm-3 the spectrum also narrows.

    Morrison's dispersion (Martin et al. 1994 as WRF codes it, F:1558) makes
    the shape parameter about 8.3 at 100 cm-3 and about 4.1 at 300 cm-3,
    and the narrower spectrum takes back part of the cube-root rise.
    """

    ocean, mu_ocean = _effc_um(OCEAN_M3)
    land, mu_land = _effc_um(LAND_M3)
    wrf, mu_wrf = _effc_um(WRF_M3)
    assert 8.0 < mu_ocean < 8.6 and 3.9 < mu_land < 4.3
    expected = (3.0 ** (1 / 3)) * _shape_term(mu_ocean) / _shape_term(mu_land)
    assert ocean / land == pytest.approx(expected, rel=1e-12)
    # The rise is real and smaller than the bare cube root.
    assert 1.0 < ocean / land < 3.0 ** (1 / 3)
    # Against WRF's 250 cm-3: the ratio the ocean columns see.
    against_wrf = ocean / wrf
    assert against_wrf == pytest.approx(
        (2.5 ** (1 / 3)) * _shape_term(mu_ocean) / _shape_term(mu_wrf),
        rel=1e-12)
    assert 1.25 < against_wrf < 2.5 ** (1 / 3)


def test_autoconversion_answers_to_the_number_as_kk2000_says():
    """PRC = 1350 qc^2.47 (N/cm-3)^-1.79 (morrison.cu:413-414, F:1667ff).

    On a saturated warm cloud with no rain on entry, autoconversion is the
    only rain source, so the rain each level holds after the call at
    100 cm-3 against 250 cm-3 is 2.5^1.79, about 5.16 times larger (the
    rain that falls on through the step falls in the same proportion), and
    the cloud keeps less water.

    The surface total is NOT that ratio, and the difference is a third
    channel: Morrison sediments cloud droplets at their Stokes speed
    (F:3440-3441), so fewer, larger droplets in a cloud that touches the
    ground fall out faster.  Measured here at dt = 60 s: 4.86e-4 against
    7.40e-4 kg m-2 reaching the surface.
    """

    dt = 60.0
    column = _warm_cloud_column()
    wrf = npref.np_morrison_column(**column, dt=dt, droplet_number_m3=WRF_M3)
    ocean = npref.np_morrison_column(**column, dt=dt,
                                     droplet_number_m3=OCEAN_M3)
    assert np.all(wrf["qr"] > 0.0)
    np.testing.assert_allclose(ocean["qr"] / wrf["qr"], 2.5 ** 1.79,
                               rtol=1e-3)
    np.testing.assert_allclose(ocean["nr"] / wrf["nr"], 2.5 ** 1.79,
                               rtol=1e-3)
    assert np.all(ocean["qc"] < wrf["qc"])
    assert ocean["precip_step"] > wrf["precip_step"]


# ------------------------------------------------------- the runtime plane

def _batch(land_fraction, sea_ice=None):
    land_fraction = np.asarray(land_fraction, np.float32).reshape(1, -1)
    sea_ice = (np.zeros_like(land_fraction) if sea_ice is None
               else np.asarray(sea_ice, np.float32).reshape(1, -1))
    return SimpleNamespace(
        xp=np,
        surface=SimpleNamespace(land_fraction=land_fraction,
                                sea_ice_fraction=sea_ice),
        surface_shape=land_fraction.shape)


def test_the_runtime_builds_no_plane_under_wrfs_number():
    owner = SimpleNamespace(options=_options())
    assert NativePhysicsRuntime._droplet_number_plane(
        owner, _batch([0.0, 1.0])) is None


def test_the_runtime_plane_follows_the_models_own_land_split():
    """Land above one half is land (xland below 1.5); sea ice is ocean."""

    owner = SimpleNamespace(options=_options(
        morrison_droplet_number="land_ocean"))
    land_fraction = [0.0, 0.3, 0.5, 0.51, 0.9, 1.0, 0.0]
    sea_ice = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    plane = NativePhysicsRuntime._droplet_number_plane(
        owner, _batch(land_fraction, sea_ice))
    assert plane.dtype == np.float32 and plane.flags.c_contiguous
    assert plane.shape == (1, 7)
    expected = np.array([[OCEAN_M3, OCEAN_M3, OCEAN_M3, LAND_M3, LAND_M3,
                          LAND_M3, OCEAN_M3]], np.float32)
    np.testing.assert_array_equal(plane, expected)


def test_the_runtime_plane_carries_the_configured_values():
    owner = SimpleNamespace(options=_options(
        morrison_droplet_number="land_ocean",
        morrison_droplet_number_land_cm3=150.0,
        morrison_droplet_number_ocean_cm3=50.0))
    plane = NativePhysicsRuntime._droplet_number_plane(
        owner, _batch([0.0, 1.0]))
    np.testing.assert_array_equal(
        plane, np.array([[50.0e6, 150.0e6]], np.float32))
