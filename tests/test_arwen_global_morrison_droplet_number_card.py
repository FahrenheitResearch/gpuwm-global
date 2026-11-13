"""The droplet-number departure on a card: the WRF build stays the WRF build.

The CPU half of this lives in ``test_arwen_global_morrison_droplet_number.py``;
these two nodes need the compiled kernels, so they sit in their own module
(the suite marks a whole module ``gpu`` when it imports cupy anywhere, and
the CPU half must run in the CPU tier).

* The ``MORR_DROPLET_PLANE`` build handed WRF's 250 cm-3 on every column is
  the WRF build, word for word, in every field the scheme writes.
* On a mixed plane each column leaves with its own droplet number, a column
  holding 250 cm-3 is the WRF build's column word for word, ocean columns
  hold larger droplets than land columns, and both agree with the float64
  mirror handed the same number.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from gpuwm.core import constants as c

from arwen_global.core import npref

WRF_M3 = 250.0e6
LAND_M3 = 300.0e6
OCEAN_M3 = 100.0e6


def _card():
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - no driver, no device: no card
        return False


needs_a_card = pytest.mark.skipif(
    not _card(), reason="needs a CUDA device and cupy")


def _device_batch(cp, nz=6, ny=2, nx=3):
    """A small warm-and-cold cloudy batch, every column the same profile."""

    pressure = np.linspace(95000.0, 45000.0, nz)
    temperature = np.linspace(290.0, 250.0, nz)
    pii = (pressure / c.P0) ** c.RCP
    ew = np.minimum(0.99 * pressure,
                    npref._np_morrison_polysvp(temperature, False))
    qvs = c.EP2 * ew / (pressure - ew)
    rho = pressure / (c.RD * temperature)

    def volume(profile):
        return cp.asarray(np.broadcast_to(
            np.asarray(profile, np.float32)[:, None, None], (nz, ny, nx)).copy())

    fields = {
        "theta": volume(temperature / pii), "qv": volume(qvs),
        "qc": volume(np.full(nz, 4.0e-4)), "qr": volume(np.full(nz, 1.0e-5)),
        "qi": volume(np.where(temperature < 268.0, 2.0e-5, 0.0)),
        "qs": volume(np.zeros(nz)), "qg": volume(np.zeros(nz)),
        "nc": volume(np.full(nz, WRF_M3) / rho),
        "nr": volume(np.full(nz, 1.0e3)),
        "ni": volume(np.where(temperature < 268.0, 1.0e4, 0.0)),
        "ns": volume(np.zeros(nz)), "ng": volume(np.zeros(nz)),
        "rho": volume(rho), "pii": volume(pii), "pressure": volume(pressure),
        "dz": volume(np.full(nz, 400.0)),
    }
    surface = {name: cp.zeros((ny, nx), cp.float32) for name in (
        "rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv",
        "sr")}
    effective = {name: cp.zeros((nz, ny, nx), cp.float32)
                 for name in ("effc", "effr", "effi", "effs")}
    return fields, surface, effective


def _launch(cp, plane=None):
    from arwen_global.core.morrison import launch_morrison

    fields, surface, effective = _device_batch(cp)
    kwargs = {} if plane is None else {"droplet_number_m3": plane}
    launch_morrison(
        fields["theta"], fields["qv"], fields["qc"], fields["qr"],
        fields["qi"], fields["qs"], fields["qg"], fields["nc"], fields["nr"],
        fields["ni"], fields["ns"], fields["ng"], fields["rho"],
        fields["pii"], fields["pressure"], fields["dz"],
        surface["rainnc"], surface["rainncv"], surface["snownc"],
        surface["snowncv"], surface["graupelnc"], surface["graupelncv"],
        surface["sr"], 60.0, **effective, **kwargs)
    cp.cuda.runtime.deviceSynchronize()
    out = {name: cp.asnumpy(value) for name, value in fields.items()}
    out.update({name: cp.asnumpy(value) for name, value in surface.items()})
    out.update({name: cp.asnumpy(value) for name, value in effective.items()})
    return out


@pytest.mark.gpu
@needs_a_card
def test_the_plane_build_handed_wrfs_number_is_the_wrf_build_bit_for_bit():
    import cupy as cp

    wrf = _launch(cp)
    plane = cp.full((2, 3), WRF_M3, dtype=cp.float32)
    built = _launch(cp, plane)
    for key in wrf:
        np.testing.assert_array_equal(
            wrf[key].view(np.uint32), built[key].view(np.uint32), err_msg=key)


@pytest.mark.gpu
@needs_a_card
def test_each_column_leaves_the_kernel_with_its_own_number():
    import cupy as cp

    numbers = np.array([[OCEAN_M3, LAND_M3, OCEAN_M3],
                        [LAND_M3, OCEAN_M3, WRF_M3]], np.float32)
    out = _launch(cp, cp.asarray(numbers))
    wrf = _launch(cp)
    fields, _surface, _effective = _device_batch(cp)
    pressure = cp.asnumpy(fields["pressure"]).astype(np.float64)
    temperature = (cp.asnumpy(fields["theta"]).astype(np.float64)
                   * cp.asnumpy(fields["pii"]).astype(np.float64))
    rhoa = pressure / (c.RD * temperature)
    np.testing.assert_allclose(out["nc"], numbers[None] / rhoa, rtol=2e-6)
    # The WRF-valued column is the WRF build's column, word for word.
    for key in ("qc", "qr", "nc", "effc", "theta", "qv"):
        np.testing.assert_array_equal(out[key][:, 1, 2], wrf[key][:, 1, 2],
                                      err_msg=key)
    # Ocean columns hold larger droplets than land columns wherever both
    # hold cloud water.
    cloudy = (out["qc"][:, 0, 0] > 1.0e-6) & (out["qc"][:, 0, 1] > 1.0e-6)
    assert cloudy.any()
    assert np.all(out["effc"][cloudy, 0, 0] > out["effc"][cloudy, 0, 1])
    # And against the float64 mirror, column by column, on the cloud fields.
    for (j, i) in ((0, 0), (0, 1)):
        column = {name: cp.asnumpy(value)[:, j, i].astype(np.float64)
                  for name, value in fields.items()}
        mirror = npref.np_morrison_column(
            **column, dt=60.0, droplet_number_m3=float(numbers[j, i]))
        np.testing.assert_allclose(out["nc"][:, j, i], mirror["nc"], rtol=2e-6)
        np.testing.assert_allclose(out["effc"][:, j, i], mirror["effc"],
                                   rtol=2e-3)
