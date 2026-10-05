"""The semi-Lagrangian step split by latitude band, and across cards.

Under the default ``sl_si`` core a multi-card run used to give the second
card almost nothing to do: the trajectory, the departure gather and the
tendencies took the whole grid on every rank (audit 2026-10-05, MG-1).
The banded step runs them one latitude band at a time behind a row halo,
on the bands a card owns, and meets the replicated spectral state at the
Fourier waist.  These gates hold it to the answer:

*   BAND-SL-1: the band window's gather, departure search, Lipschitz
    diagnostic and parallel transport read the whole grid's bits at every
    row they compute.
*   BAND-SL-2: a departure point outside the band's halo is refused by
    name, and the step recomputes the band behind a wider halo with the
    same answer.
*   BAND-SL-3: the banded step is the resident step, over several steps
    and at several band counts, every spectral and grid array and every
    metric.
*   BAND-SL-4: two cards over loopback TCP return one card's bits at the
    same band schedule, and do split the work: each card runs only its
    own bands.
"""
from __future__ import annotations

import hashlib
import socket
import threading

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global import cards  # noqa: E402
from arwen_global.dynamics import MoistHybridModel  # noqa: E402
from arwen_global.semi_implicit import VerticalModeSemiImplicit  # noqa: E402
from arwen_global.semilag import SemiLagrangianOptions  # noqa: E402
from arwen_global.semilag import step as step_module  # noqa: E402
from arwen_global.semilag.halo import default_halo_rows  # noqa: E402
from arwen_global.semilag.state import RowTrajectory  # noqa: E402
from arwen_global.semilag.interpolate import gather_batch  # noqa: E402
from arwen_global.semilag.tables import (  # noqa: E402
    HaloEscape, SphericalGridTables,
)
from arwen_global.semilag.trajectory import (  # noqa: E402
    CartesianWind, cartesian_wind, departure_points, lipschitz,
)
from arwen_global.semilag.vectors import transport_to_arrival  # noqa: E402
from arwen_global.spectral.transform import SphericalHarmonicTransform  # noqa: E402
from arwen_global.vertical import HybridCoordinate  # noqa: E402

from test_arwen_global_semilag_step import _baroclinic  # noqa: E402

SL = "sl_si"


def _transform(truncation=21):
    return SphericalHarmonicTransform.create(
        truncation, backend="numpy", precision="float64"
    )


def _model(transform, vertical, *, bands=1, exchange=None, options=None):
    return MoistHybridModel(
        transform=transform, vertical=vertical,
        surface_geopotential=np.zeros(transform.grid.shape), physics=None,
        diffusion=None,
        semi_implicit=VerticalModeSemiImplicit(off_centring_weight=0.55),
        integrator=SL, mass_fixer=True, water_fixer=False,
        positivity_repair=False, maximum_cfl=1.0e9, sponge_base_pa=0.0,
        semilag=options or SemiLagrangianOptions(),
        latitude_bands=bands, cards=exchange,
    )


def _digest(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _inventory(model, bundle) -> dict[str, str]:
    out = {}
    atmosphere = bundle.atmosphere
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure",
                 "qv"):
        out["atmosphere." + name] = _digest(getattr(atmosphere, name))
    for name, value in atmosphere.grid_tracers().items():
        out["tracer." + name] = _digest(value)
    trajectory = model.trajectory_state()
    if trajectory is not None:
        for name, value in trajectory.arrays().items():
            out["trajectory." + name] = _digest(value)
    return out


def _scalars(metrics, prefix=""):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (bool, str)):
            out[prefix + key] = value
        elif isinstance(value, (int, float)):
            out[prefix + key] = float(value)
        elif isinstance(value, dict):
            out.update(_scalars(value, prefix + key + "."))
    return out


def _run(bands: int, steps: int, *, exchange=None, options=None,
         truncation=21, nlev=8, dt_s=900.0):
    transform = _transform(truncation)
    if exchange is not None:
        transform.row_exchange = exchange
    vertical = HybridCoordinate.pressure_blend(nlev, 100.0)
    model = _model(transform, vertical, bands=bands, exchange=exchange,
                   options=options)
    bundle = _baroclinic(transform, vertical, wind_m_s=35.0)
    trace = []
    for _ in range(steps):
        bundle, metrics = model.step(bundle, dt_s)
        trace.append(_scalars(metrics))
    return model, bundle, trace


# ------------------------------------------------------------ BAND-SL-1


def _wind(tables, transform, nlev, dtype):
    lat = transform.grid.lat_rad[None, :, None]
    lon = transform.grid.lon_rad[None, None, :]
    column = np.zeros((nlev, 1, 1))
    u = (40.0 * np.cos(lat) + 5.0 * np.sin(3.0 * lon) * np.cos(lat) ** 2
         + column).astype(dtype)
    v = (10.0 * np.sin(2.0 * lon) * np.cos(lat) ** 2 + column).astype(dtype)
    s = (1.0e-4 * np.cos(lon) * np.cos(lat) + column).astype(dtype)
    vx, vy, vz = cartesian_wind(u, v, tables)
    return CartesianWind(vx, vy, vz, s)


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
def test_band_sl_1_a_band_reads_the_whole_grids_bits_at_its_rows(dtype):
    transform = _transform()
    tables = SphericalGridTables.create(transform.grid, xp=np, dtype=dtype)
    nlat = transform.grid.nlat
    nlev = 6
    wind = _wind(tables, transform, nlev, dtype)
    dt = 3.0 * 3600.0
    whole_stencil, whole_diag = departure_points(wind, tables, dt)
    rng = np.random.default_rng(7)
    fields = [rng.standard_normal((nlev, *transform.grid.shape)).astype(dtype)
              for _ in range(3)]
    whole_quintic = gather_batch(fields, whole_stencil, monotone=False, order=6)
    whole_limited = gather_batch(fields, whole_stencil,
                                 monotone="quasi_monotone", deficit=True)
    whole_u, whole_v = transport_to_arrival(*whole_quintic, whole_stencil,
                                            tables)
    whole_lipschitz = lipschitz(wind, tables, dt)
    folded = 0.0
    for a0, a1, halo in ((0, 8, 6), (8, 16, 6), (24, nlat, 5), (10, 13, 9)):
        s0, s1 = max(0, a0 - halo), min(nlat, a1 + halo)
        window = tables.window(slice(s0, s1), slice(a0, a1))
        held = CartesianWind(*(np.ascontiguousarray(x[:, s0:s1])
                               for x in wind.arrays()))
        stencil, diag = departure_points(held, tables, dt, window=window)
        for name in ("xi", "phi", "level"):
            assert np.array_equal(getattr(stencil, name),
                                  getattr(whole_stencil, name)[:, a0:a1]), name
        band_fields = [np.ascontiguousarray(x[:, s0:s1]) for x in fields]
        quintic = gather_batch(band_fields, stencil, monotone=False, order=6)
        limited = gather_batch(band_fields, stencil,
                               monotone="quasi_monotone", deficit=True)
        for got, want in zip(quintic, whole_quintic):
            assert np.array_equal(got, want[:, a0:a1])
        for (got, cut), (want, want_cut) in zip(limited, whole_limited):
            assert np.array_equal(got, want[:, a0:a1])
            assert np.array_equal(cut, want_cut[:, a0:a1])
        u, v = transport_to_arrival(*quintic, stencil, tables)
        assert np.array_equal(u, whole_u[:, a0:a1])
        assert np.array_equal(v, whole_v[:, a0:a1])
        band_lipschitz = lipschitz(held, tables, dt, window=window)
        assert band_lipschitz.lipschitz <= whole_lipschitz.lipschitz
        folded = max(folded, band_lipschitz.jacobian_balanced_s)
        assert diag.displacement_max_m <= whole_diag.displacement_max_m
    # The bands above cover every row, so their maxima fold to the whole's.
    assert folded == whole_lipschitz.jacobian_balanced_s


# ------------------------------------------------------------ BAND-SL-2


def test_band_sl_2_a_departure_outside_the_halo_is_refused_by_name():
    transform = _transform()
    tables = SphericalGridTables.create(transform.grid, xp=np,
                                        dtype=np.float64)
    nlev = 6
    wind = _wind(tables, transform, nlev, np.float64)
    # Two days of a 10 m/s meridional wind is about three rows of T21.
    dt = 48.0 * 3600.0
    window = tables.window(slice(9, 15), slice(10, 14))
    held = CartesianWind(*(np.ascontiguousarray(x[:, 9:15])
                           for x in wind.arrays()))
    with pytest.raises(HaloEscape, match="left the band's row halo"):
        departure_points(held, tables, dt, window=window)
    # The gather refuses on its own, too, when handed a stencil whose
    # departure rows lie outside what the band holds.
    whole, _diag = departure_points(wind, tables, dt)
    from arwen_global.semilag.interpolate import Stencil

    stencil = Stencil(
        xi=np.ascontiguousarray(whole.xi[:, 10:14]),
        phi=np.ascontiguousarray(whole.phi[:, 10:14]),
        level=np.ascontiguousarray(whole.level[:, 10:14]),
        tables=tables, window=window,
    )
    field = np.ones((nlev, 6, transform.grid.nlon))
    with pytest.raises(HaloEscape, match="gather"):
        gather_batch([field], stencil, monotone=False)


def test_band_sl_2_the_step_widens_an_escaped_band_and_keeps_the_answer(
        monkeypatch):
    """A halo too narrow for the flow costs a recomputation, not an answer."""
    _m, reference, reference_trace = _run(4, 3)
    monkeypatch.setattr(step_module, "default_halo_rows",
                        lambda nlat, dt, radius: 1)
    model, narrow, narrow_trace = _run(4, 3)
    assert model.semilag_halo_rows[0] == 1
    assert model.semilag_halo_rows[1] > 1, "no band escaped a one-row halo"
    assert _inventory(model, narrow) == _inventory(_m, reference)
    assert narrow_trace == reference_trace


def test_the_default_halo_is_the_audits_width_at_t533():
    radius = 6.371229e6
    assert default_halo_rows(801, 300.0, radius) == 8
    assert default_halo_rows(1200, 300.0, radius) == 9
    assert default_halo_rows(1200, 150.0, radius) == 7
    assert default_halo_rows(384, 300.0, radius) == 6


# ------------------------------------------------------------ BAND-SL-3


@pytest.mark.parametrize("bands", (2, 3, 4))
def test_band_sl_3_the_banded_step_is_the_resident_step(bands):
    resident_model, resident, resident_trace = _run(1, 3)
    banded_model, banded, banded_trace = _run(bands, 3)
    assert resident_model.pipeline.resident
    assert not banded_model.pipeline.resident
    want = _inventory(resident_model, resident)
    got = _inventory(banded_model, banded)
    assert set(want) == set(got)
    differing = sorted(k for k in want if want[k] != got[k])
    assert not differing, f"bands={bands} moved {differing}"
    assert banded_trace == resident_trace


def test_band_sl_3_the_banded_step_runs_its_bands_behind_a_halo(monkeypatch):
    """Positive evidence: the windowed gather is what ran, once per band."""
    windows = []
    original = step_module._band_pass

    def counting(model, atmosphere, sources, mass_grid, window, **kwargs):
        windows.append((window.source.start, window.source.stop,
                        window.arrival.start, window.arrival.stop))
        return original(model, atmosphere, sources, mass_grid, window,
                        **kwargs)

    monkeypatch.setattr(step_module, "_band_pass", counting)
    model, _bundle, _trace = _run(4, 1)
    nlat = model.transform.grid.nlat
    arrivals = sorted((a0, a1) for _s0, _s1, a0, a1 in windows)
    assert arrivals == [(s.start, s.stop) for s in model.pipeline.slices()]
    assert all(s1 - s0 < nlat for s0, s1, _a0, _a1 in windows)


@pytest.mark.parametrize("coupling", ("arrival", "trajectory_average"))
def test_band_sl_3_the_physics_coupling_arms_band_the_same(coupling):
    options = SemiLagrangianOptions(physics_coupling=coupling)
    m1, resident, t1 = _run(1, 2, options=options)
    m4, banded, t4 = _run(4, 2, options=options)
    assert _inventory(m1, resident) == _inventory(m4, banded)
    assert t1 == t4


def test_band_sl_3_the_additive_fixer_bands_the_same():
    options = SemiLagrangianOptions(tracer_fixer="bermejo_conde_additive")
    m1, resident, t1 = _run(1, 2, options=options)
    m3, banded, t3 = _run(3, 2, options=options)
    assert _inventory(m1, resident) == _inventory(m3, banded)
    assert t1 == t3


# ------------------------------------------------------------ BAND-SL-4


def _loopback(body, world=2, attempts=4):
    last = None
    for _ in range(int(attempts)):
        try:
            return _loopback_once(body, world)
        except (ConnectionError, OSError) as exc:
            last = exc
    raise last


def _loopback_once(body, world):
    ports, holders = [], []
    for _ in range(world - 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        ports.append(sock.getsockname()[1])
        holders.append(sock)
    for sock in holders:
        sock.close()
    ports.append(0)
    addresses = [f"127.0.0.1:{p}" for p in ports]
    results, errors = {}, {}

    def run(rank):
        try:
            transport = cards.TcpCards(rank, world, addresses,
                                       chunk_bytes=1 << 16)
            try:
                results[rank] = body(rank, transport)
            finally:
                transport.close()
        except BaseException as exc:  # noqa: BLE001
            errors[rank] = exc

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300.0)
    if errors:
        raise errors[sorted(errors)[0]]
    return results


@pytest.mark.parametrize("world, bands, weights", (
    (2, 4, (1.0, 1.0)),
    (2, 4, (3.0, 1.0)),
    (3, 6, (1.0, 1.0, 1.0)),
))
def test_band_sl_4_cards_return_one_cards_bits(world, bands, weights):
    """BIT-5 for the semi-Lagrangian core: P cards over loopback TCP, the
    same band schedule, three steps, every array the checkpoint hashes
    and every metric, against one card."""
    one_model, one, one_trace = _run(bands, 3)
    want = _inventory(one_model, one)
    nlat = one_model.transform.grid.nlat

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, bands, weights=weights)
        exchange = cards.RowExchange(session)
        model, bundle, trace = _run(bands, 3, exchange=exchange)
        # The card holds only its own rows of the second time level.
        held = model.trajectory_held()
        first, last = model.pipeline.local_rows()
        assert isinstance(held, RowTrajectory)
        assert (held.first, held.last) == (first, last)
        assert held.arrays["u_prev"].shape[1] == last - first
        assert held.arrays["n_lnps"].shape[0] == last - first
        trajectory = model.trajectory_state()
        # Each card holds only its own rows of the second time level; the
        # checkpoint assembles the globe (runner._card_gather_of), which
        # is what this does before hashing.
        assembled = {
            name: cards.gather_named_arrays(
                session, {name: np.asarray(value)}, nlat,
                model.transform.grid.nlon)[name]
            for name, value in trajectory.arrays().items()
        }
        inventory = _inventory(model, bundle)
        for name, value in assembled.items():
            inventory["trajectory." + name] = _digest(value)
        return inventory, trace, model.pipeline.local_slices()

    out = _loopback(body, world)
    for rank in range(world):
        inventory, trace, local = out[rank]
        differing = sorted(k for k in want if want[k] != inventory[k])
        assert not differing, f"rank {rank} of {world} moved {differing}"
        assert trace == one_trace, f"rank {rank} metrics moved"
        assert len(local) < bands, "a card ran every band"
    owned = sorted(s.start for rank in range(world) for s in out[rank][2])
    assert owned == [s.start for s in one_model.pipeline.slices()]


# ------------------------------------------------------------ MG-6, the sizer


def _wall(truncation, **changes):
    import dataclasses
    from pathlib import Path

    from arwen_global.config import load_config

    path = (Path(__file__).resolve().parents[1] / "src" / "arwen_global"
            / "configs"
            / f"arwen_global_gdas_t{truncation}_native_sl_si_wall.toml")
    return dataclasses.replace(load_config(str(path)), **changes)


def test_the_sizers_halo_is_the_steps_halo():
    from arwen_global import sizing

    for nlat in (32, 384, 576, 801, 1200, 1536):
        for dt in (60.0, 150.0, 225.0, 300.0, 600.0):
            assert sizing.semilag_halo_rows(nlat, dt, 6.371229e6) == (
                default_halo_rows(nlat, dt, 6.371229e6))


def test_mg6_the_gather_transient_divides_with_the_band_count():
    """T799 was refused because the gather's 10.3 GiB whole-grid transient
    entered every banded plan whole.  The banded step reads one band and
    its halo, so the plan carries that and no more."""
    from arwen_global import sizing

    estimate = sizing.estimate_global_memory(_wall(799))
    whole = estimate.semilag_gather_bytes
    assert round(whole / 2**30, 1) == 10.3
    assert estimate.semilag_halo_rows == 9
    assert sizing.semilag_gather_bytes_at(estimate, 1) == whole
    previous = whole
    for bands in (2, 4, 8, 16, 32):
        rows = -(-estimate.nlat // bands) + 2 * estimate.semilag_halo_rows
        at = sizing.semilag_gather_bytes_at(estimate, bands)
        assert at == -(-whole * rows // estimate.nlat)
        assert at < previous
        previous = at
    assert sizing.semilag_gather_bytes_at(estimate, 32) < 0.05 * whole


def test_mg6_the_card_count_divides_the_trajectory_state_and_nothing_else():
    from arwen_global import sizing

    one = sizing.estimate_global_memory(_wall(799))
    eight = sizing.estimate_global_memory(_wall(799, cards=8))
    assert eight.cards == 8
    for bands in (8, 16, 32):
        drop = (sizing.banded_device_peak_bytes(one, bands)
                - sizing.banded_device_peak_bytes(eight, bands))
        assert drop == int(one.trajectory_bytes * (1.0 - 1.0 / 8))


def test_mg6_a_multi_card_plan_gives_every_card_a_band():
    from arwen_global import sizing

    cfg = _wall(533, cards=8)
    plan = sizing.plan_run_memory(cfg, 200 * 2**30)
    assert plan.bands is not None and plan.bands >= 8


def test_band_sl_3_a_retried_search_bands_the_same():
    """A step whose search misses the convergence test is searched again
    at the most iterations.  The banded step decides that on the maximum
    over every band, so every band reruns and the answer, the fold
    determinants and the retry count are the resident step's."""
    options = SemiLagrangianOptions(trajectory_iterations=1,
                                    trajectory_convergence_cells=1.0e-3)
    m1, resident, t1 = _run(1, 2, options=options)
    m3, banded, t3 = _run(3, 2, options=options)
    assert any(row["semilag_trajectory_retried"] == 1.0 for row in t1)
    assert _inventory(m1, resident) == _inventory(m3, banded)
    assert t1 == t3
    for row in t3:
        assert 0.0 < row["semilag_fold_determinant"] <= 1.0


def test_mg6_the_banded_peak_carries_what_the_banded_step_holds_whole():
    """The model under-read every measured banded sl_si peak (T533 L40, every
    slice parked: 11.82 GiB live at 16 bands and 11.56 at 32, read 10.01 and
    9.43), so a 22 GiB card was offered a T533 plan nobody had run.  The
    missing bytes are the arrays the banded step holds whole beside its
    bands: the ten advected species and, under the default fixer, the
    assembled departure stencil."""
    from arwen_global import sizing

    cfg = _wall(533)
    estimate = sizing.estimate_global_memory(cfg)
    assert estimate.semilag_banded_held_bytes == (
        19 * estimate.nlev * estimate.nlat * estimate.nlon * 4)
    parked = sum(sizing._spill_census_estimate(cfg, estimate).values())
    # The lane's readings on the tree before the fixer, and this tree's
    # (RTX 5090, 2026-10-05: 13.01 GiB at 16 bands, 12.89 at 32).
    for bands, measured_gib in ((16, 11.82), (32, 11.56), (16, 13.01),
                                (32, 12.889)):
        peak = sizing.banded_device_peak_bytes(estimate, bands,
                                               spilled_bytes=parked)
        assert peak / 2**30 >= measured_gib, (bands, peak / 2**30)
    # One band is the fitted envelope, which already carries them.
    assert (sizing.banded_device_peak_bytes(estimate, 1)
            == estimate.device_peak_bytes)
