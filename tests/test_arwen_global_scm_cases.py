"""Known answers for the single-column case bench (tools/arwen_global_scm_cases.py).

CPU only.  The case reader, the forcing operators, the saturation split,
the DYCOMS-II RF01 radiation formula and the forcing-only run are checked
against hand-computed answers; the harness's two module substitutions are
checked by driving the native suite through the CPU fakes of
test_arwen_global_level5_native.  The cases themselves, against the shipped
kernels, run on a card (the tool's ``run`` command).
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np
import pytest

from tools import arwen_global_scm_cases as scm


def test_profile_is_linear_in_height_and_time_and_zero_or_held_above():
    p = scm.Profile(
        times=[0.0, 100.0],
        heights=[[0.0, 1000.0], [0.0, 1000.0]],
        values=[[1.0, 3.0], [5.0, 7.0]],
        above="zero",
    )
    assert p.at(0.0, [500.0]) == pytest.approx([2.0])
    assert p.at(50.0, [500.0]) == pytest.approx([4.0])
    assert p.at(500.0, [0.0, 1000.0]) == pytest.approx([5.0, 7.0])   # held after the last time
    assert p.at(0.0, [1000.0, 1000.5]) == pytest.approx([3.0, 0.0])  # zero strictly above
    held = scm.Profile.constant([0.0, 1000.0], [1.0, 3.0])
    assert held.at(0.0, [-10.0, 5000.0]) == pytest.approx([1.0, 3.0])
    s = scm.Series([0.0, 10.0], [2.0, 4.0])
    assert [s.at(-1.0), s.at(5.0), s.at(99.0)] == pytest.approx([2.0, 3.0, 4.0])


def _bomex_shaped():
    """The BOMEX/REF DEPHY definition, written out as the reader sees it."""
    def init(name, z, v):
        return {name: np.asarray([v], float), f"zh_{name}": np.asarray([z], float)}

    def forc(name, z, v):
        return {name: np.asarray([v, v], float), f"zh_{name}": np.asarray([z, z], float),
                f"time_{name}": np.asarray([0.0, 86400.0])}

    arrays = {"t0": np.asarray([0.0]), "ps": np.asarray([101500.0]),
              "lat": np.asarray([15.0, 15.0]), "lon": np.asarray([-56.5, -56.5]),
              "time_lat": np.asarray([0.0, 86400.0])}
    arrays.update(init("ua", [0.0, 700.0, 3000.0], [-8.75, -8.75, -4.61]))
    arrays.update(init("va", [0.0, 700.0, 3000.0], [0.0, 0.0, 0.0]))
    arrays.update(init("thetal", [0.0, 520.0, 1480.0, 2000.0, 3000.0],
                       [298.7, 298.7, 302.4, 308.2, 311.85]))
    arrays.update(init("qt", [0.0, 520.0, 1480.0, 2000.0, 3000.0],
                       [0.017, 0.0163, 0.0107, 0.0042, 0.003]))
    zg = [0.0, 3000.0]
    arrays.update(forc("ug", zg, [-10.0, -10.0 + 1.8e-3 * 3000.0]))
    arrays.update(forc("vg", zg, [0.0, 0.0]))
    arrays.update(forc("wa", [0.0, 1500.0, 2100.0], [0.0, -0.0065, 0.0]))
    arrays.update(forc("tnthetal_rad", [0.0, 1500.0, 3000.0], [-2.0 / 86400, -2.0 / 86400, 0.0]))
    arrays.update(forc("tnqt_adv", [0.0, 300.0, 500.0], [-1.2e-8, -1.2e-8, 0.0]))
    for name, value in (("hfss", 8e-3 * 1004.0), ("hfls", 5.2e-5 * 2.5e6),
                        ("tskin", 300.4), ("ustar", 0.28)):
        arrays[name] = np.asarray([value, value])
        arrays[f"time_{name}"] = np.asarray([0.0, 86400.0])
    attrs = {"start_date": "1969-06-24 00:00:00", "end_date": "1969-06-25 00:00:00",
             "radiation": "tend", "surface_type": "ocean", "ini_thetal": 1, "ini_qt": 1,
             "adv_qt": 1, "forc_z": 1, "forc_wa": 1, "forc_geo": 1,
             "surface_forcing_temp": "surface_flux", "surface_forcing_moisture": "surface_flux",
             "surface_forcing_wind": "ustar", "title": "BOMEX shaped"}
    return attrs, arrays


def test_dephy_reader_returns_the_case_at_known_heights():
    attrs, arrays = _bomex_shaped()
    case = scm.case_from_dephy(attrs, arrays, name="bomex", source="test",
                               duration_s=6 * 3600.0, window_s=(3 * 3600.0, 6 * 3600.0))
    assert case.temperature_kind == "thetal" and case.water_kind == "qt"
    assert case.temperature0.at(0.0, [1000.0]) == pytest.approx([298.7 + 0.5 * 3.7])
    assert case.water0.at(0.0, [1000.0]) == pytest.approx([0.0163 - 0.5 * 0.0056])
    assert case.w.at(3600.0, [750.0, 1800.0, 2200.0]) == pytest.approx([-0.00325, -0.00325, 0.0])
    assert case.t_rad.at(0.0, [2250.0]) == pytest.approx([-1.0 / 86400])
    assert case.q_adv.at(0.0, [400.0, 600.0]) == pytest.approx([-0.6e-8, 0.0])
    assert case.ug.at(0.0, [1000.0]) == pytest.approx([-8.2])
    assert case.sensible_w_m2.at(100.0) == pytest.approx(8.032)
    assert case.ustar_m_s.at(100.0) == pytest.approx(0.28)
    assert case.skin_kind == "ts" and case.skin.at(0.0) == pytest.approx(300.4)
    assert case.coriolis_s == pytest.approx(2 * 7.292e-5 * math.sin(math.radians(15.0)))
    assert case.start_utc == datetime(1969, 6, 24, tzinfo=timezone.utc)


def test_dephy_reader_refuses_what_it_cannot_apply():
    attrs, arrays = _bomex_shaped()
    with pytest.raises(ValueError, match="nudging_qv"):
        scm.case_from_dephy({**attrs, "nudging_qv": 3600.0}, arrays, name="x", source="t")
    with pytest.raises(ValueError, match="forc_wap"):
        scm.case_from_dephy({**attrs, "forc_wap": 1}, arrays, name="x", source="t")


def test_subsidence_is_exact_upwind_on_a_linear_profile():
    z = np.asarray([3000.0, 2000.0, 1200.0, 500.0, 100.0])      # top to bottom
    f = (300.0 + 0.005 * z)[:, None, None]
    sinking = scm.subsidence_tendency(f, z, np.full(5, -0.01))
    assert sinking[1:, 0, 0] == pytest.approx(np.full(4, 0.01 * 0.005))
    assert sinking[0, 0, 0] == 0.0                     # no level above the top
    rising = scm.subsidence_tendency(f, z, np.full(5, 0.02))
    assert rising[:-1, 0, 0] == pytest.approx(np.full(4, -0.02 * 0.005))
    assert rising[-1, 0, 0] == 0.0


def test_geostrophic_term_turns_the_ageostrophic_wind_exactly():
    f = 1.0e-4
    quarter = math.pi / 2.0 / f
    u, v = scm.rotate_toward_geostrophic(np.asarray(11.0), np.asarray(0.0),
                                         np.asarray(10.0), np.asarray(0.0), f, quarter)
    assert (float(u), float(v)) == pytest.approx((10.0, -1.0))
    u, v = scm.rotate_toward_geostrophic(np.asarray(10.0), np.asarray(0.0),
                                         np.asarray(10.0), np.asarray(0.0), f, 3600.0)
    assert (float(u), float(v)) == (10.0, 0.0)


def test_saturation_split_keeps_the_liquid_temperature():
    p = np.asarray([93_000.0, 93_000.0])
    t_liquid = np.asarray([283.1, 283.1])
    qt = np.asarray([9.0e-3, 5.0e-3])
    t = scm.saturation_adjusted_temperature(t_liquid, qt, p)
    ql = np.maximum(qt - scm.saturation_specific_humidity(t, p), 0.0)
    assert ql[0] > 2.0e-4 and ql[1] == 0.0
    assert t - scm.LV / scm.CP * ql == pytest.approx(t_liquid, abs=1e-9)
    assert t[1] == pytest.approx(283.1, abs=1e-9)


def test_dycoms_formula_flux_known_answers():
    z_half = np.asarray([1000.0, 900.0, 800.0, 0.0])
    # No liquid: F0 + F1 below the inversion; the subsidence term above it.
    flux = scm.dycoms_net_flux(z_half, np.zeros(3), 840.0, 1.1)
    assert flux[2:] == pytest.approx([92.0, 92.0])
    above = 1.1 * scm.DYCOMS_CP * scm.DYCOMS_DIVERGENCE * (
        0.25 * 160.0 ** (4 / 3) + 840.0 * 160.0 ** (1 / 3))
    assert flux[0] == pytest.approx(92.0 + above)
    # A 0.05 kg/m2 layer between 900 and 800 m: F0 is screened below it, F1 above it.
    flux = scm.dycoms_net_flux(z_half, np.asarray([0.0, 0.05, 0.0]), 900.0, 1.1)
    screen = math.exp(-85.0 * 0.05)
    assert flux[1] == pytest.approx(70.0 + 22.0 * screen)
    assert flux[2] == pytest.approx(70.0 * screen + 22.0)


def _cpu_setup():
    coordinate, dt_s, _options = scm.production_setup(
        datetime(2001, 7, 11, 7, tzinfo=timezone.utc))
    return coordinate, dt_s


@pytest.fixture(scope="module")
def template():
    return scm.template_water_column()


def _warming_case(rate_k_day=1.0):
    z = [0.0, 5000.0]
    return scm.SCMCase(
        name="warming", title="t", reference="r", source="s",
        latitude_deg=10.0, longitude_deg=20.0,
        start_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        duration_s=3600.0, window_s=(0.0, 3600.0), surface_pressure_pa=100_000.0,
        u0=scm.Profile.constant(z, [0.0, 0.0]), v0=scm.Profile.constant(z, [0.0, 0.0]),
        temperature_kind="theta", temperature0=scm.Profile.constant(z, [300.0, 320.0]),
        water_kind="qv", water0=scm.Profile.constant(z, [0.01, 0.002]),
        radiation="off", surface_type="ocean",
        t_adv_kind="theta",
        t_adv=scm.Profile.constant(z, [rate_k_day / 86400.0] * 2, above="zero"),
        skin_kind="ts", skin=scm.Series.constant(300.0),
    )


def test_forcing_alone_adds_exactly_the_prescribed_tendency(template):
    coordinate, dt_s = _cpu_setup()
    record = scm.run_case(_warming_case(), xp=np, dtype=np.float64, suite=False,
                          coordinate=coordinate, dt_s=dt_s, options={}, template=template)
    assert record["steps"] == 12
    theta0 = np.asarray(record["profiles"]["theta"][0])
    theta = np.asarray(record["profiles"]["theta"][-1])
    z = np.asarray(record["levels"]["z_full_m_initial"])
    # Twelve steps; the first stored profile is after one step.
    assert (theta - theta0)[z <= 5000.0] == pytest.approx(11 * 300.0 / 86400.0, rel=1e-9)
    assert (theta - theta0)[z > 5000.0] == pytest.approx(0.0, abs=1e-12)


def test_the_suite_sees_the_case_surface_and_the_spec_arm_drops_its_radiation(template):
    from test_arwen_global_level5_native import FAKE_OLR, _fake_modules, _options

    coordinate, dt_s = _cpu_setup()

    def run(heating, arm):
        return scm.run_case(
            scm.dycoms_rf01(), arm=arm, xp=np, dtype=np.float32, coordinate=coordinate,
            dt_s=dt_s, options=_options(), modules=_fake_modules(radiation_heating=heating),
            template=template, max_steps=2)

    quiet, loud = run(0.0, "spec"), run(1.0e-3, "spec")
    last = quiet["series"][-1]
    assert last["sensible_w_m2"] == pytest.approx(15.0)
    assert last["latent_w_m2"] == pytest.approx(115.0, rel=1e-6)
    assert last["olr_w_m2"] == pytest.approx(FAKE_OLR)
    assert quiet["radiation_heating_applied"] is False
    assert quiet["profiles"]["theta"] == loud["profiles"]["theta"]
    # The dropped rates are still read: 1e-3 K/s of theta over the column
    # below 1500 m, as W/m2, on the final column.
    z = np.asarray(loud["levels"]["z_full_m_final"])
    p_half = scm.initial_column(scm.dycoms_rf01(), coordinate)["p_half"]
    exner = (np.sqrt(p_half[:-1] * p_half[1:]) / scm.P0) ** scm.KAPPA_DRY
    expected = (1.0e-3 * scm.CP * exner * np.diff(p_half) / scm.GRAVITY)[z < 1500.0].sum()
    assert loud["series"][-1]["rrtmgp_lw_heating_below_1500m_w_m2"] == pytest.approx(
        expected, rel=1e-5)
    assert quiet["series"][-1]["rrtmgp_lw_heating_below_1500m_w_m2"] == 0.0
    native = run(1.0e-3, "native-radiation")
    assert native["radiation_heating_applied"] is True
    assert native["profiles"]["theta"] != loud["profiles"]["theta"]


def test_a_prescribed_flux_without_a_skin_sets_the_skin_it_implies(template):
    from dataclasses import replace

    coordinate, dt_s = _cpu_setup()
    case = replace(_warming_case(), skin=None, skin_kind=None,
                   sensible_w_m2=scm.Series.constant(100.0),
                   latent_w_m2=scm.Series.constant(0.0))
    bench = scm.Bench(case=case, coordinate=coordinate, dt_s=dt_s, xp=np, dtype=np.float64)
    bench.seed_surface(*template)
    theta1 = bench.state["theta"][-1, 0, 0]
    exner1 = bench.exner[-1, 0, 0]
    ps = bench.p_half[-1, 0, 0]
    rho = ps / (scm.RD * theta1 * exner1)
    expected = (theta1 + 100.0 / (rho * scm.CP * 0.01)) * (ps / scm.P0) ** scm.KAPPA_DRY
    assert bench.surface.temperature_k[0, 0] == pytest.approx(expected, rel=1e-12)
    # A flux large enough to pass the 20 K bound is held at it.
    big = replace(case, sensible_w_m2=scm.Series.constant(1.0e4))
    bench = scm.Bench(case=big, coordinate=coordinate, dt_s=dt_s, xp=np, dtype=np.float64)
    bench.seed_surface(*template)
    assert bench.surface.temperature_k[0, 0] == pytest.approx(theta1 * exner1 + 20.0)


def test_an_option_override_reaches_the_suite(template):
    from test_arwen_global_level5_native import _fake_modules, _options

    coordinate, dt_s = _cpu_setup()
    record = scm.run_case(
        scm.dycoms_rf01(), xp=np, dtype=np.float32, coordinate=coordinate, dt_s=dt_s,
        options=_options(), modules=_fake_modules(), template=template, max_steps=1,
        option_overrides={"cumulus": "none"})
    assert record["options"]["cumulus"] == "none"
    assert record["steps_completed"] == 1 and record["failure"] is None


def test_score_places_each_reading_against_its_range():
    record = {"case": "rico", "window_means": {
        "radiation_cloud_cover": 0.5, "lwp_g_m2": 19.4, "rwp_g_m2": 0.05,
        "sensible_w_m2": float("nan")}}
    rows = {row["metric"]: row["verdict"] for row in scm.score(record)}
    assert rows["cloud cover"] == "above"
    assert rows["liquid water path (g/m2)"] == "inside"
    assert rows["rain water path (g/m2)"] == "below"
    assert rows["surface sensible heat flux (W/m2)"] == "no reading"
    assert rows["surface latent heat flux (W/m2)"] == "no reading"


def test_native_radiation_arm_is_refused_where_the_case_folds_radiation_in(template):
    coordinate, dt_s = _cpu_setup()
    with pytest.raises(ValueError, match="count it twice"):
        scm.run_case(_warming_case(), arm="native-radiation", xp=np, dtype=np.float64,
                     suite=False, coordinate=coordinate, dt_s=dt_s, options={},
                     template=template)


def test_the_score_table_names_a_case_with_no_published_range():
    rico = {"case": "rico", "arm": "spec", "label": None,
            "window_means": {"radiation_cloud_cover": 0.21, "lwp_g_m2": 1.0}}
    fire = {"case": "fire", "arm": "spec", "label": "own", "window_means": {"lwp_g_m2": 160.0}}
    lines = scm.score_table([rico, fire])
    assert lines[0].startswith("| case | arm | options | metric | model |")
    assert "| rico | spec | shipped | cloud cover | 0.21 | 0.09 | 0.19 | 0.38 | inside |" in lines
    assert "| rico | spec | shipped | liquid water path (g/m2) | 1 | 4.8 | 19.4 | 47.7 | below |" in lines
    # Every RICO reference row is in the table, and FIRE says it has none.
    assert sum(line.startswith("| rico |") for line in lines) == sum(
        1 for row in scm.REFERENCES if row[0] == "rico")
    assert [line for line in lines if line.startswith("| fire |")] == [
        "| fire | spec | own | (no published range read for this case) | | | | | |"]


def test_a_case_file_is_found_in_the_fetched_and_the_repository_layout(tmp_path):
    flat = tmp_path / "flat" / "BOMEX_REF" / "BOMEX_REF_DEF_driver.nc"
    nested = tmp_path / "repo" / "BOMEX" / "REF" / "BOMEX_REF_DEF_driver.nc"
    for path in (flat, nested):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
    assert scm.dephy_path(tmp_path / "flat", "BOMEX_REF") == flat
    assert scm.dephy_path(tmp_path / "repo", "BOMEX_REF") == nested
    # Neither present: the fetched layout is named, so the reader's error is
    # about the path `fetch` writes.
    assert scm.dephy_path(tmp_path / "none", "RICO_SHORT") == (
        tmp_path / "none" / "RICO_SHORT" / "RICO_SHORT_DEF_driver.nc")


def test_fetch_keeps_only_the_pinned_bytes(tmp_path, monkeypatch):
    import hashlib

    payload = {dephy: f"case {dephy}".encode() for dephy in scm.DEPHY_FILES}
    monkeypatch.setattr(scm, "DEPHY_FILES", {
        dephy: (place, hashlib.sha256(payload[dephy]).hexdigest())
        for dephy, (place, _) in scm.DEPHY_FILES.items()})
    asked = []

    def opener(url):
        asked.append(url)
        return payload[url.rsplit("/", 1)[1].removesuffix("_DEF_driver.nc")]

    written = scm.fetch_dephy(tmp_path, opener=opener)
    assert len(written) == 6 and len(asked) == 6
    assert all(f"/{scm.DEPHY_COMMIT}/" in url for url in asked)
    assert any(url.endswith("/RICO/SHORT/RICO_SHORT_DEF_driver.nc") for url in asked)
    assert (tmp_path / "LBA_REF" / "LBA_REF_DEF_driver.nc").read_bytes() == b"case LBA_REF"
    # Already there with the right bytes: nothing is downloaded again.
    scm.fetch_dephy(tmp_path, opener=opener)
    assert len(asked) == 6

    # Bytes that differ from the pin are refused and never written.
    def tampered(url):
        return b"not the case file"

    with pytest.raises(ValueError, match="sha256"):
        scm.fetch_dephy(tmp_path / "other", opener=tampered)
    assert not (tmp_path / "other").exists()


def test_the_pinned_case_files_are_the_ones_the_cases_read():
    named = {str(spec["dephy"]) for spec in scm.CASES.values() if "dephy" in spec}
    assert named == set(scm.DEPHY_FILES)
    for place, digest in scm.DEPHY_FILES.values():
        assert len(digest) == 64 and int(digest, 16) >= 0
        assert place.count("/") == 1
