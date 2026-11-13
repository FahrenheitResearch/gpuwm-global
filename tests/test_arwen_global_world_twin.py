"""Twins and the noise floor: the ``world`` command group (worlds plan,
package 1.5).

The row's two tests are here by name: a twin of an all-Earth world differs
by exactly zero, and the perturbation is the same bytes for the same seed.
Around them: the presets parse, ``world new`` and ``world twin`` write
configs that load, the perturbed start restarts, and the difference rule
(offset removed over unchanged ground, shown only above the noise) holds
on planes whose answer is known.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from arwen_global.checkpoint import read_checkpoint
from arwen_global.cli import build_parser
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.world import parse_world
from arwen_global.world.twin import (
    MAXIMUM_AMPLITUDE,
    PERTURBED_ARRAY,
    PRESETS,
    ROUNDING_AMPLITUDE,
    add_world_table,
    checkpoint_difference,
    control_config_text,
    describe_lines,
    difference_planes,
    difference_runs,
    height_on_pressure,
    perturb_arrays,
    perturb_checkpoint,
    preset_lines,
    rounding_factors,
    sea_level_pressure,
    strip_world_table,
    tape_planes,
    world_main,
)

SMOKE = config_root() / "arwen_global_moist_smoke.toml"

ALL_EARTH = """
[world]
label = "earth"

[world.terrain]
scale = 1.0
air_mass = "conserve"

[world.sun]
obliquity_deg = 23.5
brightness = 1.0
day_length_h = 24.0

[world.sea]
level_m = 0.0
land = "keep"

[world.air]
co2_ppm = "dated"
"""

FLAT = """
[world]
label = "flat-earth"

[world.terrain]
scale = 0.0
air_mass = "conserve"
"""


def _world_cmd(*argv: str) -> tuple[int, list[str]]:
    lines: list[str] = []
    args = build_parser().parse_args(["world", *argv])
    return world_main(args, say=lines.append), lines


def _ten_step_config(tmp_path: Path, extra: str, tag: str,
                     integrator: str = "ssprk3") -> Path:
    text = SMOKE.read_text(encoding="utf-8")
    text = text.replace("duration_s = 40.0", "duration_s = 100.0")
    text = text.replace("output_interval_s = 20.0", "output_interval_s = 100.0")
    if integrator == "sl_si":
        # The shipped integrator: its step-0 checkpoint carries no second
        # time level and the restart takes the cold run's start-up step.
        # It runs fully implicit (sl_si refuses an explicit remainder).
        text = text.replace('integrator = "ssprk3"', 'integrator = "sl_si"')
        text = text.replace("weight = 0.5", "weight = 1.0")
    path = tmp_path / f"ten-{tag}.toml"
    path.write_text(text + ("\n" + extra if extra else ""), encoding="utf-8")
    return path


def _run(cfg_path: Path, outdir: Path, restart: Path | None = None) -> Path:
    from arwen_global.runner import run

    cfg = load_config(cfg_path)
    result = run(cfg, outdir, restart=restart)
    assert result["status"] == "pass"
    return outdir / "arwen_global_step00000010.npz"


# -- presets and configs ---------------------------------------------------


def test_every_preset_parses_to_a_world_that_is_not_earth():
    tokens = set()
    for name, item in PRESETS.items():
        world = item.world
        assert world.label == name
        assert not world.is_earth, name
        tokens.add(world.token)
        assert item.label_class in (
            "illustration",
            "science for days 2 to 6 near the ranges; illustration elsewhere")
    assert len(tokens) == len(PRESETS), "two presets are the same world"
    lines = preset_lines()
    assert [line.split()[0] for line in lines] == list(PRESETS)


def test_world_list_prints_every_preset():
    code, lines = _world_cmd("list")
    assert code == 0
    for name in PRESETS:
        assert any(line.startswith(name + " ") for line in lines), name


def test_world_new_writes_a_config_that_loads_as_the_preset(tmp_path):
    out = tmp_path / "flat.toml"
    code, lines = _world_cmd("new", "flat-earth", "--from", str(SMOKE), "--out", str(out))
    assert code == 0, lines
    cfg = load_config(out)
    assert cfg.world == PRESETS["flat-earth"].world
    assert cfg.config_hash != load_config(SMOKE).config_hash
    with pytest.raises(FileExistsError):
        _world_cmd("new", "flat-earth", "--from", str(SMOKE), "--out", str(out))
    # A config that already has a world is not given a second one.
    with pytest.raises(ValueError, match="already has a \\[world\\] table"):
        add_world_table(out.read_text(encoding="utf-8"), PRESETS["dim-sun"].world)
    with pytest.raises(ValueError, match="no shipped world is called"):
        _world_cmd("new", "atlantis", "--from", str(SMOKE), "--out",
                   str(tmp_path / "x.toml"))


def test_the_control_config_is_the_base_config_byte_for_byte_in_identity(tmp_path):
    base_text = SMOKE.read_text(encoding="utf-8")
    world_text = base_text + "\n" + FLAT
    control = control_config_text(world_text)
    assert "[world" not in control
    path = tmp_path / "control.toml"
    path.write_text(control, encoding="utf-8")
    assert load_config(path).config_hash == load_config(SMOKE).config_hash
    # No world table: nothing is taken out.
    assert strip_world_table(base_text).rstrip("\n") == base_text.rstrip("\n")


def test_a_world_table_in_the_middle_is_taken_out_and_the_rest_kept():
    text = ('[arwen_global]\nname = "x"\n\n[world]\nlabel = "w"\n\n'
            '[world.sun]\nbrightness = 0.9\n\n[grid]\ntruncation = 3\n')
    control = control_config_text(text)
    assert control == '[arwen_global]\nname = "x"\n\n[grid]\ntruncation = 3\n'


def test_an_inline_world_table_is_refused_rather_than_kept():
    with pytest.raises(ValueError, match="inline"):
        control_config_text('world = { label = "w", sun = { brightness = 0.9 } }\n'
                            '[grid]\ntruncation = 3\n')


def test_world_twin_writes_the_control_and_prints_six_commands(tmp_path):
    world = tmp_path / "flat.toml"
    world.write_text(SMOKE.read_text(encoding="utf-8") + "\n" + FLAT, encoding="utf-8")
    code, lines = _world_cmd("twin", str(world), "--seed", "5")
    assert code == 0
    control = tmp_path / "flat.control.toml"
    assert load_config(control).world.is_earth
    assert load_config(control).config_hash == load_config(SMOKE).config_hash
    steps = [line for line in lines if line.lstrip()[:2] in {f"{i}." for i in range(1, 7)}]
    assert len(steps) == 6
    assert any("world perturb" in line and "--seed 5" in line for line in steps)
    assert any("world diff" in line for line in steps)


def test_a_shipped_experiment_twins_by_its_bare_name_outside_the_package(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, lines = _world_cmd("twin", "arwen_global_moist_smoke")
    assert code == 0
    control = tmp_path / "arwen_global_moist_smoke.control.toml"
    assert load_config(control).config_hash == load_config(SMOKE).config_hash
    assert not (config_root() / "arwen_global_moist_smoke.control.toml").exists()
    code, _ = _world_cmd("twin", str(SMOKE), "--outdir", str(tmp_path / "elsewhere"))
    assert code == 0
    assert (tmp_path / "elsewhere" / "arwen_global_moist_smoke.control.toml").is_file()


def test_describe_names_the_identity_the_changes_and_the_sentence(tmp_path):
    world = tmp_path / "flat.toml"
    world.write_text(SMOKE.read_text(encoding="utf-8") + "\n" + FLAT, encoding="utf-8")
    cfg = load_config(world)
    lines = describe_lines(cfg)
    text = "\n".join(lines)
    assert cfg.world.identity_hash in text
    assert "terrain.scale" in text
    assert "It is not this world's climate." in text
    assert "preset: flat-earth" in text
    earth = describe_lines(load_config(SMOKE))
    assert any("identity is 'none'" in line for line in earth)


# -- the perturbation: the same bytes for the same seed -------------------

#: sha256 of ``rounding_factors(1, (3, 4, 5)).tobytes()``, measured once and
#: pinned: PCG64's raw stream after SeedSequence is fixed by its published
#: definition, so a host or NumPy version that moved it fails here.
PINNED_FACTORS_SHA256 = (
    "d6bfd10b192b954147f7f9ce1d99d373ca2193b53b1cf37cabb4b34112e405bd")


def test_the_perturbation_is_the_same_bytes_for_the_same_seed():
    a = rounding_factors(1, (3, 4, 5))
    b = rounding_factors(1, (3, 4, 5))
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != rounding_factors(2, (3, 4, 5)).tobytes()
    assert np.all(np.abs(a - 1.0) <= ROUNDING_AMPLITUDE)
    assert np.all(a != 1.0)
    assert np.all(rounding_factors(1, (3, 4, 5), 0.0) == 1.0)
    assert hashlib.sha256(a.tobytes()).hexdigest() == PINNED_FACTORS_SHA256


def test_the_perturbation_refuses_what_is_not_rounding_level():
    with pytest.raises(ValueError, match="amplitude"):
        rounding_factors(1, (2,), 10.0 * MAXIMUM_AMPLITUDE)
    with pytest.raises(ValueError, match="seed"):
        rounding_factors(-1, (2,))


def test_only_the_potential_temperature_moves_and_a_real_coefficient_stays_real():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(4, 10)) + 1j * rng.normal(size=(4, 10))
    theta[:, :3] = theta[:, :3].real  # m = 0 coefficients are real
    arrays = {PERTURBED_ARRAY: theta, "atmosphere__vorticity": theta.copy()}
    moved = perturb_arrays(arrays, seed=9)
    assert moved["atmosphere__vorticity"] is arrays["atmosphere__vorticity"]
    assert np.all(moved[PERTURBED_ARRAY][:, :3].imag == 0.0)
    ratio = np.abs(moved[PERTURBED_ARRAY] / theta - 1.0)
    assert ratio.max() <= ROUNDING_AMPLITUDE * (1 + 1e-9) and ratio.max() > 0.0


def test_a_float32_state_moves_by_at_most_one_unit_both_ways():
    """Every shipped T255 run holds its state in float32.  The move is
    rounded once from double precision, so it is at most one unit in the
    last place and goes up as often as down."""
    rng = np.random.default_rng(1)
    theta = (300.0 * rng.uniform(0.5, 1.5, size=(40, 2000))
             + 1j * rng.normal(size=(40, 2000))).astype(np.complex64)
    moved = perturb_arrays({PERTURBED_ARRAY: theta}, seed=11)[PERTURBED_ARRAY]
    assert moved.dtype == np.complex64
    real_before, real_after = theta.real, moved.real
    ulps = (real_after.view(np.int32).astype(np.int64)
            - real_before.view(np.int32).astype(np.int64))
    assert set(np.unique(ulps)) <= {-1, 0, 1}
    # A move of up to 2**-24 relative crosses half a unit for 1 - ln 2
    # (about 31 percent) of uniformly spread mantissas, half each way.
    up, down = int(np.sum(ulps == 1)), int(np.sum(ulps == -1))
    assert up > 0.1 * ulps.size and down > 0.1 * ulps.size
    assert abs(up - down) < 0.02 * ulps.size


@pytest.mark.parametrize("integrator", ["ssprk3", "sl_si"])
def test_a_perturbed_start_is_the_same_bytes_and_restarts(tmp_path, integrator):
    config = _ten_step_config(tmp_path, "", "control", integrator)
    cfg = load_config(config)
    assert cfg.integrator == integrator
    final = _run(config, tmp_path / "control")
    start = tmp_path / "control" / "arwen_global_step00000000.npz"

    one = perturb_checkpoint(cfg, start, tmp_path / "p1.npz", seed=3)
    two = perturb_checkpoint(cfg, start, tmp_path / "p2.npz", seed=3)
    assert one["checkpoint"]["self_sha256"] == two["checkpoint"]["self_sha256"]
    other = perturb_checkpoint(cfg, start, tmp_path / "p3.npz", seed=4)
    assert other["checkpoint"]["self_sha256"] != one["checkpoint"]["self_sha256"]
    (meta_a, a), (meta_s, s) = read_checkpoint(tmp_path / "p1.npz"), read_checkpoint(start)
    changed = sorted(name for name in a if not np.array_equal(a[name], s[name]))
    assert changed == [PERTURBED_ARRAY]
    assert one["coefficients_changed"] > 0
    with pytest.raises(FileExistsError):
        perturb_checkpoint(cfg, start, tmp_path / "p1.npz", seed=3)

    # The perturbed control runs from it under the control's own config,
    # and ends near the control but not on it.
    moved_final = _run(config, tmp_path / "perturbed", restart=tmp_path / "p1.npz")
    delta = checkpoint_difference(moved_final, final)
    assert max(delta.values()) > 0.0
    assert delta["atmosphere__log_surface_pressure"] < 1e-6

    # At amplitude zero the restart is the control, bit for bit: the
    # noise floor carries no artefact of the restart itself.
    perturb_checkpoint(cfg, start, tmp_path / "p0.npz", seed=3, amplitude=0.0)
    still = _run(config, tmp_path / "unmoved", restart=tmp_path / "p0.npz")
    assert max(checkpoint_difference(still, final).values()) == 0.0


# -- a twin of an all-Earth world differs by exactly zero -----------------


def test_a_twin_of_an_all_earth_world_differs_by_exactly_zero(tmp_path):
    world = _ten_step_config(tmp_path, ALL_EARTH, "world")
    control = tmp_path / "ten-world.control.toml"
    control.write_text(control_config_text(world.read_text(encoding="utf-8")),
                       encoding="utf-8")
    assert load_config(world).config_hash == load_config(control).config_hash
    _run(world, tmp_path / "w")
    _run(control, tmp_path / "c")
    cfg = load_config(control)
    perturb_checkpoint(cfg, tmp_path / "c" / "arwen_global_step00000000.npz",
                       tmp_path / "start.npz", seed=1)
    _run(control, tmp_path / "p", restart=tmp_path / "start.npz")

    receipt = difference_runs(tmp_path / "w", tmp_path / "c", tmp_path / "p",
                              tmp_path / "diff", say=lambda line: None)
    assert receipt["kind"] == "checkpoints"
    names = [row["checkpoint"] for row in receipt["rows"]]
    assert names == ["arwen_global_step00000010.npz"]
    for row in receipt["rows"]:
        assert row["largest_world_minus_control"] == 0.0
        assert row["largest_perturbed_minus_control"] > 0.0
        for values in row["arrays"].values():
            assert values["world_minus_control"] == 0.0


# -- the difference rule ---------------------------------------------------


def _planes(seed: int, shape=(18, 36)):
    rng = np.random.default_rng(seed)
    lat = np.linspace(-85.0, 85.0, shape[0])
    base = {
        "PSFC": 100000.0 + 500.0 * rng.normal(size=shape),
        "SLP": 101000.0 + 500.0 * rng.normal(size=shape),
        "Z500": 5600.0 + 50.0 * rng.normal(size=shape),
        "T2": 280.0 + rng.normal(size=shape),
    }
    return lat, base


def test_the_offset_is_removed_over_unchanged_ground_and_only_signal_is_shown():
    lat, control = _planes(1)
    shape = control["PSFC"].shape
    hgt_c = np.zeros(shape)
    hgt_c[5:8, 10:14] = 3000.0          # a range
    hgt_w = np.zeros(shape)             # removed
    noise = {k: 0.5 * np.ones(shape) for k in control}
    perturbed = {k: control[k] + noise[k] for k in control}
    world = {k: control[k].copy() for k in control}
    for key in ("PSFC", "SLP"):
        world[key] -= 2600.0             # the conserve rule's offset
    world["Z500"] -= 210.0
    world["SLP"][0, 0] += 100.0          # one real signal, far away
    world["T2"][3, 3] += 0.25            # one change below the noise

    planes, table = difference_planes(world, control, perturbed, latitude=lat,
                                      terrain_world=hgt_w, terrain_control=hgt_c)
    assert table["_terrain"]["offset_removed"] is True
    slp = table["SLP"]["offset_removed"]
    assert slp == pytest.approx(-2600.0 + 100.0 * _weight_share(lat, shape, (0, 0), hgt_c))
    assert table["Z500"]["offset_removed"] == pytest.approx(-210.0)
    assert table["T2"]["offset_removed"] == 0.0
    shown_slp = ~np.ma.getmaskarray(planes["SLP"]["diff"])
    assert shown_slp[0, 0]
    assert shown_slp.sum() == 1
    # Surface pressure over moved ground is the ground moving, never shown.
    assert np.ma.getmaskarray(planes["PSFC"]["diff"])[5:8, 10:14].all()
    # A change smaller than the noise floor is not drawn.
    assert np.ma.getmaskarray(planes["T2"]["diff"]).all()
    assert table["Z500"]["signal_to_noise"] == pytest.approx(0.0, abs=1e-9)


def _weight_share(lat, shape, cell, hgt_c):
    weight = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], shape)
    keep = hgt_c <= 1.0
    return weight[cell] / weight[keep].sum()


def test_with_the_terrain_kept_no_offset_is_removed():
    lat, control = _planes(2)
    world = {k: v + 3.0 for k, v in control.items()}
    planes, table = difference_planes(world, control, control, latitude=lat)
    assert table["_terrain"]["offset_removed"] is False
    assert table["SLP"]["offset_removed"] == 0.0
    # A noise floor of exactly zero: every nonzero difference is shown.
    assert not np.ma.getmaskarray(planes["SLP"]["diff"]).any()
    assert table["Z500"]["signal_to_noise"] is None


def test_identical_twins_show_nothing_and_differ_by_zero():
    lat, control = _planes(3)
    planes, table = difference_planes(control, control, control, latitude=lat)
    for name, plane in planes.items():
        assert np.all(plane["raw"] == 0.0), name
        assert np.ma.getmaskarray(plane["diff"]).all(), name
        assert table[name]["rms_difference"] == 0.0


def test_z500_is_exact_on_an_isothermal_column():
    """Geopotential linear in log pressure at the half levels, full levels
    at the half levels' geometric means: the 500 hPa height is the
    isothermal closed form."""
    rd, g, t0, ps = 287.04, 9.80665, 250.0, 101325.0
    p_half = np.array([ps, 85000.0, 70000.0, 55000.0, 40000.0, 25000.0])
    phi_half = rd * t0 * np.log(ps / p_half)
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    shape = (2, 3)
    pb = np.broadcast_to(p_full[:, None, None], (5, *shape)).copy()
    phb = np.broadcast_to(phi_half[:, None, None], (6, *shape)).copy()
    z = height_on_pressure(pb, phb)
    assert np.allclose(z, rd * t0 * math.log(ps / 50000.0) / g, rtol=0, atol=1e-6)
    planes = tape_planes({"PB": pb, "PHB": phb, "PSFC": np.full(shape, ps),
                          "HGT": np.zeros(shape), "T2": np.full(shape, t0)})
    assert np.allclose(planes["Z500"], z)
    assert np.array_equal(planes["SLP"], planes["PSFC"])


def test_sea_level_pressure_rises_over_high_ground():
    slp = sea_level_pressure(np.array([70000.0]), np.array([3000.0]), np.array([270.0]))
    assert 98000.0 < slp[0] < 106000.0


# -- difference tapes ------------------------------------------------------


def _write_tape(path: Path, fields: dict[str, np.ndarray], lat: np.ndarray,
                lon: np.ndarray) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    ny, nx = lat.shape
    with netCDF4.Dataset(str(path), "w") as tape:
        tape.createDimension("Time", None)
        tape.createDimension("DateStrLen", 19)
        tape.createDimension("south_north", ny)
        tape.createDimension("west_east", nx)
        tape.createDimension("bottom_top", 3)
        tape.createDimension("bottom_top_stag", 4)
        times = tape.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.array(list("2026-11-12_00:00:00"), dtype="S1")
        for name, value in {"XLAT": lat, "XLONG": lon, **fields}.items():
            dims = {2: ("Time", "south_north", "west_east"),
                    3: ("Time", "bottom_top" if value.shape[0] == 3 else "bottom_top_stag",
                        "south_north", "west_east")}[value.ndim]
            variable = tape.createVariable(name, "f4", dims)
            variable[0] = value.astype(np.float32)


def test_difference_tapes_are_written_masked_with_their_table(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    ny, nx = 6, 12
    lat = np.broadcast_to(np.linspace(-75.0, 75.0, ny)[:, None], (ny, nx)).copy()
    lon = np.broadcast_to(np.linspace(-180.0, 150.0, nx)[None, :], (ny, nx)).copy()
    pb = np.broadcast_to(np.array([95000.0, 60000.0, 30000.0])[:, None, None],
                         (3, ny, nx)).copy()
    phb = np.broadcast_to(np.array([0.0, 5000.0, 40000.0, 90000.0])[:, None, None],
                          (4, ny, nx)).copy()
    hgt_c = np.zeros((ny, nx))
    hgt_c[2, 4] = 2500.0
    control = {"HGT": hgt_c, "PSFC": np.full((ny, nx), 100000.0),
               "T2": np.full((ny, nx), 280.0), "PB": pb, "PHB": phb,
               "RAINNC": np.ones((ny, nx)), "RAINC": np.zeros((ny, nx))}
    perturbed = {**control, "T2": control["T2"] + 0.01}
    world = {**control, "HGT": np.zeros((ny, nx)),
             "PSFC": control["PSFC"] - 2000.0, "T2": control["T2"] + 0.5}
    dirs = {}
    for role, fields in (("world", world), ("control", control), ("perturbed", perturbed)):
        dirs[role] = tmp_path / role
        dirs[role].mkdir()
        _write_tape(dirs[role] / "wrfout_d01_2026-11-12_00_00_00", fields, lat, lon)
    lines: list[str] = []
    receipt = difference_runs(dirs["world"], dirs["control"], dirs["perturbed"],
                              tmp_path / "out", label="flat-earth", say=lines.append)
    assert receipt["kind"] == "tapes" and len(receipt["rows"]) == 1
    row = receipt["rows"][0]["fields"]
    assert row["PSFC"]["offset_removed"] == pytest.approx(-2000.0, abs=0.01)
    assert row["T2"]["shown_area_fraction"] == pytest.approx(1.0)
    assert any("uniform offset" in line for line in lines)
    out = tmp_path / "out" / "worlddiff_wrfout_d01_2026-11-12_00_00_00"
    with netCDF4.Dataset(str(out)) as tape:
        assert tape.WORLD_LABEL == "flat-earth"
        t2 = tape.variables["T2_DIFF"][0]
        assert np.allclose(t2, 0.5, atol=1e-5)
        psfc = tape.variables["PSFC_DIFF"][0]
        assert np.ma.getmaskarray(psfc)[2, 4]
        assert float(tape.variables["HGT_CHANGE"][0][2, 4]) == -2500.0
        assert "".join(c.decode() for c in tape.variables["Times"][0]) == \
            "2026-11-12_00:00:00"
    with pytest.raises(FileExistsError):
        difference_runs(dirs["world"], dirs["control"], dirs["perturbed"],
                        tmp_path / "out", say=lines.append)


def test_parse_world_of_every_preset_round_trips_through_its_table():
    for item in PRESETS.values():
        world = item.world
        import tomllib

        assert parse_world(tomllib.loads(world.to_toml())["world"]) == world
