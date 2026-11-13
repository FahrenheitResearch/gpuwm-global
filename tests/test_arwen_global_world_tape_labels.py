"""A world's tape says which world it is; an Earth tape is unchanged.

THE BREAKAGE THESE GATES PREVENT.  A tape is what a launcher, the Rust plots
and the engine's renderer read, and a tape of Earth with its mountains
removed is the same shape as a tape of Earth.  Without the world on the tape
every map of it reads as an ordinary forecast, a flattened Tibet included,
and on a world with another day length the valid time no longer says where
the sun is.  So a world's tape carries its name, its label (science or
illustration, for that tape's lead time), its identity and the settings that
hash to it, the hash of its world statement, whether its coast is its own,
the plan's weather-not-climate sentence and the sun's true position; its
TITLE and the model name the renderer prints say the world too.

And an Earth tape, with no [world] table or with an all-Earth one, carries
none of it and is the same bytes as before worlds existed.

The labels and the sun's position are pinned against closed forms with no
device; one tiny CPU run is exported as Earth twice and as a world once.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_netcdf_writer, requires_render_kernels  # noqa: E402

from arwen_global import wrfout_export as we
from arwen_global.clock import solar_declination_and_equation_of_time
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.world import EARTH, parse_world

START = "2026-08-30_18:00:00"
SMOKE = config_root() / "arwen_global_moist_smoke.toml"
DAY = 86_400.0
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
orbit = "earth"

[world.sea]
level_m = 0.0
land = "keep"
temperature = "analysis"
ice = "analysis"

[world.air]
co2_ppm = "dated"
ch4_ppb = "dated"
n2o_ppb = "dated"
"""
#: The world the exported run uses: on its side, with a two-day day, so the
#: tilt and the turn of the sun both show in the sub-solar point.
SIDEWAYS = """
[world]
label = "on-its-side"

[world.sun]
obliquity_deg = 90.0
day_length_h = 48.0
"""
def _world(text: str):
    import tomllib

    return parse_world(tomllib.loads(text)["world"])


UNDATED = SimpleNamespace(dated=False, start_utc=None)
DATED = SimpleNamespace(dated=True, start_utc=datetime(2026, 6, 21, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# the attributes on their own
# ---------------------------------------------------------------------------

def test_earth_stamps_nothing_and_keeps_its_title():
    valid = datetime(2026, 8, 30, 18)
    assert we.world_tape_attributes(EARTH, clock=UNDATED, valid=valid, seconds=0.0) == {}
    assert we.world_tape_attributes(_world(ALL_EARTH), clock=UNDATED, valid=valid,
                                    seconds=0.0) == {}
    assert we.world_tape_title(EARTH, "science") == we.TAPE_TITLE
    assert we.TAPE_TITLE == "Arwen Global research model"
    with pytest.raises(ValueError, match="Earth has no world label"):
        we.world_tape_class(EARTH, 0.0)


def test_a_worlds_tape_carries_every_attribute_and_its_settings_hash_to_its_identity():
    world = _world(SIDEWAYS)
    attrs = we.world_tape_attributes(
        world, clock=UNDATED, valid=datetime(2026, 8, 30, 18), seconds=3600.0,
        statement_sha256="ab" * 32)
    assert tuple(attrs) == we.WORLD_TAPE_ATTRS
    assert attrs[we.WORLD_ATTR] == "on-its-side"
    assert attrs[we.WORLD_CLASS_ATTR] == "illustration"
    assert attrs[we.WORLD_IDENTITY_ATTR] == world.identity_hash
    settings = attrs[we.WORLD_SETTINGS_ATTR]
    assert hashlib.sha256(settings.encode()).hexdigest() == world.identity_hash
    assert json.loads(settings) == world.identity
    assert attrs[we.WORLD_CHANGED_ATTR] == "sun.obliquity_deg, sun.day_length_h"
    assert attrs[we.WORLD_STATEMENT_ATTR] == "ab" * 32
    assert attrs[we.WORLD_COAST_ATTR] == "earth"
    assert "first 0.04167 days" in attrs[we.WORLD_SENTENCE_ATTR]
    assert "It is not this world's climate." in attrs[we.WORLD_SENTENCE_ATTR]
    assert "48-hour day" in attrs[we.SUN_POSITION_ATTR]
    for name in (we.SUBSOLAR_LAT_ATTR, we.SUBSOLAR_LON_ATTR):
        assert attrs[name].dtype == np.float32
    unrecorded = we.world_tape_attributes(
        world, clock=UNDATED, valid=datetime(2026, 8, 30, 18), seconds=0.0)
    assert unrecorded[we.WORLD_STATEMENT_ATTR] == we.WORLD_STATEMENT_UNRECORDED
    assert we.world_tape_title(world, "illustration") == (
        "Arwen Global research model, world on-its-side (illustration)")


def test_the_model_name_on_a_worlds_tape_is_one_the_renderer_prints():
    # The renderer drops a model name over 32 characters or holding a
    # parenthesis and prints "WRF" instead (RENDERER_MODEL_LABEL_MAX).
    for label, expected in (("on-its-side", "WOOF Global, world on-its-side"),
                            ("pangea-soil25", "WOOF Global, world pangea-soil25"),
                            ("a-longer-world-name", "WOOF Global, a-longer-world-name"),
                            ("a-much-longer-world-name-still", "WOOF Global")):
        world = _world(f'[world]\nlabel = "{label}"\n[world.sun]\nobliquity_deg = 90.0\n')
        name = we.world_model_label(world)
        assert name == expected
        assert len(name) <= we.RENDERER_MODEL_LABEL_MAX
        assert "(" not in name and ")" not in name


def test_lowered_mountains_are_science_from_day_two_to_day_six_only():
    flat = _world('[world]\nlabel = "flat-earth"\n[world.terrain]\nscale = 0.0\n')
    expected = {0.5: "illustration", 2.0: "science", 4.0: "science",
                6.0: "science", 6.5: "illustration", 10.0: "illustration"}
    for days, label in expected.items():
        assert we.world_tape_class(flat, days * DAY)[0] == label, days
    assert "near the lowered ranges only" in we.world_tape_class(flat, 3 * DAY)[1]
    # Taller mountains, or lowered ones with anything else changed, are an
    # illustration on every day.
    tall = _world('[world]\nlabel = "tall"\n[world.terrain]\nscale = 1.5\n')
    both = _world('[world]\nlabel = "both"\n[world.terrain]\nscale = 0.0\n'
                  '[world.sun]\nbrightness = 1.05\n')
    for world in (tall, both):
        assert we.world_tape_class(world, 3 * DAY)[0] == "illustration"
    co2 = _world('[world]\nlabel = "hot"\n[world.air]\nco2_ppm = 4000.0\n')
    assert "unvalidated above eight times" in we.world_tape_class(co2, DAY)[1]


def test_a_world_whose_land_mask_is_not_earths_says_its_coast_is_its_own():
    cases = {
        '[world]\nlabel = "w"\n[world.sea]\nland = "none"\n[world.terrain]\nscale = 0.0\n': "own",
        '[world]\nlabel = "f"\n[world.sea]\nlevel_m = 70.0\n': "own",
        '[world]\nlabel = "t"\n[world.terrain]\nscale = 0.0\n': "earth",
        SIDEWAYS: "earth",
    }
    for text, coast in cases.items():
        attrs = we.world_tape_attributes(
            _world(text), clock=UNDATED, valid=datetime(2026, 8, 30, 18), seconds=0.0)
        assert attrs[we.WORLD_COAST_ATTR] == coast, text


# ---------------------------------------------------------------------------
# the sun's position, against closed forms
# ---------------------------------------------------------------------------

def test_the_undated_sun_stands_at_the_equinox_and_turns_with_the_day_length():
    valid = datetime(2026, 8, 30, 18)
    bright = _world('[world]\nlabel = "b"\n[world.sun]\nbrightness = 1.1\n')
    # Longitudes run from -180 up to, not including, 180.
    for seconds, longitude in ((0.0, -180.0), (6 * 3600.0, 90.0), (18 * 3600.0, -90.0)):
        lat, lon, _how = we.subsolar_point(bright, UNDATED, valid, seconds)
        assert lat == 0.0
        assert lon == pytest.approx(longitude, abs=1e-12)
    # Six hours into a 12-hour day the sun has gone half way round, a
    # quarter turn further west than Earth's; on a 48-hour day, an eighth.
    for day_h, longitude in ((12.0, 0.0), (48.0, 135.0)):
        world = _world(f'[world]\nlabel = "d"\n[world.sun]\nday_length_h = {day_h}\n')
        _lat, lon, how = we.subsolar_point(world, UNDATED, valid, 6 * 3600.0)
        assert lon == pytest.approx(longitude, abs=1e-9), day_h
        assert "no longer says where it is day" in how


def test_the_dated_sun_is_the_models_own_where_the_hour_angle_is_zero():
    """At Earth's tilt and day the sub-solar point is where the radiation's
    own cosine of the zenith angle (physics/reference.py) is one."""
    bright = _world('[world]\nlabel = "b"\n[world.sun]\nbrightness = 1.1\n')
    for valid in (datetime(2026, 6, 21, 0), datetime(2026, 11, 12, 9, 40, 20),
                  datetime(2027, 3, 1, 23, 59, 59)):
        lat, lon, how = we.subsolar_point(bright, DATED, valid, 12345.0)
        instant = valid.replace(tzinfo=timezone.utc)
        declination, equation = solar_declination_and_equation_of_time(instant)
        assert lat == math.degrees(declination)
        hour = (instant - instant.replace(hour=0, minute=0, second=0)).total_seconds() / 3600.0
        hour_angle = math.radians((60.0 * hour + equation) / 4.0 - 180.0 + lon)
        cosine = (math.sin(math.radians(lat)) * math.sin(declination)
                  + math.cos(math.radians(lat)) * math.cos(declination) * math.cos(hour_angle))
        assert cosine == pytest.approx(1.0, abs=1e-12)
        assert -180.0 <= lon < 180.0
        assert "radconst" in how


def test_the_tilt_scales_the_sine_of_the_declination_and_an_equinox_holds_it_at_zero():
    valid = datetime(2026, 6, 21, 0)
    earth_dec, _eq = solar_declination_and_equation_of_time(valid.replace(tzinfo=timezone.utc))
    for tilt in (0.0, 10.0, 60.0, 90.0):
        world = _world(f'[world]\nlabel = "t"\n[world.sun]\nobliquity_deg = {tilt}\n')
        lat, _lon, _how = we.subsolar_point(world, DATED, valid, 0.0)
        assert math.sin(math.radians(lat)) == pytest.approx(
            math.sin(math.radians(tilt)) * math.sin(earth_dec) / math.sin(math.radians(23.5)),
            abs=1e-12)
    # Within a day of the June solstice a planet on its side has the sun
    # within a few degrees of its north pole.
    sideways = _world('[world]\nlabel = "s"\n[world.sun]\nobliquity_deg = 90.0\n')
    assert we.subsolar_point(sideways, DATED, valid, 0.0)[0] > 85.0
    equinox = _world('[world]\nlabel = "e"\n[world.sun]\norbit = "perpetual-equinox"\n')
    assert we.subsolar_point(equinox, DATED, valid, 0.0)[0] == 0.0


def _turn_apart(a: float, b: float) -> float:
    gap = (a - b) % 360.0
    return min(gap, 360.0 - gap)


def test_the_tapes_sun_is_the_radiations_sun_once_the_sun_and_clock_land():
    """Packages 1.3 (``world/sun.py``) and 3.1 (``world/clock.py``) hold the
    sun the radiation actually runs.  Once they are merged the tape's
    position must be theirs; until then this skips."""
    sun_module = pytest.importorskip("arwen_global.world.sun")
    for tilt in (23.5, 60.0, 90.0):
        world = _world(f'[world]\nlabel = "t"\n[world.sun]\nobliquity_deg = {tilt}\n'
                       'brightness = 1.05\n')
        theirs = sun_module.WorldSun(obliquity_deg=tilt, brightness=1.05)
        for valid in (datetime(2026, 6, 21, 0), datetime(2026, 11, 12, 9, 40, 20)):
            lat, lon, _how = we.subsolar_point(world, DATED, valid, 0.0)
            their_lat, their_lon = theirs.subsolar_point(valid)
            assert lat == pytest.approx(their_lat, abs=1e-9)
            assert _turn_apart(lon, their_lon) < 1e-9
    clock_module = pytest.importorskip("arwen_global.world.clock")
    for day_h in (12.0, 48.0):
        world = _world(f'[world]\nlabel = "d"\n[world.sun]\nday_length_h = {day_h}\n')
        clock = clock_module.PlanetClock.from_world(world)
        valid = datetime(2026, 11, 12, 9, 40, 20)
        for elapsed in (0.0, 3 * 3600.0, 5 * DAY + 17.0):
            _lat, earth_lon, _how = we.subsolar_point(
                _world('[world]\nlabel = "b"\n[world.sun]\nbrightness = 1.1\n'),
                DATED, valid, elapsed)
            _lat, lon, _how = we.subsolar_point(world, DATED, valid, elapsed)
            assert _turn_apart(lon, clock.subsolar_longitude_deg(earth_lon, elapsed)) < 1e-6


def test_an_eccentric_orbit_writes_no_position_and_says_why():
    world = _world('[world]\nlabel = "x"\n[world.sun]\norbit = "eccentric"\n'
                   'eccentricity = 0.1\n')
    assert we.subsolar_point(world, DATED, datetime(2026, 6, 21), 0.0)[:2] == (None, None)
    attrs = we.world_tape_attributes(world, clock=DATED, valid=datetime(2026, 6, 21),
                                     seconds=0.0)
    assert we.SUBSOLAR_LAT_ATTR not in attrs and we.SUBSOLAR_LON_ATTR not in attrs
    assert attrs[we.SUN_POSITION_ATTR].startswith("not written")


def test_the_statement_hash_and_another_worlds_checkpoint():
    text = "world on-its-side: the sun is tilted to 90 degrees"
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert we._recorded_statement_sha256({}, text) == digest
    assert we._recorded_statement_sha256({"world": {"statement": text}}, None) == digest
    assert we._recorded_statement_sha256({"world": {"statement_sha256": "cd" * 32}},
                                         None) == "cd" * 32
    assert we._recorded_statement_sha256({}, None) is None
    # The cold start's provenance (package 1.6) is read before the record.
    stated = {"world_statement": {"sha256": "ef" * 32}}
    assert we._recorded_statement_sha256(
        {"world": {"statement_sha256": "cd" * 32}}, None, stated) == "ef" * 32
    assert we._recorded_statement_sha256({}, text, stated) == digest
    world = _world(SIDEWAYS)
    we._refuse_another_worlds_checkpoint(world, {}, "old.npz")   # no record: config hash rules
    we._refuse_another_worlds_checkpoint(world, {"world": world.record()}, "same.npz")
    with pytest.raises(ValueError, match="was integrated in the world 'none'"):
        we._refuse_another_worlds_checkpoint(world, {"world": EARTH.record()}, "earth.npz")


# ---------------------------------------------------------------------------
# a real (tiny, CPU) run, exported
# ---------------------------------------------------------------------------

def _config(tmp: Path, extra: str, name: str):
    path = tmp / name
    path.write_text(SMOKE.read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
    return load_config(path)


@pytest.fixture(scope="module")
def tapes(tmp_path_factory):
    """The smoke run as Earth, exported under no table and under an
    all-Earth one, and the same run as a world, exported."""
    from arwen_global.runner import run

    root = tmp_path_factory.mktemp("world-tapes")
    bare = load_config(SMOKE)
    earth = _config(root, ALL_EARTH, "earth.toml")
    world = _config(root, SIDEWAYS, "world.toml")
    assert earth.config_hash == bare.config_hash != world.config_hash
    run(bare, root / "run")
    # The smoke runs the reference physics, which has no place for a
    # world's sun and refuses one by name (package 1.3, held by
    # tests/test_arwen_global_world_sun.py).  The tape's sun attributes are
    # computed from the world table at export, not read from the radiation,
    # so the world run here passes its physics through unchanged.
    # The export rebuilds the model, so it runs under the same pass-through.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("arwen_global.world.radiation.attach_sun",
                      lambda physics, world: physics)
        run(world, root / "world-run")
        checkpoints = sorted((root / "run").glob("arwen_global_step*.npz"))
        world_checkpoints = sorted((root / "world-run").glob("arwen_global_step*.npz"))
        assert len(checkpoints) >= 3 and len(world_checkpoints) == len(checkpoints)
        kw = dict(nlat=12, nlon=24, start_date=START)
        return SimpleNamespace(
            root=root, world=world.world, checkpoints=world_checkpoints,
            bare=we.export_wrfout(bare, checkpoints, root / "bare", **kw),
            again=we.export_wrfout(bare, checkpoints, root / "again", **kw),
            earth=we.export_wrfout(earth, checkpoints, root / "earth", **kw),
            sideways=we.export_wrfout(world, world_checkpoints, root / "sideways",
                                      world_statement="the statement", **kw))


@requires_netcdf_writer
@requires_render_kernels
def test_an_earth_tape_is_the_same_bytes_with_or_without_an_all_earth_world(tapes):
    from netCDF4 import Dataset

    assert len(tapes.bare) == len(tapes.earth) == len(tapes.again) >= 3
    for bare, again, earth in zip(tapes.bare, tapes.again, tapes.earth):
        assert bare.name == earth.name
        assert bare.read_bytes() == again.read_bytes()   # the writer is deterministic
        assert bare.read_bytes() == earth.read_bytes()
        with Dataset(earth) as ds:
            assert not set(ds.ncattrs()) & set(we.WORLD_TAPE_ATTRS)
            assert not any(name.startswith(("ARWEN_WORLD", "ARWEN_SUBSOLAR", "ARWEN_SUN"))
                           for name in ds.ncattrs())
            assert ds.getncattr("TITLE") == we.TAPE_TITLE
            assert ds.getncattr(we.MODEL_LABEL_ATTR) == we.MODEL_LABEL
    receipts = [json.loads((tapes.root / name / we.EXPORT_RECEIPT_NAME).read_text(
        encoding="utf-8")) for name in ("bare", "earth")]
    for receipt in receipts:
        assert "world" not in receipt
        assert all("world" not in row for row in receipt["tapes"])


@requires_netcdf_writer
@requires_render_kernels
def test_a_world_tape_carries_every_attribute_and_the_world_in_its_title(tapes):
    from netCDF4 import Dataset

    world = tapes.world
    start = datetime.strptime(START, "%Y-%m-%d_%H:%M:%S")
    digest = hashlib.sha256(b"the statement").hexdigest()
    receipt = json.loads((tapes.root / "sideways" / we.EXPORT_RECEIPT_NAME).read_text(
        encoding="utf-8"))
    assert [row["tape"] for row in receipt["tapes"]] == [str(t) for t in tapes.sideways]
    lons = set()
    for tape, earth, row in zip(tapes.sideways, tapes.bare, receipt["tapes"]):
        assert tape.name == earth.name
        with Dataset(earth) as ds:
            earth_names = set(ds.ncattrs())
        with Dataset(tape) as ds:
            names = set(ds.ncattrs())
            # Every Earth attribute and every world one, and nothing else.
            assert names == earth_names | set(we.WORLD_TAPE_ATTRS)
            assert ds.getncattr("TITLE") == (
                "Arwen Global research model, world on-its-side (illustration)")
            assert ds.getncattr(we.MODEL_LABEL_ATTR) == (
                f"{we.MODEL_LABEL}, world on-its-side")
            assert ds.getncattr(we.WORLD_ATTR) == "on-its-side"
            assert ds.getncattr(we.WORLD_CLASS_ATTR) == "illustration"
            assert ds.getncattr(we.WORLD_IDENTITY_ATTR) == world.identity_hash
            settings = ds.getncattr(we.WORLD_SETTINGS_ATTR)
            assert hashlib.sha256(settings.encode()).hexdigest() == world.identity_hash
            assert ds.getncattr(we.WORLD_STATEMENT_ATTR) == digest
            assert ds.getncattr(we.WORLD_COAST_ATTR) == "earth"
            assert "It is not this world's climate." in ds.getncattr(we.WORLD_SENTENCE_ATTR)
            valid = datetime.strptime(row["valid"], "%Y-%m-%d_%H:%M:%S")
            elapsed = (valid - start).total_seconds()
            lat, lon, how = we.subsolar_point(world, None, valid, elapsed)
            assert ds.getncattr(we.SUBSOLAR_LAT_ATTR) == np.float32(lat) == 0.0
            assert ds.getncattr(we.SUBSOLAR_LON_ATTR) == np.float32(lon)
            # The undated fixture's sun at 00 UTC of an equinox, turning
            # once in 48 hours: 7.5 degrees west per model hour.
            turned = (float(ds.getncattr(we.SUBSOLAR_LON_ATTR))
                      - (180.0 - 7.5 * elapsed / 3600.0)) % 360.0
            assert min(turned, 360.0 - turned) < 1e-4
            assert ds.getncattr(we.SUN_POSITION_ATTR) == how
            lons.add(float(ds.getncattr(we.SUBSOLAR_LON_ATTR)))
    assert len(lons) == len(tapes.sideways)              # the sun moves tape to tape
    assert receipt["world"]["identity"] == world.identity_hash
    assert receipt["world"]["label"] == "on-its-side"
    assert receipt["world"]["tape_attributes"] == list(we.WORLD_TAPE_ATTRS)
    assert [row["world"]["class"] for row in receipt["tapes"]] == (
        ["illustration"] * len(tapes.sideways))
    assert all(row["world"]["statement_sha256"] == digest for row in receipt["tapes"])
