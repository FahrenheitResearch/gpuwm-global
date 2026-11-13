"""Twins and the noise floor: the ``world`` command group (plan 4.9, package 1.5).

One forecast of a changed world beside one forecast of Earth is not a
measurement after about a week: two forecasts from the same analysis drift
apart on their own.  So every twin here is THREE runs from one analysis:

1. the control: the world's own config with the ``[world]`` table taken
   out, so it is Earth and nothing else differs;
2. the perturbed control: the control restarted from its own step-0
   checkpoint with the potential temperature moved at rounding level
   (:func:`perturb_checkpoint`), the same bytes for the same seed;
3. the world.

A difference tape then shows world minus control only where it exceeds
perturbed control minus control, column by column, and the uniform
pressure offset an air-mass rule leaves over unchanged ground (about 26 hPa
for a world without mountains under ``air_mass = "conserve"``) is removed
and printed rather than drawn as a map-wide signal.

Nothing here runs when there is no world: a config with no ``[world]``
table never reaches this module, and a twin of an all-Earth world is the
same config twice, whose difference is exactly zero.

Commands (``gpuwm-global world ...``):

- ``list``: the shipped presets, one line each.
- ``new PRESET --from BASE --out OUT``: BASE with the preset's table added.
- ``describe CONFIG``: the world's identity, what it changes and the one
  sentence; the full world statement is package 1.6's.
- ``twin CONFIG``: writes the control config beside CONFIG and prints the
  six commands of the twin.
- ``perturb CONFIG CHECKPOINT OUT --seed N``: the perturbed control's start.
- ``diff WORLD CONTROL PERTURBED --out DIR``: the masked difference tapes
  and their table (or, for run directories holding no tapes, the largest
  per-array checkpoint difference).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib

import numpy as np

from .spec import EARTH, WorldSpec, parse_world

#: The array the perturbed control's start is moved in: the potential
#: temperature's spectral coefficients.  Mass (log surface pressure), wind
#: and water are left alone, so the two controls hold the same air.
PERTURBED_ARRAY = "atmosphere__theta"

#: The default relative size of the move: float32's unit roundoff, 2**-24
#: (about 6.0e-8; about 1.8e-5 K on a 300 K coefficient).  Each
#: coefficient is multiplied by ``1 + amplitude * r`` with ``r`` uniform in
#: [-1, 1), so a real field stays real and nothing is added where the
#: field has no amplitude.
ROUNDING_AMPLITUDE = 2.0 ** -24

#: The largest relative move this door accepts: anything larger is no
#: longer a rounding-level perturbation and would be a different start.
MAXIMUM_AMPLITUDE = 1.0e-4

#: Beside every perturbed checkpoint: what was moved, by how much, from what.
PERTURBATION_RECEIPT_SUFFIX = ".perturbation.json"
PERTURBATION_SCHEMA = "arwen-global.world-perturbation/v1"

#: The difference door's receipt, in its output directory.
DIFFERENCE_RECEIPT_NAME = "world-difference.json"
DIFFERENCE_SCHEMA = "arwen-global.world-difference/v1"
#: Difference tapes are named after the tape they difference.
DIFFERENCE_PREFIX = "worlddiff_"

#: Terrain that moved by more than this between the world's and the
#: control's tape is "changed ground": the uniform offset is measured
#: elsewhere and surface pressure is not differenced there.
TERRAIN_TOLERANCE_M = 1.0

#: Fields whose world-minus-control carries the air-mass rule's uniform
#: offset when the terrain moves.
OFFSET_FIELDS = ("SLP", "PSFC", "Z500")

#: The band the plan's signal-to-noise ratio is read over (section 6).
SNR_BAND_DEG = (20.0, 80.0)

GRAVITY_M_S2 = 9.80665
DRY_AIR_GAS_CONSTANT = 287.04
STANDARD_LAPSE_K_M = 0.0065
Z500_PRESSURE_PA = 50000.0

#: The sentence every world result carries (plan section 1).
ONE_SENTENCE = (
    "This is the weather of the first {days} after Earth was changed, over "
    "a sea surface that does not respond. It is not this world's climate.")

#: The run's own config copy (cli.RUN_CONFIG_COPY_NAME, by value so this
#: module does not import the command line).
_RUN_CONFIG_COPY_NAME = "arwen-global-run-config.toml"
_CHECKPOINT_GLOB = "arwen_global_step*.npz"
_TAPE_GLOB = "wrfout_d0*"


# -- presets ---------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    """A shipped world: its ``[world]`` table and the label it carries."""

    name: str
    summary: str
    label_class: str
    table: dict

    @property
    def world(self) -> WorldSpec:
        return parse_world({"label": self.name, **self.table})


_SCIENCE_NEAR_RANGES = (
    "science for days 2 to 6 near the ranges; illustration elsewhere")

#: The presets that need no folder of their own.  An invented world
#: (Pangea, a painted map) is a folder the user builds, not a preset.
#: A preset whose part is built by a later package parses here and is
#: refused by that part's own door until it is built.
PRESETS: dict[str, Preset] = {p.name: p for p in (
    Preset("flat-earth", "Earth with no mountains, Earth's amount of air",
           _SCIENCE_NEAR_RANGES,
           {"terrain": {"scale": 0.0, "air_mass": "conserve"}}),
    Preset("half-mountains", "every mountain at half its height",
           _SCIENCE_NEAR_RANGES,
           {"terrain": {"scale": 0.5, "air_mass": "conserve"}}),
    Preset("no-tibet", "no Tibetan plateau or Himalaya, the rest of Earth kept",
           _SCIENCE_NEAR_RANGES,
           {"terrain": {"scale": 0.0, "air_mass": "conserve",
                        "box": {"south_deg": 25.0, "north_deg": 42.0,
                                "west_deg": 70.0, "east_deg": 105.0,
                                "taper_deg": 3.0}}}),
    Preset("no-rockies", "no Rocky Mountains, the rest of Earth kept",
           _SCIENCE_NEAR_RANGES,
           {"terrain": {"scale": 0.0, "air_mass": "conserve",
                        "box": {"south_deg": 30.0, "north_deg": 62.0,
                                "west_deg": -125.0, "east_deg": -100.0,
                                "taper_deg": 3.0}}}),
    Preset("no-seasons", "no tilt: the sun stays over the equator all year",
           "illustration", {"sun": {"obliquity_deg": 0.0}}),
    Preset("tilt-54", "a 54 degree tilt", "illustration",
           {"sun": {"obliquity_deg": 54.0}}),
    Preset("on-its-side", "a 90 degree tilt, Earth on its side", "illustration",
           {"sun": {"obliquity_deg": 90.0}}),
    Preset("dim-sun", "a sun 6 percent dimmer", "illustration",
           {"sun": {"brightness": 0.94}}),
    Preset("bright-sun", "a sun 6 percent brighter", "illustration",
           {"sun": {"brightness": 1.06}}),
    Preset("ice-age-air", "glacial CO2, 190 ppm", "illustration",
           {"air": {"co2_ppm": 190.0}}),
    Preset("hothouse-2x", "CO2 at 560 ppm", "illustration",
           {"air": {"co2_ppm": 560.0}}),
    Preset("hothouse-4x", "CO2 at 1120 ppm", "illustration",
           {"air": {"co2_ppm": 1120.0}}),
    Preset("flood-70m", "the sea 70 m higher over today's land", "illustration",
           {"sea": {"level_m": 70.0}}),
    Preset("waterworld", "no land, flat, Earth's zonal-mean sea",
           "illustration",
           {"terrain": {"scale": 0.0, "air_mass": "conserve"},
            "sea": {"land": "none", "temperature": "zonal-mean", "ice": "zonal"}}),
)}


def preset(name: str) -> Preset:
    if name not in PRESETS:
        raise ValueError(
            f"no shipped world is called {name!r}; the presets are "
            + ", ".join(PRESETS))
    return PRESETS[name]


def preset_lines() -> list[str]:
    width = max(len(name) for name in PRESETS)
    return [f"{p.name:<{width}}  {p.summary} [{p.label_class}]"
            for p in PRESETS.values()]


def preset_of(world: WorldSpec) -> Preset | None:
    """The preset whose settings this world has (the label aside)."""
    for item in PRESETS.values():
        if item.world.identity_hash == world.identity_hash:
            return item
    return None


# -- config text -----------------------------------------------------------

_HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$")


def _is_world_header(name: str) -> bool:
    head = name.split(".", 1)[0].strip().strip('"').strip("'")
    return head == "world"


def strip_world_table(text: str) -> str:
    """``text`` with the ``[world]`` table and its sub-tables taken out.

    Line-based: a ``[world]`` or ``[world.*]`` header starts a run of lines
    that ends at the next header of any other table.  The result is
    checked by parsing (:func:`control_config_text`), so a file this rule
    misreads is refused, never written wrong."""
    out: list[str] = []
    inside = False
    for line in text.splitlines(keepends=True):
        match = _HEADER.match(line)
        if match is not None:
            inside = _is_world_header(match.group(1))
        if not inside:
            out.append(line)
    stripped = "".join(out)
    return stripped.rstrip("\n") + "\n" if stripped.strip() else stripped


def control_config_text(text: str) -> str:
    """The control config of a world config: the same file with no
    ``[world]`` table.  Refused unless every other setting parses back
    unchanged."""
    raw = tomllib.loads(text)
    control = strip_world_table(text)
    parsed = tomllib.loads(control)
    if "world" in parsed:
        raise ValueError(
            "the [world] table could not be taken out of this config line by "
            "line (is it written inline, as world = {...}?); write it as a "
            "[world] table")
    expected = {key: value for key, value in raw.items() if key != "world"}
    if parsed != expected:
        raise ValueError(
            "taking the [world] table out of this config changed another "
            "setting; write the [world] table and its sub-tables last in "
            "the file")
    return control


def add_world_table(text: str, world: WorldSpec) -> str:
    """``text`` (a config with no ``[world]`` table) with ``world``'s table
    appended."""
    if "world" in tomllib.loads(text):
        raise ValueError(
            "this config already has a [world] table; edit it, or start "
            "from a config without one")
    out = text.rstrip("\n") + "\n\n" + world.to_toml()
    if parse_world(tomllib.loads(out).get("world")) != world:
        raise AssertionError("the written [world] table does not parse back")
    return out


def twin_paths(world_config: Path, outdir: Path | None = None) -> dict[str, Path]:
    """Where ``world twin`` writes: beside the world's config (so a path
    the config gives relative to its own folder stays right), or in
    ``outdir`` when given.  A shipped experiment's folder is inside the
    installed package and is never written: its twin goes to the working
    directory unless ``outdir`` says otherwise."""
    from ..configs_dir import config_root

    world_config = Path(world_config)
    folder = outdir
    if folder is None:
        folder = world_config.parent
        if folder.resolve() == config_root().resolve():
            folder = Path.cwd()
    folder = Path(folder)
    stem = world_config.stem
    return {
        "control": folder / f"{stem}.control.toml",
        "runs": folder / f"{stem}.twin",
    }


def twin_commands(world_config: Path, control_config: Path, runs: Path, *,
                  seed: int, console: str = "gpuwm-global") -> list[str]:
    """The six commands of a twin, in order."""
    control_dir = runs / "control"
    perturbed_dir = runs / "control-perturbed"
    world_dir = runs / "world"
    start = runs / "control-perturbed-start.npz"
    tapes = "tapes"
    return [
        f"{console} run {control_config} --outdir {control_dir}",
        (f"{console} world perturb {control_config} "
         f"{control_dir / 'arwen_global_step00000000.npz'} {start} --seed {seed}"),
        (f"{console} run {control_config} --outdir {perturbed_dir} "
         f"--restart {start}"),
        f"{console} run {world_config} --outdir {world_dir}",
        (f"{console} export <config> <run>/arwen_global_step*.npz "
         f"--outdir <run>/{tapes}   (once for each of the three runs, "
         "each with its own config)"),
        (f"{console} world diff {world_dir / tapes} {control_dir / tapes} "
         f"{perturbed_dir / tapes} --out {runs / 'difference'}"),
    ]


# -- the perturbed control -------------------------------------------------


def rounding_factors(seed: int, shape: tuple[int, ...],
                     amplitude: float = ROUNDING_AMPLITUDE) -> np.ndarray:
    """``1 + amplitude * r``, ``r`` uniform in [-1, 1), from ``seed``.

    The bits are PCG64's raw output after NumPy's SeedSequence, both fixed
    by their published definitions rather than by a distribution's
    implementation, so the same seed gives the same bytes on every host
    and NumPy version that ships them."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"the perturbation seed must be a whole number >= 0, got {seed!r}")
    amplitude = float(amplitude)
    if not (math.isfinite(amplitude) and 0.0 <= amplitude <= MAXIMUM_AMPLITUDE):
        raise ValueError(
            f"the perturbation amplitude must lie in [0, {MAXIMUM_AMPLITUDE:g}] "
            f"(relative), got {amplitude:g}: larger is not a rounding-level "
            "perturbation but a different start")
    count = int(np.prod(shape, dtype=np.int64))
    bits = np.random.PCG64(np.random.SeedSequence(int(seed))).random_raw(count)
    unit = (np.asarray(bits, dtype=np.uint64) >> np.uint64(11)).astype(np.float64)
    unit *= 2.0 ** -53
    return (1.0 + amplitude * (2.0 * unit - 1.0)).reshape(shape)


def perturb_arrays(arrays: dict[str, np.ndarray], *, seed: int,
                   amplitude: float = ROUNDING_AMPLITUDE) -> dict[str, np.ndarray]:
    """A copy of a checkpoint's arrays with :data:`PERTURBED_ARRAY` moved
    at rounding level; every other array is the same object."""
    if PERTURBED_ARRAY not in arrays:
        raise ValueError(
            f"this checkpoint carries no {PERTURBED_ARRAY}: it is not a "
            "state the perturbed control can start from")
    source = arrays[PERTURBED_ARRAY]
    factors = rounding_factors(seed, source.shape, amplitude)
    # The product is taken in double precision and rounded once to the
    # stored precision.  A float32 state (every shipped T255 run) then
    # moves by at most one unit in the last place, up or down with equal
    # chance; multiplying in float32 would round every factor above 1 back
    # to 1 and move the field one way only.
    wide = np.complex128 if np.iscomplexobj(source) else np.float64
    out = dict(arrays)
    out[PERTURBED_ARRAY] = np.multiply(source.astype(wide), factors).astype(source.dtype)
    return out


def perturb_checkpoint(cfg, source: str | Path, output: str | Path, *,
                       seed: int, amplitude: float = ROUNDING_AMPLITUDE,
                       overwrite: bool = False) -> dict:
    """Write the perturbed control's start: ``source`` (a checkpoint of
    ``cfg``'s run, normally its step 0) with the potential temperature
    moved at rounding level.  The result carries ``cfg``'s config hash, so
    ``gpuwm-global run CONFIG --restart OUTPUT`` continues from it; the
    receipt beside it says what was moved."""
    from ..checkpoint import read_checkpoint, write_checkpoint_arrays

    source, output = Path(source), Path(output)
    receipt_path = output.with_name(output.name + PERTURBATION_RECEIPT_SUFFIX)
    for path in (output, receipt_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    if source.resolve() == output.resolve():
        raise ValueError("the perturbed checkpoint may not replace its source")
    metadata, arrays = read_checkpoint(
        source, expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator)
    if any(name.startswith("trajectory__") for name in arrays):
        raise ValueError(
            f"{source} carries a second time level (step {metadata['step']}); "
            "perturb the run's step-0 checkpoint, whose start-up step reads "
            "only the state that is moved")
    moved = perturb_arrays(arrays, seed=seed, amplitude=amplitude)
    write_checkpoint_arrays(
        output, moved, step=int(metadata["step"]), time_s=float(metadata["time_s"]),
        physics_state_schema=metadata["physics_state_schema"],
        physics_metadata=metadata["physics_metadata"],
        config_hash=metadata["config_hash"], trackers=metadata["run_trackers"],
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator)
    written, _ = read_checkpoint(
        output, expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator)
    before, after = arrays[PERTURBED_ARRAY], moved[PERTURBED_ARRAY]
    receipt = {
        "schema": PERTURBATION_SCHEMA,
        "array": PERTURBED_ARRAY,
        "rule": "each coefficient times (1 + amplitude * r), r uniform in [-1, 1) "
                "from PCG64(SeedSequence(seed)).random_raw",
        "seed": int(seed),
        "amplitude": float(amplitude),
        "largest_change": float(np.max(np.abs(after - before))) if before.size else 0.0,
        "coefficients_changed": int(np.count_nonzero(after != before)),
        "coefficients": int(before.size),
        "perturbed_array_sha256": written["arrays"][PERTURBED_ARRAY]["sha256"],
        "step": int(metadata["step"]),
        "time_s": float(metadata["time_s"]),
        "config_hash": metadata["config_hash"],
        "source": {"path": source.name, "self_sha256": metadata["self_sha256"]},
        "checkpoint": {"path": output.name, "self_sha256": written["self_sha256"]},
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    receipt["receipt_path"] = str(receipt_path)
    return receipt


# -- differences -----------------------------------------------------------


def checkpoint_difference(world: str | Path, control: str | Path) -> dict[str, float]:
    """The largest absolute difference of every array two checkpoints
    share, by name.  Exactly 0.0 everywhere for a twin of Earth."""
    from ..checkpoint import read_checkpoint

    meta_a, a = read_checkpoint(world)
    meta_b, b = read_checkpoint(control)
    if set(a) != set(b):
        raise ValueError(
            f"{world} and {control} carry different arrays: "
            f"{sorted(set(a) ^ set(b))}")
    out = {}
    for name in sorted(a):
        if a[name].shape != b[name].shape:
            raise ValueError(f"{name} differs in shape between the two checkpoints")
        delta = np.abs(a[name].astype(np.complex128 if np.iscomplexobj(a[name])
                                      else np.float64) - b[name])
        out[name] = float(delta.max()) if delta.size else 0.0
    return out


def sea_level_pressure(psfc: np.ndarray, hgt: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Surface pressure reduced to sea level through a column at the screen
    temperature warming 6.5 K/km downward (the simple reduction, not WRF's
    post-processor's).  Equal to ``psfc`` where the ground is at sea level."""
    hgt = np.asarray(hgt, dtype=np.float64)
    mean_t = np.asarray(t2, dtype=np.float64) + 0.5 * STANDARD_LAPSE_K_M * hgt
    return np.asarray(psfc, dtype=np.float64) * np.exp(
        GRAVITY_M_S2 * hgt / (DRY_AIR_GAS_CONSTANT * mean_t))


def height_on_pressure(pressure: np.ndarray, geopotential_half: np.ndarray,
                       level_pa: float = Z500_PRESSURE_PA) -> np.ndarray:
    """Geopotential height (m) on one pressure surface from a tape's
    full-level pressure (``nz, ny, nx``, bottom first) and half-level
    geopotential (``nz + 1, ny, nx``): full-level geopotential is the mean
    of its two half levels, interpolated linearly in log pressure.  A
    column whose lowest level is above ``level_pa`` takes its lowest
    level's height; one whose top is below it, its top's."""
    p = np.asarray(pressure, dtype=np.float64)
    phi_half = np.asarray(geopotential_half, dtype=np.float64)
    if phi_half.shape[0] != p.shape[0] + 1:
        raise ValueError("half-level geopotential must have one level more than pressure")
    phi = 0.5 * (phi_half[:-1] + phi_half[1:])
    nz = p.shape[0]
    below = np.sum(p > level_pa, axis=0)  # levels with higher pressure
    k = np.clip(below - 1, 0, nz - 2)
    lo = np.take_along_axis(p, k[None], 0)[0]
    hi = np.take_along_axis(p, (k + 1)[None], 0)[0]
    phi_lo = np.take_along_axis(phi, k[None], 0)[0]
    phi_hi = np.take_along_axis(phi, (k + 1)[None], 0)[0]
    weight = (math.log(level_pa) - np.log(lo)) / (np.log(hi) - np.log(lo))
    weight = np.clip(weight, 0.0, 1.0)
    return (phi_lo + weight * (phi_hi - phi_lo)) / GRAVITY_M_S2


#: What a difference tape carries, with units.  Each is read or derived
#: from a tape's own variables; one a tape lacks is left out and named.
DIFFERENCE_FIELDS: dict[str, str] = {
    "SLP": "Pa", "PSFC": "Pa", "Z500": "m", "T2": "K", "Q2": "kg kg-1",
    "U10": "m s-1", "V10": "m s-1", "TSK": "K", "PRECIP": "kg m-2",
}


def tape_planes(variables: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The difference fields of one tape, from its variables (each with
    the leading time dimension already removed)."""
    v = variables
    out: dict[str, np.ndarray] = {}
    for name in ("PSFC", "T2", "Q2", "U10", "V10", "TSK"):
        if name in v:
            out[name] = np.asarray(v[name], dtype=np.float64)
    if {"PSFC", "HGT", "T2"} <= set(v):
        out["SLP"] = sea_level_pressure(v["PSFC"], v["HGT"], v["T2"])
    if {"PB", "PHB"} <= set(v):
        pressure = np.asarray(v["PB"], np.float64) + np.asarray(v.get("P", 0.0), np.float64)
        phi = np.asarray(v["PHB"], np.float64) + np.asarray(v.get("PH", 0.0), np.float64)
        out["Z500"] = height_on_pressure(pressure, phi)
    if "RAINNC" in v or "RAINC" in v:
        out["PRECIP"] = (np.asarray(v.get("RAINNC", 0.0), np.float64)
                         + np.asarray(v.get("RAINC", 0.0), np.float64))
    return out


def _weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    total = float(weight.sum())
    return float((values * weight).sum() / total) if total > 0.0 else 0.0


def _weighted_rms(values: np.ndarray, weight: np.ndarray) -> float:
    return math.sqrt(max(_weighted_mean(values * values, weight), 0.0))


def difference_planes(world: dict[str, np.ndarray], control: dict[str, np.ndarray],
                      perturbed: dict[str, np.ndarray], *, latitude: np.ndarray,
                      terrain_world: np.ndarray | None = None,
                      terrain_control: np.ndarray | None = None,
                      offset: str = "auto") -> tuple[dict, dict]:
    """World minus control, shown only above the noise floor.

    ``world``, ``control`` and ``perturbed`` map field names to planes on
    one grid (:func:`tape_planes`); ``latitude`` is that grid's latitude in
    degrees (a plane, or one value per row).  For each field present in
    all three:

    - ``diff = world - control``, less the uniform offset for the
      pressure fields (:data:`OFFSET_FIELDS`) when the offset is on;
    - ``noise = |perturbed - control|``, the noise floor of this column;
    - ``shown`` where ``|diff| > noise`` (strictly), and for surface
      pressure only where the ground did not move.

    The offset is the area-weighted mean of ``world - control`` over the
    columns whose terrain did not move.  ``offset="auto"`` removes it only
    when the terrain moved somewhere (it is then the air-mass rule's,
    not weather); "on" and "off" force it.

    Returns ``(planes, table)``: ``planes[name]`` holds ``diff`` (a
    masked array, masked where not shown), ``noise`` and ``raw`` (the
    difference before masking, after the offset), and ``table[name]``
    the offset, the area-weighted RMS of the shown-or-not difference and
    of the noise, and the shown fraction of the globe's area."""
    if offset not in ("auto", "on", "off"):
        raise ValueError(f"offset must be 'auto', 'on' or 'off', got {offset!r}")
    fields = [name for name in DIFFERENCE_FIELDS
              if name in world and name in control and name in perturbed]
    if not fields:
        raise ValueError("the three tapes share none of the difference fields "
                         + ", ".join(DIFFERENCE_FIELDS))
    shape = np.shape(world[fields[0]])
    lat = np.asarray(latitude, dtype=np.float64)
    if lat.ndim == 1:
        lat = np.broadcast_to(lat[:, None], shape)
    if lat.shape != shape:
        raise ValueError(f"latitude has shape {lat.shape}, the planes {shape}")
    weight = np.cos(np.deg2rad(lat))
    weight = np.where(weight > 0.0, weight, 0.0)
    if terrain_world is not None and terrain_control is not None:
        moved = np.abs(np.asarray(terrain_world, np.float64)
                       - np.asarray(terrain_control, np.float64)) > TERRAIN_TOLERANCE_M
    else:
        moved = np.zeros(shape, dtype=bool)
    remove = offset == "on" or (offset == "auto" and bool(moved.any()))
    band = (lat >= SNR_BAND_DEG[0]) & (lat <= SNR_BAND_DEG[1])
    planes: dict[str, dict] = {}
    table: dict[str, dict] = {}
    for name in fields:
        w = np.asarray(world[name], np.float64)
        c = np.asarray(control[name], np.float64)
        p = np.asarray(perturbed[name], np.float64)
        if not (w.shape == c.shape == p.shape == shape):
            raise ValueError(f"{name} differs in shape between the three tapes")
        raw = w - c
        noise = np.abs(p - c)
        uniform = 0.0
        if remove and name in OFFSET_FIELDS:
            keep = ~moved
            uniform = _weighted_mean(raw[keep], weight[keep]) if keep.any() else 0.0
            raw = raw - uniform
        valid = ~moved if name == "PSFC" else np.ones(shape, dtype=bool)
        shown = valid & (np.abs(raw) > noise)
        diff = np.ma.masked_array(raw, mask=~shown)
        planes[name] = {"diff": diff, "noise": noise, "raw": raw}
        row = {
            "units": DIFFERENCE_FIELDS[name],
            "offset_removed": uniform,
            "rms_difference": _weighted_rms(np.where(valid, raw, 0.0), weight * valid),
            "rms_noise": _weighted_rms(np.where(valid, noise, 0.0), weight * valid),
            "shown_area_fraction": (_weighted_mean(shown.astype(np.float64), weight)),
            "largest_shown": float(np.max(np.abs(raw[shown]))) if shown.any() else 0.0,
        }
        if name == "Z500" and band.any():
            signal = _weighted_rms(raw[band], weight[band])
            floor = _weighted_rms(noise[band], weight[band])
            row["band_deg"] = list(SNR_BAND_DEG)
            row["band_rms_difference"] = signal
            row["band_rms_noise"] = floor
            row["signal_to_noise"] = (signal / floor) if floor > 0.0 else None
        table[name] = row
    table["_terrain"] = {
        "moved_area_fraction": _weighted_mean(moved.astype(np.float64), weight),
        "offset_rule": offset,
        "offset_removed": remove,
    }
    return planes, table


# -- tapes -----------------------------------------------------------------


def _netcdf():
    try:
        import netCDF4
    except ModuleNotFoundError as exc:  # pragma: no cover - a declared dependency
        raise ModuleNotFoundError(
            "world diff reads and writes tapes through netCDF4, which this "
            "install lacks") from exc
    return netCDF4


_TAPE_VARIABLES = ("XLAT", "XLONG", "HGT", "LANDMASK", "PSFC", "T2", "Q2",
                   "U10", "V10", "TSK", "RAINC", "RAINNC", "PB", "P", "PHB", "PH")


def read_tape(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """The variables :func:`tape_planes` reads from one tape (first time
    only), and the tape's global attributes."""
    netCDF4 = _netcdf()
    out: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(str(path), "r") as tape:
        tape.set_auto_mask(False)
        for name in _TAPE_VARIABLES:
            if name in tape.variables:
                value = np.asarray(tape.variables[name][:])
                dims = tape.variables[name].dimensions
                if dims and dims[0] == "Time":
                    value = value[0]
                out[name] = value
        if "Times" in tape.variables:
            raw = np.asarray(tape.variables["Times"][:])
            out["Times"] = raw
        attrs = {key: tape.getncattr(key) for key in tape.ncattrs()}
    if "XLAT" not in out:
        raise ValueError(f"{path} has no XLAT: not a render tape")
    return out, attrs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_difference_tape(path: Path, *, world_vars: dict, control_vars: dict,
                          planes: dict, table: dict, attributes: dict) -> Path:
    """One difference tape: ``<FIELD>_DIFF`` (world minus control, missing
    where it does not beat the noise), ``<FIELD>_NOISE`` (the noise floor),
    ``HGT_CHANGE`` (world terrain minus Earth's), the world's ``LANDMASK``
    and the grid, on the tape's own dimensions."""
    netCDF4 = _netcdf()
    ny, nx = np.shape(world_vars["XLAT"])
    temporary = path.with_name(f".{path.name}.partial")
    with netCDF4.Dataset(str(temporary), "w", format="NETCDF4") as tape:
        tape.createDimension("Time", None)
        tape.createDimension("south_north", ny)
        tape.createDimension("west_east", nx)
        dims = ("Time", "south_north", "west_east")
        if "Times" in world_vars:
            times = np.asarray(world_vars["Times"])
            tape.createDimension("DateStrLen", times.shape[-1])
            variable = tape.createVariable("Times", "S1", ("Time", "DateStrLen"))
            variable[:] = times.reshape(1, -1)

        def put(name, value, units, description, fill=False):
            kwargs = {"fill_value": netCDF4.default_fillvals["f4"]} if fill else {}
            variable = tape.createVariable(name, "f4", dims, zlib=True, **kwargs)
            variable.units = units
            variable.description = description
            variable[0] = value.astype(np.float32) if not np.ma.isMaskedArray(value) \
                else np.ma.masked_array(value.data.astype(np.float32), mask=value.mask)

        put("XLAT", np.asarray(world_vars["XLAT"], np.float64), "degree_north", "latitude")
        put("XLONG", np.asarray(world_vars["XLONG"], np.float64), "degree_east", "longitude")
        if "LANDMASK" in world_vars:
            put("LANDMASK", np.asarray(world_vars["LANDMASK"], np.float64), "1",
                "the world's land mask")
        if "HGT" in world_vars and "HGT" in control_vars:
            put("HGT_CHANGE", np.asarray(world_vars["HGT"], np.float64)
                - np.asarray(control_vars["HGT"], np.float64), "m",
                "world terrain minus Earth's")
        for name, plane in planes.items():
            units = DIFFERENCE_FIELDS[name]
            row = table[name]
            put(f"{name}_DIFF", plane["diff"], units,
                f"world minus control, shown only where it exceeds the noise "
                f"floor; uniform offset removed: {row['offset_removed']:.6g} {units}",
                fill=True)
            put(f"{name}_NOISE", plane["noise"], units,
                "perturbed control minus control, absolute: the noise floor")
        for key, value in attributes.items():
            tape.setncattr(key, value)
    temporary.replace(path)
    return path


def _tapes_in(directory: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted(Path(directory).glob(_TAPE_GLOB))
            if path.is_file()}


def _checkpoints_in(directory: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted(Path(directory).glob(_CHECKPOINT_GLOB))}


def _world_label(directory: Path) -> str | None:
    """The world's label from the run's own config copy, when one is found
    in the tape directory or its parent."""
    for folder in (Path(directory), Path(directory).parent):
        copy = folder / _RUN_CONFIG_COPY_NAME
        if copy.is_file():
            try:
                raw = tomllib.loads(copy.read_text(encoding="utf-8"))
                return parse_world(raw.get("world")).label
            except (OSError, ValueError, tomllib.TOMLDecodeError):
                return None
    return None


def difference_runs(world_dir: str | Path, control_dir: str | Path,
                    perturbed_dir: str | Path, out: str | Path, *,
                    offset: str = "auto", label: str | None = None,
                    overwrite: bool = False, say=print) -> dict:
    """The difference door.  Three directories of tapes give one
    difference tape per valid time all three share and a receipt with
    the table; three directories of checkpoints (no tapes) give the
    largest per-array difference per shared step, world minus control and
    perturbed control minus control, with no files written but the
    receipt."""
    world_dir, control_dir, perturbed_dir = (Path(world_dir), Path(control_dir),
                                             Path(perturbed_dir))
    out = Path(out)
    for folder in (world_dir, control_dir, perturbed_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} is not a directory")
    receipt_path = out / DIFFERENCE_RECEIPT_NAME
    if receipt_path.exists() and not overwrite:
        raise FileExistsError(f"{receipt_path} exists; pass --overwrite to replace it")
    label = label or _world_label(world_dir) or "unnamed-world"
    tapes = [_tapes_in(d) for d in (world_dir, control_dir, perturbed_dir)]
    if any(tapes):
        rows = _difference_tapes(tapes, out, offset=offset, label=label,
                                 overwrite=overwrite, say=say)
        kind = "tapes"
    else:
        rows = _difference_checkpoints(
            [_checkpoints_in(d) for d in (world_dir, control_dir, perturbed_dir)], say=say)
        kind = "checkpoints"
    receipt = {
        "schema": DIFFERENCE_SCHEMA,
        "kind": kind,
        "world_label": label,
        "noise_rule": "world minus control is shown only where its magnitude "
                      "exceeds |perturbed control minus control| in the same column",
        "offset_rule": offset,
        "world": str(world_dir), "control": str(control_dir),
        "perturbed": str(perturbed_dir),
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def _shared(names: list[dict[str, Path]], what: str, say) -> list[str]:
    common = sorted(set(names[0]) & set(names[1]) & set(names[2]))
    if not common:
        raise ValueError(
            f"the world, control and perturbed control share no {what}: a "
            "twin pairs files by name, so each must come from the same "
            "analysis, step and output interval")
    left_out = sorted((set(names[0]) | set(names[1]) | set(names[2])) - set(common))
    if left_out:
        say(f"world diff: {len(left_out)} {what} not in all three runs are left "
            f"out: {', '.join(left_out[:6])}{' ...' if len(left_out) > 6 else ''}")
    return common


def _difference_checkpoints(found: list[dict[str, Path]], say) -> list[dict]:
    rows = []
    for name in _shared(found, "checkpoints", say):
        signal = checkpoint_difference(found[0][name], found[1][name])
        noise = checkpoint_difference(found[2][name], found[1][name])
        row = {"checkpoint": name,
               "largest_world_minus_control": max(signal.values()),
               "largest_perturbed_minus_control": max(noise.values()),
               "arrays": {key: {"world_minus_control": signal[key],
                                "perturbed_minus_control": noise[key]}
                          for key in signal}}
        say(f"world diff: {name}: largest |world - control| "
            f"{row['largest_world_minus_control']:.6g}, largest |perturbed - "
            f"control| {row['largest_perturbed_minus_control']:.6g}")
        rows.append(row)
    return rows


def _difference_tapes(found: list[dict[str, Path]], out: Path, *, offset: str,
                      label: str, overwrite: bool, say) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in _shared(found, "tapes", say):
        target = out / f"{DIFFERENCE_PREFIX}{name}"
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite to replace it")
        variables = []
        for group in found:
            values, _attrs = read_tape(group[name])
            variables.append(values)
        w, c, p = variables
        for other in (c, p):
            if (np.shape(other["XLAT"]) != np.shape(w["XLAT"])
                    or not np.array_equal(other["XLAT"], w["XLAT"])
                    or not np.array_equal(other["XLONG"], w["XLONG"])):
                raise ValueError(f"{name}: the three tapes are not on one grid")
        planes, table = difference_planes(
            tape_planes(w), tape_planes(c), tape_planes(p),
            latitude=np.asarray(w["XLAT"], np.float64),
            terrain_world=w.get("HGT"), terrain_control=c.get("HGT"), offset=offset)
        sources = {role: _sha256_file(group[name])
                   for role, group in zip(("world", "control", "perturbed"), found)}
        attributes = {
            "TITLE": f"Arwen Global world difference: {label} minus Earth",
            "WORLD_LABEL": label,
            "WORLD_DIFFERENCE_RULE": (
                "world minus control, shown only where it exceeds |perturbed "
                "control minus control|"),
            "WORLD_ONE_SENTENCE": ONE_SENTENCE.format(days="days"),
            "WORLD_OFFSET_REMOVED": int(bool(table["_terrain"]["offset_removed"])),
            **{f"WORLD_OFFSET_{key}": float(table[key]["offset_removed"])
               for key in OFFSET_FIELDS if key in table},
            **{f"SOURCE_SHA256_{role.upper()}": digest
               for role, digest in sources.items()},
        }
        write_difference_tape(target, world_vars=w, control_vars=c, planes=planes,
                              table=table, attributes=attributes)
        rows.append({"tape": name, "difference_tape": target.name,
                     "sources_sha256": sources, "fields": table})
        say(f"world diff: {name}")
        for key, row in table.items():
            if key.startswith("_"):
                continue
            line = (f"  {key:<6} rms {row['rms_difference']:.4g} {row['units']}, "
                    f"noise {row['rms_noise']:.4g}, shown on "
                    f"{100.0 * row['shown_area_fraction']:.1f}% of the globe")
            if row["offset_removed"]:
                line += f", uniform offset {row['offset_removed']:.4g} {row['units']} removed"
            if "signal_to_noise" in row:
                ratio = row["signal_to_noise"]
                line += (f"; {SNR_BAND_DEG[0]:g}-{SNR_BAND_DEG[1]:g}N signal to noise "
                         + ("inf (the noise floor is zero)" if ratio is None
                            else f"{ratio:.3g}"))
            say(line)
    return rows


# -- the command group -----------------------------------------------------

WORLD_HELP = (
    "other worlds: list the shipped presets, write a world config, describe "
    "a world, and run a twin with its own noise floor (world minus Earth, "
    "drawn only where it beats a perturbed Earth)")


def add_world_arguments(parser) -> None:
    from ..configs_dir import config_argument

    legs = parser.add_subparsers(dest="world_command", required=True)
    legs.add_parser("list", help="the shipped worlds, one line each")

    new = legs.add_parser("new", help="write a config: a base config with a "
                                      "shipped world's [world] table added")
    new.add_argument("preset", help="a shipped world's name (world list)")
    new.add_argument("--from", dest="base", type=config_argument, required=True,
                     help="the Earth config to start from (a path or a shipped "
                          "experiment's name)")
    new.add_argument("--out", type=Path, required=True, help="the config to write")
    new.add_argument("--label", default=None,
                     help="the world's label (default: the preset's name)")
    new.add_argument("--overwrite", action="store_true")

    describe = legs.add_parser("describe", help="what a world changes, its "
                                                "identity and its label, before any card time")
    describe.add_argument("config", type=config_argument)

    twin = legs.add_parser("twin", help="write the control config beside a world "
                                        "config and print the twin's six commands")
    twin.add_argument("config", type=config_argument,
                      help="the world config (a path, or a shipped experiment's name)")
    twin.add_argument("--outdir", type=Path, default=None,
                      help="where the control config and the run folders go "
                           "(default: beside the world config; a shipped "
                           "experiment's go to the working directory)")
    twin.add_argument("--seed", type=int, default=1,
                      help="the perturbed control's seed (default 1)")
    twin.add_argument("--overwrite", action="store_true")

    perturb = legs.add_parser("perturb", help="the perturbed control's start: a "
                                              "checkpoint with its potential temperature "
                                              "moved at rounding level")
    perturb.add_argument("config", type=config_argument,
                         help="the control's config (the checkpoint's own)")
    perturb.add_argument("checkpoint", type=Path, help="the control's step-0 checkpoint")
    perturb.add_argument("output", type=Path, help="the perturbed checkpoint to write")
    perturb.add_argument("--seed", type=int, required=True)
    perturb.add_argument("--amplitude", type=float, default=ROUNDING_AMPLITUDE,
                         help="relative size of the move (default 2**-24, float32's "
                              "unit roundoff)")
    perturb.add_argument("--overwrite", action="store_true")

    diff = legs.add_parser("diff", help="world minus control where it beats the "
                                        "noise floor: difference tapes and their table")
    diff.add_argument("world", type=Path, help="the world's tapes (or checkpoints)")
    diff.add_argument("control", type=Path, help="the control's tapes (or checkpoints)")
    diff.add_argument("perturbed", type=Path,
                      help="the perturbed control's tapes (or checkpoints)")
    diff.add_argument("--out", type=Path, required=True)
    diff.add_argument("--offset", choices=("auto", "on", "off"), default="auto",
                      help="remove the uniform pressure and height offset over "
                           "unchanged ground: auto (default) when the terrain moved")
    diff.add_argument("--label", default=None,
                      help="the world's label (default: from the world run's config copy)")
    diff.add_argument("--overwrite", action="store_true")


def _days(duration_s: float) -> str:
    days = duration_s / 86400.0
    if days == 1.0:
        return "day"
    return f"{days:g} days"


def describe_lines(cfg, config_path: str | Path | None = None) -> list[str]:
    """What ``world describe`` prints: the identity, the changes and the
    one sentence, then the world statement (package 1.6) when this build
    carries it."""
    world = cfg.world
    head = [f"world: {world.label}"]
    if world.is_earth:
        head.append("Earth: every setting is Earth's; the identity is 'none' and "
                    "nothing in the run differs from a config with no [world] table")
        head.append(f"config hash: {cfg.config_hash}")
        return head
    found = preset_of(world)
    head += [
        f"identity: {world.identity_hash}",
        f"token: {world.token}",
        f"config hash: {cfg.config_hash}",
        "changes: " + ", ".join(world.changed_settings()),
    ]
    if found is not None:
        head.append(f"preset: {found.name} ({found.summary}) [{found.label_class}]")
    statement = _statement_text(cfg)
    if not statement:
        # Package 1.6's statement carries this sentence itself.
        head.append(ONE_SENTENCE.format(days=_days(cfg.duration_s)))
    if statement:
        head.append("")
        head.append(statement.rstrip("\n"))
    else:
        head.append("the world statement (label, four margins, refusals: "
                    "its module) is not in this build")
    return head


def _statement_text(cfg) -> str | None:
    try:
        from . import statement
    except ImportError:
        return None
    render = getattr(statement, "statement_text", None)
    if render is not None:
        return str(render(cfg))
    # Package 1.6 ships describe_config(cfg) -> WorldStatement; without a
    # frame its statement says which margins wait for the frame.
    describe_config = getattr(statement, "describe_config", None)
    return None if describe_config is None else str(describe_config(cfg).text())


def world_main(args, say=print) -> int:
    from ..config import load_config

    command = args.world_command
    if command == "list":
        for line in preset_lines():
            say(line)
        say("an invented world (Pangea, a painted map) is a folder of its own: "
            "[world.surface] folder")
        return 0
    if command == "new":
        chosen = preset(args.preset)
        table = {"label": args.label or chosen.name, **chosen.table}
        world = parse_world(table)
        out = Path(args.out)
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        base = Path(args.base)
        text = add_world_table(base.read_text(encoding="utf-8"), world)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        cfg = load_config(out)
        say(f"world new: {out}: {world.label} ({chosen.summary}) "
            f"[{chosen.label_class}], identity {world.token}, config hash "
            f"{cfg.config_hash[:12]}")
        if out.resolve().parent != base.resolve().parent:
            say("world new: note: the config was written in another folder than "
                f"{base.name}; a path it gives relative to its folder now points "
                "elsewhere")
        return 0
    if command == "describe":
        cfg = load_config(args.config)
        for line in describe_lines(cfg, args.config):
            say(line)
        return 0
    if command == "twin":
        source = Path(args.config)
        cfg = load_config(source)
        paths = twin_paths(source, getattr(args, "outdir", None))
        if paths["control"].parent.resolve() != source.resolve().parent:
            say("world twin: note: the control config is written in another "
                f"folder than {source.name}; a path it gives relative to its "
                "folder now points elsewhere")
        control = paths["control"]
        if control.exists() and not args.overwrite:
            raise FileExistsError(f"{control} exists; pass --overwrite to replace it")
        text = control_config_text(source.read_text(encoding="utf-8"))
        control.parent.mkdir(parents=True, exist_ok=True)
        control.write_text(text, encoding="utf-8")
        control_cfg = load_config(control)
        if not control_cfg.world.is_earth:
            raise AssertionError("the control config still carries a world")
        say(f"world twin: {cfg.world.label} against Earth, with a perturbed Earth "
            "as the noise floor")
        say(f"world twin: control config {control} (config hash "
            f"{control_cfg.config_hash[:12]}; the world's is {cfg.config_hash[:12]})")
        if cfg.world.is_earth:
            say("world twin: this world is Earth: the world run and the control "
                "are the same config and differ by exactly zero")
        say("world twin: the commands, in order:")
        for index, line in enumerate(
                twin_commands(source, control, paths["runs"], seed=args.seed), 1):
            say(f"  {index}. {line}")
        return 0
    if command == "perturb":
        cfg = load_config(args.config)
        receipt = perturb_checkpoint(cfg, args.checkpoint, args.output,
                                     seed=args.seed, amplitude=args.amplitude,
                                     overwrite=args.overwrite)
        say(f"world perturb: {args.output}: {receipt['array']} moved at relative "
            f"{receipt['amplitude']:.3g} by seed {receipt['seed']} "
            f"({receipt['coefficients_changed']} of {receipt['coefficients']} "
            f"coefficients, largest change {receipt['largest_change']:.3g}); "
            f"receipt {receipt['receipt_path']}")
        say(f"world perturb: continue it with: run {args.config} --outdir <dir> "
            f"--restart {args.output}")
        return 0
    if command == "diff":
        receipt = difference_runs(args.world, args.control, args.perturbed, args.out,
                                  offset=args.offset, label=args.label,
                                  overwrite=args.overwrite, say=say)
        say(f"world diff: {len(receipt['rows'])} {receipt['kind']}; receipt "
            f"{receipt['receipt_path']}")
        return 0
    raise AssertionError(command)


__all__ = [
    "DIFFERENCE_FIELDS", "EARTH", "MAXIMUM_AMPLITUDE", "ONE_SENTENCE",
    "PERTURBED_ARRAY", "PRESETS", "Preset", "ROUNDING_AMPLITUDE",
    "add_world_arguments", "add_world_table", "checkpoint_difference",
    "control_config_text", "describe_lines", "difference_planes",
    "difference_runs", "height_on_pressure", "perturb_arrays",
    "perturb_checkpoint", "preset", "preset_lines", "read_tape",
    "rounding_factors", "sea_level_pressure", "strip_world_table",
    "tape_planes", "twin_commands", "twin_paths", "world_main",
    "write_difference_tape",
]
