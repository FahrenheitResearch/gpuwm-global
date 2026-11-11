"""Regional parent frames: Arwen Global checkpoints as a mapped NetCDF source.

The regional engine (``gpuwm``) builds initial and lateral boundary data from
any model whose output it can read through ``gpuwm prep --source mapped``: one
NetCDF file per valid time on a regular lat-lon grid, described by a mapping
document (which variable is which canonical field, in which units) and a
composition document (where terrain and soil come from).  This door writes
that whole set from this model's checkpoints -- the frames, the mapping, the
composition, the provenance note and a receipt carrying the command to hand
them to the regional engine -- so the global forecast can be the one-way
parent of a regional run with no change on the regional side.

It replaces nothing: ``export`` still writes render tapes (WRF-staggered
winds, no CF coordinates) and ``export-parent`` still writes the hash-bound
NPZ of the regional bridge.  The frames here differ from a render tape in
three ways the regional intake needs: CF 1-D coordinate variables, earth
winds on mass points sampled from the model's own wind synthesis (a tape's
C-staggered U and V averaged back differ by up to several m/s at fronts), and
a unit attribute on every variable.

Everything is written bottom to top, the regional engine's order.
"""
from __future__ import annotations

from calendar import timegm
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .constants import (
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
    WATER_SPECIES,
)

#: One frame per valid time: ``arwen_global_parent_YYYY-MM-DD_HH_MM_SS.nc``.
FRAME_PREFIX = "arwen_global_parent_"
MAPPING_NAME = "arwen-global-parent.mapping.json"
COMPOSITION_NAME = "arwen-global-parent.composition.json"
PROVENANCE_NAME = "arwen-global-parent.provenance.json"
RECEIPT_NAME = "arwen-global-mapped-parent-receipt.json"
#: The role names the composition declares for the in-band terrain: the same
#: frames are passed a second time under the data role, and the provenance
#: note under the provenance role.
SUPPLEMENT_ROLE = "arwen_global_in_band_surface"
PROVENANCE_ROLE = "arwen_global_in_band_surface_provenance"
PROVENANCE_SCHEMA = "arwen-global-parent-in-band-terrain-provenance-v1"
FRAME_CONVENTION = "arwen-global-mapped-parent-frame/v1"

TIME_UNITS = "seconds since 1970-01-01 00:00:00"
TIME_CALENDAR = "standard"
_THETA_OFFSET_K = 300.0
#: Noah's four layers, named by their top in centimetres (0-10-40-100-200).
SOIL_LAYER_TOPS_CM = (0, 10, 40, 100)
_SOIL_LAYER_BOUNDS_M = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))

#: The file contract: NetCDF name, unit string, description.  The regional
#: decoder compares unit attributes as exact strings, so these strings and
#: the mapping's are one table.
MASS_FIELDS = (
    ("T", "K", "potential temperature minus 300 K"),
    ("PB", "Pa", "full pressure at layer midpoints"),
    ("QVAPOR", "kg kg-1", "water vapour mixing ratio (per kg of dry air)"),
    ("QCLOUD", "kg kg-1", "cloud water mixing ratio"),
    ("QRAIN", "kg kg-1", "rain mixing ratio"),
    ("QICE", "kg kg-1", "cloud ice mixing ratio"),
    ("QSNOW", "kg kg-1", "snow mixing ratio"),
    ("QGRAUP", "kg kg-1", "graupel mixing ratio"),
    ("U_EARTH", "m s-1", "eastward wind at mass points"),
    ("V_EARTH", "m s-1", "northward wind at mass points"),
)
INTERFACE_FIELDS = (
    ("PHB", "m2 s-2", "hydrostatic geopotential at layer interfaces"),
    ("W", "m s-1", "vertical velocity of the air through the hybrid coordinate "
                   "surfaces at layer interfaces (zero at the model top and at "
                   "the surface; not the velocity relative to the ground)"),
)
SURFACE_FIELDS = (
    ("PSFC", "Pa", "surface pressure"),
    ("HGT", "m", "model terrain height (spectrally truncated orography)"),
    ("TSK", "K", "skin temperature"),
    ("T2", "K", "2 m air temperature"),
    ("Q2", "kg kg-1", "2 m water vapour mixing ratio (per kg of dry air)"),
    ("U10", "m s-1", "10 m eastward wind"),
    ("V10", "m s-1", "10 m northward wind"),
    ("LANDMASK", "1", "land mask (1 land, 0 water): land fraction >= 0.5"),
    ("SEAICE", "1", "sea-ice fraction"),
    ("SNOW", "kg m-2", "snow water equivalent"),
    ("SNOWH", "m", "snow depth"),
)
SOIL_FIELDS = (
    ("TSLB", "K", "soil temperature, Noah layers"),
    ("SMOIS", "m3 m-3", "volumetric soil moisture, Noah layers"),
)
FRAME_VARIABLES = tuple(
    name for group in (MASS_FIELDS, INTERFACE_FIELDS, SURFACE_FIELDS, SOIL_FIELDS)
    for name, _units, _text in group
)
#: What a frame's ``W`` is, in one sentence set: the provenance note of the
#: frames carries these words.  The render tape (``wrfout_export``) adds the
#: wind's climb along the surfaces to this array and describes its own W.
VERTICAL_VELOCITY_POLICY = (
    "W is the air's vertical velocity through the model's hybrid "
    "coordinate surfaces, -(eta-dot dp/d-eta) / (rho g) at layer "
    "interfaces, from the state's own hybrid-layer mass continuity. "
    "It is zero at the model top and at the surface by construction "
    "and leaves out the motion of the surfaces themselves, "
    "B (dps/dt + V . grad ps): over sloping ground it is not the "
    "velocity relative to the ground"
)


def frame_filename(valid: datetime) -> str:
    return f"{FRAME_PREFIX}{valid.strftime('%Y-%m-%d_%H_%M_%S')}.nc"


# ---------------------------------------------------------------------------
# the three documents the regional intake reads beside the frames
# ---------------------------------------------------------------------------

def _selector(name: str, **extra) -> dict[str, object]:
    return {"format": "netcdf", "name": [name], **extra}


def _direct(name: str, units: str, axes: tuple[str, ...], location: str, *,
            target_units: str | None = None, scale: float = 1.0,
            offset: float = 0.0, staggering: str = "none",
            dependency_only: bool = False) -> dict[str, object]:
    field: dict[str, object] = {
        "selectors": [_selector(name)],
        "units": {"source": units, "target": target_units or units,
                  "scale": scale, "offset": offset},
        "source_axes": ["time", *axes],
        "target_axes": list(axes),
        "location": location,
        "staggering": staggering,
        "missing": {"kind": "reject"},
    }
    if dependency_only:
        field["dependency_only"] = True
    return field


def _derived(derivation: str, units: str, axes: tuple[str, ...],
             location: str) -> dict[str, object]:
    return {
        "derivation": derivation,
        "units": {"source": units, "target": units},
        "source_axes": list(axes),
        "target_axes": list(axes),
        "location": location,
        "staggering": "none",
        "missing": {"kind": "reject"},
    }


def _soil(name: str, units: str) -> dict[str, object]:
    return {
        "selectors": [
            _selector(name, layer_dimension="soil_layers_stag",
                      layer_value=top, layer_units="cm")
            for top in SOIL_LAYER_TOPS_CM
        ],
        "selector_stack_axis": "soil",
        "units": {"source": units, "target": units, "scale": 1.0, "offset": 0.0},
        "source_axes": ["time", "soil", "y", "x"],
        "target_axes": ["soil", "y", "x"],
        "location": "soil",
        "staggering": "none",
        "missing": {"kind": "preserve_mask"},
    }


_REQUIRED_FIELDS = (
    ("air_temperature", ("vertical", "y", "x"), "mass", "K"),
    ("specific_humidity", ("vertical", "y", "x"), "mass", "kg kg-1"),
    ("eastward_wind", ("vertical", "y", "x"), "mass", "m s-1"),
    ("northward_wind", ("vertical", "y", "x"), "mass", "m s-1"),
    ("geopotential_height", ("vertical", "y", "x"), "mass", "m"),
    ("air_pressure", ("vertical", "y", "x"), "mass", "Pa"),
    ("surface_pressure", ("y", "x"), "surface", "Pa"),
    ("terrain_height", ("y", "x"), "surface", "m"),
    ("skin_temperature", ("y", "x"), "surface", "K"),
    ("air_temperature_2m", ("y", "x"), "surface", "K"),
    ("specific_humidity_2m", ("y", "x"), "surface", "kg kg-1"),
    ("eastward_wind_10m", ("y", "x"), "surface", "m s-1"),
    ("northward_wind_10m", ("y", "x"), "surface", "m s-1"),
    ("land_fraction", ("y", "x"), "surface", "1"),
    ("soil_temperature", ("soil", "y", "x"), "soil", "K"),
    ("volumetric_soil_moisture", ("soil", "y", "x"), "soil", "m3 m-3"),
    ("sea_surface_temperature", ("y", "x"), "surface", "K"),
    ("vertical_velocity", ("vertical", "y", "x"), "mass", "m s-1"),
)
_POLICY_CONTROLLED = (
    "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio", "snow_mixing_ratio",
    "graupel_or_hail_mixing_ratio", "vertical_velocity",
    "snow_water_equivalent", "snow_depth", "sea_ice_fraction",
)


def parent_mapping(nlev: int, *, boundary_interval_s: int = 3600,
                   target_vertical_levels: int = 49) -> dict[str, object]:
    """The ``rw-wps.mapping.v1`` document for frames of ``nlev`` model levels.

    It follows the regional engine's packaged model-level NetCDF profile
    (its hex-history mapping) field for field, with this model's own level
    count, wind names and constants: heights are geopotential over this
    model's gravity, and temperature comes back from theta with this
    model's kappa and reference pressure.  ``boundary_interval_s`` is the
    base cadence; a run may use any whole multiple of it.
    """
    if nlev < 2:
        raise ValueError("a mapped parent needs at least two model levels")
    if boundary_interval_s < 1:
        raise ValueError("the boundary interval must be a positive number of seconds")
    mass = ("vertical", "y", "x")
    half = ("half_level", "y", "x")
    plane = ("y", "x")
    mixing = {
        "cloud_water_mixing_ratio": "QCLOUD",
        "rain_water_mixing_ratio": "QRAIN",
        "cloud_ice_mixing_ratio": "QICE",
        "snow_mixing_ratio": "QSNOW",
        "graupel_or_hail_mixing_ratio": "QGRAUP",
    }
    fields: dict[str, object] = {
        "potential_temperature": _direct(
            "T", "K", mass, "mass", offset=_THETA_OFFSET_K, dependency_only=True),
        "air_pressure": _direct("PB", "Pa", mass, "mass"),
        "air_temperature": _derived("temperature-from-theta", "K", mass, "mass"),
        "water_vapor_mixing_ratio": _direct(
            "QVAPOR", "kg kg-1", mass, "mass", dependency_only=True),
        "specific_humidity": _derived(
            "humidity-from-mixing-ratio", "kg kg-1", mass, "mass"),
        **{canonical: _direct(name, "kg kg-1", mass, "mass")
           for canonical, name in mixing.items()},
        "eastward_wind": _direct("U_EARTH", "m s-1", mass, "mass"),
        "northward_wind": _direct("V_EARTH", "m s-1", mass, "mass"),
        "interface_height": _direct(
            "PHB", "m2 s-2", half, "mass", target_units="m",
            scale=1.0 / GRAVITY_M_S2, dependency_only=True),
        "geopotential_height": _derived("height-from-interfaces", "m", mass, "mass"),
        "surface_pressure": _direct("PSFC", "Pa", plane, "surface"),
        "terrain_height": _direct("HGT", "m", plane, "surface"),
        "skin_temperature": _direct("TSK", "K", plane, "surface"),
        "air_temperature_2m": _direct("T2", "K", plane, "surface"),
        "water_vapor_mixing_ratio_2m": _direct(
            "Q2", "kg kg-1", plane, "surface", dependency_only=True),
        "specific_humidity_2m": _derived(
            "humidity-2m-from-mixing-ratio", "kg kg-1", plane, "surface"),
        "eastward_wind_10m": _direct("U10", "m s-1", plane, "surface"),
        "northward_wind_10m": _direct("V10", "m s-1", plane, "surface"),
        "land_fraction": _direct("LANDMASK", "1", plane, "surface"),
        "sea_ice_fraction": _direct("SEAICE", "1", plane, "surface"),
        "snow_water_equivalent": _direct("SNOW", "kg m-2", plane, "surface"),
        "snow_depth": _direct("SNOWH", "m", plane, "surface"),
        "soil_temperature": _soil("TSLB", "K"),
        "volumetric_soil_moisture": _soil("SMOIS", "m3 m-3"),
        "sea_surface_temperature": _derived("water-skin-is-sst", "K", plane, "surface"),
        "interface_vertical_velocity": _direct(
            "W", "m s-1", half, "mass", staggering="z", dependency_only=True),
        "vertical_velocity": _derived("w-from-interfaces", "m s-1", mass, "mass"),
    }
    return {
        "schema": "rw-wps.mapping.v1",
        "name": f"arwen-global-parent-netcdf-{nlev}-level",
        "format": "netcdf",
        "coordinates": {
            "horizontal": {
                "kind": "variables",
                "latitude": {"format": "netcdf", "name": ["lat"],
                             "standard_name": "latitude"},
                "longitude": {"format": "netcdf", "name": ["lon"],
                              "standard_name": "longitude"},
            },
            "vertical": {
                "kind": "model_level",
                "units": "1",
                "positive": "up",
                "levels": list(range(1, nlev + 1)),
                "interface_levels": list(range(1, nlev + 2)),
                "selector": _selector("bottom_top"),
                "interface_selector": _selector("bottom_top_stag"),
            },
            "time": {
                "kind": "dimension",
                "units": TIME_UNITS,
                "calendar": TIME_CALENDAR,
                "selector": {"name": ["Time"], "standard_name": "time"},
            },
        },
        "fields": fields,
        "derivations": [
            {"name": "temperature-from-theta",
             "operation": "air_temperature_from_potential_temperature",
             "potential_temperature": "potential_temperature",
             "pressure": "air_pressure",
             "reference_pressure_pa": float(REFERENCE_PRESSURE_PA),
             "kappa": float(KAPPA)},
            {"name": "humidity-from-mixing-ratio",
             "operation": "specific_humidity_from_mixing_ratio",
             "mixing_ratio": "water_vapor_mixing_ratio"},
            {"name": "humidity-2m-from-mixing-ratio",
             "operation": "specific_humidity_from_mixing_ratio",
             "mixing_ratio": "water_vapor_mixing_ratio_2m"},
            {"name": "height-from-interfaces",
             "operation": "height_from_interfaces",
             "source": "interface_height"},
            {"name": "water-skin-is-sst", "operation": "copy",
             "source": "skin_temperature"},
            {"name": "w-from-interfaces", "operation": "mass_from_interfaces",
             "source": "interface_vertical_velocity"},
        ],
        "target": {
            "name": (f"gpuwm/WRF-real Arwen Global parent ({nlev} model "
                     "levels) NetCDF source"),
            "physics_suite": "source-independent",
            "max_dom": 4,
            "require_lateral_boundaries": True,
            "target_vertical_levels": int(target_vertical_levels),
            "soil_layer_count": len(SOIL_LAYER_TOPS_CM),
            "boundary_interval_seconds": int(boundary_interval_s),
            "accept_boundary_interval_multiples": True,
            "required_fields": [
                {"name": name, "axes": list(axes), "location": location,
                 "target_units": units}
                for name, axes, location, units in _REQUIRED_FIELDS
            ],
            "pressure_requirement": "air_pressure",
            "policy_controlled_fields": list(_POLICY_CONTROLLED),
            "initialization_policies": {},
        },
    }


def parent_composition() -> dict[str, object]:
    """The ``gpuwm-mapped-composition-v2`` document: terrain is the frames'
    own HGT (passed again as an in-band supplement) and soil is Noah's four
    layers, one to one."""

    def layer_selectors(top_cm: int) -> dict[str, object]:
        return {
            canonical: _selector(name, layer_dimension="soil_layers_stag",
                                 layer_value=top_cm, layer_units="cm")
            for canonical, name in (("soil_temperature", "TSLB"),
                                    ("volumetric_soil_moisture", "SMOIS"))
        }

    return {
        "schema": "gpuwm-mapped-composition-v2",
        "name": "arwen-global-parent-with-in-band-terrain-and-noah4-soil",
        "mapping_binding": "input_manifest_sha256",
        "soil_layers": {
            "temperature_field": "soil_temperature",
            "moisture_field": "volumetric_soil_moisture",
            "depth_units": "m",
            "source_layers": [
                {"top": top, "bottom": bottom,
                 "selectors": layer_selectors(top_cm)}
                for (top, bottom), top_cm in zip(_SOIL_LAYER_BOUNDS_M, SOIL_LAYER_TOPS_CM)
            ],
            "target_layers": [
                {"top": top, "bottom": bottom} for top, bottom in _SOIL_LAYER_BOUNDS_M
            ],
            "remap": {
                "kind": "conservative_layer_means",
                "source_value_location": "layer_mean",
                "target_value_location": "layer_mean",
                "coverage": "require_complete",
            },
            "missing": {
                "land": "reject",
                "ocean": {"stage": "after_horizontal_interpolation",
                          "temperature": "skin_temperature", "moisture": 1.0},
            },
        },
        "supplements": {
            "terrain_height": {
                "data_role": SUPPLEMENT_ROLE,
                "provenance_role": PROVENANCE_ROLE,
                "format": "netcdf",
                "field": "terrain_height",
                "selector_authority": "mapping_field_exact",
                "grid_alignment": "exact_coordinate_subset",
                "time_alignment": "valid_time_exact",
                "require_invariant_across_time": True,
            },
        },
    }


def parent_provenance(*, run_name: str, config_hash: str, start: str,
                      grid: str, nlev: int) -> dict[str, object]:
    """What the parent is and how each field was made, in sentences."""
    return {
        "schema": PROVENANCE_SCHEMA,
        "source": (
            f"Arwen Global forecast '{run_name}' (config hash {config_hash}) "
            f"started {start} UTC, exported from its own checkpoints onto {grid}"
        ),
        "field": "terrain_height",
        "selector": {"format": "netcdf", "name": ["HGT"]},
        "time_policy": (
            "one frame per checkpoint; HGT is the run's spectrally truncated "
            "orography sampled once and written identically into every frame"
        ),
        "vertical_policy": (
            f"{nlev} hybrid model levels, bottom to top: full pressure in PB "
            "(geometric mean of the two interface pressures a + b ps), "
            "hydrostatic interface geopotential in PHB (divide by g = "
            f"{GRAVITY_M_S2} for height), theta in T + {_THETA_OFFSET_K:g} K, "
            f"temperature from theta with kappa = {KAPPA!r} and p0 = "
            f"{REFERENCE_PRESSURE_PA:g} Pa"
        ),
        "humidity_policy": (
            "QVAPOR, the condensate species and Q2 are dry-air mixing ratios "
            "q / (1 - q_v) from the model's specific humidities"
        ),
        "wind_policy": (
            "U_EARTH / V_EARTH: earth-relative winds at mass points, "
            "synthesized from vorticity and divergence on the Gaussian grid "
            "and sampled bilinearly (no staggering)"
        ),
        "surface_policy": (
            "TSK over water is the model's water skin temperature (its "
            "prescribed or held SST), so SST is taken from TSK; SEAICE, SNOW, "
            "SNOWH, TSLB and SMOIS are the land surface's own state; T2, Q2, "
            "U10 and V10 are the physics suite's screen-level fields when the "
            "checkpoint carries them, else a similarity diagnostic made at "
            "export (each frame's ARWEN_SURFACE_DIAGNOSTICS attribute says "
            "which); LANDMASK is land fraction >= 0.5 after bilinear sampling"
        ),
        "vertical_velocity_policy": VERTICAL_VELOCITY_POLICY,
        "number_moment_policy": (
            "number concentrations are not carried; the regional scheme "
            "initializes them from the mass fields"
        ),
        "static_policy": (
            "land use, soil categories and vegetation are not carried; the "
            "regional run takes them, and its own terrain, from its geography"
        ),
        "input_identity": (
            "frame paths and SHA-256 values are owned by the mapped input "
            "manifest; this export's receipt lists them beside the checkpoint "
            "each came from"
        ),
    }


# ---------------------------------------------------------------------------
# the frame writer (no model, no device: arrays in, one NetCDF file out)
# ---------------------------------------------------------------------------

def _expected_shapes(nlev: int, ny: int, nx: int) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for name, _units, _text in MASS_FIELDS:
        shapes[name] = (nlev, ny, nx)
    for name, _units, _text in INTERFACE_FIELDS:
        shapes[name] = (nlev + 1, ny, nx)
    for name, _units, _text in SURFACE_FIELDS:
        shapes[name] = (ny, nx)
    for name, _units, _text in SOIL_FIELDS:
        shapes[name] = (len(SOIL_LAYER_TOPS_CM), ny, nx)
    return shapes


def check_parent_frame(latitude, longitude, fields) -> int:
    """Refuse a frame the regional intake would refuse, or silently misread,
    before it is written.  Returns the frame's level count."""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
        raise ValueError("a parent frame needs 1-D latitude and longitude of at least two points")
    if not (np.all(np.diff(lat) > 0.0) and np.all(np.diff(lon) > 0.0)):
        raise ValueError(
            "parent frame latitude must run south to north and longitude west "
            "to east, strictly increasing")
    missing = [name for name in FRAME_VARIABLES if name not in fields]
    if missing:
        raise ValueError(f"parent frame is missing {missing}")
    extra = sorted(set(fields) - set(FRAME_VARIABLES))
    if extra:
        raise ValueError(f"parent frame carries undeclared variables {extra}")
    nlev = int(np.asarray(fields["T"]).shape[0])
    for name, shape in _expected_shapes(nlev, lat.size, lon.size).items():
        value = np.asarray(fields[name])
        if value.shape != shape:
            raise ValueError(
                f"parent frame {name} has shape {value.shape}, expected {shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"parent frame {name} holds a non-finite value")
    pressure = np.asarray(fields["PB"])
    if not np.all(pressure[:-1] > pressure[1:]):
        raise ValueError(
            "parent frame PB must decrease upward on every column (the frame "
            "is written bottom to top)")
    height = np.asarray(fields["PHB"])
    if not np.all(height[1:] > height[:-1]):
        raise ValueError(
            "parent frame PHB must increase upward on every column")
    return nlev


def write_parent_frame(path: str | Path, valid: datetime, latitude, longitude,
                       fields, *, attrs=None) -> Path:
    """Write one frame.  ``fields`` maps every name in :data:`FRAME_VARIABLES`
    to its array, bottom to top; ``valid`` is naive UTC."""
    from netCDF4 import Dataset

    nlev = check_parent_frame(latitude, longitude, fields)
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    dataset = Dataset(temporary, "w", format="NETCDF4_CLASSIC")
    try:
        for dimension, size in (
            ("Time", None), ("bottom_top", nlev), ("bottom_top_stag", nlev + 1),
            ("lat", lat.size), ("lon", lon.size),
            ("soil_layers_stag", len(SOIL_LAYER_TOPS_CM)),
        ):
            dataset.createDimension(dimension, size)

        def coordinate(name, values, **attributes):
            variable = dataset.createVariable(name, "f8", (name,))
            variable.setncatts(attributes)
            variable[:] = values

        coordinate("lat", lat, units="degrees_north", standard_name="latitude")
        coordinate("lon", lon, units="degrees_east", standard_name="longitude")
        coordinate("Time", [timegm(valid.timetuple())], units=TIME_UNITS,
                   calendar=TIME_CALENDAR, standard_name="time")
        coordinate("bottom_top", np.arange(1, nlev + 1), units="1")
        coordinate("bottom_top_stag", np.arange(1, nlev + 2), units="1")
        coordinate("soil_layers_stag", SOIL_LAYER_TOPS_CM, units="cm")

        for group, dimensions in (
            (MASS_FIELDS, ("Time", "bottom_top", "lat", "lon")),
            (INTERFACE_FIELDS, ("Time", "bottom_top_stag", "lat", "lon")),
            (SURFACE_FIELDS, ("Time", "lat", "lon")),
            (SOIL_FIELDS, ("Time", "soil_layers_stag", "lat", "lon")),
        ):
            for name, units, text in group:
                variable = dataset.createVariable(name, "f4", dimensions)
                variable.units = units
                variable.long_name = text
                variable[0] = np.asarray(fields[name], dtype=np.float32)
        dataset.setncatts({
            "title": "Arwen Global regional parent frame",
            "Conventions": FRAME_CONVENTION,
            "valid_time": valid.strftime("%Y-%m-%d_%H:%M:%S"),
            **{str(key): value for key, value in (attrs or {}).items()},
        })
    except BaseException:
        dataset.close()
        temporary.unlink(missing_ok=True)
        raise
    dataset.close()
    os.replace(temporary, target)
    return target


def _write_json(path: Path, payload: dict[str, object]) -> str:
    text = json.dumps(payload, indent=1, sort_keys=False, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_parent_documents(outdir: str | Path, *, nlev: int,
                           boundary_interval_s: int, run_name: str,
                           config_hash: str, start: str, grid: str,
                           overwrite: bool = False) -> dict[str, dict[str, str]]:
    """Write the mapping, composition and provenance beside the frames and
    return each one's path and SHA-256."""
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    documents = {
        "mapping": (MAPPING_NAME, parent_mapping(
            nlev, boundary_interval_s=boundary_interval_s)),
        "composition": (COMPOSITION_NAME, parent_composition()),
        "provenance": (PROVENANCE_NAME, parent_provenance(
            run_name=run_name, config_hash=config_hash, start=start,
            grid=grid, nlev=nlev)),
    }
    written = {}
    for role, (name, payload) in documents.items():
        path = output / name
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} exists; pass --overwrite to replace it")
        written[role] = {"path": str(path), "sha256": _write_json(path, payload)}
    return written


def regional_prep_command(frames, outdir: str | Path) -> list[str]:
    """The regional engine's preparation command for these frames, as an
    argument list: everything but the experiment's own config, namelist,
    geography root and output folder, which belong to the regional run.

    Every path is absolute.  The command is run later, from the regional
    run's folder and not from the one the export ran in, and a relative
    ``--outdir`` copied into it would point at files that are not there."""
    output = Path(os.path.abspath(outdir))
    frames = [Path(os.path.abspath(frame)) for frame in frames]
    command = [
        "gpuwm", "prep", "--source", "mapped", "--source-format", "netcdf",
        "--mapping", str(output / MAPPING_NAME),
        "--composition", str(output / COMPOSITION_NAME),
        "--provenance", f"{PROVENANCE_ROLE}={output / PROVENANCE_NAME}",
    ]
    for frame in frames:
        command += ["--input", str(frame)]
    for frame in frames:
        command += ["--supplement", f"{SUPPLEMENT_ROLE}={frame}"]
    command += ["--author-input-manifest", str(output / "inputs.json")]
    return command


# ---------------------------------------------------------------------------
# the export itself
# ---------------------------------------------------------------------------

def diagnose_interface_w(model, transform, atmosphere):
    """The air's vertical velocity (m/s) THROUGH the model's coordinate
    surfaces, on its interfaces, top to surface, on the Gaussian grid.

    ``vertical.continuity`` returns the mass flux across each hybrid
    interface, eta-dot dp/d-eta in Pa/s, which is zero at the top and at the
    surface whatever the flow; this is that flux over rho g, positive up
    (the parent export's recipe).  It is not the pressure velocity Dp/Dt and
    so not the hydrostatic w: the part that moves with the surfaces,
    B (dps/dt + V . grad ps), is left out.  Uniform flow up a slope has
    w = V . grad h at the ground and zero here.  The frame, the provenance
    note and the page all say so; a reader who needs the velocity relative
    to the ground must add that term."""
    xp = transform.backend.xp
    native = model.grid_state(atmosphere)
    _, div_mass = model._mass_flux_divergence(native["dp"], native["u"], native["v"])
    ps_t = -xp.sum(div_mass, axis=0)
    omega_half, _closure = model.vertical.continuity(div_mass, ps_t, transform.backend)
    rho_full = native["p_full"] / (model.gas_constant * native["virtual_temperature"])
    rho_half = xp.empty_like(omega_half)
    rho_half[0] = rho_full[0]
    rho_half[-1] = rho_full[-1]
    rho_half[1:-1] = 0.5 * (rho_full[:-1] + rho_full[1:])
    return -omega_half / xp.maximum(rho_half * GRAVITY_M_S2, 1.0e-12)


def plan_parent_frames(checkpoints, start: datetime, outdir: str | Path, *,
                       overwrite: bool = False) -> list[dict[str, object]]:
    """Every frame this export will write, decided before any is written.

    Reads each checkpoint's header only (no arrays, no device) and refuses,
    with nothing written, the three things that used to surface in the
    middle of the series: a checkpoint that is not later than the one before
    it, a frame that already exists when ``overwrite`` is off, and a parent
    frame already in the folder that this export will not write.  The last
    one is refused with ``overwrite`` too: the documents and the receipt
    describe exactly the frames of one export, and a stale frame beside them
    is picked up by any reader that lists the folder.
    """
    from .checkpoint import read_checkpoint_header

    output = Path(outdir)
    plan: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        header = read_checkpoint_header(checkpoint)
        seconds = float(header["time_s"])
        if abs(seconds - round(seconds)) > 1.0e-6:
            raise ValueError(f"checkpoint time {seconds} s is not a whole second")
        valid = start + timedelta(seconds=round(seconds))
        if plan and valid <= plan[-1]["valid"]:
            raise ValueError(
                f"checkpoint {checkpoint} (valid {valid}) is not later than "
                "the one before it; pass the checkpoints in time order")
        plan.append({"checkpoint": Path(checkpoint), "seconds": seconds,
                     "valid": valid, "path": output / frame_filename(valid)})
    planned = {row["path"].name for row in plan}
    present = sorted(path.name for path in output.glob(f"{FRAME_PREFIX}*.nc"))
    stale = [name for name in present if name not in planned]
    if stale:
        shown = ", ".join(stale[:3]) + (" ..." if len(stale) > 3 else "")
        raise FileExistsError(
            f"{output} holds {len(stale)} parent frame(s) this export does not "
            f"write ({shown}); its documents and receipt would describe only "
            "part of the folder. Export into an empty folder, or remove them")
    if not overwrite:
        clash = [name for name in present if name in planned]
        if clash:
            raise FileExistsError(
                f"parent frame {output / clash[0]} exists"
                + (f" (and {len(clash) - 1} more of this series)" if len(clash) > 1 else "")
                + "; pass --overwrite to replace it")
    return plan


def _uniform_interval(valid_times: list[datetime]) -> int | None:
    if len(valid_times) < 2:
        return None
    steps = {int(round((b - a).total_seconds()))
             for a, b in zip(valid_times[:-1], valid_times[1:])}
    if len(steps) != 1:
        return None
    (step,) = steps
    return step if step > 0 else None


def export_mapped_parent(
    cfg,
    checkpoints,
    outdir: str | Path,
    *,
    nlat: int,
    nlon: int,
    bbox: tuple[float, float, float, float] | None = None,
    start_date: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write one parent frame per checkpoint on a regular ``nlat`` x ``nlon``
    grid (or its ``bbox`` window, degrees with longitude in -180..180), plus
    the mapping, composition, provenance and receipt.  Returns the receipt.

    The checkpoints must be in time order.  A series the regional engine can
    use as lateral boundaries is uniformly spaced; a series that is not is
    still written (a single frame is a valid initial state) and the receipt
    says so.

    The whole series is checked from the checkpoint headers before the first
    frame is regridded (:func:`plan_parent_frames`): order, frames that
    already exist, and frames in the folder that are not part of this
    export.  On the card a refusal found at the ninth frame used to cost the
    eight before it, and left them in the folder with no receipt.
    """
    from . import render_kernels
    from .pins import pins_hash
    from .receipt import write_receipt
    from .runner import (
        adopt_checkpoint_clock, build_model_and_cold_state, build_transform,
    )
    from .checkpoint import read_checkpoint, state_from_checkpoint
    from .spectral.sampling import regular_latlon_coordinates
    from .spectral.vector import VorticityDivergenceOperator
    from .wrfout_export import (
        SURFACE_DIAGNOSTICS_ATTR, TAPE_WATER_CONVENTION, WATER_CONVENTION_ATTR,
        _gaussian_to_regular, _screen_level_fields, _snow_planes,
        parse_start_date, tape_start, tape_water_fields,
    )

    checkpoints = [Path(item) for item in checkpoints]
    if not checkpoints:
        raise ValueError("no checkpoint to export")
    given = parse_start_date(start_date)
    render_kernels.library()
    # Absolute from here on: the receipt's paths and its preparation command
    # are read from another folder.
    output = Path(os.path.abspath(outdir))
    checkpoints = [Path(os.path.abspath(item)) for item in checkpoints]
    transform = build_transform(cfg)
    model, _cold = build_model_and_cold_state(
        cfg, transform, scratch_destination=output)
    clock = model.forecast_clock
    start = tape_start(clock, given)
    backend = transform.backend
    vector = VorticityDivergenceOperator(transform)

    plan = plan_parent_frames(checkpoints, start, output, overwrite=overwrite)

    lat, lon = regular_latlon_coordinates(nlat, nlon, include_poles=False)
    lon_signed = np.where(lon >= 180.0, lon - 360.0, lon)
    order = np.argsort(lon_signed)
    lon_out = lon_signed[order]
    lon_sample = lon[order]
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = (float(v) for v in bbox)
        if not (lat_min < lat_max and lon_min < lon_max):
            raise ValueError(
                "bbox must be (lat_min, lat_max, lon_min, lon_max) with "
                "min < max on both axes")
        lat_sel = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_sel = np.where((lon_out >= lon_min) & (lon_out <= lon_max))[0]
        if lat_sel.size < 2 or lon_sel.size < 2:
            raise ValueError(
                f"bbox {bbox} selects {lat_sel.size}x{lon_sel.size} points; "
                "a parent frame needs at least 2x2")
        lat = lat[lat_sel]
        lon_out = lon_out[lon_sel]
        lon_sample = lon_sample[lon_sel]
    if lat[0] > lat[-1]:
        raise ValueError("the regular grid's latitudes do not run south to north")
    sampler = _gaussian_to_regular(transform.grid, lat, lon_sample)

    def regrid(values, post: str = "none") -> np.ndarray:
        return sampler(backend.to_numpy(values), post)

    def up(field: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(field[::-1]).astype(np.float32)

    ny_out, nx_out = lat.size, lon_out.size
    a_half = np.asarray(cfg.a_half_pa, dtype=np.float64)
    b_half = np.asarray(cfg.b_half, dtype=np.float64)
    nlev = a_half.size - 1
    phi_surface = regrid(model.surface_geopotential)
    terrain_m = (phi_surface / GRAVITY_M_S2).astype(np.float32)
    grid_text = (
        f"a regular {nlat}x{nlon} lat-lon grid of cell centres "
        f"({180.0 / nlat:g} x {360.0 / nlon:g} deg)"
        + ("" if bbox is None else
           f", window {lat[0]:.4f}..{lat[-1]:.4f}N {lon_out[0]:.4f}..{lon_out[-1]:.4f}E")
    )

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    valid_times: list[datetime] = []
    paths: list[Path] = []
    for planned in plan:
        checkpoint = planned["checkpoint"]
        metadata, arrays = read_checkpoint(
            checkpoint,
            expected_config_hash=cfg.config_hash,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
        )
        bundle = state_from_checkpoint(metadata, arrays, backend)
        adopt_checkpoint_clock(model, bundle, f"checkpoint {checkpoint}")
        model.enforce(bundle)
        atmosphere = bundle.atmosphere
        seconds = float(metadata["time_s"])
        if seconds != planned["seconds"]:
            raise ValueError(
                f"checkpoint {checkpoint} reads {seconds} s with its arrays "
                f"and {planned['seconds']} s in its header")
        valid = planned["valid"]
        path = planned["path"]

        theta = regrid(transform.inverse(atmosphere.theta))
        ps = regrid(transform.inverse(atmosphere.log_surface_pressure), "exp")
        u_grid, v_grid = vector.wind_from_vordiv(
            atmosphere.vorticity, atmosphere.divergence)
        u = regrid(u_grid)
        v = regrid(v_grid)
        species = {
            name: regrid(
                transform.inverse(atmosphere.qv) if name == "qv"
                else getattr(atmosphere, name),
                "floor_zero",
            )
            for name in WATER_SPECIES
        }
        w_half = regrid(diagnose_interface_w(model, transform, atmosphere))
        p_half, p_full = render_kernels.hybrid_pressure(a_half, b_half, ps)
        temperature, virtual = render_kernels.virtual_temperature(
            theta, species, p_full,
            reference_pressure_pa=REFERENCE_PRESSURE_PA, kappa=KAPPA)
        phi_half = render_kernels.hydrostatic(
            virtual, p_half, phi_surface, gas_constant=model.gas_constant)

        surface = bundle.surface
        screen, screen_source = _screen_level_fields(
            cfg, bundle, backend, regrid, temperature, species, u, v, p_half, p_full)
        mixing, q2 = tape_water_fields(species, screen["q2"])
        snow = _snow_planes(bundle, regrid, (ny_out, nx_out))
        land = regrid(surface.land_fraction)
        fields = {
            "T": up(theta - _THETA_OFFSET_K),
            "PB": up(p_full),
            "QVAPOR": up(mixing["qv"]),
            "QCLOUD": up(mixing["qc"]),
            "QRAIN": up(mixing["qr"]),
            "QICE": up(mixing["qi"]),
            "QSNOW": up(mixing["qs"]),
            "QGRAUP": up(mixing["qg"]),
            "U_EARTH": up(u),
            "V_EARTH": up(v),
            "PHB": up(phi_half),
            "W": up(w_half),
            "PSFC": ps.astype(np.float32),
            "HGT": terrain_m,
            "TSK": regrid(surface.temperature_k).astype(np.float32),
            "T2": np.asarray(screen["t2"], dtype=np.float32),
            "Q2": np.asarray(q2, dtype=np.float32),
            "U10": np.asarray(screen["u10"], dtype=np.float32),
            "V10": np.asarray(screen["v10"], dtype=np.float32),
            "LANDMASK": (land >= 0.5).astype(np.float32),
            "SEAICE": np.clip(regrid(surface.sea_ice_fraction), 0.0, 1.0).astype(np.float32),
            "SNOW": np.maximum(snow["SNOW"], 0.0).astype(np.float32),
            "SNOWH": np.maximum(snow["SNOWH"], 0.0).astype(np.float32),
            "TSLB": regrid(surface.soil_temperature_k)[:len(SOIL_LAYER_TOPS_CM)].astype(np.float32),
            "SMOIS": regrid(surface.soil_water_fraction)[:len(SOIL_LAYER_TOPS_CM)].astype(np.float32),
        }
        write_parent_frame(
            path, valid, lat, lon_out, fields,
            attrs={
                "START_DATE": start.strftime("%Y-%m-%d_%H:%M:%S"),
                "forecast_seconds": np.float64(seconds),
                "source_checkpoint": checkpoint.name,
                "source_checkpoint_self_sha256": str(metadata.get("self_sha256", "")),
                "config_hash": cfg.config_hash,
                SURFACE_DIAGNOSTICS_ATTR: screen_source,
                WATER_CONVENTION_ATTR: TAPE_WATER_CONVENTION,
            },
        )
        paths.append(path)
        valid_times.append(valid)
        rows.append({
            "frame": str(path),
            "sha256": _file_sha256(path),
            "valid": valid.strftime("%Y-%m-%d_%H:%M:%S"),
            "forecast_seconds": seconds,
            "checkpoint": str(checkpoint),
            "checkpoint_self_sha256": str(metadata.get("self_sha256", "")),
            "step": int(metadata["step"]),
            "surface_diagnostics": screen_source,
            "ranges": {
                name: [float(np.min(fields[name])), float(np.max(fields[name]))]
                for name in ("T", "PB", "QVAPOR", "U_EARTH", "V_EARTH", "W",
                             "PSFC", "HGT", "TSK", "T2")
            },
        })

    interval = _uniform_interval(valid_times)
    documents = write_parent_documents(
        output, nlev=nlev, boundary_interval_s=interval or 3600,
        run_name=cfg.name, config_hash=cfg.config_hash,
        start=start.strftime("%Y-%m-%d_%H:%M:%S"), grid=grid_text,
        # The documents are a function of the frames; a re-export into the
        # same folder rewrites them with the frames it replaced.
        overwrite=True,
    )
    receipt_path = write_receipt(output / RECEIPT_NAME, {
        "name": cfg.name,
        "config_hash": cfg.config_hash,
        "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
        "physics_mode": cfg.physics_mode,
        "convention": FRAME_CONVENTION,
        "start_date": start.strftime("%Y-%m-%d_%H:%M:%S"),
        "start_date_source": (
            clock.source if clock is not None and clock.dated
            else "start-date argument"),
        "grid": {
            "kind": "regular-latlon-cell-centres", "nlat": int(nlat),
            "nlon": int(nlon), "bbox": None if bbox is None else [float(v) for v in bbox],
            "ny": int(ny_out), "nx": int(nx_out),
            "latitude": [float(lat[0]), float(lat[-1])],
            "longitude": [float(lon_out[0]), float(lon_out[-1])],
        },
        "levels": int(nlev),
        "constants": {"gravity_m_s2": GRAVITY_M_S2, "kappa": KAPPA,
                      "reference_pressure_pa": REFERENCE_PRESSURE_PA},
        "water_convention": TAPE_WATER_CONVENTION,
        "boundary_interval_seconds": interval,
        "lateral_boundary_series": (
            "uniform" if interval else
            "not a boundary series: fewer than two frames, or unevenly spaced"),
        "documents": documents,
        "roles": {"supplement": SUPPLEMENT_ROLE, "provenance": PROVENANCE_ROLE},
        "regional_prep_command": regional_prep_command(paths, output),
        "frames": rows,
    })
    return json.loads(Path(receipt_path).read_text(encoding="utf-8"))


__all__ = [
    "COMPOSITION_NAME", "FRAME_PREFIX", "FRAME_VARIABLES", "MAPPING_NAME",
    "PROVENANCE_NAME", "PROVENANCE_ROLE", "RECEIPT_NAME", "SUPPLEMENT_ROLE",
    "VERTICAL_VELOCITY_POLICY", "check_parent_frame", "diagnose_interface_w", "export_mapped_parent",
    "frame_filename", "parent_composition", "parent_mapping",
    "parent_provenance", "plan_parent_frames", "regional_prep_command",
    "write_parent_documents",
    "write_parent_frame",
]
