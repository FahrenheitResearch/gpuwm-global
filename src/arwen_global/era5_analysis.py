"""ERA5 as a global cold start: the fetch, and what the cold start needs that ERA5 does not carry.

Until this module the global model cold-started from one kind of input, a
whole-globe GDAS 0.25-degree analysis, and the GDAS archive the engine's
fetch route reads begins at 2021-03-22 12Z.  A run started on an earlier
date, or on an hour that is not 00/06/12/18Z, had no input at all.  ERA5
covers every hour from 1940 to about six days ago, and the engine already
reads it without a key: `gpuwm.era5_arco.retrieve_era5_arco` asks Google's
public ARCO Zarr store through the `rw_zarr` program in the engine's bridge
bundle and writes one NetCDF, ``era5-combined.nc``.

THE DECODE IS THE ENGINE'S, through a NetCDF mapping this package ships
(``data/package-mappings/rw-wps-era5-arco-global-analysis-netcdf.mapping.json``,
bare id ``era5-arco-global``).  Nothing here reads a level or a wind.  What
this module adds is the part of the initial state GDAS carries and ERA5 does
not, each one a stated rule recorded in the run's provenance under
``era5_completion``:

* SOIL OVER WATER.  ERA5 writes soil temperature and soil water over the sea
  (water near 0, down to -0.013).  The GDAS product leaves those points
  missing and the cold start fills them from the skin before it regrids.
  The engine's ``landmask_water`` missing policy would do the same, but
  engine 2.9.13 applies it only on the GRIB route and ignores it on NetCDF,
  so the mapping declares the soil as the file holds it (no missing values)
  and every soil value on a point whose ERA5 land-sea mask is below 0.5 is
  set missing here, the shape of the GDAS bitmap; the cold start's own water
  fill then applies.  Unmasked, every coastal land column would mix ocean
  "soil" into its bilinear regrid.
* SEA ICE ON LAND AND ON THE COAST.  ERA5's sea fields (sea-ice cover and
  sea surface temperature) are missing on land, and the cold start refuses a
  hole in a sea-ice plane.  Land becomes 0.  The mask of the sea fields is
  not the land-sea mask at 0.5: on 2019-05-20 00Z, 2,900 points whose land
  fraction is below 0.5 (0.19 to 0.50) carry no sea value.  They are COAST,
  not lakes (their mean lake cover is 0.07, and ERA5's real lakes, the Great
  Lakes and Great Bear Lake among them, do carry a sea-ice value), so each
  takes the mean of the sea values beside it (the 3 x 3 finite-neighbour
  mean, :data:`SEA_FIELD_FILL_PASSES` passes; every one of the 2,900 touched
  a sea point).  ERA5's FLake ice depth is defined at every point, lake or
  not, and read as "frozen" on these points it froze 147 open-sea coast
  points (west Svalbard) on that hour.  It answers only for a water point no
  sea value reaches, which is enclosed water: 1 where its ice depth is above
  0, else 0.
* SEA SURFACE TEMPERATURE.  The cold start reads the skin temperature of the
  analysis's water points as the water temperature it holds for the run.
  GDAS's mask is 0 or 1 and its skin on a water point is water.  ERA5's mask
  is a fraction and its skin is the mean over the point's tiles, so the skin
  of a point with land fraction 0.3 is three tenths ground: read as water it
  started an open-water column in the Gulf of California 10.8 K above ERA5's
  SST, and 1,747 columns more than 1 K off.  So an ice-free ERA5 water point
  takes ERA5's own SST (``SST`` in the engine's file, filled on the coast
  points like the ice) as its skin; a point with ice keeps the skin, which
  is the ice-and-water composite the frozen column reads.
* SEA-ICE THICKNESS.  Not in ERA5 at all.  A sea-ice point takes 3.0 m,
  WRF's ``seaice_thickness_default``, which is what WRF does when its input
  carries no thickness; frozen enclosed water takes ERA5's own lake ice
  depth.  3.0 m is too thick for first-year Antarctic ice (about 1 m) and for
  a frozen lake ERA5 carries in its sea-ice cover; it is a stand-in, and the
  receipt says so.
* SNOW DEPTH.  ERA5 carries snow water (``sd``, metres of water, which the
  mapping scales to kg m-2) and snow DENSITY, not depth.  The fetch asks the
  same `rw_zarr` program for ``snow_density`` (about 0.5 MB more) and writes
  it beside the analysis as ``era5-snow-density.nc``; depth is water over
  density.  A snowy point with no readable density gets no depth here, and
  the seeding then derives one at 200 kg m-3 and counts it.
* SNOW ON SEA ICE.  ERA5 has none (its snow is zero on its sea-ice points),
  so the pack starts bare, and the frame says so (``snow_on_sea_ice =
  False``): the seeding then averages snow over the analysis's land points
  only and leaves every frozen water column bare.  Its GDAS rule, the mean
  over land AND sea-ice points, read ERA5's bare ice as analysed snow and
  mixed the glacier beside it onto the pack: 2,336 sea-ice columns started
  with snow, up to 8,178 kg m-2 (31 m deep).  Nothing is invented for it.

ERA5's own snow is not capped.  It puts 10 m of water on glaciers and deep
packs on some high-latitude and mountain land; GDAS carries far less.  A cap
would be a rule this module made up, so there is none.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

__all__ = [
    "COMPLETION_SCHEMA",
    "ERA5_FIRST_HOUR",
    "ERA5_GLOBAL_DOC",
    "ERA5_GLOBAL_MAPPING_ID",
    "ERA5_GLOBAL_MAPPING_NAME",
    "ERA5_GLOBAL_QUICKSTART_CONFIG",
    "LAKE_ICE_DEPTH_VARIABLE",
    "SEA_FIELD_FILL_PASSES",
    "SEA_ICE_THICKNESS_STANDIN_M",
    "SEA_SURFACE_TEMPERATURE_VARIABLE",
    "SNOW_DENSITY_FILE",
    "SNOW_DENSITY_VARIABLE",
    "WHOLE_GLOBE_AREA",
    "complete_era5_fields",
    "complete_era5_frame",
    "era5_fetch_epilogue",
    "fetch_era5_analysis",
    "is_era5_mapping",
]

#: The bare id a config names; it globs exactly one file.
ERA5_GLOBAL_MAPPING_ID = "era5-arco-global"
#: The ``name`` inside that mapping, which is what the cold start keys on.
ERA5_GLOBAL_MAPPING_NAME = "ecmwf-era5-arco-netcdf-global-analysis"
#: The shipped experiment an ERA5 start is a copy of.  No ERA5 experiment
#: ships: the ``[initial]`` table names one hour's file, so a shipped copy
#: would name a date nobody fetched, and the quickstart with that table
#: replaced is the whole configuration.
ERA5_GLOBAL_QUICKSTART_CONFIG = "arwen_global_t255_quickstart"
#: The page, in this repository.
ERA5_GLOBAL_DOC = "docs/ARWEN_GLOBAL_ERA5.md"

#: The supplement the fetch writes beside the analysis, and its variable.
SNOW_DENSITY_FILE = "era5-snow-density.nc"
SNOW_DENSITY_VARIABLE = "SNOW_DENSITY"
#: ERA5's FLake ice depth (m), in the engine's combined file.
LAKE_ICE_DEPTH_VARIABLE = "LAKE_ICE_DEPTH"
#: ERA5's sea surface temperature (K), in the engine's combined file; missing
#: wherever the sea-ice cover is.
SEA_SURFACE_TEMPERATURE_VARIABLE = "SST"
#: How many 3 x 3 neighbour passes fill a water point ERA5's sea fields
#: leave out from the sea values beside it.  One pass reached 2,899 of the
#: 2,900 on 2019-05-20 00Z and the second the last; a point still out of
#: reach after four is enclosed water, not coast.
SEA_FIELD_FILL_PASSES = 4
#: WRF's ``seaice_thickness_default`` (m): its rule when the input carries
#: no ice thickness.  ERA5 carries none.
SEA_ICE_THICKNESS_STANDIN_M = 3.0
#: ERA5's land-sea mask is a fraction; a point is land at 0.5 and above, the
#: split the cold start's own analysis-mask tests use.
LAND_THRESHOLD = 0.5

#: The engine's area spelling for the whole globe (`gpuwm.fetch.parse_area`).
WHOLE_GLOBE_AREA = "-90,0,90,360"
#: The first hour the ARCO store holds.
ERA5_FIRST_HOUR = dt.datetime(1940, 1, 1, 0)

COMPLETION_SCHEMA = "arwen-global-era5-completion-v2"


# ------------------------------------------------------------- the completion

def is_era5_mapping(mapping_path) -> bool:
    """Whether ``mapping_path`` is the ERA5 global analysis mapping, by name."""

    try:
        document = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return document.get("name") == ERA5_GLOBAL_MAPPING_NAME


def complete_era5_fields(fields: dict, *, lake_ice_depth_m, snow_density_kg_m3,
                         sea_surface_temperature_k):
    """The cold start's surface fields from ERA5's, by the rules in this
    module's docstring.

    ``fields`` maps the decoded names to arrays (the soil fields
    ``(layer, y, x)``, the rest ``(y, x)``); ``snow_water_equivalent`` is in
    kg m-2, as the mapping scales it.  Returns ``(fields, record)``: a new
    dict with the soil fields masked over water, ``sea_ice_fraction``
    without holes, ``skin_temperature`` reading ERA5's SST on the ice-free
    water points, and ``sea_ice_thickness`` and ``snow_depth`` added; the
    input arrays are not modified.
    """

    from .analysis_initial import _grow_over_primary_land

    land_fraction = np.asarray(fields["land_fraction"], dtype=np.float64)
    shape = land_fraction.shape
    land = land_fraction >= LAND_THRESHOLD
    water = ~land

    def plane(name, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != shape:
            raise ValueError(
                f"ERA5 {name} has shape {values.shape}; the analysis surface "
                f"plane is {shape}")
        return values

    out = dict(fields)
    soil_values_masked = 0
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        stack = np.array(fields[name], dtype=np.float64, copy=True)
        if stack.ndim != 3 or stack.shape[1:] != shape:
            raise ValueError(
                f"ERA5 {name} has shape {stack.shape}; a (layer, y, x) stack "
                f"on the {shape} analysis grid was expected")
        soil_values_masked += int(np.count_nonzero(np.isfinite(stack[:, water])))
        stack[:, water] = np.nan
        out[name] = stack

    ice = plane("sea_ice_fraction", fields["sea_ice_fraction"])
    lake_ice = plane("lake_ice_depth", lake_ice_depth_m)
    lake_ice = np.where(np.isfinite(lake_ice), np.maximum(lake_ice, 0.0), 0.0)
    missing_ice = ~np.isfinite(ice)
    # A water point ERA5's sea fields leave out is coast: it reads the sea
    # beside it.  Only a point no sea value reaches is enclosed water, and
    # only there does the FLake ice depth answer.
    coast = missing_ice & water
    grown_ice, _filled, _left = _grow_over_primary_land(
        ice, coast, SEA_FIELD_FILL_PASSES)
    reached = coast & np.isfinite(grown_ice)
    enclosed = coast & ~reached
    frozen_enclosed = enclosed & (lake_ice > 0.0)
    ice_out = np.where(missing_ice, 0.0, ice)
    ice_out = np.where(reached, np.clip(np.where(reached, grown_ice, 0.0), 0.0, 1.0), ice_out)
    ice_out = np.where(frozen_enclosed, 1.0, ice_out)
    sea_ice = (ice_out > 0.0) & ~enclosed
    thickness = np.where(sea_ice, SEA_ICE_THICKNESS_STANDIN_M, 0.0)
    thickness = np.where(frozen_enclosed, lake_ice, thickness)
    out["sea_ice_fraction"] = ice_out
    out["sea_ice_thickness"] = thickness

    # The water temperature of an ice-free water point is ERA5's SST, not
    # its skin, which is the mean over the point's land and water tiles.
    skin = plane("skin_temperature", fields["skin_temperature"])
    sst = plane("sea_surface_temperature", sea_surface_temperature_k)
    grown_sst, _filled, _left = _grow_over_primary_land(
        sst, water & ~np.isfinite(sst), SEA_FIELD_FILL_PASSES)
    takes_sst = water & (ice_out <= 0.0) & np.isfinite(grown_sst)
    skin_out = np.where(takes_sst, np.where(takes_sst, grown_sst, 0.0), skin)
    skin_change = np.where(takes_sst, skin_out - skin, 0.0)
    out["skin_temperature"] = skin_out

    swe = plane("snow_water_equivalent", fields["snow_water_equivalent"])
    density = plane("snow_density", snow_density_kg_m3)
    snowy = np.isfinite(swe) & (swe > 0.0)
    readable = snowy & np.isfinite(density) & (density > 0.0)
    depth = np.zeros(shape, dtype=np.float64)
    np.divide(swe, density, out=depth, where=readable)
    # A masked water equivalent stays masked in the depth, so the seeding
    # reads both planes' bitmap the same way.
    depth = np.where(np.isfinite(swe), depth, np.nan)
    out["snow_depth"] = depth

    def _range(values, where):
        picked = np.asarray(values)[where]
        picked = picked[np.isfinite(picked)]
        if picked.size == 0:
            return None
        return [float(picked.min()), float(picked.max())]

    record = {
        "schema": COMPLETION_SCHEMA,
        "mapping": ERA5_GLOBAL_MAPPING_NAME,
        "land_rule": f"a point is land where the ERA5 land-sea mask is >= {LAND_THRESHOLD:g}",
        "soil_over_water": {
            "rule": "soil temperature and soil water set missing on ERA5 water "
                    "points (the GDAS bitmap's shape); the cold start's water "
                    "fill then applies.  ERA5 writes values there, and engine "
                    "2.9.13 does not apply landmask_water to NetCDF",
            "values_masked": soil_values_masked,
            "water_points": int(np.count_nonzero(water)),
        },
        "sea_ice_fraction": {
            "rule": "missing on land -> 0; missing on a water point (coast "
                    "the ERA5 sea fields leave out) -> the mean of the sea "
                    f"values beside it, 3 x 3, {SEA_FIELD_FILL_PASSES} passes; "
                    "a water point no sea value reaches (enclosed water) -> "
                    "1 where ERA5 lake_ice_depth > 0, else 0",
            "missing_on_land": int(np.count_nonzero(missing_ice & land)),
            "coast_points_without_a_sea_value": int(np.count_nonzero(coast)),
            "coast_points_filled_from_the_sea_beside": int(np.count_nonzero(reached)),
            "coast_points_filled_with_ice": int(np.count_nonzero(reached & (ice_out > 0.0))),
            "enclosed_water_points": int(np.count_nonzero(enclosed)),
            "frozen_enclosed_water_points": int(np.count_nonzero(frozen_enclosed)),
            "frozen_enclosed_water_ice_depth_m": _range(lake_ice, frozen_enclosed),
            "sea_ice_points": int(np.count_nonzero(sea_ice)),
        },
        "sea_surface_temperature": {
            "rule": "an ice-free water point takes ERA5's SST as its skin "
                    "(filled on the coast points like the ice); a point with "
                    "ice, and a water point no SST reaches, keeps ERA5's skin",
            "water_points_taking_sst": int(np.count_nonzero(takes_sst)),
            "water_points_keeping_skin": int(np.count_nonzero(water & ~takes_sst)),
            "ice_free_water_points_no_sst_reaches": int(np.count_nonzero(
                water & (ice_out <= 0.0) & ~takes_sst)),
            "sst_minus_skin_k": {
                "mean": float(skin_change[takes_sst].mean()) if takes_sst.any() else None,
                "range": _range(skin_change, takes_sst),
                "points_changed_by_more_than_1_k": int(np.count_nonzero(
                    np.abs(skin_change) > 1.0)),
            },
        },
        "sea_ice_thickness": {
            "rule": f"not in ERA5: {SEA_ICE_THICKNESS_STANDIN_M:g} m on every "
                    "sea-ice point (WRF seaice_thickness_default); ERA5 "
                    "lake_ice_depth on frozen enclosed water",
            "stand_in_m": SEA_ICE_THICKNESS_STANDIN_M,
            "stand_in": True,
        },
        "snow_depth": {
            "rule": "ERA5 snow water (kg m-2) / ERA5 snow_density (kg m-3); a "
                    "snowy point without a readable density gets none here and "
                    "the seeding derives it at its own default density",
            "snowy_points": int(np.count_nonzero(snowy)),
            "snowy_points_without_density": int(np.count_nonzero(snowy & ~readable)),
            "density_on_snow_kg_m3": _range(density, readable),
            "depth_m": _range(depth, readable),
            "snow_water_kg_m2": _range(swe, snowy),
        },
        "snow_on_sea_ice": "none: ERA5 carries no snow on its sea-ice points, "
                           "so the pack starts bare; the seeding averages "
                           "snow over land points only and leaves every "
                           "frozen water column bare (its count is in "
                           "surface_seeding)",
        "snow_cap": "none: ERA5's snow is taken as analysed, including 10 m "
                    "of water on glaciers",
    }
    return out, record


class _Field:
    __slots__ = ("values", "source")

    def __init__(self, values, source="primary"):
        self.values = values
        self.source = source


class Era5Frame:
    """A decoded ERA5 frame with the completed surface fields."""

    #: Read by the cold start's surface seeding: ERA5's sea-ice points carry
    #: no snow, so snow is averaged over land points only and no frozen
    #: water column starts with any.
    snow_on_sea_ice = False

    def __init__(self, frame, fields: dict):
        self.latitude = frame.latitude
        self.longitude = frame.longitude
        self.vertical_kind = getattr(frame, "vertical_kind", "pressure")
        self.vertical_values = frame.vertical_values
        self.fields = fields
        for name in ("mapping_sha256", "input_sha256", "source_cycle", "valid_time"):
            setattr(self, name, getattr(frame, name, None))


def _naive_utc(value) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, dt.datetime):
        value = dt.datetime(value.year, value.month, value.day,
                            getattr(value, "hour", 0), getattr(value, "minute", 0),
                            getattr(value, "second", 0))
    if value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return value


def _read_plane(path: Path, variable: str, frame) -> np.ndarray:
    """One surface plane of a bridge-written NetCDF, checked against the
    decoded frame's grid and valid time."""

    from netCDF4 import Dataset, num2date

    if not path.is_file():
        raise FileNotFoundError(
            f"the ERA5 cold start needs {path.name} beside the analysis "
            f"({path}); `gpuwm-global fetch-analysis --source era5` writes it")
    with Dataset(path) as dataset:
        if variable not in dataset.variables:
            raise ValueError(f"{path.name} carries no {variable} variable")
        latitude = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
        longitude = np.asarray(dataset.variables["longitude"][:], dtype=np.float64)
        if (latitude.shape != np.shape(frame.latitude)
                or longitude.shape != np.shape(frame.longitude)
                or not np.allclose(latitude, np.asarray(frame.latitude), atol=1e-6)
                or not np.allclose(longitude, np.asarray(frame.longitude), atol=1e-6)):
            raise ValueError(
                f"{path.name} is not on the decoded analysis grid; it was "
                "written for another fetch")
        times = dataset.variables["time"]
        stamps = num2date(np.asarray(times[:]), times.units,
                          only_use_cftime_datetimes=False,
                          only_use_python_datetimes=True)
        stamps = [_naive_utc(stamp) for stamp in np.atleast_1d(stamps)]
        valid = _naive_utc(getattr(frame, "valid_time", None))
        if valid is None or len(stamps) != 1 or stamps[0] != valid:
            raise ValueError(
                f"{path.name} holds {[str(s) for s in stamps]} and the analysis "
                f"is valid at {valid}; the supplement must be the same hour")
        values = dataset.variables[variable][0]
        return np.asarray(np.ma.filled(np.ma.asarray(values, dtype=np.float64), np.nan))


def complete_era5_frame(frame, analysis_path):
    """The decoded ERA5 ``frame``, completed for the cold start.

    Reads ERA5's lake ice depth and sea surface temperature from the
    analysis file itself and its snow density from :data:`SNOW_DENSITY_FILE`
    beside it, each checked against the frame's grid and valid time, and
    returns ``(frame, record)``.
    """

    analysis_path = Path(analysis_path)
    lake_ice = _read_plane(analysis_path, LAKE_ICE_DEPTH_VARIABLE, frame)
    sst = _read_plane(analysis_path, SEA_SURFACE_TEMPERATURE_VARIABLE, frame)
    density_path = analysis_path.parent / SNOW_DENSITY_FILE
    density = _read_plane(density_path, SNOW_DENSITY_VARIABLE, frame)
    plain = {name: field.values for name, field in frame.fields.items()}
    completed, record = complete_era5_fields(
        plain, lake_ice_depth_m=lake_ice, snow_density_kg_m3=density,
        sea_surface_temperature_k=sst)
    record["snow_density_file"] = str(density_path.resolve())
    fields = {name: _Field(values) for name, values in completed.items()}
    return Era5Frame(frame, fields), record


# ------------------------------------------------------------------ the fetch

def _engine_combined_name() -> str:
    from gpuwm import fetch as engine_fetch

    namer = getattr(engine_fetch, "era5_combined_name", None)
    return namer("arco") if namer is not None else "era5-combined.nc"


def _snow_density_request(cycle: dt.datetime) -> dict:
    from gpuwm import fetch as engine_fetch
    from gpuwm.era5_arco import ARCO_STORE

    south, west, north, east = (float(v) for v in WHOLE_GLOBE_AREA.split(","))
    return {
        "store": ARCO_STORE,
        "times": [cycle.isoformat() + "Z"],
        "area": [south, west, north, east],
        # rw_zarr refuses a request whose declared levels differ from the
        # store's 37, even for a surface-only field.
        "expected_levels_hpa": list(engine_fetch.ERA5_PRESSURE_LEVELS_HPA),
        "specific_humidity_authority": False,
        "fields": [{"source": "snow_density", "output": SNOW_DENSITY_VARIABLE,
                    "units": "kg m-3", "pressure": False}],
        "derived": [],
    }


def fetch_era5_analysis(out: Path, cycle: dt.datetime, *, progress=print) -> dict:
    """Fetch one whole-globe ERA5 hour and its snow density into ``out``.

    The analysis is the engine's ARCO retrieval (`retrieve_era5_arco`, which
    keeps its own receipt and reuses a verified file of the same request);
    the snow density is one more `rw_zarr` request, written beside it as
    :data:`SNOW_DENSITY_FILE` with its request and result documents.
    Returns a record naming both files.
    """

    from gpuwm.era5_arco import retrieve_era5_arco
    from gpuwm.zarr_bridge import extract_regular_zarr

    cycle = _naive_utc(cycle)
    if cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError("an ERA5 start is an exact UTC hour")
    if cycle < ERA5_FIRST_HOUR:
        raise ValueError(
            f"ERA5 begins at {ERA5_FIRST_HOUR:%Y-%m-%d %HZ}; {cycle:%Y-%m-%d %HZ} "
            "is before it")
    out = Path(out).expanduser().resolve()
    analysis = retrieve_era5_arco(
        cycle=cycle, hours=0, area=WHOLE_GLOBE_AREA, out=out, progress=progress)
    request = _snow_density_request(cycle)
    target = out / SNOW_DENSITY_FILE
    stage = out / f".{SNOW_DENSITY_FILE}.partial"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        progress("fetch-analysis: ERA5 snow density (the depth ERA5 does not carry)")
        result = extract_regular_zarr(
            request, request_path=stage / "request.json",
            output=stage / SNOW_DENSITY_FILE, progress=progress)
        for name, final in (("request.json", "era5-snow-density.request.json"),
                            ("request.result.json", "era5-snow-density.result.json")):
            if (stage / name).exists():
                os.replace(stage / name, out / final)
        os.replace(stage / SNOW_DENSITY_FILE, target)
    finally:
        for leftover in stage.glob("*"):
            leftover.unlink()
        stage.rmdir()
    return {"analysis": Path(analysis), "snow_density": target,
            "cycle": cycle, "provisional_times": result.get("provisional_times", [])}


def era5_fetch_epilogue(*, analysis: Path) -> list[str]:
    """What a completed ERA5 fetch says comes next."""

    path = Path(analysis).as_posix()
    return [
        "fetch-analysis: next: cold-start the global model from this ERA5 "
        "hour.",
        f"  # copy {ERA5_GLOBAL_QUICKSTART_CONFIG} (gpuwm-global configs "
        "--paths prints where it is)\n"
        "  # and replace its [initial] table with this fetch's output:\n"
        "  #   [initial]\n"
        "  #   mode = \"analysis\"\n"
        f"  #   analysis_grib = \"{path}\"\n"
        f"  #   analysis_mapping = \"{ERA5_GLOBAL_MAPPING_ID}\"\n"
        "  #   surface_water_kg_m2 = 500.0",
        "  gpuwm-global run CONFIG --outdir out/arwen-global-era5",
        f"  # {SNOW_DENSITY_FILE} stays beside the analysis: the cold start\n"
        "  # reads snow depth from it.  What ERA5 lacks and what stands in\n"
        f"  # is {ERA5_GLOBAL_DOC}.",
    ]
