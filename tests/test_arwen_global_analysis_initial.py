from __future__ import annotations

from arwen_global.configs_dir import config_root as _shipped_configs
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


from arwen_global.analysis_initial import (
    DRY_AIR_GAS_CONSTANT,
    _global_regridder,
    analysis_initial_state,
    resolve_analysis_mapping,
    surface_virtual_temperature,
)
from arwen_global.config import load_config
from arwen_global.constants import GRAVITY_M_S2
from arwen_global.runner import build_transform
from arwen_global.spectral.transform import SphericalHarmonicTransform

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
#: The two spellings that resolve from an INSTALL.  The third,
#: `gpuwm/authorities/rw-wps-...json`, is a checkout-relative path: it named
#: a real file only while this model lived inside the engine's tree, and
#: these gates were written there.  Every shipped config carries the bare id.
MAPPING = "gdas-global"
MAPPING_NAME = "rw-wps-gdas-global-analysis-grib2.mapping.json"


@dataclass
class _Field:
    values: np.ndarray


@dataclass
class _Frame:
    latitude: np.ndarray
    longitude: np.ndarray
    vertical_kind: str
    vertical_values: np.ndarray
    fields: dict
    mapping_sha256: str = "test-mapping"
    input_sha256: str = "test-input"
    source_cycle: datetime = datetime(2026, 8, 30, 18)
    valid_time: datetime = datetime(2026, 8, 30, 18)


def _synthetic_frame(
    nlat=37, nlon=72, descending=True, ridge_sigma_deg=15.0,
    terrain_zonal_wavenumber=0,
):
    lat = np.linspace(90.0, -90.0, nlat) if descending else np.linspace(-90.0, 90.0, nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    levels = np.asarray([10000.0, 30000.0, 50000.0, 70000.0, 85000.0, 100000.0])
    lat2 = np.deg2rad(lat)[:, None] * np.ones((1, nlon))
    lon2 = np.deg2rad(lon)[None, :] * np.ones((nlat, 1))
    shape3 = (levels.size, nlat, nlon)
    temperature = 220.0 + 70.0 * (levels / 100000.0)[:, None, None] * np.cos(lat2)[None] ** 2
    humidity = 0.01 * (levels / 100000.0)[:, None, None] ** 3 * np.ones(shape3)
    u = 20.0 * np.cos(lat2)[None] * np.ones(shape3)
    v = np.zeros(shape3)
    envelope = np.exp(-((np.rad2deg(lat2) - 30.0) / ridge_sigma_deg) ** 2)
    if terrain_zonal_wavenumber:
        terrain = 700.0 * (
            1.0 + np.cos(terrain_zonal_wavenumber * lon2)
        ) * envelope
    else:
        terrain = 1500.0 * envelope
    ps = 101000.0 * np.exp(-terrain / 8000.0)
    land = (terrain > 100.0).astype(np.float64)
    skin = 288.0 - 30.0 * np.sin(lat2) ** 2
    soil_t = np.where(land[None] > 0.5, skin[None] - 1.0, np.nan) * np.ones((4, 1, 1))
    soil_m = np.where(land[None] > 0.5, 0.3, np.nan) * np.ones((4, 1, 1))
    # The surface seeding's four planes, GDAS-shaped: sea ice on the water
    # poleward of 70 degrees (1.2 m thick), snow on the ridge's land north
    # of 38 degrees (40 kg/m2 over 0.2 m), the snow planes masked (NaN)
    # on open water.
    lat_deg = np.rad2deg(lat2)
    ice = np.where((land < 0.5) & (np.abs(lat_deg) >= 70.0), 1.0, 0.0)
    thickness = np.where(ice >= 0.5, 1.2, 0.0)
    snowy = (land >= 0.5) & (lat_deg >= 38.0)
    open_water = (land < 0.5) & (ice < 0.5)
    swe = np.where(open_water, np.nan, np.where(snowy, 40.0, 0.0))
    snow_depth = np.where(open_water, np.nan, np.where(snowy, 0.2, 0.0))
    fields = {
        "sea_ice_fraction": _Field(ice),
        "sea_ice_thickness": _Field(thickness),
        "snow_water_equivalent": _Field(swe),
        "snow_depth": _Field(snow_depth),
        "air_temperature": _Field(temperature),
        "specific_humidity": _Field(humidity),
        "eastward_wind": _Field(u),
        "northward_wind": _Field(v),
        "surface_pressure": _Field(ps),
        "terrain_height": _Field(terrain),
        "skin_temperature": _Field(skin),
        "land_fraction": _Field(land),
        "soil_temperature": _Field(soil_t),
        "volumetric_soil_moisture": _Field(soil_m),
    }
    return _Frame(lat, lon, "pressure", levels, fields)


def _analysis_cfg(tmp_path, *, statics='[statics]\nsource = "synthetic"\n'):
    """An analysis-mode config on the smoke grid.

    These gates drive a synthetic frame, so the planet is the declared
    synthetic one; ``statics=""`` leaves the table out and takes the
    analysis-mode default (real statics, refused without a cache).
    """
    text = Path(CONFIG).read_text(encoding="utf-8")
    replaced = text.replace(
        "[initial]\n",
        "[initial]\nmode = \"analysis\"\n"
        "analysis_grib = \"unused.grib\"\n"
        f"analysis_mapping = \"{MAPPING}\"\n",
        1,
    )
    kept = []
    for line in replaced.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(key) for key in (
            "surface_pressure_pa", "surface_temperature_k", "top_temperature_k",
            "qv_surface", "zonal_wind_m_s", "perturbation_amplitude",
            "zonal_wavenumber", "terrain_amplitude_m",
        )):
            continue
        kept.append(line)
    path = tmp_path / "analysis.toml"
    path.write_text("\n".join(kept) + "\n" + statics, encoding="utf-8")
    return load_config(path)


# The mapping refusal fires BEFORE the statics refusal this test grades, so
# unlike its neighbours it cannot be answered by the refusal-to-skip rule in
# conftest: `pytest.raises` turns the engine's own sentence into an
# AssertionError of this file's making.  It is marked instead.
def test_an_analysis_run_defaults_to_real_statics_and_refuses_without_a_cache(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    cfg = _analysis_cfg(tmp_path, statics="")
    assert cfg.statics.source == "real" and not cfg.statics.declared
    transform = build_transform(cfg)
    with pytest.raises(FileNotFoundError, match="gpuwm-global statics"):
        analysis_initial_state(cfg, transform, frame=_synthetic_frame())


def test_a_declared_synthetic_planet_is_recorded_as_synthetic(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    state, _phi, provenance = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame()
    )
    from arwen_global.statics import (
        SURFACE_STATICS_METADATA_KEY, SYNTHETIC_CONVENTION, frozen_water_columns,
        water_columns,
    )

    statics = provenance["statics"]
    assert statics["source"] == "synthetic" and statics["declared"] is True
    host = transform.backend.to_numpy
    # One vegetation and soil class on land; the runtime's water columns
    # carry the MODIS water class and water soil, as real.exe leaves them;
    # the water columns the analysis freezes over carry the ice class and
    # ice soil (the synthetic frame plants ice poleward of 70 degrees).
    land_fraction = host(state.surface.land_fraction)
    ice = host(state.surface.sea_ice_fraction)
    water = water_columns(land_fraction, sea_ice_fraction=ice)
    frozen = frozen_water_columns(ice)
    assert water.any() and (~water).any() and frozen.any()
    assert not np.any(water & frozen) and np.all(land_fraction[frozen] <= 0.5)
    categories = host(state.surface.landuse_category)
    assert np.all(categories[~water & ~frozen] == 7)
    assert np.all(categories[water] == 17) and np.all(categories[frozen] == 15)
    soil = host(state.surface.soil_category_top)
    assert np.all(soil[~water & ~frozen] == 8)
    assert np.all(soil[water] == 14) and np.all(soil[frozen] == 16)
    assert np.allclose(host(state.surface.leaf_area_index), 3.0)
    assert state.physics_state.metadata[SURFACE_STATICS_METADATA_KEY] == (
        SYNTHETIC_CONVENTION.as_metadata("synthetic")
    )
    assert statics["convention"] == SYNTHETIC_CONVENTION.as_metadata("synthetic")
    assert np.allclose(
        host(state.surface.deep_soil_temperature_k),
        host(state.surface.soil_temperature_k)[-1],
    )


def test_analysis_state_is_complete_finite_and_hydrostatic(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    state, phi_surface, provenance = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame()
    )
    nlev = cfg.vertical.nlev
    shape = (nlev, *transform.spectral_shape)
    assert state.atmosphere.theta.shape == shape
    assert state.atmosphere.log_surface_pressure.shape == transform.spectral_shape
    assert state.atmosphere.qv.shape == shape
    # The condensate species and number moments are grid tracers: real,
    # nonnegative arrays on the Gaussian grid.
    for name in ("qc", "nc", "ng"):
        value = getattr(state.atmosphere, name)
        assert value.shape == (nlev, *transform.grid.shape)
        assert value.dtype.kind == "f"
        assert float(np.min(value)) >= 0.0
    grid_theta = transform.backend.to_numpy(transform.inverse(state.atmosphere.theta))
    assert np.isfinite(grid_theta).all()
    assert float(grid_theta.min()) > 200.0
    ps = np.exp(transform.backend.to_numpy(
        transform.inverse(state.atmosphere.log_surface_pressure)
    ))
    assert np.isfinite(ps).all() and 40000.0 < ps.min() and ps.max() < 110000.0
    # Condensate and moments start at exact spectral zero: the pgrb2 ladder
    # carries no condensate analyses.
    assert np.all(transform.backend.to_numpy(state.atmosphere.qc) == 0.0)
    assert np.all(transform.backend.to_numpy(state.atmosphere.ng) == 0.0)
    assert np.isfinite(transform.backend.to_numpy(phi_surface)).all()
    assert provenance["mode"] == "analysis"
    assert provenance["analysis_levels"] == 6


def test_ocean_masked_soil_is_filled_from_skin_temperature(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    state, _phi, _prov = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame()
    )
    soil = transform.backend.to_numpy(state.surface.soil_temperature_k)
    moisture = transform.backend.to_numpy(state.surface.soil_water_fraction)
    assert np.isfinite(soil).all()
    assert np.isfinite(moisture).all()
    assert soil.shape[0] == 4 and moisture.shape[0] == 4


def test_regional_subset_is_refused_by_name(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame()
    frame = _Frame(
        frame.latitude, frame.longitude[:40], frame.vertical_kind,
        frame.vertical_values,
        {name: _Field(np.asarray(field.values)[..., :40])
         for name, field in frame.fields.items()},
    )
    with pytest.raises(ValueError, match="not the\n?.*globe|globe"):
        analysis_initial_state(cfg, transform, frame=frame)


def test_missing_required_field_is_refused_by_name(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame()
    del frame.fields["skin_temperature"]
    with pytest.raises(ValueError, match="skin_temperature"):
        analysis_initial_state(cfg, transform, frame=frame)


def test_non_pressure_vertical_kind_is_refused(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame()
    frame = _Frame(
        frame.latitude, frame.longitude, "hybrid_sigma_pressure",
        frame.vertical_values, frame.fields,
    )
    with pytest.raises(ValueError, match="pressure"):
        analysis_initial_state(cfg, transform, frame=frame)


def test_surface_pressure_moves_hypsometrically_with_terrain_smoothing(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    # Zonal wavenumber 4 is resolvable on the T3 Gaussian grid (nlon=8) but
    # beyond the T3 truncation, so the spectrally smoothed model terrain
    # must DROP the +-700 m alternation entirely and the surface pressure
    # must move hypsometrically against the dropped relief.
    frame = _synthetic_frame(terrain_zonal_wavenumber=4)
    state, phi_model, _prov = analysis_initial_state(cfg, transform, frame=frame)
    ps_model = np.exp(transform.backend.to_numpy(
        transform.inverse(state.atmosphere.log_surface_pressure)
    ))
    gauss_lat = transform.grid.latitude_deg
    gauss_lon = transform.grid.longitude_deg
    envelope = np.exp(-((gauss_lat - 30.0) / 15.0) ** 2)[:, None]
    terrain_src = 700.0 * (
        1.0 + np.cos(4.0 * np.deg2rad(gauss_lon))[None, :]
    ) * envelope
    ps_src = 101000.0 * np.exp(-terrain_src / 8000.0)
    delta_phi = transform.backend.to_numpy(phi_model) - 9.80665 * terrain_src
    strong = np.abs(delta_phi) > 0.5 * np.abs(delta_phi).max()
    assert np.abs(delta_phi).max() > 2000.0
    agreement = np.mean(
        np.sign(ps_model - ps_src)[strong] == -np.sign(delta_phi)[strong]
    )
    assert agreement > 0.9


def test_surface_pressure_reduction_uses_the_air_at_the_source_surface():
    # A plateau column: the analysis surface sits at 600 hPa, the profile
    # above it is a 6.5 K/km atmosphere, and the isobaric levels BELOW the
    # surface carry the analysis's own below-ground continuation - 28 K
    # warmer at 1000 hPa than the air at the surface (the audit's Tibetan
    # column: T1000 312.2 K vs Tsfc 284.3 K).
    levels = np.asarray([30000.0, 50000.0, 60000.0, 70000.0, 85000.0, 100000.0])
    ln_source = np.log(levels)
    t_surface = 284.3
    scale_height_m = DRY_AIR_GAS_CONSTANT * t_surface / 9.80665
    heights = -scale_height_m * np.log(levels / 60000.0)
    temperature = (t_surface - 0.0065 * heights)[:, None, None]
    humidity = np.clip(0.008 * (levels / 60000.0) ** 2, 0.0, 0.02)[:, None, None]
    ps_src = np.full((1, 1), 60000.0)
    virtual = surface_virtual_temperature(temperature, humidity, ln_source, ps_src)
    expected = t_surface * (1.0 + 0.608 * 0.008)
    assert virtual.shape == (1, 1)
    assert virtual[0, 0] == pytest.approx(expected, abs=1.0e-9)
    # The bottom source level is 27.9 K warmer: the value the reduction used
    # before the fix, over a layer that lies between 600 and ~800 hPa.
    bottom = temperature[-1, 0, 0] * (1.0 + 0.608 * humidity[-1, 0, 0])
    assert bottom - virtual[0, 0] > 25.0
    # Smoothing the plateau down 2000 m (the audit's dz) reduces ps by
    # 1780 Pa less with the surface air than with the underground level.
    dphi = 9.80665 * 2000.0
    ps_fix = 60000.0 * np.exp(dphi / (DRY_AIR_GAS_CONSTANT * virtual[0, 0]))
    ps_old = 60000.0 * np.exp(dphi / (DRY_AIR_GAS_CONSTANT * bottom))
    assert ps_fix - ps_old > 1500.0
    # A sea-level column below the bottom source level continues the
    # bottom layer's lapse rate rather than holding 1000 hPa's value.
    ps_low = np.full((1, 1), 101300.0)
    low = surface_virtual_temperature(temperature, humidity, ln_source, ps_low)
    slope = (temperature[-1, 0, 0] - temperature[-2, 0, 0]) / (ln_source[-1] - ln_source[-2])
    t_low = temperature[-1, 0, 0] + slope * (np.log(101300.0) - ln_source[-1])
    assert low[0, 0] == pytest.approx(t_low * (1.0 + 0.608 * humidity[-1, 0, 0]), abs=1.0e-9)


def test_analysis_state_reduces_surface_pressure_with_surface_air(tmp_path):
    # End to end through the pipeline: the receipted ps adjustment equals
    # the hypsometric reduction computed with the surface-interpolated Tv
    # on the regridded source fields, not with the bottom source level.
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame(terrain_zonal_wavenumber=4)
    # Make the below-ground levels hot, the way an isobaric analysis
    # continues under a plateau, so the two choices separate clearly.
    temperature = np.asarray(frame.fields["air_temperature"].values).copy()
    temperature[-1] += 25.0
    frame.fields["air_temperature"] = _Field(temperature)
    _state, phi_model, provenance = analysis_initial_state(cfg, transform, frame=frame)
    assert provenance["surface_pressure_reduction"].startswith("hypsometric-virtual-temperature-interpolated")
    regrid = _global_regridder(frame.latitude, frame.longitude, transform.grid)
    ps_src = regrid(frame.fields["surface_pressure"].values)
    phi_src = GRAVITY_M_S2 * regrid(frame.fields["terrain_height"].values)
    ln_source = np.log(np.asarray(frame.vertical_values))
    t_src = regrid(temperature)
    q_src = np.clip(regrid(frame.fields["specific_humidity"].values), 0.0, None)
    dphi = phi_src - transform.backend.to_numpy(phi_model)
    with_surface_air = ps_src * np.exp(
        dphi / (DRY_AIR_GAS_CONSTANT * surface_virtual_temperature(t_src, q_src, ln_source, ps_src))
    ) - ps_src
    with_bottom_level = ps_src * np.exp(
        dphi / (DRY_AIR_GAS_CONSTANT * t_src[-1] * (1.0 + 0.608 * q_src[-1]))
    ) - ps_src
    assert provenance["surface_pressure_adjustment_pa"]["max"] == pytest.approx(
        float(with_surface_air.max()), rel=1.0e-9
    )
    assert provenance["surface_pressure_adjustment_pa"]["min"] == pytest.approx(
        float(with_surface_air.min()), rel=1.0e-9
    )
    assert abs(float(with_surface_air.max()) - float(with_bottom_level.max())) > 50.0


def test_analysis_config_refuses_analytic_shape_keys(tmp_path):
    text = Path(CONFIG).read_text(encoding="utf-8").replace(
        "[initial]\n", "[initial]\nmode = \"analysis\"\nanalysis_grib = \"x.grib\"\n", 1
    )
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys in \\[initial\\]"):
        load_config(path)


def test_analysis_config_requires_grib(tmp_path):
    cfg_text = Path(CONFIG).read_text(encoding="utf-8")
    kept = [
        line for line in cfg_text.splitlines()
        if not any(line.strip().startswith(key) for key in (
            "surface_pressure_pa", "surface_temperature_k", "top_temperature_k",
            "qv_surface", "zonal_wind_m_s", "perturbation_amplitude",
            "zonal_wavenumber", "terrain_amplitude_m",
        ))
    ]
    text = "\n".join(kept).replace("[initial]", "[initial]\nmode = \"analysis\"")
    path = tmp_path / "bad.toml"
    path.write_text(text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="analysis_grib"):
        load_config(path)


def test_mapping_id_resolution_refuses_ambiguity():
    with pytest.raises(ValueError, match="exactly one"):
        resolve_analysis_mapping("gdas")
    assert resolve_analysis_mapping(MAPPING).name == MAPPING_NAME


def test_the_bare_id_and_the_file_name_resolve_to_one_mapping(
    tmp_path, monkeypatch
):
    """One mapping file, two spellings, both resolving from an install.

    Configurations used to name the authority by its checkout-relative path,
    which resolves to nothing inside an installed wheel -- there is no
    ``gpuwm/authorities/...`` under the working directory of a user who pip
    installed, and the resolver says so by name rather than substituting a
    file the caller did not ask for.  The two spellings that DO resolve are
    the bare id a config carries and the file name a reader module opens,
    and both go through one resolver, so they cannot answer differently.

    Both must name the same file, or the two would be different runs.  And
    because the node chains launch these configs from a checkout root, the
    bare id is exercised from there too, next to a directory whose name
    could shadow it.
    """

    packaged = resolve_analysis_mapping(MAPPING)
    by_name = resolve_analysis_mapping(MAPPING_NAME)
    assert packaged.resolve() == by_name.resolve()
    assert packaged.read_bytes() == by_name.read_bytes()
    # Packaged, not working-directory-relative: absolute, and under an
    # authorities directory that ships beside a module.
    assert packaged.is_absolute()
    assert packaged.parent.name == "authorities"

    checkout_root = Path(__file__).resolve().parents[1]
    for cwd in (checkout_root, tmp_path):
        monkeypatch.chdir(cwd)
        assert resolve_analysis_mapping(MAPPING).resolve() == packaged.resolve(), cwd

    # The checkout spelling is refused BY NAME, never substituted: a caller
    # who typed a path meant that file, and quietly answering with another is
    # how a run gets initialized from a source nobody named.
    with pytest.raises(FileNotFoundError):
        resolve_analysis_mapping("gpuwm/authorities/" + MAPPING_NAME)


def test_every_shipped_global_analysis_config_names_the_bare_id():
    """A path spelling in a shipped config is an install-time refusal."""

    for path in (
        str(_shipped_configs() / "arwen_global_t255_quickstart.toml"),
        str(_shipped_configs() / "arwen_global_gdas_t63_48h.toml"),
        str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml"),
        str(_shipped_configs() / "arwen_global_gdas_t533_24h.toml"),
    ):
        cfg = load_config(path)
        assert cfg.initial_mode == "analysis", path
        assert cfg.analysis_mapping == "gdas-global", path
        assert resolve_analysis_mapping(cfg.analysis_mapping).exists(), path
        # And the GRIB each names is relative, so the config is runnable
        # from a working directory the fetch door wrote into.
        assert not Path(cfg.analysis_grib).is_absolute(), path


# GI-4 (audit 2026-10-05): an analysis whose top lies below the model's top
# full level held the top value constant up into the lid, so the IFS
# open-data start (14 levels topping at 10 hPa) put 228.6 K at 1, 2, 5 and
# 10 hPa where the standard atmosphere has 270.7, 260 and 245 K.
_IFS_LEVELS_PA = np.asarray([
    1000.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 40000.0,
    50000.0, 60000.0, 70000.0, 85000.0, 92500.0, 100000.0,
])


def test_temperature_above_the_analysis_top_follows_the_standard_atmosphere():
    from arwen_global.analysis_initial import analysis_to_model_levels
    from arwen_global.vertical import standard_atmosphere_temperature_k

    standard = standard_atmosphere_temperature_k(_IFS_LEVELS_PA)
    # A column 6 K colder than the standard atmosphere everywhere (a winter
    # stratosphere): the departure at the analysis top is carried upward.
    source = (standard - 6.0)[:, None, None] * np.ones((1, 2, 3))
    targets = np.asarray([100.0, 200.0, 500.0, 2000.0, 50000.0])
    p_full = targets[:, None, None] * np.ones((1, 2, 3))
    remapped, record = analysis_to_model_levels(
        {"air_temperature": source, "specific_humidity": np.full_like(source, 3.0e-6),
         "eastward_wind": np.full_like(source, 12.0)},
        _IFS_LEVELS_PA, p_full,
    )
    expected = standard_atmosphere_temperature_k(targets[:3]) - 6.0
    np.testing.assert_allclose(remapped["air_temperature"][:3, 0, 0], expected, atol=1.0e-9)
    # Inside the analysis the remap is the ln p interpolation it always was.
    np.testing.assert_allclose(
        remapped["air_temperature"][4, 0, 0],
        np.interp(np.log(50000.0), np.log(_IFS_LEVELS_PA), source[:, 0, 0]), atol=1.0e-9)
    # 1 hPa is about 270 K in the standard atmosphere, not the 10 hPa value.
    assert remapped["air_temperature"][0, 0, 0] > 260.0
    # Vapour and wind are held at the analysis top value.
    np.testing.assert_array_equal(remapped["specific_humidity"][:3], 3.0e-6)
    np.testing.assert_array_equal(remapped["eastward_wind"][:3], 12.0)
    assert record["analysis_top_pa"] == 1000.0
    assert record["model_full_levels_above_analysis_top"] == 3
    assert "standard atmosphere" in record["temperature_above_top"]


def test_a_float32_column_below_the_analysis_top_keeps_its_bits(monkeypatch):
    """The GI-4 extension must not move a level it does not extend.  The
    remap took ln p in the model's own pressure dtype; casting a float32
    run's p_full to float64 before the log moved every level of the
    default float32 GDAS configs by up to about 1e-4 K."""
    import arwen_global.analysis_initial as analysis_initial
    from coldstart_oracle import numpy_to_model_levels

    monkeypatch.setattr(analysis_initial, "_to_model_levels", numpy_to_model_levels)
    levels = np.asarray([1000.0, 5000.0, 10000.0, 25000.0, 50000.0,
                         70000.0, 85000.0, 100000.0])
    rng = np.random.default_rng(7)
    source = (np.linspace(220.0, 290.0, levels.size)[:, None, None]
              + rng.normal(0.0, 3.0, (levels.size, 4, 5)))
    p_full = (np.geomspace(2000.0, 98000.0, 31)[:, None, None]
              * (1.0 + 0.01 * rng.random((1, 4, 5)))).astype(np.float32)
    remapped, record = analysis_initial.analysis_to_model_levels(
        {"air_temperature": source}, levels, p_full)
    assert record["model_full_levels_above_analysis_top"] == 0
    expected = numpy_to_model_levels(
        source, np.log(levels), np.log(p_full), extrapolate_below=True)
    np.testing.assert_array_equal(remapped["air_temperature"], expected)


def test_an_analysis_reaching_the_lid_records_no_extension():
    from arwen_global.analysis_initial import analysis_to_model_levels

    levels = np.asarray([50.0, 1000.0, 10000.0, 50000.0, 100000.0])
    source = np.linspace(260.0, 290.0, levels.size)[:, None, None] * np.ones((1, 1, 1))
    p_full = np.asarray([120.0, 5000.0, 90000.0])[:, None, None]
    remapped, record = analysis_to_model_levels(
        {"air_temperature": source}, levels, p_full)
    assert record["model_full_levels_above_analysis_top"] == 0
    assert np.all(np.isfinite(remapped["air_temperature"]))


def test_the_analysis_receipt_names_the_levels_above_the_analysis_top(tmp_path):
    from arwen_global.vertical import standard_atmosphere_temperature_k

    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    state, _phi, provenance = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame()
    )
    record = provenance["analysis_top"]
    # The synthetic frame tops at 100 hPa under a 1 hPa lid.
    assert record["analysis_top_pa"] == 10000.0
    assert record["model_full_levels_above_analysis_top"] > 0
    grid_theta = transform.backend.to_numpy(transform.inverse(state.atmosphere.theta))
    ps = np.exp(transform.backend.to_numpy(
        transform.inverse(state.atmosphere.log_surface_pressure)))
    p_full = transform.backend.to_numpy(
        cfg.vertical.pressure(ps, transform.backend)["p_full"])
    temperature = grid_theta * (p_full / 100000.0) ** (287.0 / 1004.0)
    # The top full level is warmer than the 100 hPa analysis by the
    # standard atmosphere's warming between them (spectral truncation of
    # theta costs a fraction of a kelvin here).
    top_level = p_full[0].mean()
    rise = standard_atmosphere_temperature_k(top_level) - standard_atmosphere_temperature_k(10000.0)
    assert rise > 5.0
    analysed_top = 220.0 + 70.0 * 0.1 * np.cos(np.deg2rad(transform.grid.latitude_deg))[:, None] ** 2
    np.testing.assert_allclose(
        temperature[0], np.broadcast_to(analysed_top + rise, temperature[0].shape), atol=1.5)


# GI-7 (audit 2026-10-05): the IFS open-data soil layers are 0-7, 7-28,
# 28-100 and 100-289 cm; Noah's are 0-10, 10-40, 40-100 and 100-200 cm.
_IFS_MAPPING_FILE = (
    Path(__file__).resolve().parents[1] / "src" / "arwen_global" / "data"
    / "authorities" / "rw-wps-ecmwf-open-data-global-forecast-grib2.mapping.json"
)


def _layered_soil_frame():
    frame = _synthetic_frame()
    land = np.asarray(frame.fields["land_fraction"].values) > 0.5
    layer_t = np.asarray([280.0, 285.0, 290.0, 295.0])[:, None, None]
    layer_m = np.asarray([0.10, 0.20, 0.30, 0.40])[:, None, None]
    frame.fields["soil_temperature"] = _Field(np.where(land[None], layer_t, np.nan))
    frame.fields["volumetric_soil_moisture"] = _Field(np.where(land[None], layer_m, np.nan))
    return frame


def _cfg_with_mapping(tmp_path, mapping):
    cfg = _analysis_cfg(tmp_path)
    import dataclasses

    return dataclasses.replace(cfg, analysis_mapping=str(mapping))


def test_ifs_soil_layers_reach_noah_by_depth_not_by_position(tmp_path):
    cfg = _cfg_with_mapping(tmp_path, _IFS_MAPPING_FILE)
    transform = build_transform(cfg)
    state, _phi, provenance = analysis_initial_state(
        cfg, transform, frame=_layered_soil_frame())
    host = transform.backend.to_numpy
    land = host(state.surface.land_fraction) >= 1.0
    assert land.any()
    soil_t = host(state.surface.soil_temperature_k)[:, land]
    soil_m = host(state.surface.soil_water_fraction)[:, land]
    # Noah 0-10 cm: 7 cm of the 0-7 layer and 3 cm of the 7-28 layer;
    # 10-40 cm: 18 cm of 7-28 and 12 cm of 28-100; 40-100 cm inside 28-100;
    # 100-200 cm inside 100-289.
    expected_t = [0.7 * 280 + 0.3 * 285, (18 * 285 + 12 * 290) / 30, 290.0, 295.0]
    expected_m = [0.7 * 0.1 + 0.3 * 0.2, (18 * 0.2 + 12 * 0.3) / 30, 0.30, 0.40]
    for k in range(4):
        np.testing.assert_allclose(soil_t[k], expected_t[k], atol=1.0e-6)
        np.testing.assert_allclose(soil_m[k], expected_m[k], atol=1.0e-9)
    record = provenance["soil_layers"]["soil_temperature"]
    assert record["remapped"] is True
    assert record["source_bounds_m"] == [[0.0, 0.07], [0.07, 0.28], [0.28, 1.0], [1.0, 2.89]]
    assert record["model_bounds_m"] == [[0.0, 0.1], [0.1, 0.4], [0.4, 1.0], [1.0, 2.0]]


def test_gdas_soil_layers_match_noah_and_pass_unchanged(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    state, _phi, provenance = analysis_initial_state(
        cfg, transform, frame=_layered_soil_frame())
    host = transform.backend.to_numpy
    land = host(state.surface.land_fraction) >= 1.0
    soil_t = host(state.surface.soil_temperature_k)[:, land]
    for k, value in enumerate([280.0, 285.0, 290.0, 295.0]):
        np.testing.assert_allclose(soil_t[k], value, atol=1.0e-9)
    assert provenance["soil_layers"]["soil_temperature"]["remapped"] is False


def test_soil_layer_remap_keeps_column_water_and_refuses_an_unknown_index():
    from arwen_global.analysis_initial import (
        noah_soil_layer_bounds_m, remap_soil_layers, soil_layer_bounds_m,
    )

    gdas = resolve_analysis_mapping(MAPPING)
    assert soil_layer_bounds_m(gdas) == noah_soil_layer_bounds_m()
    ifs = soil_layer_bounds_m(_IFS_MAPPING_FILE, "volumetric_soil_moisture")
    rng = np.random.default_rng(7)
    stack = rng.uniform(0.05, 0.45, size=(4, 3, 5))
    out, record = remap_soil_layers(stack, ifs, noah_soil_layer_bounds_m())
    # Water in the top 2 m is kept: the source's 0-2 m water equals Noah's.
    source_dz = np.asarray([0.07, 0.21, 0.72, 1.0])
    noah_dz = np.asarray([0.1, 0.3, 0.6, 1.0])
    np.testing.assert_allclose(
        np.tensordot(source_dz, stack, axes=1), np.tensordot(noah_dz, out, axes=1),
        rtol=1.0e-12)
    np.testing.assert_allclose(np.sum(record["weights"], axis=1), 1.0)


def test_a_soil_level_index_without_a_depth_row_is_refused(tmp_path):
    import json

    from arwen_global.analysis_initial import soil_layer_bounds_m

    document = json.loads(_IFS_MAPPING_FILE.read_text(encoding="utf-8"))
    document["name"] = "a-source-with-no-depth-row"
    path = tmp_path / "unknown.mapping.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="by position"):
        soil_layer_bounds_m(path)


@pytest.mark.parametrize("change", ["other_type", "mixed_types"])
def test_a_soil_selector_without_a_depth_names_the_breakage(tmp_path, change):
    """Every soil-depth refusal names what it prevents: layers handed to
    Noah by position (the refusal law: a refusal names its breakage)."""
    import json

    from arwen_global.analysis_initial import soil_layer_bounds_m

    document = json.loads(_IFS_MAPPING_FILE.read_text(encoding="utf-8"))
    selector = document["fields"]["soil_temperature"]["selectors"][0]
    if change == "other_type":
        selector["level_type"] = selector["second_level_type"] = 1
    else:
        selector["second_level_type"] = 106
    path = tmp_path / "changed.mapping.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="by position"):
        soil_layer_bounds_m(path)
