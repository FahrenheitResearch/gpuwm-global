"""Other worlds on ArWen Global.

The ``[world]`` table of a run config is parsed into one immutable
:class:`WorldSpec` (``arwen_global.world.spec``), carried on the run config
as ``ArwenGlobalConfig.world``.  Every part of the worlds track reads a
world through that object; with ``world.is_earth`` True no world code may
run, and a run is bit-identical to a run whose config has no ``[world]``
table.
"""
from __future__ import annotations

from .spec import (
    AIR_MASS_RULES,
    DATED,
    EARTH,
    EARTH_IDENTITY,
    LAND_COVERS,
    LAUNCHER_RESERVED_KEYS,
    NOT_SUPPORTED_YET,
    ORBITS,
    SEA_ICE_RULES,
    SEA_LAND_RULES,
    SEA_TEMPERATURE_PROFILES,
    SEA_TEMPERATURE_SOURCES,
    WORLD_KEYS,
    WORLD_SCHEMA,
    Air,
    Sea,
    Sun,
    Surface,
    Terrain,
    TerrainBox,
    WorldSpec,
    parse_world,
    world_identity_of,
)

__all__ = [
    "AIR_MASS_RULES", "DATED", "EARTH", "EARTH_IDENTITY", "LAND_COVERS",
    "LAUNCHER_RESERVED_KEYS",
    "NOT_SUPPORTED_YET", "ORBITS", "SEA_ICE_RULES",
    "SEA_LAND_RULES", "SEA_TEMPERATURE_PROFILES", "SEA_TEMPERATURE_SOURCES",
    "WORLD_KEYS", "WORLD_SCHEMA", "Air", "Sea", "Sun", "Surface", "Terrain",
    "TerrainBox", "WorldSpec", "parse_world", "world_identity_of",
]
