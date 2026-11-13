"""Refusals downstream: the doors that must never run on another world.

A world is a forecast of something that is not Earth.  Three families of
door downstream of the forecast assume Earth, and on a world they would
run without complaint and hand back a wrong answer with nothing saying so
(other-worlds plan, sections 2.3 and 4.11):

``regional``
    The regional parent export.  The regional engine has no ``[world]``
    table: it takes terrain, coast and land use from Earth's geography and
    runs Earth's sun, rotation and greenhouse gases, so a regional run fed
    by a world's parent would put the world's weather on modern Earth.
``assimilation``
    Cycling and assimilation.  An analysis draws the state toward
    observations of Earth; every cycle would pull the world back toward
    Earth.
``scorecard``
    The scorecards.  They score a forecast of Earth against Earth's
    observations or analyses; on a world every number would measure how
    far the world is from Earth, not how good the forecast is.

Each door calls :func:`refuse_on_world` with what it holds -- the config's
:class:`~arwen_global.world.spec.WorldSpec`, a run receipt, or a
checkpoint's metadata -- and is refused by name with
:class:`WorldRefusedError` when that names any world but Earth.  An Earth
run (no ``[world]`` table, or an all-Earth one) carries no world record, so
for Earth nothing here runs past one comparison and no door's output
changes by a byte.

Where the world is read from a stored payload (the scorecards read a run
receipt; the export and the cycle read checkpoint headers), the record is
the one the run writes (``WorldSpec.record()``) under the key ``"world"``:
in the receipt at the top level and in a checkpoint's physics metadata.
A receipt written before that record existed still carries the world's
settings in its config identity (``config_identity["world"]``, present
only for a world that is not Earth), and that is read too.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from .spec import (
    Air,
    EARTH_IDENTITY,
    Sea,
    Sun,
    Surface,
    Terrain,
    WorldSpec,
    world_identity_of,
)

#: The key a run receipt and a checkpoint's physics metadata carry a
#: world's record under (the same name the receipt and the checkpoint
#: writers use).
WORLD_RECORD_KEY = "world"

#: Why each family of door is refused on a world.  The texts are part of
#: the contract: the tests assert them word for word.
WHY: dict[str, str] = {
    "regional": (
        "the regional engine has no [world] table: it takes terrain, coast "
        "and land use from Earth's geography and runs Earth's sun, rotation "
        "and greenhouse gases, so a regional run from this parent would put "
        "this world's weather on modern Earth with nothing saying so.  A "
        "world's own geography for the regional engine is not built yet;"
        " until then no world has a regional zoom"
    ),
    "assimilation": (
        "assimilation draws the state toward observations of Earth, measured "
        "over Earth's terrain, coast, sun and air, so every analysis would "
        "pull this world back toward Earth and its departures would measure "
        "the world's distance from Earth, not the analysis.  Cycling and "
        "assimilation stay refused on every world"
    ),
    "scorecard": (
        "a scorecard scores a forecast of Earth against Earth's observations "
        "or analyses, and this run is not a forecast of Earth, so every "
        "score would measure how far the world is from Earth, not how good "
        "the forecast is.  Scorecards stay refused on every world; compare "
        "a world with its own control run instead"
    ),
}


class WorldRefusedError(ValueError):
    """A door that assumes Earth was asked to run on another world."""


def stored_world_record(payload) -> dict | None:
    """The world a stored payload says it belongs to, as a record
    ``{"label", "identity", "settings"}``, or None for Earth.

    ``payload`` is a run receipt, a checkpoint's whole metadata, a
    checkpoint's physics metadata, or None.  A payload with no world record
    (every receipt and checkpoint of an Earth run, and every one written
    before worlds existed) reads as Earth.  A receipt with no record but a
    ``config`` identity that holds a world's settings is read from those."""
    if not isinstance(payload, dict):
        return None
    for holder in (payload, payload.get("physics_metadata")):
        if not isinstance(holder, dict):
            continue
        record = holder.get(WORLD_RECORD_KEY)
        if isinstance(record, dict):
            if world_identity_of(record) == EARTH_IDENTITY:
                return None
            return record
    config = payload.get("config")
    if isinstance(config, dict) and isinstance(config.get(WORLD_RECORD_KEY), dict):
        settings = config[WORLD_RECORD_KEY]
        raw = json.dumps(settings, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        return {"label": None, "identity": hashlib.sha256(raw).hexdigest(),
                "settings": settings}
    return None


def _changed_from_settings(settings) -> list[str]:
    """The dotted settings that differ from Earth's in a stored identity."""
    if not isinstance(settings, dict):
        return []
    earth = {"terrain": asdict(Terrain()), "sun": asdict(Sun()),
             "sea": asdict(Sea()), "air": asdict(Air()),
             "surface": asdict(Surface())}
    out: list[str] = []
    for part, defaults in earth.items():
        values = settings.get(part)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key in defaults and value != defaults[key]:
                out.append(f"{part}.{key}")
    if "terrain.air_mass" in out and "terrain.scale" not in out \
            and "terrain.cap_m" not in out and "surface.folder" not in out:
        out.remove("terrain.air_mass")
    return out


def describe_world(subject) -> str | None:
    """One phrase naming the world ``subject`` holds, or None for Earth.

    ``subject`` is a :class:`WorldSpec`, a stored payload
    (:func:`stored_world_record`), or None."""
    if subject is None:
        return None
    if isinstance(subject, WorldSpec):
        if subject.is_earth:
            return None
        label, token = subject.label, subject.token
        changed = subject.changed_settings()
    else:
        record = stored_world_record(subject)
        if record is None:
            return None
        label = record.get("label")
        token = world_identity_of(record)[:12]
        changed = _changed_from_settings(record.get("settings"))
    name = f"world {label!r}" if label else "an unlabelled world"
    text = f"{name} (world identity {token}"
    if changed:
        text += ", changing " + ", ".join(changed)
    return text + ")"


def world_refusal(subject, *, door: str, family: str, source: str) -> str | None:
    """The refusal text for running ``door`` on what ``subject`` holds, or
    None when it holds Earth.  ``family`` is a key of :data:`WHY`;
    ``source`` names where the world was read (a config, a receipt, a
    checkpoint) so the operator knows which input to change."""
    if family not in WHY:
        raise ValueError(f"unknown refusal family {family!r}; one of {sorted(WHY)}")
    world = describe_world(subject)
    if world is None:
        return None
    return f"{door} is refused on {world}, read from {source}: {WHY[family]}"


def refuse_on_world(subject, *, door: str, family: str, source: str) -> None:
    """Raise :class:`WorldRefusedError` when ``subject`` names a world that
    is not Earth (see :func:`world_refusal`); return quietly for Earth."""
    reason = world_refusal(subject, door=door, family=family, source=source)
    if reason is not None:
        raise WorldRefusedError(reason)


__all__ = [
    "WHY",
    "WORLD_RECORD_KEY",
    "WorldRefusedError",
    "describe_world",
    "refuse_on_world",
    "stored_world_record",
    "world_refusal",
]
