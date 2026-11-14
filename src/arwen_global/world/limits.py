"""The limits of the model as data, and the label each world earns.

A world is measured against the model's limits (the radiation's 355 K skin
range, the 115,000 Pa pressure ceiling, the 450 hPa terrain floor, the
water store every column pays its evaporation from, ...), against Earth's
own figures measured with this model, and against the rules of worlds plan
section 2 that give it one of three labels: ``science``, ``illustration``
or ``not supported yet`` (``earth`` for Earth itself).  All of that is held
in ``arwen_global/data/world-limits.json``, so a measured bound changes a
row and not code (plan section 4.8).

The same file carries fifty-two **fixture worlds**, each with the label
and the exact set of rules it must earn: Mars is refused by three named
rules, eight times pre-industrial CO2 is labelled unvalidated, a world with
no mountains is science.  :func:`check_fixtures` runs them all.

What is classified is a **request**: the ``[world]`` table as a user would
write it (a nested dict, or a parsed :class:`~.spec.WorldSpec`), which may
also ask for things the table refuses (``gravity_m_s2``, ``radius_km``,
``year_length_d``, ``air.composition``, ``air.surface_pressure_pa``, another
star, a tidally locked planet, a responding ocean ...), plus two optional
sub-tables the table never holds:

- ``run``: what the run is asked to do (``regional_zoom``, ``assimilation``,
  ``climate``);
- ``facts``: numbers known about this world's run, from the preflight or
  the run plan (``run_days``, ``surface_pressure_min_pa``,
  ``surface_pressure_max_pa``, ``skin_temperature_max_k``,
  ``air_temperature_min_k``, ``ocean_store_days``).

Every key a request may hold is listed in the data file; any other key is
refused, so a misspelt fixture cannot pass by matching nothing.

The contract part of a request (the keys :mod:`.spec` accepts) is also run
through :func:`~.spec.parse_world`.  The parser's verdict is kept on the
result; when the parser refuses a request and no rule of the table does,
the request is ``not supported yet`` under the rule ``table_range`` with the
parser's own sentence, so the labels can never call "runs" what the table
refuses to parse.

A label can only get worse during a run, never better: :func:`worse` is
the one comparison every caller uses.

Nothing here runs for a run whose world is Earth: the model never calls
this module when ``cfg.world.is_earth``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import json
import math
from typing import Any, Mapping

from .spec import WORLD_KEYS, WorldSpec, parse_world

#: The data file's schema.
LIMITS_SCHEMA = "arwen-global.world-limits/v1"
#: The data file, under ``arwen_global/data``.
LIMITS_FILE = "world-limits.json"

EARTH_CLASS = "earth"
SCIENCE = "science"
ILLUSTRATION = "illustration"
NOT_SUPPORTED_YET = "not supported yet"
#: Best first, worst last: a world takes the worst label of its rules.
LABEL_ORDER = (EARTH_CLASS, SCIENCE, ILLUSTRATION, NOT_SUPPORTED_YET)

#: The rule a request earns when the [world] table refuses it and no row of
#: the data file does.
TABLE_RANGE_RULE = "table_range"

#: Keys computed from the request, which a rule may read.
DERIVED_KEYS = frozenset({
    "world.is_earth", "world.changes_coast", "air.co2_multiple",
    "sun.rotation_factor",
})

_OPS = ("==", "!=", "<", "<=", ">", ">=", "present", "absent")
_COMPARE_FROM = ("value", "limit", "measured", "ref")
#: A request key carrying one of these values asks for nothing new.
_EARTH_REQUEST_VALUES = (None, False, "sun")


# -- the rows ---------------------------------------------------------------


@dataclass(frozen=True)
class Limit:
    """One limit of the model a world is measured against.  The fields are
    those of the preflight's own ``Limit`` (package 1.6)."""

    value: float
    unit: str
    name: str
    source: str
    #: A margin smaller than this (and not negative) prints as "close".
    caution: float = 0.0


@dataclass(frozen=True)
class Measured:
    """One figure measured on Earth with this model."""

    value: float
    unit: str
    source: str


@dataclass(frozen=True)
class Rule:
    """One row of plan section 2: when it fires, the label it gives and the
    limit sentence that goes with it."""

    id: str
    label: str
    setting: str
    limit: str
    source: str
    when: Mapping[str, Any]

    def keys(self) -> frozenset[str]:
        """Every request key this rule reads."""
        return frozenset(_condition_keys(self.when))


@dataclass(frozen=True)
class Fixture:
    id: str
    request: Mapping[str, Any]
    label: str
    rules: tuple[str, ...]


@dataclass(frozen=True)
class Hit:
    """One rule a world earned."""

    id: str
    label: str
    setting: str
    limit: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label, "setting": self.setting,
                "limit": self.limit}


@dataclass(frozen=True)
class Classification:
    """A world's label and every rule that gave it."""

    label: str
    hits: tuple[Hit, ...]
    #: The [world] table parser's refusal of the request's contract part,
    #: or None when it parses.
    contract_error: str | None

    @property
    def rules(self) -> tuple[str, ...]:
        return tuple(hit.id for hit in self.hits)

    @property
    def refusals(self) -> tuple[Hit, ...]:
        """The rules that make the world ``not supported yet``."""
        return tuple(hit for hit in self.hits if hit.label == NOT_SUPPORTED_YET)

    @property
    def refused(self) -> bool:
        return self.label == NOT_SUPPORTED_YET

    def record(self) -> dict[str, object]:
        return {"label": self.label,
                "rules": [hit.as_dict() for hit in self.hits],
                "contract_error": self.contract_error}


@dataclass(frozen=True)
class LimitsTable:
    """The parsed data file."""

    schema: str
    labels: tuple[str, ...]
    result_sentence: str
    limits: Mapping[str, Limit]
    measured_earth: Mapping[str, Measured]
    request_keys: Mapping[str, str]
    fact_keys: Mapping[str, str]
    fallback: Hit
    rules: tuple[Rule, ...]
    fixtures: tuple[Fixture, ...]

    def rule(self, rule_id: str) -> Rule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)

    def known_keys(self) -> frozenset[str]:
        """Every dotted key a request or a rule may name."""
        return (contract_keys() | frozenset(self.request_keys)
                | frozenset(self.fact_keys) | DERIVED_KEYS)


# -- loading ------------------------------------------------------------------


def contract_keys() -> frozenset[str]:
    """Every dotted key the [world] table accepts (``label`` aside)."""
    out: set[str] = set()
    for path, keys in WORLD_KEYS.items():
        for key in keys:
            dotted = f"{path}.{key}" if path else key
            if dotted in WORLD_KEYS or dotted == "label":
                continue
            out.add(dotted)
    return frozenset(out)


def _condition_keys(cond: Mapping[str, Any]):
    for group in ("all", "any"):
        if group in cond:
            for item in cond[group]:
                yield from _condition_keys(item)
            return
    if "not" in cond:
        yield from _condition_keys(cond["not"])
        return
    yield cond["key"]
    if "ref" in cond:
        yield cond["ref"]


def _check_condition(cond, where: str, table_keys, limits, measured) -> None:
    if not isinstance(cond, dict):
        raise ValueError(f"{where}: a condition must be an object")
    groups = [g for g in ("all", "any", "not") if g in cond]
    if groups:
        if len(cond) != 1:
            raise ValueError(f"{where}: {groups[0]!r} must stand alone")
        if groups[0] == "not":
            _check_condition(cond["not"], where, table_keys, limits, measured)
            return
        items = cond[groups[0]]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{where}: {groups[0]!r} needs a non-empty list")
        for item in items:
            _check_condition(item, where, table_keys, limits, measured)
        return
    if cond.get("key") not in table_keys:
        raise ValueError(f"{where}: unknown key {cond.get('key')!r}")
    op = cond.get("op")
    if op not in _OPS:
        raise ValueError(f"{where}: unknown op {op!r}")
    sources = [s for s in _COMPARE_FROM if s in cond]
    extra = set(cond) - {"key", "op", *_COMPARE_FROM}
    if extra:
        raise ValueError(f"{where}: unknown fields {sorted(extra)}")
    if op in ("present", "absent"):
        if sources:
            raise ValueError(f"{where}: {op!r} compares with nothing")
        return
    if len(sources) != 1:
        raise ValueError(f"{where}: {op!r} needs exactly one of {list(_COMPARE_FROM)}")
    if "limit" in cond and cond["limit"] not in limits:
        raise ValueError(f"{where}: unknown limit {cond['limit']!r}")
    if "measured" in cond and cond["measured"] not in measured:
        raise ValueError(f"{where}: unknown measured figure {cond['measured']!r}")
    if "ref" in cond and cond["ref"] not in table_keys:
        raise ValueError(f"{where}: unknown ref {cond['ref']!r}")


def parse_table(raw: Mapping[str, Any]) -> LimitsTable:
    """Parse and check the data file's content.  Every rule must name known
    keys, limits and labels, and every fixture must name known rules."""
    if raw.get("schema") != LIMITS_SCHEMA:
        raise ValueError(f"world limits: schema must be {LIMITS_SCHEMA!r}, "
                         f"got {raw.get('schema')!r}")
    labels = tuple(raw["labels"])
    if labels != LABEL_ORDER:
        raise ValueError(f"world limits: labels must be {list(LABEL_ORDER)} "
                         "in that order (best first)")
    limit_fields = {f.name for f in fields(Limit)}
    limits: dict[str, Limit] = {}
    for key, row in raw["limits"].items():
        if set(row) - limit_fields:
            raise ValueError(f"world limits: limit {key!r} has unknown fields")
        limit = Limit(**row)
        if not math.isfinite(float(limit.value)):
            raise ValueError(f"world limits: limit {key!r} is not finite")
        limits[key] = limit
    measured: dict[str, Measured] = {}
    for key, row in raw["measured_earth"].items():
        measured[key] = Measured(**row)
    request_keys = dict(raw["request_keys"])
    fact_keys = dict(raw["fact_keys"])
    overlap = set(request_keys) & contract_keys()
    if overlap:
        raise ValueError(f"world limits: request keys {sorted(overlap)} are "
                         "already [world] table keys")
    for key in fact_keys:
        if not key.startswith("facts."):
            raise ValueError(f"world limits: fact key {key!r} must start with 'facts.'")
    for key in request_keys:
        if key.startswith("facts."):
            raise ValueError(f"world limits: request key {key!r} is a fact key")
    table_keys = (contract_keys() | frozenset(request_keys)
                  | frozenset(fact_keys) | DERIVED_KEYS)
    rules: list[Rule] = []
    seen: set[str] = {TABLE_RANGE_RULE}
    fb = raw["fallback"]
    fallback = Hit(fb["id"], fb["label"], fb["setting"], fb["limit"])
    if fallback.label not in labels or fallback.id in seen:
        raise ValueError("world limits: bad fallback row")
    seen.add(fallback.id)
    for row in raw["rules"]:
        rule = Rule(id=row["id"], label=row["label"], setting=row["setting"],
                    limit=row["limit"], source=row["source"], when=row["when"])
        if rule.id in seen:
            raise ValueError(f"world limits: rule id {rule.id!r} is not unique")
        seen.add(rule.id)
        if rule.label not in labels or rule.label == EARTH_CLASS:
            raise ValueError(f"world limits: rule {rule.id!r} has label {rule.label!r}")
        _check_condition(rule.when, f"world limits: rule {rule.id!r}",
                         table_keys, limits, measured)
        rules.append(rule)
    fixtures: list[Fixture] = []
    fixture_ids: set[str] = set()
    for row in raw["fixtures"]:
        fixture = Fixture(id=row["id"], request=row["request"], label=row["label"],
                          rules=tuple(row["rules"]))
        if fixture.id in fixture_ids:
            raise ValueError(f"world limits: fixture id {fixture.id!r} is not unique")
        fixture_ids.add(fixture.id)
        if fixture.label not in labels:
            raise ValueError(f"world limits: fixture {fixture.id!r} label {fixture.label!r}")
        unknown = set(fixture.rules) - seen
        if unknown:
            raise ValueError(f"world limits: fixture {fixture.id!r} names unknown "
                             f"rules {sorted(unknown)}")
        fixtures.append(fixture)
    return LimitsTable(
        schema=raw["schema"], labels=labels,
        result_sentence=raw["result_sentence"],
        limits=limits, measured_earth=measured,
        request_keys=request_keys, fact_keys=fact_keys, fallback=fallback,
        rules=tuple(rules), fixtures=tuple(fixtures))


@lru_cache(maxsize=1)
def load_table() -> LimitsTable:
    """The shipped data file, parsed and checked once per process."""
    from importlib.resources import files

    text = files("arwen_global").joinpath("data", LIMITS_FILE).read_text(encoding="utf-8")
    return parse_table(json.loads(text))


def limits() -> Mapping[str, Limit]:
    """The model's limits, by key (the preflight's ``LIMITS``)."""
    return load_table().limits


def measured_earth() -> Mapping[str, Measured]:
    """Earth's own figures measured with this model (the preflight's
    ``MEASURED_EARTH``)."""
    return load_table().measured_earth


# -- requests -------------------------------------------------------------------


def _flatten(table: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in table.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(value, dotted))
        else:
            out[dotted] = value
    return out


def _earth_defaults() -> dict[str, Any]:
    return _spec_values(WorldSpec())


def _spec_values(world: WorldSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in ("terrain", "sun", "sea", "air", "surface"):
        value = getattr(world, part)
        for item in fields(value):
            if part == "terrain" and item.name == "box":
                box = value.box
                for sub in ("south_deg", "north_deg", "west_deg", "east_deg", "taper_deg"):
                    out[f"terrain.box.{sub}"] = None if box is None else getattr(box, sub)
                continue
            out[f"{part}.{item.name}"] = getattr(value, item.name)
    return out


def _unflatten(flat: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dotted, value in flat.items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def request_values(request: WorldSpec | Mapping[str, Any] | None,
                   table: LimitsTable | None = None) -> tuple[dict[str, Any], str | None]:
    """The flat key-to-value view a rule reads, and the [world] table
    parser's refusal of the request's contract part (None when it parses).

    Contract keys the request leaves out take Earth's value; request and
    fact keys it leaves out are absent (None)."""
    table = table or load_table()
    defaults = _earth_defaults()
    if request is None:
        request = {}
    if isinstance(request, WorldSpec):
        values = dict(defaults)
        values.update(_spec_values(request))
        contract_error = None
        label = request.label
        given: dict[str, Any] = {}
    else:
        if not isinstance(request, Mapping):
            raise TypeError(f"not a world request: {type(request).__name__}")
        flat = _flatten(request)
        label = flat.pop("label", None)
        known = table.known_keys() - DERIVED_KEYS
        unknown = sorted(set(flat) - known)
        if unknown:
            raise ValueError(
                f"unknown keys in a world request: {', '.join(unknown)}; the "
                f"[world] table's keys, the request keys and the fact keys of "
                f"{LIMITS_FILE} are accepted")
        values = dict(defaults)
        given = {k: v for k, v in flat.items() if k not in contract_keys()}
        values.update(flat)
        contract = {k: v for k, v in flat.items() if k in contract_keys()}
        contract_table = _unflatten(contract)
        if label is not None:
            contract_table["label"] = label
        try:
            parsed = parse_world(contract_table)
        except ValueError as exc:
            contract_error = str(exc)
            if (contract and "label" not in contract_table
                    and "world.label is required" in contract_error):
                # A label is a name, not a setting: check the settings.
                try:
                    parse_world(dict(contract_table, label="unnamed"))
                    contract_error = None
                except ValueError as again:
                    contract_error = str(again)
        else:
            contract_error = None
            # The parser's canonical values (e.g. a perpetual equinox has
            # no tilt) are what the model runs.
            values.update(_spec_values(parsed))
    for key in table.request_keys:
        values.setdefault(key, None)
    for key in table.fact_keys:
        values.setdefault(key, None)
    # Derived.
    contract_earth = all(
        _same(values[k], defaults[k]) for k in defaults
        if k != "terrain.air_mass")
    asks_more = any(
        v not in _EARTH_REQUEST_VALUES for k, v in given.items()
        if k in table.request_keys and not k.startswith("run."))
    values["world.is_earth"] = contract_earth and not asks_more
    level = values["sea.level_m"]
    values["world.changes_coast"] = bool(
        (_is_number(level) and level != 0.0) or values["sea.land"] != "keep"
        or values["surface.folder"] is not None)
    co2 = values["air.co2_ppm"]
    preindustrial = table.limits["co2_preindustrial_ppm"].value
    values["air.co2_multiple"] = co2 / preindustrial if _is_number(co2) else None
    day = values["sun.day_length_h"]
    values["sun.rotation_factor"] = 24.0 / day if _is_number(day) and day > 0 else None
    return values, contract_error


def _is_number(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _same(a, b) -> bool:
    if _is_number(a) and _is_number(b):
        return float(a) == float(b)
    return a == b


# -- the rules --------------------------------------------------------------------


def _operand(cond, values, table: LimitsTable):
    if "value" in cond:
        return cond["value"]
    if "limit" in cond:
        return table.limits[cond["limit"]].value
    if "measured" in cond:
        return table.measured_earth[cond["measured"]].value
    return values.get(cond["ref"])


def _holds(cond, values, table: LimitsTable) -> bool:
    if "all" in cond:
        return all(_holds(c, values, table) for c in cond["all"])
    if "any" in cond:
        return any(_holds(c, values, table) for c in cond["any"])
    if "not" in cond:
        return not _holds(cond["not"], values, table)
    have = values.get(cond["key"])
    op = cond["op"]
    if op == "present":
        return have is not None
    if op == "absent":
        return have is None
    want = _operand(cond, values, table)
    if have is None or want is None:
        return False
    if op in ("==", "!="):
        # True is not 1.0 here: a switch and a number never compare equal.
        equal = (_same(have, want)
                 and isinstance(have, bool) == isinstance(want, bool))
        return equal if op == "==" else not equal
    # An ordering holds only between two numbers: "dated" is not less than
    # anything.
    if not (_is_number(have) and _is_number(want)):
        return False
    have, want = float(have), float(want)
    return {"<": have < want, "<=": have <= want,
            ">": have > want, ">=": have >= want}[op]


def worse(a: str, b: str) -> str:
    """The worse of two labels.  A label can only get worse during a run:
    every update of a run's label goes through this."""
    for label in (a, b):
        if label not in LABEL_ORDER:
            raise ValueError(f"not a world label: {label!r}")
    return max(a, b, key=LABEL_ORDER.index)


def classify(request: WorldSpec | Mapping[str, Any] | None,
             table: LimitsTable | None = None) -> Classification:
    """The label a world earns and every rule that gave it, in table order.

    ``request`` is a parsed :class:`~.spec.WorldSpec`, or a nested dict as
    a user would write the ``[world]`` table, optionally with request keys
    the table refuses and the ``run`` and ``facts`` sub-tables (see the
    module docstring).  ``None`` is Earth."""
    table = table or load_table()
    values, contract_error = request_values(request, table)
    hits = [Hit(rule.id, rule.label, rule.setting, rule.limit)
            for rule in table.rules if _holds(rule.when, values, table)]
    if contract_error is not None and not any(
            hit.label == NOT_SUPPORTED_YET for hit in hits):
        hits.append(Hit(TABLE_RANGE_RULE, NOT_SUPPORTED_YET, "world", contract_error))
    if not hits and not values["world.is_earth"]:
        hits.append(table.fallback)
    label = EARTH_CLASS
    for hit in hits:
        label = worse(label, hit.label)
    return Classification(label=label, hits=tuple(hits), contract_error=contract_error)


def label_of(request: WorldSpec | Mapping[str, Any] | None) -> str:
    """Just the label."""
    return classify(request).label


@dataclass(frozen=True)
class FixtureResult:
    fixture: Fixture
    got: Classification

    @property
    def ok(self) -> bool:
        return (self.got.label == self.fixture.label
                and sorted(self.got.rules) == sorted(self.fixture.rules))

    def describe(self) -> str:
        return (f"{self.fixture.id}: expected {self.fixture.label!r} by "
                f"{sorted(self.fixture.rules)}, got {self.got.label!r} by "
                f"{sorted(self.got.rules)}")


def check_fixtures(table: LimitsTable | None = None) -> tuple[FixtureResult, ...]:
    """Classify every fixture world of the data file against what it
    expects.  A failed fixture's ``describe()`` says how it differs."""
    table = table or load_table()
    return tuple(FixtureResult(f, classify(f.request, table)) for f in table.fixtures)
