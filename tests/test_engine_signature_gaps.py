"""A name that resolves can still be a different callable.

`tests/test_engine_compat.py` holds the seam for symbols the engine does not
carry.  This file holds the other half of the same boundary: engine callables
that DO resolve and do not accept an argument this package hands them.

Why it needed its own instrument.  The carve's boundary table asked, symbol by
symbol, whether the engine had the name.  Eighty-two of eighty-six resolved and
the table called the rest of the boundary clean.  Then
`tools/measure_engine_signatures.py` compared the parsed signatures of the two
engine trees and found eighteen callables that differ, eight of which do not
take an argument this package passes.

ALL EIGHT ROWS WERE RETIRED ON 2026-09-09 AND 2026-09-10, and that is the
healthy outcome for a gate.  Seven were physics: the surface layer's `vegfra`,
radiation's `column_size_bounding`, both cumulus constructors' `column_chunk`,
the YSU launcher's mixing-length flag and the two float64 mirrors' keywords.
The carve took those callables into `arwen_global.core`, so their signatures
are this package's own and no installed engine can present a different one.
The eighth was `decode_mapped_source(scratch_destination=)`: the published
engine places the same scratch by `GPUWM_COMPOSE_SCRATCH`, so
`arwen_global.mapped_source_compat` translates between the two spellings and
says in the receipt which one placed the directory.  A gate is retired when
the breakage it names cannot happen, not weakened into a warning.

SO THE TABLE IS EMPTY, and this file proves the retirement from both sides:
that the carried callables take the keywords their rows used to refuse, that
the scratch placement is translated rather than dropped, and that the
mechanism still refuses by name for a row the next measurement adds.  The
mechanism is kept because the question does not go away: an engine inside
`gpuwm>=2.7.0,<2.8` can move a signature underneath the seam.
"""
from __future__ import annotations

import inspect
import sys
import types

import pytest

from arwen_global import engine_compat
from arwen_global.engine_compat import (
    SIGNATURE_GAPS,
    MissingEngineSymbol,
    SignatureGap,
    engine_signature_gaps,
    require_engine_signature,
)


def test_the_table_is_empty_because_every_measured_row_was_retired():
    """Emptiness is the measurement, so it is asserted rather than assumed."""

    assert SIGNATURE_GAPS == ()
    assert engine_signature_gaps() == ()


def test_an_unmeasured_call_is_a_key_error_not_a_silent_pass():
    """Asking about a call nobody measured must not answer "fine"."""

    with pytest.raises(KeyError):
        require_engine_signature("gpuwm.core.noah", "load_tables")
    with pytest.raises(KeyError):
        require_engine_signature("gpuwm.mapped_source", "decode_mapped_source")


def test_the_seven_physics_rows_are_gone_and_the_callables_are_carried():
    """The retirement, measured on the objects rather than asserted.

    Each takes the keyword its row used to say a published engine refused,
    because the module that defines it is now this package's.
    """

    from arwen_global.core import gf, ntiedtke, rrtmgp

    def takes(target, keyword):
        return keyword in inspect.signature(target).parameters

    assert takes(gf.GrellFreitas, "column_chunk")
    assert takes(gf.GrellFreitas, "updraft_only_when_downdraft_dry")
    assert takes(gf.GrellFreitas, "resolved_convergence_closure")
    assert takes(ntiedtke.NewTiedtke, "column_chunk")
    assert takes(rrtmgp.RRTMGPRadiation, "column_size_bounding")
    for module in (gf, ntiedtke, rrtmgp):
        assert module.__name__.startswith("arwen_global.core.")
    # sfclay, ysu and the float64 mirror reach cupy at module scope, so their
    # signatures are read where a card is, in
    # tests/test_arwen_global_carried_core.py.  The keywords are asserted on
    # the SOURCE here rather than not at all, because the point of this test
    # is that the rows were retired for a reason.
    from pathlib import Path

    import arwen_global.core as core

    carried = Path(core.__file__).parent
    assert "vegfra" in (carried / "sfclay.py").read_text(encoding="utf-8")
    assert "free_atmosphere_mixing_length" in (
        carried / "ysu.py").read_text(encoding="utf-8")


def test_the_scratch_row_is_gone_because_the_placement_is_translated():
    """The eighth retirement: one placement, two spellings, one receipt."""

    from arwen_global import mapped_source_compat

    assert mapped_source_compat.COMPOSE_SCRATCH_ENV == "GPUWM_COMPOSE_SCRATCH"
    assert hasattr(mapped_source_compat, "engine_scratch")
    # What the translation records is proven in
    # tests/test_mapped_source_compat.py, which drives the door itself from
    # two threads and reads each receipt.


# --------------------------------------- the mechanism, on a row it is given

def _install(monkeypatch, module_name: str, name: str, target) -> None:
    """Put one callable in sys.modules under a name the gate reads."""

    module = types.ModuleType(module_name)
    setattr(module, name, target)
    monkeypatch.setitem(sys.modules, module_name, module)


_ROW = SignatureGap(
    module="gpuwm.example_seam",
    name="example_call",
    keywords=("staging_destination",),
    stops="nothing: this row exists only where a test puts it, so the gate "
          "that refuses the next real one is exercised rather than trusted")


def _with_row(monkeypatch) -> None:
    monkeypatch.setattr(engine_compat, "SIGNATURE_GAPS", (_ROW,))


def test_a_signature_that_takes_the_keyword_is_not_a_gap(monkeypatch):
    _with_row(monkeypatch)

    def example_call(mapping, files, *, staging_destination=None):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    require_engine_signature("gpuwm.example_seam", "example_call")
    assert engine_compat.engine_signature_gaps() == ()


def test_a_signature_without_the_keyword_is_refused_by_name(monkeypatch):
    _with_row(monkeypatch)

    def example_call(mapping, files, *, destination=None):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    with pytest.raises(MissingEngineSymbol) as raised:
        require_engine_signature("gpuwm.example_seam", "example_call")
    text = str(raised.value)
    assert "staging_destination" in text
    assert "gpuwm.example_seam.example_call" in text
    # And it says why the argument is not simply dropped.
    assert "the symbol resolves" in text


def test_a_module_that_cannot_be_imported_is_unanswerable_not_clean(monkeypatch):
    """A host that cannot import the module must not publish a verdict."""

    _with_row(monkeypatch)
    monkeypatch.setitem(sys.modules, "gpuwm.example_seam", None)
    assert _ROW.missing() is None
    require_engine_signature("gpuwm.example_seam", "example_call")


def test_a_star_kwargs_signature_is_unanswerable(monkeypatch):
    """`**kwargs` accepts anything at the call and refuses inside it."""

    _with_row(monkeypatch)

    def example_call(mapping, files, **kwargs):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    assert _ROW.missing() is None


def test_building_native_physics_no_longer_refuses_on_a_published_engine():
    """The gate that stood here, and why nothing stands in its place.

    THE BREAKAGE IT NAMED: a native run against a published engine was
    accepted, priced the card, allocated it, integrated its first steps and
    died inside the surface layer on `vegfra`.  The gate moved that to the
    point where the forecast is built.

    The carve moved the surface layer itself.  `build_physics` now constructs
    the bridge and returns it, and this test is what says the refusal is gone
    rather than merely untested: a native config builds against the published
    engine this suite runs on.
    """

    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root
    from arwen_global.runner import build_physics

    cfg = load_config(str(config_root()
                          / "arwen_global_level5_native_smoke.toml"))
    assert cfg.physics_mode == "arwen-native"
    bridge = build_physics(cfg, "numpy")
    assert bridge is not None


def test_the_engine_this_suite_runs_against_reports_its_own_gaps():
    """Whatever the installed engine is, the report is about THAT engine."""

    for gap, missing in engine_signature_gaps():
        assert missing, (gap.module, gap.name)
        assert set(missing) <= set(gap.keywords)
        loaded = sys.modules.get(gap.module)
        if loaded is not None and getattr(loaded, gap.name, None) is not None:
            parameters = inspect.signature(getattr(loaded, gap.name)).parameters
            assert all(name not in parameters for name in missing)


# ------------------------------------------------- and the doctor says so

def _boundary_rows():
    from arwen_global.doctor import build_report

    report = build_report()
    for title, rows in report.sections:
        if title == "engine boundary":
            return rows
    raise AssertionError("the doctor has no engine boundary section")


def test_the_doctor_carries_the_boundary_and_grades_it_the_way_the_seam_does():
    """A carried contract is a note; a refused computation is a gap.

    THE BREAKAGE THIS PREVENTS.  The doctor's last line used to read "No gaps.
    Every documented command can run on this machine" on a box where `sizing`
    could not price a card and a scorecard could not regrid its reference,
    because the report looked at Rust doors and versions and never at the
    symbols the package imports out of the engine.
    """

    from arwen_global.engine_compat import engine_gaps

    rows = {row.label: row for row in _boundary_rows()}
    for gap in engine_gaps():
        label = f"{gap.module.split('.')[-1]}.{gap.symbol}"
        assert label in rows, (label, sorted(rows))
        expected = "note" if gap.handling == "carried" else "gap"
        assert rows[label].verdict == expected, (label, rows[label].verdict)
        assert rows[label].detail, label


def test_a_signature_gap_reaches_the_doctor_as_a_gap(monkeypatch):
    _with_row(monkeypatch)

    def example_call(mapping, files, *, destination=None):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    rows = {row.label: row for row in _boundary_rows()}
    assert "example_seam.example_call" in rows
    assert rows["example_seam.example_call"].verdict == "gap"
    assert "staging_destination" in rows["example_seam.example_call"].finding


def test_an_engine_whose_signatures_are_whole_leaves_no_signature_row(monkeypatch):
    """The section retires itself, row by row, as the engine catches up."""

    _with_row(monkeypatch)

    def example_call(mapping, files, *, staging_destination=None):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    assert engine_compat.engine_signature_gaps() == ()
    labels = {row.label for row in _boundary_rows()}
    assert "example_seam.example_call" not in labels
