"""A card that runs out of memory in a command the door does not price.

THE BREAKAGE THIS PREVENTS, measured 2026-11-23 from a fresh account on an
RTX 5070 Ti (16 GB): the README's `da fresh ... --filter letkf --members 32`
ran 12 minutes and left as a CuPy `OutOfMemoryError` traceback, with nothing
that named the card it needs or what to change.  The door now ends such a
command with one sentence on stderr and exit 4 (the device refused), and a
refusal that is not the card keeps its own exit code.
"""
from __future__ import annotations

import pytest

from arwen_global import cli, da_door


class OutOfMemoryError(MemoryError):
    """CuPy's pool error, by the name `device_memory.is_out_of_memory` reads."""


def _fresh_argv(tmp_path):
    return ["da", "fresh", "arwen_global_gdas_t255_native_sl_si_24h",
            "--outdir", str(tmp_path / "fresh"), "--stream", "iem-asos",
            "--filter", "letkf", "--members", "32"]


def test_a_card_out_of_memory_in_da_fresh_is_a_sentence_and_exit_4(monkeypatch, tmp_path, capsys):
    def dies(*_args, **_kwargs):
        raise OutOfMemoryError(
            "Out of memory allocating 151,200,256 bytes (allocated so far: 15,747,409,920 bytes).")

    monkeypatch.setattr(da_door, "fresh", dies)
    code = cli.main(_fresh_argv(tmp_path))
    err = capsys.readouterr().err
    assert code == cli.EXIT_DEVICE
    assert "Traceback" not in err
    assert "ran out of memory" in err and "15,747,409,920" in err
    assert "32 members" in err and "16 GB" in err
    assert "successive-correction" in err


def test_a_refusal_that_is_not_the_card_keeps_its_exit_code(monkeypatch, tmp_path, capsys):
    def refuses(*_args, **_kwargs):
        raise FileNotFoundError("no static-field cache for this truncation")

    monkeypatch.setattr(da_door, "fresh", refuses)
    code = cli.main(_fresh_argv(tmp_path))
    err = capsys.readouterr().err
    assert code == cli.EXIT_REFUSED
    assert "no static-field cache" in err and "ran out of memory" not in err


def test_a_host_memory_error_is_not_read_as_the_card(monkeypatch, tmp_path):
    def host(*_args, **_kwargs):
        raise MemoryError("host")

    monkeypatch.setattr(da_door, "fresh", host)
    with pytest.raises(MemoryError):
        cli.main(_fresh_argv(tmp_path))


def test_the_statics_refusal_names_the_shipped_day_for_the_truncation(tmp_path):
    from arwen_global import orographic_statics, statics

    land = tmp_path / "arwen-global-statics-T127-192x384-default.npz"
    drag = tmp_path / "arwen-global-statics-T127-192x384-default-orogwd_10m.npz"
    assert "gpuwm-global statics arwen_global_gdas_t127_native_sl_si_24h\n" in statics.missing_cache_remedy(land)
    assert ("gpuwm-global statics arwen_global_gdas_t127_native_sl_si_24h --orographic"
            in orographic_statics.missing_cache_remedy(drag))
    other = tmp_path / "arwen-global-statics-T127-192x384-modis-x.npz"
    assert "<this run's config>" in statics.missing_cache_remedy(other)
