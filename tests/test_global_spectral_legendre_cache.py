"""The per-machine Legendre analysis-table cache (spectral.legendre_cache).

The claim the cache ships under is that a transform built from a cached
table holds the bits a fresh build holds, so nothing a run computes can
tell a hit from a build.  These gates read that off the tables themselves
and off a transform's output, and check that every way a file can be
wrong (another key, a torn file, a foreign file) is a miss that rebuilds.
"""
from __future__ import annotations

import numpy as np
import pytest

from arwen_global.spectral import legendre_cache
from arwen_global.spectral.legendre_cache import CACHE_ENV, AnalysisTableCache
from arwen_global.spectral.transform import SphericalHarmonicTransform

T = legendre_cache.MIN_TRUNCATION  # the smallest truncation it caches


def _build(precision="float64", truncation=T, **kwargs):
    return SphericalHarmonicTransform.create(
        truncation, backend="numpy", precision=precision, **kwargs)


def _tables(transform):
    return transform._basis.dense(), transform._analysis.dense()


def _files(folder):
    return sorted(p for p in folder.glob("*.npz"))


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_cached_table_is_the_built_table_bit_for_bit(tmp_path, monkeypatch, precision):
    monkeypatch.setenv(CACHE_ENV, "off")
    plain = _build(precision)
    assert plain.legendre_cache == "off"
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    first = _build(precision)
    assert first.legendre_cache == "written"
    assert len(_files(tmp_path)) == 1
    second = _build(precision)
    assert second.legendre_cache == "hit"
    for a, b, c in zip(_tables(plain), _tables(first), _tables(second)):
        assert _same_bits(a, b) and _same_bits(a, c)
    # And what a run computes from them: an analysis and a synthesis of
    # one field, through the transform's own calls.
    rng = np.random.default_rng(7)
    field = rng.standard_normal(plain.grid.shape).astype(plain.backend.float_dtype)
    outs = [(t.forward(field), t.inverse(t.forward(field))) for t in (plain, first, second)]
    for spec, grid in outs[1:]:
        assert _same_bits(spec, outs[0][0]) and _same_bits(grid, outs[0][1])


def test_the_file_is_written_once_and_named_by_what_it_holds(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    _build("float32")
    _build("float32")
    (path,) = _files(tmp_path)
    assert path.name.startswith(f"analysis-T{T}-nlat")
    assert "-float32-" in path.name
    assert not list(tmp_path.glob(".*.tmp")), "a temporary file was left behind"
    _build("float64")
    assert len(_files(tmp_path)) == 2  # another dtype is another table


def test_a_changed_blas_environment_is_another_key(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    monkeypatch.delenv("OPENBLAS_CORETYPE", raising=False)
    _build()
    monkeypatch.setenv("OPENBLAS_CORETYPE", "Haswell")
    assert _build().legendre_cache == "written"
    assert len(_files(tmp_path)) == 2


@pytest.mark.parametrize("damage", ["truncate", "garbage", "foreign_key"])
def test_a_damaged_or_foreign_file_is_a_miss_that_rebuilds(tmp_path, monkeypatch, damage):
    monkeypatch.setenv(CACHE_ENV, "off")
    plain = _build()
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    _build()
    (path,) = _files(tmp_path)
    data = path.read_bytes()
    if damage == "truncate":
        path.write_bytes(data[: len(data) // 2])
    elif damage == "garbage":
        # Flip bytes inside the stored arrays: the zip checksum refuses it.
        cut = len(data) // 2
        path.write_bytes(data[:cut] + bytes(b ^ 0xFF for b in data[cut:cut + 64])
                         + data[cut + 64:])
    else:
        with np.load(path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["key"] = np.array(str(arrays["key"][()]).replace("schema", "schemb"))
        with open(path, "wb") as stream:
            np.savez(stream, **arrays)
    rebuilt = _build()
    assert rebuilt.legendre_cache == "written"
    for a, b in zip(_tables(plain), _tables(rebuilt)):
        assert _same_bits(a, b)
    assert _build().legendre_cache == "hit"


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "no", ""])
def test_the_cache_can_be_turned_off(tmp_path, monkeypatch, value):
    monkeypatch.setenv(CACHE_ENV, value)
    assert legendre_cache.cache_dir() is None
    assert _build().legendre_cache == "off"


def test_the_default_folder_is_under_the_family_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(CACHE_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    folder = legendre_cache.cache_dir()
    assert folder.parts[-3:] == (".gpuwm", "cache", "arwen-global-legendre")


def test_small_partitioned_and_streamed_transforms_are_never_cached(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    assert _build(truncation=T - 1).legendre_cache == "off"
    assert _build(streaming=True).legendre_cache == "off"
    whole = _build()
    part = _build(order_partition=((0, 32),))
    assert part.legendre_cache == "off"
    assert len(_files(tmp_path)) == 1  # only the whole table's
    assert whole.legendre_cache == "written"


def test_an_unwritable_folder_leaves_the_build_as_it_was(tmp_path, monkeypatch):
    blocker = tmp_path / "file-not-folder"
    blocker.write_bytes(b"")
    monkeypatch.setenv(CACHE_ENV, str(blocker / "cache"))
    built = _build()
    assert built.legendre_cache == "unwritten"
    monkeypatch.setenv(CACHE_ENV, "off")
    for a, b in zip(_tables(_build()), _tables(built)):
        assert _same_bits(a, b)


def test_a_table_above_the_size_cap_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(legendre_cache, "MAX_TABLE_BYTES", 1024)
    assert _build().legendre_cache == "off"
    assert _files(tmp_path) == []


def test_the_key_names_every_input_of_the_solve():
    key = legendre_cache.table_key(
        T, 192, 32, np.float32, np.linspace(-0.9, 0.9, 192), np.ones(192))
    for name in ("schema", "truncation", "nlat", "band", "dtype", "nodes",
                 "source", "numpy", "blas", "machine", "env"):
        assert name in key
    assert set(legendre_cache.BLAS_ENV) == set(key["env"])
    assert isinstance(AnalysisTableCache, type)
