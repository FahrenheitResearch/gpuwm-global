"""Stored checkpoints, the offset reader, the pinned snapshot and the one
card probe per run: none of them may move a byte of any array, a hash, or
a restart (the [output] table)."""
from __future__ import annotations

from dataclasses import replace
import io
import os
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest

from arwen_global import runner as runner_module
from arwen_global.checkpoint import (
    PinnedSnapshot,
    _stored_members,
    bundle_arrays,
    normalize_trackers,
    read_checkpoint,
    write_checkpoint,
)
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.runner import _CheckpointWriter, build_model_and_cold_state, run

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _npz_members(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _write_pair(tmp_path):
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    trackers = normalize_trackers()
    trackers["maximum_spectral_cfl"] = 0.25
    paths = {}
    for compression in ("none", "deflate"):
        paths[compression] = write_checkpoint(
            tmp_path / f"{compression}.npz", state, config_hash=cfg.config_hash,
            to_numpy=model.transform.backend.to_numpy, trackers=trackers,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator, compression=compression,
        )
    return cfg, model, state, paths


def test_stored_and_deflated_checkpoints_carry_one_record_and_one_set_of_arrays(tmp_path):
    cfg, _model, _state, paths = _write_pair(tmp_path)
    with zipfile.ZipFile(paths["none"]) as archive:
        assert {i.compress_type for i in archive.infolist()} == {zipfile.ZIP_STORED}
    with zipfile.ZipFile(paths["deflate"]) as archive:
        assert {i.compress_type for i in archive.infolist()} == {zipfile.ZIP_DEFLATED}
    meta_none, arrays_none = read_checkpoint(
        paths["none"], expected_config_hash=cfg.config_hash)
    meta_deflate, arrays_deflate = read_checkpoint(
        paths["deflate"], expected_config_hash=cfg.config_hash)
    # The record, self_sha256 included, is the same: it is taken of the
    # arrays, not of the container.
    assert meta_none == meta_deflate
    assert arrays_none.keys() == arrays_deflate.keys()
    for name in arrays_none:
        assert arrays_none[name].dtype == arrays_deflate[name].dtype, name
        assert arrays_none[name].tobytes() == arrays_deflate[name].tobytes(), name
    # Any numpy reads the stored archive as the ordinary .npz it is.
    plain = _npz_members(paths["none"])
    assert set(plain) == set(arrays_none) | {"__metadata__"}


def test_the_offset_reader_returns_every_member_np_load_returns(tmp_path):
    _cfg, _model, _state, paths = _write_pair(tmp_path)
    by_offset = _stored_members(paths["none"])
    by_load = _npz_members(paths["none"])
    assert by_offset is not None
    assert by_offset.keys() == by_load.keys()
    for name, array in by_load.items():
        got = by_offset[name]
        assert got.shape == array.shape and got.dtype == array.dtype, name
        assert got.tobytes() == array.tobytes(), name
    # A deflated archive is not the offset reader's: np.load reads it.
    assert _stored_members(paths["deflate"]) is None
    # numpy writes a zip64 extra into every LOCAL header that the central
    # directory does not repeat for a small member, so the reader above
    # already crossed a local extra length that differs from the
    # directory's.  Asserted, so the case cannot silently stop being
    # covered.
    with zipfile.ZipFile(paths["none"]) as archive:
        infos = archive.infolist()
    raw = paths["none"].read_bytes()
    differs = 0
    for info in infos:
        local_extra = struct.unpack_from("<H", raw, info.header_offset + 28)[0]
        differs += local_extra != len(info.extra)
    assert differs > 0


def test_a_flipped_byte_in_a_stored_member_is_refused_by_its_hash(tmp_path):
    cfg, _model, _state, paths = _write_pair(tmp_path)
    path = paths["none"]
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("atmosphere__qv.npy")
    raw = bytearray(path.read_bytes())
    # The last byte of the member's data: past any .npy header.
    position = info.header_offset + 30 + len(info.filename.encode()) + struct.unpack_from(
        "<H", raw, info.header_offset + 28)[0] + info.file_size - 1
    raw[position] ^= 0x01
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="atmosphere__qv hash mismatch"):
        read_checkpoint(path, expected_config_hash=cfg.config_hash)


def test_a_truncated_stored_checkpoint_is_refused(tmp_path):
    _cfg, _model, _state, paths = _write_pair(tmp_path)
    path = paths["none"]
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        read_checkpoint(path)


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def test_a_member_whose_size_disagrees_with_its_header_is_refused(tmp_path):
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("a.npy", _npy_bytes(np.arange(6, dtype=np.float32)) + b"\0\0\0\0")
    with pytest.raises(ValueError, match="holds 28 array bytes where its header describes 24"):
        _stored_members(path)


def test_members_written_without_a_local_extra_and_in_fortran_order_read_back(tmp_path):
    path = tmp_path / "plain.npz"
    fortran = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    complex_ = (np.arange(4) + 1j * np.arange(4)).astype(np.complex64)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("f.npy", _npy_bytes(fortran))
        archive.writestr("c.npy", _npy_bytes(complex_))
        archive.writestr("s.npy", _npy_bytes(np.asarray("record")))
    members = _stored_members(path)
    assert members["f"].flags.f_contiguous and np.array_equal(members["f"], fortran)
    assert np.array_equal(members["c"], complex_)
    assert str(members["s"].item()) == "record"


def test_a_member_beyond_four_gib_is_read_at_its_zip64_offset(tmp_path):
    """The central directory names an offset past 4 GiB through its zip64
    extra; zipfile resolves it and the reader seeks there.  The 4 GiB in
    front is a hole, so the file costs no disk where holes are supported."""
    path = tmp_path / "far.zip"
    with path.open("wb") as stream:
        stream.truncate(4 * 1024**3 + 4096)
    blocks = getattr(os.stat(path), "st_blocks", None)
    if blocks is None or blocks * 512 > 64 * 1024**2:
        path.unlink()
        pytest.skip("this filesystem allocates sparse files")
    array = np.arange(1000, dtype=np.float32)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as archive:
        archive.writestr("far.npy", _npy_bytes(array))
    with zipfile.ZipFile(path) as archive:
        assert archive.getinfo("far.npy").header_offset > 4 * 1024**3
    members = _stored_members(path)
    assert np.array_equal(members["far"], array)
    path.unlink()


def test_the_output_table_never_reaches_the_config_hash(tmp_path):
    text = Path(CONFIG).read_text(encoding="utf-8")
    base = load_config(CONFIG)
    assert base.checkpoint_compression == "none"
    assert base.snapshot_buffer == "pinned"
    for compression in ("none", "deflate"):
        for buffer in ("pinned", "pageable"):
            variant = tmp_path / f"{compression}-{buffer}.toml"
            variant.write_text(
                text + f'\n[output]\ncheckpoint_compression = "{compression}"\n'
                f'snapshot_buffer = "{buffer}"\n', encoding="utf-8")
            cfg = load_config(variant)
            assert cfg.checkpoint_compression == compression
            assert cfg.snapshot_buffer == buffer
            assert cfg.config_hash == base.config_hash
    for bad in ('checkpoint_compression = "zstd"', 'snapshot_buffer = "mapped"',
                'keep = 2'):
        variant = tmp_path / "bad.toml"
        variant.write_text(text + f"\n[output]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="output"):
            load_config(variant)


def test_the_pinned_snapshot_holds_the_pageable_bytes_in_one_reused_buffer():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    to_numpy = model.transform.backend.to_numpy
    pageable = bundle_arrays(state, to_numpy)
    snapshot = PinnedSnapshot(model.transform.backend.xp)
    first = bundle_arrays(state, to_numpy, snapshot=snapshot)
    assert list(first) == list(pageable)
    for name in pageable:
        assert first[name].dtype == pageable[name].dtype, name
        assert first[name].tobytes() == pageable[name].tobytes(), name
    second = bundle_arrays(state, to_numpy, snapshot=snapshot)
    assert snapshot.allocations == 1
    # The second snapshot reuses the first's memory: the writer's join is
    # what makes that safe, and this is the property it relies on.
    name = next(iter(first))
    assert np.shares_memory(first[name], second[name])


def test_the_writer_with_a_snapshot_publishes_the_synchronous_archive(tmp_path):
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    to_numpy = model.transform.backend.to_numpy
    trackers = normalize_trackers()
    sync = write_checkpoint(
        tmp_path / "sync.npz", state, config_hash=cfg.config_hash,
        to_numpy=to_numpy, trackers=trackers,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    writer = _CheckpointWriter(
        cfg, to_numpy, snapshot=PinnedSnapshot(model.transform.backend.xp))
    first = writer.submit(tmp_path / "a.npz", state, trackers)
    second = writer.submit(tmp_path / "b.npz", state, trackers)
    writer.close()
    meta_sync, arrays_sync = read_checkpoint(sync)
    for path in (first, second):
        with zipfile.ZipFile(path) as archive:
            assert {i.compress_type for i in archive.infolist()} == {zipfile.ZIP_STORED}
        meta, arrays = read_checkpoint(path, expected_config_hash=cfg.config_hash)
        assert meta == meta_sync
        for name in arrays_sync:
            assert arrays[name].tobytes() == arrays_sync[name].tobytes(), name


def test_a_stored_lineage_restarts_bit_exact_from_either_container(tmp_path):
    """A straight stored run, its restart from the stored midpoint, and a
    restart from a deflated copy of that midpoint end on the same arrays;
    and a straight run with deflated checkpoints writes the same arrays."""
    cfg = load_config(CONFIG)
    deflate_cfg = replace(cfg, checkpoint_compression="deflate")
    assert deflate_cfg.config_hash == cfg.config_hash
    run(cfg, tmp_path / "stored")
    run(deflate_cfg, tmp_path / "deflated")
    midpoint = tmp_path / "stored" / "arwen_global_step00000002.npz"
    run(cfg, tmp_path / "resumed", restart=midpoint)
    copy = tmp_path / "midpoint-deflated.npz"
    with copy.open("wb") as stream:
        np.savez_compressed(stream, **_npz_members(midpoint))
    run(cfg, tmp_path / "resumed-from-deflated", restart=copy)
    end = "arwen_global_step00000004.npz"
    meta_a, arrays_a = read_checkpoint(tmp_path / "stored" / end)
    for other in ("deflated", "resumed", "resumed-from-deflated"):
        meta_b, arrays_b = read_checkpoint(tmp_path / other / end)
        assert meta_b == meta_a, other
        assert arrays_b.keys() == arrays_a.keys(), other
        for name in arrays_a:
            assert arrays_b[name].tobytes() == arrays_a[name].tobytes(), (other, name)
    for step in ("00000000", "00000002", "00000004"):
        path = tmp_path / "stored" / f"arwen_global_step{step}.npz"
        with zipfile.ZipFile(path) as archive:
            assert {i.compress_type for i in archive.infolist()} == {zipfile.ZIP_STORED}


def test_a_run_prices_its_memory_once(tmp_path, monkeypatch):
    """One prediction per run, carried by the receipt: the receipt no
    longer probes the card a second and third time."""
    calls = []
    real = runner_module._sizing_model_peak_bytes

    def counting(cfg, free_bytes=None):
        calls.append(free_bytes)
        return real(cfg, free_bytes=free_bytes)

    monkeypatch.setattr(runner_module, "_sizing_model_peak_bytes", counting)
    run(load_config(CONFIG), tmp_path / "run")
    assert len(calls) == 1


def test_the_prediction_takes_a_reading_it_is_handed(monkeypatch):
    from arwen_global import sizing

    probes = []

    def probe():
        probes.append(1)
        return 8 * 1024**3

    monkeypatch.setattr(sizing, "_free_device_bytes", probe)
    cfg = replace(load_config(CONFIG), backend="cupy")
    handed = runner_module._sizing_model_peak_bytes(cfg, free_bytes=8 * 1024**3)
    assert probes == []
    probed = runner_module._sizing_model_peak_bytes(cfg)
    assert probes == [1]
    assert handed == probed
