"""Run one long forecast as a chain of leases, resumable, on a shared card.

::

    python tools/arwen_global_longrun_chain.py start RUN_DIR --config CONFIG.toml
        [--adapter-options FRAGMENT] [--lease-days 5] [--days N]
        [--lock LOCKFILE] [--lock-mode exclusive|shared]
        [--pythonpath SRC] [--gates stop|report] [--foreground]
    python tools/arwen_global_longrun_chain.py start RUN_DIR      # resume
    python tools/arwen_global_longrun_chain.py status RUN_DIR
    python tools/arwen_global_longrun_chain.py stop RUN_DIR       # after this lease
    python tools/arwen_global_longrun_chain.py report RUN_DIR [--reference-run OTHER_RUN_DIR]
    python tools/arwen_global_longrun_chain.py pack RUN_DIR < HAVE > pull.tar

``start`` composes the run's config once (CONFIG, with the inline-table
body in FRAGMENT added to ``[physics] native_adapter_options``), writes it
as ``RUN_DIR/run.toml`` beside ``chain.json``, and then runs
``gpuwm-global run run.toml --outdir RUN_DIR/leases/Lnnn [--restart ...]
--until-s ...`` once per lease, each under ``flock LOCKFILE`` so the card
is held for one lease at a time.  A lease ends on a lease boundary (a
whole multiple of ``--lease-days`` from the start); the next one restarts
from its last checkpoint, which is the documented way to segment a long
run (docs/ARWEN_GLOBAL_LONG_RUNS.md, "Launching a long run"), and a
restarted run reproduces the straight run bit for bit.

The chain detaches (its own session; ``chain.log``) unless
``--foreground``.  It is resumed by running the same command again, or
``start RUN_DIR`` alone: it finds the newest whole checkpoint in any lease
folder (the file is published by an atomic rename, and its
``.longrun.npz`` sidecar is written before it), and runs from there to the
next lease boundary in a new folder (``Lnnn-r1`` when ``Lnnn`` was cut
off).  A second ``start`` while one runs refuses: the chain's own lock
stays held while the driver or any process it started is alive.

After every lease the chain

* stops, with the reason in ``STATUS``, when the run's exit code is not 0,
  its receipt is missing or not ``pass``, the lease's last checkpoint is
  absent, or (``--gates stop``, the default) a required long-run gate
  reads ``fail`` over the rows so far;
* prunes the checkpoints of every lease folder by the config's
  ``[output] keep_checkpoints`` rule applied to the whole run (the run
  itself prunes only its own folder), with each deleted checkpoint's
  header appended to ``RUN_DIR/arwen_global_pruned_checkpoints.jsonl``;
* links the daily summaries and mean files into ``RUN_DIR/products``,
  writes the run's diagnostics rows there as one file, copies the lease's
  receipt, and runs the long-run gate report over every lease folder;
* appends one line to ``STATUS`` (and one record to ``status.jsonl``):
  the model day reached, the walls, the card's largest memory reading,
  the receipt's gates, the long-run gates, and the lease's planetary
  albedo and net top-of-atmosphere flux from its daily summaries.

``pack`` writes, to stdout, a tar of what a pull takes (``STATUS``,
``status.jsonl``, ``chain.json``, ``run.toml``, ``chain.log``, the report,
``products/``, and each lease folder's receipt, logs and pruned record;
never a checkpoint), leaving out every file whose path and size are in the
``path<TAB>size`` lines read from stdin (what the destination holds).

Nothing here changes the forecast: the model is run exactly as by hand.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time

SCHEMA = "gpuwm.arwen-global-longrun-chain/v1"
CHAIN_JSON = "chain.json"
RUN_TOML = "run.toml"
STATUS = "STATUS"
STATUS_JSONL = "status.jsonl"
STOP_FILE = "STOP"
CHAIN_LOCK = "chain.lock"
LEASES = "leases"
PRODUCTS = "products"
PRUNED_RECORD = "arwen_global_pruned_checkpoints.jsonl"
RECEIPT = "arwen-global-receipt.json"
_STEP_RE = re.compile(r"^arwen_global_step(\d{8})\.npz$")
_LEASE_RE = re.compile(r"^L(\d{3})(?:-r(\d+))?$")
_TOOLS = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# pure pieces (tests/test_arwen_global_longrun_chain.py)
# ---------------------------------------------------------------------------

def compose_config(text: str, fragment: str | None) -> str:
    """CONFIG's text with FRAGMENT (an inline-table body, ``key = value, ...``)
    added inside ``[physics] native_adapter_options = { ... }``.  A key the
    table already holds is refused rather than overridden."""
    if fragment is None:
        return text
    body = " ".join(line.strip() for line in fragment.splitlines()
                    if line.strip() and not line.strip().startswith("#")).strip().rstrip(",")
    if not body:
        return text
    pattern = re.compile(r"^(native_adapter_options\s*=\s*\{)(.*)(\}\s*)$", re.M)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError("the config must hold exactly one one-line "
                         "native_adapter_options = { ... } to add options to")
    import tomllib

    have = tomllib.loads("t = {" + matches[0].group(2) + "}")["t"]
    new = tomllib.loads("t = {" + body + "}")["t"]
    clash = sorted(set(have) & set(new))
    if clash:
        raise ValueError(f"the fragment repeats {', '.join(clash)}, which the config sets")
    m = matches[0]
    inner = m.group(2).rstrip()
    joined = f"{m.group(1)}{inner}, {body} {m.group(3).lstrip()}"
    out = text[:m.start()] + joined + text[m.end():]
    composed = tomllib.loads(out)["physics"]["native_adapter_options"]
    if any(composed.get(k) != v for k, v in new.items()):
        raise ValueError("the fragment did not land in [physics] native_adapter_options")
    return out


def next_until_s(done_s: float, lease_s: float, total_s: float) -> float:
    """The lease boundary after model time ``done_s`` (capped at the end)."""
    k = math.floor(done_s / lease_s + 1.0e-9) + 1
    return min(total_s, k * lease_s)


def lease_index(until_s: float, lease_s: float) -> int:
    return int(math.ceil(until_s / lease_s - 1.0e-9))


def folder_name(index: int, existing) -> str:
    """``Lnnn`` for lease ``index``, or ``Lnnn-rK`` when that name is taken
    (a lease cut off and resumed)."""
    base = f"L{index:03d}"
    names = set(existing)
    if base not in names:
        return base
    k = 1
    while f"{base}-r{k}" in names:
        k += 1
    return f"{base}-r{k}"


def folder_order(name: str) -> tuple[int, int]:
    m = _LEASE_RE.match(name)
    if not m:
        return (10 ** 6, 0)
    return (int(m.group(1)), int(m.group(2) or 0))


def checkpoint_step(path: Path) -> int | None:
    m = _STEP_RE.match(Path(path).name)
    return int(m.group(1)) if m else None


def prune_plan(found: dict[int, list[Path]], *, dt_s: float, start_utc,
               keep_newest: int | None, keep_month_ends: bool,
               protected=()) -> list[Path]:
    """The files to delete so the whole run keeps what ``keep_checkpoints``
    keeps within one segment: the newest N steps, the month-ends under
    ``+month-ends``, and ``protected``.  A step held by two folders (a lease
    cut off and resumed) keeps one copy, the latest folder's."""
    from arwen_global.longrun_ops import checkpoints_to_prune

    doomed: list[Path] = []
    for step, paths in found.items():
        ordered = sorted(paths, key=lambda p: folder_order(p.parent.name))
        doomed.extend(ordered[:-1])
    if keep_newest is None:
        return doomed
    for step in checkpoints_to_prune(
            found, dt_s=dt_s, start=start_utc, keep_newest=keep_newest,
            keep_month_ends=keep_month_ends, protected=protected):
        doomed.append(sorted(found[step], key=lambda p: folder_order(p.parent.name))[-1])
    return doomed


def lease_radiation(summaries) -> dict:
    """Planetary albedo and net TOA over a lease's daily summaries (their
    metadata's global means), weighted by each day's radiation interval."""
    import numpy as np

    sw_in = sw_up = net = olr = seconds = 0.0
    days = []
    for path in summaries:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
        means = meta.get("global_means") or {}
        interval = meta.get("radiation_interval")
        if not isinstance(interval, dict) or "toa_net_w_m2" not in means:
            continue
        s = float(interval["seconds"])
        sw_in += s * float(means["toa_incoming_shortwave_w_m2"])
        sw_up += s * float(means["toa_reflected_shortwave_w_m2"])
        net += s * float(means["toa_net_w_m2"])
        olr += s * float(means["toa_outgoing_longwave_w_m2"])
        seconds += s
        days.append(meta["day"])
    if seconds == 0.0:
        return {"days": [], "albedo": None, "toa_net_w_m2": None, "olr_w_m2": None}
    return {"days": days, "albedo": sw_up / sw_in if sw_in else None,
            "toa_net_w_m2": net / seconds, "olr_w_m2": olr / seconds}


# ---------------------------------------------------------------------------
# the run folder
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _append(path: Path, line: str) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(line.rstrip("\n") + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def status_line(run: Path, text: str, record: dict | None = None) -> None:
    _append(run / STATUS, f"{_utc_now()} {text}")
    if record is not None:
        _append(run / STATUS_JSONL, json.dumps({"utc": _utc_now(), **record},
                                               sort_keys=True, default=str))
    print(f"{_utc_now()} {text}", flush=True)


def lease_folders(run: Path) -> list[Path]:
    root = run / LEASES
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and _LEASE_RE.match(p.name)),
                  key=lambda p: folder_order(p.name))


def whole_checkpoints(run: Path, *, products: bool) -> dict[int, list[Path]]:
    """Every checkpoint in the lease folders that a restart can read: its
    header loads, and when the run writes long-run products its sidecar
    loads too."""
    import numpy as np

    from arwen_global.checkpoint import read_checkpoint_header

    found: dict[int, list[Path]] = {}
    for folder in lease_folders(run):
        for path in folder.glob("arwen_global_step*.npz"):
            step = checkpoint_step(path)
            if step is None:
                continue
            try:
                header = read_checkpoint_header(path)
                if int(header["step"]) != step:
                    continue
                if products:
                    with np.load(path.with_name(path.name[:-4] + ".longrun.npz"),
                                 allow_pickle=False) as side:
                        json.loads(str(side["header"]))
            except Exception:  # noqa: BLE001 - not whole: not a restart point
                continue
            found.setdefault(step, []).append(path)
    return found


def _leftovers(run: Path) -> list[Path]:
    """What a killed run leaves behind and nothing reads: partial writes
    (``.name.partial-PID``, ``*.part``) and a sidecar whose checkpoint never
    landed (the sidecar is written first).  Called only while no lease runs."""
    out = []
    for folder in lease_folders(run):
        out += list(folder.glob(".arwen_global_*.partial-*"))
        out += list(folder.glob("*.part"))
        out += [side for side in folder.glob("arwen_global_step*.longrun.npz")
                if not side.with_name(side.name[:-len(".longrun.npz")] + ".npz").is_file()]
    return out


def _gpu_sampler(stop: threading.Event, peak: dict) -> None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        peak["card_note"] = "nvidia-smi not found"
        return
    while not stop.is_set():
        try:
            out = subprocess.run(
                [exe, "--query-gpu=memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
            used, temp, power = (float(x) for x in out.splitlines()[0].split(","))
            peak["card_max_mib"] = max(peak.get("card_max_mib", 0.0), used)
            peak["card_max_c"] = max(peak.get("card_max_c", 0.0), temp)
            peak["card_max_w"] = max(peak.get("card_max_w", 0.0), power)
        except Exception:  # noqa: BLE001 - a missing reading is not a failure
            pass
        stop.wait(2.0)


def _tail(path: Path, n: int = 4) -> str:
    try:
        lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if l.strip()]
    except OSError:
        return ""
    return " | ".join(lines[-n:])[-600:]


def _du_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _child_env(plan: dict) -> dict:
    env = dict(os.environ)
    if plan.get("pythonpath"):
        env["PYTHONPATH"] = plan["pythonpath"]
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ---------------------------------------------------------------------------
# start / resume
# ---------------------------------------------------------------------------

def _create_or_check(run: Path, args) -> dict:
    import tomllib

    path = run / CHAIN_JSON
    plan = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if args.config is None:
        if plan is None:
            raise SystemExit(f"{run} holds no chain; give --config to start one")
        return plan
    text = Path(args.config).read_text(encoding="utf-8")
    fragment = (Path(args.adapter_options).read_text(encoding="utf-8")
                if args.adapter_options else None)
    composed = compose_config(text, fragment)
    cfg = tomllib.loads(composed)
    duration = float(cfg["time"]["duration_s"])
    output = float(cfg["time"]["output_interval_s"])
    dt_s = float(cfg.get("time", {}).get("dt_s", 300.0))
    lease_s = float(args.lease_days) * 86400.0
    total_s = duration if args.days is None else min(duration, float(args.days) * 86400.0)
    if abs(lease_s / output - round(lease_s / output)) > 1e-9:
        raise SystemExit(f"--lease-days {args.lease_days} is not a whole number of "
                         f"output intervals ({output:g} s): a lease must end on a checkpoint")
    if abs(total_s / output - round(total_s / output)) > 1e-9:
        raise SystemExit("--days is not a whole number of output intervals")
    out = cfg.get("output", {})
    fresh = {
        "schema": SCHEMA,
        "config_source": str(Path(args.config).resolve()),
        "adapter_options_source": (str(Path(args.adapter_options).resolve())
                                   if args.adapter_options else None),
        "adapter_options": fragment,
        "run_config": str((run / RUN_TOML).resolve()),
        "run_config_sha256": _sha256_text(composed),
        "lease_s": lease_s, "total_s": total_s, "duration_s": duration,
        "output_interval_s": output, "dt_s": dt_s,
        "keep_checkpoints": out.get("keep_checkpoints", "all"),
        "products": bool(out.get("daily_summary")) or out.get("mean_state", "none") != "none",
        "lock": str(Path(args.lock).resolve()) if args.lock else None,
        "lock_mode": args.lock_mode,
        "pythonpath": args.pythonpath,
        "python": args.python or sys.executable,
        "gates": args.gates,
    }
    if plan is None:
        run.mkdir(parents=True, exist_ok=True)
        (run / RUN_TOML).write_text(composed, encoding="utf-8", newline="\n")
        fresh["created_utc"] = _utc_now()
        path.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return fresh
    if plan["run_config_sha256"] != fresh["run_config_sha256"]:
        raise SystemExit(
            "the composed config differs from the one this chain started with "
            f"({run / RUN_TOML}); a restart under it would be refused by its config "
            "hash.  Start a new RUN_DIR, or resume with the original --config and "
            "--adapter-options")
    for key in ("lease_s", "total_s"):
        if plan[key] != fresh[key]:
            raise SystemExit(f"{key} differs from the chain's ({plan[key]} vs {fresh[key]})")
    return plan


def _detach(run: Path) -> None:
    """Fork into a new session; the parent returns at once."""
    log = open(run / "chain.log", "ab", buffering=0)
    pid = os.fork()
    if pid:
        print(f"chain: started in the background (log {run / 'chain.log'})")
        os._exit(0)
    os.setsid()
    if os.fork():
        os._exit(0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)


def _take_chain_lock(run: Path):
    handle = open(run / CHAIN_LOCK, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        raise SystemExit(f"chain: {run} is already running ({handle.read().strip()})")
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid {os.getpid()} since {_utc_now()}\n")
    handle.flush()
    return handle


def start(args) -> int:
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    plan = _create_or_check(run, args)
    if plan.get("pythonpath"):
        sys.path.insert(0, plan["pythonpath"])
    # Refuse here, in the caller's terminal, when the chain already runs.
    _take_chain_lock(run).close()
    if not args.foreground:
        _detach(run)
    lock = _take_chain_lock(run)
    (run / STOP_FILE).unlink(missing_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    status_line(run, f"chain start pid {os.getpid()} lease {plan['lease_s'] / 86400:g} d "
                     f"to {plan['total_s'] / 86400:g} d config {plan['run_config_sha256'][:12]}",
                {"event": "start", "pid": os.getpid()})
    try:
        return _loop(run, plan, lock)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - the reason goes to STATUS
        status_line(run, f"STOP chain error {type(exc).__name__}: {exc}",
                    {"event": "stop", "reason": f"{type(exc).__name__}: {exc}"})
        return 2


def _loop(run: Path, plan: dict, lock) -> int:
    dt_s = plan["dt_s"]
    while True:
        # Nothing of this chain runs here (its lock is held), so whatever a
        # cut-off lease left half-written is garbage.
        leftovers = _leftovers(run)
        if leftovers:
            status_line(run, "removed what a cut-off run left: " + ", ".join(
                f"{p.parent.name}/{p.name}" for p in leftovers),
                {"event": "leftovers", "removed": [str(p) for p in leftovers]})
        for leftover in leftovers:
            leftover.unlink(missing_ok=True)
        found = whole_checkpoints(run, products=plan["products"])
        done_step = max(found) if found else None
        done_s = 0.0 if done_step is None else done_step * dt_s
        if done_s >= plan["total_s"] - 1e-6:
            _after_lease(run, plan, None, found, protected=[done_step])
            status_line(run, f"DONE day {done_s / 86400:.2f} of {plan['total_s'] / 86400:g}",
                        {"event": "done", "step": done_step})
            return 0
        if (run / STOP_FILE).is_file():
            status_line(run, f"STOPPED on request at day {done_s / 86400:.2f}",
                        {"event": "stopped", "step": done_step})
            return 0
        until_s = next_until_s(done_s, plan["lease_s"], plan["total_s"])
        index = lease_index(until_s, plan["lease_s"])
        folders = lease_folders(run)
        name = folder_name(index, [f.name for f in folders])
        restart = None
        if done_step:
            restart = sorted(found[done_step], key=lambda p: folder_order(p.parent.name))[-1]
        cut = [f.name for f in folders if f.name.startswith(f"L{index:03d}")]
        if cut:
            status_line(run, f"resume lease {index} in {name} from step {done_step or 0} "
                             f"(day {done_s / 86400:.2f}); cut-off folder(s) {', '.join(cut)}",
                        {"event": "resume", "lease": index, "folder": name,
                         "from_step": done_step, "cut": cut})
        code = _one_lease(run, plan, lock, index, name, restart, until_s)
        if code != 0:
            return code


def _one_lease(run: Path, plan: dict, lock, index: int, name: str,
               restart: Path | None, until_s: float) -> int:
    folder = run / LEASES / name
    folder.mkdir(parents=True, exist_ok=False)
    python = plan["python"]
    model = [python, "-m", "arwen_global", "run", str(run / RUN_TOML),
             "--outdir", str(folder), "--until-s", f"{until_s:.0f}"]
    if restart is not None:
        model += ["--restart", str(restart)]
    command = model
    if plan.get("lock"):
        command = ["flock"] + (["-s"] if plan["lock_mode"] == "shared" else []) + \
                  [plan["lock"]] + model
    (folder / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    peak: dict = {}
    stop = threading.Event()
    sampler = threading.Thread(target=_gpu_sampler, args=(stop, peak), daemon=True)
    queued = time.time()
    with open(folder / "stdout.log", "wb") as log:
        child = subprocess.Popen(command, cwd=str(folder), env=_child_env(plan),
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
        sampler.start()
        rc = child.wait()
    stop.set()
    sampler.join(timeout=15)
    wall = time.time() - queued
    receipt = None
    if (folder / RECEIPT).is_file():
        try:
            receipt = json.loads((folder / RECEIPT).read_text(encoding="utf-8"))
        except ValueError:
            receipt = None
    until_step = int(round(until_s / plan["dt_s"]))
    last = folder / f"arwen_global_step{until_step:08d}.npz"
    reason = None
    if rc != 0:
        reason = f"exit code {rc}"
    elif receipt is None:
        reason = "no receipt"
    elif receipt.get("status") != "pass":
        reason = f"receipt {receipt.get('status')}"
    elif not last.is_file():
        reason = f"no checkpoint at step {until_step}"
    record: dict = {"event": "lease", "lease": index, "folder": name,
                    "from_step": None if restart is None else
                    int(re.search(r"(\d{8})", restart.name).group(1)),
                    "until_step": until_step, "rc": rc, "wall_s": round(wall, 1), **peak}
    if receipt is not None:
        gates = receipt.get("gates") or {}
        record["receipt_status"] = receipt.get("status")
        record["receipt_wall_s"] = receipt.get("wall_seconds")
        record["step_s"] = (receipt.get("step_wall") or {}).get("step_s")
        record["gates_passed"] = sum(1 for g in gates.values() if g.get("passed"))
        record["gates_total"] = len(gates)
        record["gates_failed"] = sorted(k for k, g in gates.items() if not g.get("passed"))
        clock = receipt.get("forecast_clock") or {}
        record["start_utc"] = clock.get("start_utc")
    if reason is not None:
        failure = None
        if receipt is not None and receipt.get("error_type"):
            failure = (f"{receipt['error_type']}: {receipt.get('error_message')} "
                       f"(completed step {receipt.get('completed_step')})")
        elif receipt is not None and record.get("gates_failed"):
            failure = "gates failed: " + ", ".join(record["gates_failed"])
        record["reason"] = reason
        record["detail"] = str(failure)[:600] if failure else _tail(folder / "stdout.log")
        status_line(run, f"STOP lease {index} ({name}) {reason}: {record['detail']}",
                    {**record, "event": "stop"})
        return 1
    found = whole_checkpoints(run, products=plan["products"])
    summary = _after_lease(run, plan, folder, found, protected=[until_step])
    record.update(summary)
    # The in-situ ledger's trip snapshots (a state dump when one of its
    # watches fires; informational, a few hundred MB each, not pulled).
    record["insitu_trips"] = sorted(p.name for p in folder.glob("insitu_trip_*.npz"))
    lr = summary.get("longrun_gates") or {}
    failed = sorted(k for k, g in lr.items() if g == "fail")
    lr_text = " ".join(f"{k}={v}" for k, v in lr.items()) or "-"
    rad = summary.get("radiation") or {}
    albedo = rad.get("albedo")
    net = rad.get("toa_net_w_m2")
    start_utc = record.get("start_utc")
    valid = ""
    if start_utc:
        when = dt.datetime.fromisoformat(start_utc.replace("Z", "+00:00")) + \
            dt.timedelta(seconds=until_s)
        valid = f" ({when:%Y-%m-%dT%HZ})"
    text = (f"lease {index} {name} day {until_s / 86400:.2f}/{plan['total_s'] / 86400:g}{valid} "
            f"wall {wall:.0f}s step {record.get('step_s') or float('nan'):.4f}s "
            f"card_max {peak.get('card_max_mib', float('nan')):.0f}MiB "
            f"{peak.get('card_max_c', float('nan')):.0f}C "
            f"gates {record.get('gates_passed')}/{record.get('gates_total')} pass "
            f"LR {lr_text} "
            f"albedo {'-' if albedo is None else f'{albedo:.4f}'} "
            f"toa_net {'-' if net is None else f'{net:+.2f}'}W/m2 "
            f"({len(rad.get('days') or [])} d) "
            f"insitu_trips {len(record['insitu_trips'])} "
            f"kept {summary['run_dir_gb']:.2f}GB free {summary['free_gb']:.0f}GB")
    status_line(run, text, record)
    if failed and plan.get("gates", "stop") == "stop":
        status_line(run, f"STOP long-run gate(s) {', '.join(failed)} read fail over the "
                         f"rows so far (report: {run / 'report.txt'})",
                    {"event": "stop", "reason": "longrun gate fail", "gates": failed})
        return 1
    return 0


def _after_lease(run: Path, plan: dict, folder: Path | None, found, protected) -> dict:
    """Prune across the lease folders, collect the products, report."""
    from arwen_global.longrun_ops import _append_jsonl, pruned_header

    keep = plan.get("keep_checkpoints", "all")
    newest, month_ends = None, False
    m = re.fullmatch(r"newest:(\d+)(\+month-ends)?", str(keep))
    if m:
        newest, month_ends = int(m.group(1)), bool(m.group(2))
    start_utc = None
    for f in lease_folders(run):
        try:
            clock = json.loads((f / RECEIPT).read_text(encoding="utf-8")).get(
                "forecast_clock") or {}
        except (OSError, ValueError):
            continue
        if clock.get("start_utc"):
            start_utc = clock["start_utc"]
            break
    pruned = 0
    if keep != "all":
        for path in prune_plan(found, dt_s=plan["dt_s"], start_utc=start_utc,
                               keep_newest=newest, keep_month_ends=month_ends,
                               protected=[p for p in protected if p is not None]):
            _append_jsonl(run / PRUNED_RECORD, {**pruned_header(path, start=start_utc),
                                                "folder": path.parent.name})
            path.unlink(missing_ok=True)
            path.with_name(path.name[:-4] + ".longrun.npz").unlink(missing_ok=True)
            pruned += 1
    products = run / PRODUCTS
    (products / "receipts").mkdir(parents=True, exist_ok=True)
    folders = lease_folders(run)
    for f in folders:
        for item in sorted(f.glob("arwen_global_daily_*.npz")) + \
                sorted(f.glob("arwen_global_mean_*.npz")):
            target = products / item.name
            if target.exists() and os.path.samefile(target, item):
                continue
            tmp = products / (item.name + ".link")
            tmp.unlink(missing_ok=True)
            try:
                os.link(item, tmp)
            except OSError:
                shutil.copy2(item, tmp)
            os.replace(tmp, target)
        if (f / RECEIPT).is_file():
            shutil.copy2(f / RECEIPT, products / "receipts" / f"{f.name}.json")
    with_rows = [f for f in folders if (f / "diagnostics.jsonl").is_file()]
    summary: dict = {"pruned": pruned}
    if with_rows:
        from arwen_global.longrun_gates import read_rows

        rows = read_rows(with_rows)
        tmp = products / "diagnostics.jsonl.part"
        with open(tmp, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(tmp, products / "diagnostics.jsonl")
        report = subprocess.run(
            [plan["python"], str(_TOOLS / "arwen_global_longrun_report.py"),
             *map(str, with_rows), "--json", str(run / "report.json")],
            capture_output=True, text=True, env=_child_env(plan), cwd=str(run))
        (run / "report.txt").write_text(report.stdout + report.stderr, encoding="utf-8")
        try:
            gates = json.loads((run / "report.json").read_text(encoding="utf-8"))["gates"]
            summary["longrun_gates"] = {k: g["verdict"] for k, g in gates.items()}
        except (OSError, ValueError, KeyError):
            summary["longrun_gates"] = {"report": f"error rc {report.returncode}"}
    if folder is not None:
        receipt = json.loads((folder / RECEIPT).read_text(encoding="utf-8"))
        written = ((receipt.get("long_run_outputs") or {}).get("products") or {}).get(
            "written") or []
        days = [folder / n for n in written if n.startswith("arwen_global_daily_")]
        summary["radiation"] = lease_radiation([p for p in days if p.is_file()])
        summary["written"] = written
    summary["run_dir_gb"] = _du_bytes(run) / 1e9
    summary["free_gb"] = shutil.disk_usage(run).free / 1e9
    summary["checkpoints_kept"] = sorted(
        f"{p.parent.name}/{p.name}" for p in run.glob(f"{LEASES}/*/arwen_global_step*.npz"))
    return summary


# ---------------------------------------------------------------------------
# status / stop / report / pack
# ---------------------------------------------------------------------------

def status(args) -> int:
    run = Path(args.run_dir).resolve()
    if (run / STATUS).is_file():
        sys.stdout.write((run / STATUS).read_text(encoding="utf-8"))
    handle = open(run / CHAIN_LOCK, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        print("chain: not running")
    except BlockingIOError:
        handle.seek(0)
        print(f"chain: running ({handle.read().strip()})")
        current = lease_folders(run)
        if current and (current[-1] / "status.json").is_file():
            st = json.loads((current[-1] / "status.json").read_text(encoding="utf-8"))
            print(f"chain: {current[-1].name} step {st.get('step')} state {st.get('state')} "
                  f"updated {st.get('updated_utc')}")
    return 0


def stop(args) -> int:
    run = Path(args.run_dir).resolve()
    (run / STOP_FILE).write_text(_utc_now() + "\n", encoding="utf-8")
    print(f"chain: {run} stops after the lease that is running")
    return 0


def report(args) -> int:
    run = Path(args.run_dir).resolve()
    plan = json.loads((run / CHAIN_JSON).read_text(encoding="utf-8"))
    folders = [f for f in lease_folders(run) if (f / "diagnostics.jsonl").is_file()]
    command = [plan["python"], str(_TOOLS / "arwen_global_longrun_report.py"), *map(str, folders)]
    if args.reference_run:
        ref = Path(args.reference_run).resolve()
        command += ["--reference-run",
                    *[str(f) for f in lease_folders(ref) if (f / "diagnostics.jsonl").is_file()]]
    command += ["--json", str(args.json or run / "report-final.json")]
    return subprocess.run(command, env=_child_env(plan)).returncode


def pullable(run: Path) -> list[Path]:
    names = [STATUS, STATUS_JSONL, CHAIN_JSON, RUN_TOML, "chain.log", "report.txt",
             "report.json", "report-final.json", PRUNED_RECORD]
    out = [run / n for n in names if (run / n).is_file()]
    products = run / PRODUCTS
    if products.is_dir():
        out += sorted(p for p in products.rglob("*") if p.is_file()
                      and not p.name.endswith((".part", ".link")))
    for f in lease_folders(run):
        for n in (RECEIPT, "run.log", "stdout.log", "command.txt", "status.json",
                  "arwen-global-run-config.toml", "arwen-global-run.json", PRUNED_RECORD):
            if (f / n).is_file():
                out.append(f / n)
    return out


def pack(args) -> int:
    run = Path(args.run_dir).resolve()
    have: dict[str, int] = {}
    if not sys.stdin.isatty():
        for line in sys.stdin.read().splitlines():
            if "\t" in line:
                path, size = line.rsplit("\t", 1)
                try:
                    have[path.strip()] = int(size)
                except ValueError:
                    pass
    sent = 0
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
        for path in pullable(run):
            rel = path.relative_to(run).as_posix()
            if have.get(rel) == path.stat().st_size and rel not in (
                    STATUS, STATUS_JSONL, "chain.log", "report.txt", "report.json"):
                continue
            tar.add(path, arcname=rel, recursive=False)
            sent += path.stat().st_size
    print(f"pack: {sent / 1e6:.1f} MB", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start", help="start, or resume, a chain")
    p.add_argument("run_dir")
    p.add_argument("--config", default=None, help="the run's config (first start)")
    p.add_argument("--adapter-options", default=None,
                   help="a file holding an inline-table body (key = value, ...) added to "
                        "[physics] native_adapter_options")
    p.add_argument("--lease-days", type=float, default=5.0)
    p.add_argument("--days", type=float, default=None,
                   help="stop the chain at this model day (default: the config's duration)")
    p.add_argument("--lock", default=None, help="hold this lock file (flock) for each lease")
    p.add_argument("--lock-mode", choices=("exclusive", "shared"), default="exclusive",
                   help="shared lets two chains (two arms) hold the card together, "
                        "while every exclusive holder waits")
    p.add_argument("--pythonpath", default=None, help="PYTHONPATH for the run (a source tree)")
    p.add_argument("--python", default=None, help="interpreter for the run (default: this one)")
    p.add_argument("--gates", choices=("stop", "report"), default="stop",
                   help="stop when a required long-run gate reads fail (default), or report only")
    p.add_argument("--foreground", action="store_true")
    for name, helptext in (("status", "print STATUS and whether the chain runs"),
                           ("stop", "stop after the lease that is running")):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("run_dir")
    q = sub.add_parser("report", help="the long-run gate report over every lease")
    q.add_argument("run_dir")
    q.add_argument("--reference-run", default=None, help="the other arm's RUN_DIR (LR-5)")
    q.add_argument("--json", default=None)
    q = sub.add_parser("pack", help="tar of what a pull takes, to stdout")
    q.add_argument("run_dir")
    args = parser.parse_args(argv)
    return {"start": start, "status": status, "stop": stop, "report": report,
            "pack": pack}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
