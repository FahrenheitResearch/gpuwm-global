"""The lease chain's bookkeeping: the config it composes, where each lease
ends, what it prunes across lease folders, the radiation it reports and what
a pull takes.

THE BREAKAGE THIS PREVENTS.  `tools/arwen_global_longrun_chain.py` drives a
90-day run as 18 leases on a rented card.  A fragment that lands outside
``native_adapter_options`` would run the arm without the options it was
launched for; a lease that ends off an output step leaves no checkpoint to
restart from; a cross-folder prune that ignored the month-ends or the
restart point would delete the state a resume needs; and a pull that
carried a checkpoint would move gigabytes through a desktop for nothing.
Each is checked here on a case with a known answer.  No device, no model.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tomllib

import numpy as np
import pytest

TREE = Path(__file__).resolve().parents[1]

CONFIG = """[arwen_global]
name = "x"

[time]
duration_s = 7776000.0
output_interval_s = 21600.0

[physics]
mode = "arwen-native"
native_adapter_options = { radiation = "rrtmgp", dx_m = 52100.0 }

[output]
keep_checkpoints = "newest:2+month-ends"
"""


@pytest.fixture(scope="module")
def chain():
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_longrun_chain", TREE / "tools" / "arwen_global_longrun_chain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fragment_lands_in_native_adapter_options_and_nowhere_else(chain):
    keys = ('morrison_droplet_number = "land_ocean", '
            'radiation_liquid_radius = "wrf_relcalc",\n'
            '# a comment line\nradiation_cloud_inhomogeneity = "tiedtke1996"\n')
    out = tomllib.loads(chain.compose_config(CONFIG, keys))
    options = out["physics"]["native_adapter_options"]
    assert options == {"radiation": "rrtmgp", "dx_m": 52100.0,
                       "morrison_droplet_number": "land_ocean",
                       "radiation_liquid_radius": "wrf_relcalc",
                       "radiation_cloud_inhomogeneity": "tiedtke1996"}
    assert out["output"] == tomllib.loads(CONFIG)["output"]
    assert chain.compose_config(CONFIG, None) == CONFIG
    assert chain.compose_config(CONFIG, "# nothing\n") == CONFIG
    with pytest.raises(ValueError, match="repeats dx_m"):
        chain.compose_config(CONFIG, "dx_m = 1.0")
    with pytest.raises(ValueError, match="exactly one"):
        chain.compose_config("[physics]\nmode = 'x'\n", "a = 1")


def test_a_lease_ends_on_the_next_boundary_and_a_cut_lease_resumes_to_it(chain):
    day = 86400.0
    lease, total = 5 * day, 90 * day
    assert chain.next_until_s(0.0, lease, total) == lease
    assert chain.next_until_s(lease, lease, total) == 2 * lease
    # Cut off at day 7.25: the resumed segment finishes lease 2, not 7.25 + 5.
    assert chain.next_until_s(7.25 * day, lease, total) == 2 * lease
    assert chain.next_until_s(88.0 * day, lease, total) == total
    assert chain.lease_index(2 * lease, lease) == 2
    assert chain.lease_index(total, lease) == 18
    assert chain.folder_name(2, ["L001"]) == "L002"
    assert chain.folder_name(2, ["L001", "L002"]) == "L002-r1"
    assert chain.folder_name(2, ["L001", "L002", "L002-r1"]) == "L002-r2"
    names = ["L002-r1", "L010", "L002", "L001"]
    assert sorted(names, key=chain.folder_order) == ["L001", "L002", "L002-r1", "L010"]


def test_the_run_wide_prune_keeps_the_newest_two_the_month_ends_and_the_restart(chain, tmp_path):
    sys.path.insert(0, str(TREE / "src"))
    dt_s = 300.0
    per_6h = 72
    start = "2026-08-27T00:00:00Z"
    found = {}
    # Leases of 2 days from 08-27: L001 steps 0..576, L002 576..1152 (cut at
    # 864), L002-r1 864..1152, L003 1152..1728.  Six-hourly checkpoints.
    layout = {"L001": range(per_6h, 577, per_6h), "L002": range(648, 865, per_6h),
              "L002-r1": range(936, 1153, per_6h), "L003": range(1224, 1729, per_6h)}
    for folder, steps in layout.items():
        for step in steps:
            found.setdefault(step, []).append(tmp_path / folder / f"arwen_global_step{step:08d}.npz")
    # The resumed copy of step 1152 the next lease also holds.
    found[1152].append(tmp_path / "L003" / "arwen_global_step00001152.npz")
    doomed = chain.prune_plan(found, dt_s=dt_s, start_utc=start, keep_newest=2,
                              keep_month_ends=True, protected=[1728])
    kept = {s for s, paths in found.items() for p in paths if p not in doomed}
    # 08-31 18Z is step 4.75 days = 1368, the last checkpoint of August.
    assert kept == {1368, 1656, 1728}
    # The duplicate 1152 goes twice: the older folder's copy and the step itself.
    assert tmp_path / "L002-r1" / "arwen_global_step00001152.npz" in doomed
    assert tmp_path / "L003" / "arwen_global_step00001152.npz" in doomed
    assert len(doomed) == len(set(doomed))
    every = chain.prune_plan(found, dt_s=dt_s, start_utc=start, keep_newest=None,
                             keep_month_ends=False)
    assert every == [tmp_path / "L002-r1" / "arwen_global_step00001152.npz"]


def test_a_resume_clears_only_what_a_killed_write_left(chain, tmp_path):
    # As the rehearsal's kill -9 left lease 2: the sidecar of step 936 was
    # written, its checkpoint was still a partial file.
    folder = tmp_path / "leases" / "L002"
    folder.mkdir(parents=True)
    names = ["arwen_global_step00000864.npz", "arwen_global_step00000864.longrun.npz",
             "arwen_global_step00000936.longrun.npz",
             ".arwen_global_step00000936.npz.partial-85583",
             "arwen_global_daily_2026-08-04.npz.part", "arwen_global_daily_2026-08-03.npz"]
    for name in names:
        (folder / name).write_text("x")
    gone = sorted(p.name for p in chain._leftovers(tmp_path))
    assert gone == sorted(names[2:5])


def test_the_lease_radiation_is_the_interval_weighted_mean_of_its_days(chain, tmp_path):
    def day(name, seconds, incoming, reflected, net, olr):
        meta = {"day": name, "radiation_interval": {"seconds": seconds},
                "global_means": {"toa_incoming_shortwave_w_m2": incoming,
                                 "toa_reflected_shortwave_w_m2": reflected,
                                 "toa_net_w_m2": net, "toa_outgoing_longwave_w_m2": olr}}
        path = tmp_path / f"arwen_global_daily_{name}.npz"
        np.savez(path, metadata=np.array(json.dumps(meta)))
        return path

    first = tmp_path / "arwen_global_daily_2026-08-01.npz"
    np.savez(first, metadata=np.array(json.dumps(
        {"day": "2026-08-01", "radiation_interval": "absent", "global_means": {}})))
    files = [first, day("2026-08-02", 86400.0, 340.0, 100.0, -10.0, 240.0),
             day("2026-08-03", 43200.0, 346.0, 106.0, -4.0, 236.0)]
    out = chain.lease_radiation(files)
    assert out["days"] == ["2026-08-02", "2026-08-03"]
    assert out["albedo"] == pytest.approx((2 * 100.0 + 106.0) / (2 * 340.0 + 346.0), rel=1e-15)
    assert out["toa_net_w_m2"] == pytest.approx((2 * -10.0 - 4.0) / 3.0, rel=1e-15)
    assert chain.lease_radiation([first])["albedo"] is None


def test_a_pull_never_carries_a_checkpoint_and_skips_what_the_destination_holds(
        chain, tmp_path, monkeypatch):
    run = tmp_path / "run"
    lease = run / "leases" / "L001"
    (run / "products" / "receipts").mkdir(parents=True)
    lease.mkdir(parents=True)
    for path, text in {run / "STATUS": "a\n", run / "chain.json": "{}",
                       lease / "arwen-global-receipt.json": "{}",
                       lease / "arwen_global_step00000072.npz": "ckpt",
                       lease / "arwen_global_step00000072.longrun.npz": "side",
                       lease / "arwen_global_daily_2026-08-01.npz": "day",
                       run / "products" / "arwen_global_daily_2026-08-01.npz": "day",
                       run / "products" / "arwen_global_mean_2026-08.npz": "mean",
                       run / "products" / "receipts" / "L001.json": "{}"}.items():
        path.write_text(text)

    def packed(have: str) -> list[str]:
        out = io.BytesIO()

        class Out:
            buffer = out

        monkeypatch.setattr(sys, "stdin", io.StringIO(have))
        monkeypatch.setattr(sys, "stdout", Out)
        chain.main(["pack", str(run)])
        out.seek(0)
        with tarfile.open(fileobj=out) as tar:
            return sorted(tar.getnames())

    everything = packed("")
    assert everything == sorted([
        "STATUS", "chain.json", "leases/L001/arwen-global-receipt.json",
        "products/arwen_global_daily_2026-08-01.npz", "products/arwen_global_mean_2026-08.npz",
        "products/receipts/L001.json"])
    # Held at the same size: skipped, except STATUS, which is always sent.
    have = "STATUS\t2\nchain.json\t2\nproducts/arwen_global_mean_2026-08.npz\t4\n"
    assert packed(have) == sorted([
        "STATUS", "leases/L001/arwen-global-receipt.json",
        "products/arwen_global_daily_2026-08-01.npz", "products/receipts/L001.json"])
