#!/usr/bin/env bash
# WOOF Global multi-card box test: placement, bandwidth, 1-vs-P identity.
#
# Written for an eight-card box (8x RTX 5090 or any eight CUDA cards in one
# host).  Every stage runs the shipped code paths: the wire probe places
# each rank exactly as the run door does, and the identity stage launches
# the `python -m arwen_global run` door once per rank.
#
#   stage 0  environment: cards, topology, CuPy and NCCL versions
#   stage 1  placement: 8 ranks with CUDA_VISIBLE_DEVICES unset must land
#            on 8 distinct cards (receipt UUIDs and nvidia-smi's own view)
#   stage 2  bandwidth: row gather and neighbour halo at P = 2, 4, 8 over
#            NCCL and over the TCP mesh, every byte checked
#   stage 3  identity: T255 and T383, P = 1, 2, 4, 8 on the default
#            transport (NCCL where it opens), and P = 8 on the TCP mesh;
#            every rank's final checkpoint must be the one-card checkpoint
#            array for array and by self_sha256 (gates BIT-5, BIT-6)
#
# usage:
#   PY=/path/venv/bin/python TREE=/path/gpuwm-global OUT=/path/out \
#     bash tools/arwen_global_box_cards_test.sh
# optional:
#   CONFIG_T255, CONFIG_T383  configs to run (default: the shipped analytic
#                             baroclinic wave on sl_si at T255, and the same
#                             file re-truncated to T383; point them at the
#                             GDAS configs when the analysis is on the box)
#   UNTIL_T255, UNTIL_T383    model seconds per identity run (21600, 10800)
#   CARDS                     card counts for stage 3 (default 1,2,4,8)
#   BANDS                     latitude bands for every run (default 16)
#   PROBE_MB                  wire probe buffer, MiB (default 1024)
#   STAGES                    which stages to run (default "0 1 2 3")
#
# Needs CuPy with NCCL importable for the NCCL arms (pip install
# "nvidia-nccl-cu13>=2.27.7,<3" next to cupy-cuda13x); without it the auto
# arms run the TCP mesh and say so in every receipt.  Exit status 0 when
# every check of every stage that ran passed.
set -u
PY=${PY:-python3}
TREE=${TREE:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${OUT:-$PWD/woof-global-box-cards}
CARDS=${CARDS:-1,2,4,8}
BANDS=${BANDS:-16}
PROBE_MB=${PROBE_MB:-1024}
UNTIL_T255=${UNTIL_T255:-21600}
UNTIL_T383=${UNTIL_T383:-10800}
STAGES=${STAGES:-"0 1 2 3"}
CONFIGS=$TREE/src/arwen_global/configs
CONFIG_T255=${CONFIG_T255:-$CONFIGS/arwen_global_jw06_t255_sl_si.toml}
export PYTHONPATH=$TREE/src${PYTHONPATH:+:$PYTHONPATH}
export NCCL_IB_DISABLE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
unset CUDA_VISIBLE_DEVICES
mkdir -p "$OUT"
FAIL=0
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/box.log"; }

if [ -z "${CONFIG_T383:-}" ]; then
  CONFIG_T383=$OUT/t383.toml
  sed -e 's/^truncation = .*/truncation = 383/' "$CONFIG_T255" > "$CONFIG_T383"
fi
NCARDS=$(nvidia-smi -L | wc -l)

probe() {  # probe <world> <transport> <tag>
  local world=$1 transport=$2 tag=$3 addrs="" pids=() r rc=0 REPEATS=10
  [ "$tag" = placement ] && REPEATS=50
  for r in $(seq 0 $((world - 1))); do addrs="$addrs,127.0.0.1:$((29800 + r))"; done
  addrs=${addrs#,}
  mkdir -p "$OUT/probe/$tag"
  for r in $(seq 0 $((world - 1))); do
    "$PY" "$TREE/tools/arwen_global_cards_wire_probe.py" --rank "$r" --world "$world" \
      --addresses "$addrs" --transport "$transport" --mb "$PROBE_MB" --repeats "$REPEATS" \
      --json "$OUT/probe/$tag/rank$r.json" > "$OUT/probe/$tag/rank$r.log" 2>&1 &
    pids+=($!)
  done
  local sampler=""
  if [ "$tag" = placement ]; then
    # nvidia-smi's own view of which card each rank's process holds,
    # sampled every second for as long as the ranks live.
    ( while true; do
        nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null
        sleep 1
      done ) > "$OUT/probe/$tag/nvidia-smi-apps.csv" &
    sampler=$!
  fi
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
  return $rc
}

summarise_probe() {  # summarise_probe <tag>
  "$PY" - "$OUT/probe/$1" <<'PYEOF'
import glob, json, sys
rows = [json.load(open(p)) for p in sorted(glob.glob(sys.argv[1] + "/rank*.json"))]
uuids = [((r.get("devices") or [{}])[r["rank"]] or {}).get("uuid") for r in rows]
ok = all(all(r["checks"].values()) for r in rows) and len(set(uuids)) == len(uuids)
print(json.dumps({
    "ranks": len(rows), "ok": ok,
    "transport": sorted({r["transport"] for r in rows}),
    "distinct_cards": len(set(uuids)),
    "pci": [((r.get("devices") or [{}])[r["rank"]] or {}).get("pci_bus_id") for r in rows],
    "row_gather_gb_s_per_rank": [round(r["row_gather_received_gb_s"], 2) for r in rows],
    "halo_gb_s_per_rank": [None if r["halo_received_gb_s"] is None else round(r["halo_received_gb_s"], 2) for r in rows],
    "wire_achieved_gb_s": [round((r.get("wire") or {}).get("achieved_gb_s") or 0, 2) for r in rows],
}))
sys.exit(0 if ok else 1)
PYEOF
}

for stage in $STAGES; do
  case $stage in
  0)
    say "stage 0: environment"
    { nvidia-smi -L; nvidia-smi topo -m; "$PY" -c "import cupy, sys; print('cupy', cupy.__version__, 'python', sys.version.split()[0]); from cupy.cuda import nccl; print('nccl available', nccl.available, nccl.get_version() if nccl.available else None)"; } \
      > "$OUT/env.txt" 2>&1
    cat "$OUT/env.txt" | tee -a "$OUT/box.log"
    ;;
  1)
    say "stage 1: placement, $NCARDS ranks, CUDA_VISIBLE_DEVICES unset"
    probe "$NCARDS" auto placement || FAIL=1
    summarise_probe placement | tee -a "$OUT/box.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL=1
    "$PY" - "$OUT/probe/placement" "$NCARDS" <<'PYEOF' | tee -a "$OUT/box.log"
import csv, glob, json, sys
folder, ncards = sys.argv[1], int(sys.argv[2])
ranks = {json.load(open(p))["rank"] for p in glob.glob(folder + "/rank*.json")}
seen = {}
for row in csv.reader(open(folder + "/nvidia-smi-apps.csv")):
    if len(row) > 1 and row[0].strip().isdigit():
        seen.setdefault(row[0].strip(), set()).add(row[1].strip())
cards = set().union(*seen.values()) if seen else set()
one_each = all(len(v) == 1 for v in seen.values())
print(f"nvidia-smi saw {len(seen)} rank processes on {len(cards)} distinct cards "
      f"(each process on one card: {one_each}); {len(ranks)} rank receipts")
sys.exit(0 if len(seen) == ncards and len(cards) == ncards and one_each else 1)
PYEOF
    [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL=1
    ;;
  2)
    for world in 2 4 8; do
      [ "$world" -le "$NCARDS" ] || continue
      for transport in nccl tcp; do
        say "stage 2: wire probe P=$world transport=$transport ${PROBE_MB} MiB"
        probe "$world" "$transport" "p$world-$transport" || FAIL=1
        summarise_probe "p$world-$transport" | tee -a "$OUT/box.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL=1
      done
    done
    ;;
  3)
    for pair in "t255:$CONFIG_T255:$UNTIL_T255" "t383:$CONFIG_T383:$UNTIL_T383"; do
      name=${pair%%:*}; rest=${pair#*:}; config=${rest%%:*}; until=${rest##*:}
      say "stage 3: identity $name cards=$CARDS until=${until}s (default transport)"
      "$PY" "$TREE/tools/arwen_global_cards_identity.py" --config "$config" \
        --cards "$CARDS" --until-s "$until" --latitude-bands "$BANDS" \
        --out "$OUT/identity/$name-auto" --python "$PY" --base-port 29900 \
        --json "$OUT/identity/$name-auto.json" 2>&1 | tee -a "$OUT/box.log"
      [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL=1
      say "stage 3: identity $name cards=1,8 on the TCP mesh"
      "$PY" "$TREE/tools/arwen_global_cards_identity.py" --config "$config" \
        --cards 1,8 --until-s "$until" --latitude-bands "$BANDS" --transport tcp \
        --out "$OUT/identity/$name-tcp" --python "$PY" --base-port 30100 \
        --json "$OUT/identity/$name-tcp.json" 2>&1 | tee -a "$OUT/box.log"
      [ "${PIPESTATUS[0]}" -eq 0 ] || FAIL=1
    done
    ;;
  esac
done
say "box cards test finished, fail=$FAIL"
exit $FAIL
