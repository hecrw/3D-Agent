#!/usr/bin/env bash
# Full camera-ready rerun of every experiment (JAC-ECC 2026, paper 162).
# Laid out for one Linux box with 4 GPUs (e.g. 4x RTX A6000 48 GB, sm_86).
# Run from the repo root. Each stage is resumable: rerun it after a crash.
# All output lands in eval/runs/$RUN_ID/. Nothing from earlier runs is reused.
#
#   bash eval/rerun_all.sh preflight        # checks GPUs, envs, repos, keys, network
#   bash eval/rerun_all.sh servers          # TRELLIS.2 servers: GPU0 :8200, GPU1 :8201
#   bash eval/rerun_all.sh smoke            # 1 prompt x {raw, all_on} — verify before the full run
#
#   # the three below can run AT THE SAME TIME (one tmux pane each):
#   bash eval/rerun_all.sh agent generated  # agent ablation, generated arm  -> server :8200 (GPU0)
#   bash eval/rerun_all.sh agent retrieved  # agent ablation, retrieved arm  -> server :8201 (GPU1)
#   bash eval/rerun_all.sh baselines retrieved   # direct backbones on GPU2
#
#   bash eval/rerun_all.sh baselines generated   # GPU2, after the generated arm (needs its concepts)
#   bash eval/rerun_all.sh score            # CLIP + Gen3DEval for baseline meshes
#   bash eval/rerun_all.sh ulip             # ULIP-2 for every results CSV, GPU3
#   bash eval/rerun_all.sh check            # fails loudly if any cell is missing/errored/unenforced
#   bash eval/rerun_all.sh analyze
#   bash eval/rerun_all.sh stop-servers
#
# The agent may also route to Hunyuan3D-2 / PartCrafter, which it calls on
# Modal: the Modal apps must be deployed and `modal` authed on this box.
set -euo pipefail

RUN_ID="${RUN_ID:-camready_20261111}"
RUN_DIR="eval/runs/$RUN_ID"
PY="${PY:-.venv/bin/python}"                 # agent/eval env (Python 3.11)
TRELLIS_REPO="${TRELLIS_REPO:-$HOME/TRELLIS.2}"
HUNYUAN_REPO="${HUNYUAN_REPO:-$HOME/Hunyuan3D-2}"
PARTCRAFTER_REPO="${PARTCRAFTER_REPO:-$HOME/PartCrafter}"
TRELLIS_ENV="${TRELLIS_ENV:-trellis}"        # conda env names
HUNYUAN_ENV="${HUNYUAN_ENV:-hunyuan}"
PARTCRAFTER_ENV="${PARTCRAFTER_ENV:-partcrafter}"
ULIP_ENV="${ULIP_ENV:-ulip}"
MAX_PASSES="${MAX_PASSES:-3}"                # resume passes to retry errored cells

# GPU layout (override if you want a different split)
GPU_TRELLIS_GEN="${GPU_TRELLIS_GEN:-0}"; PORT_GEN="${PORT_GEN:-8200}"
GPU_TRELLIS_RET="${GPU_TRELLIS_RET:-1}"; PORT_RET="${PORT_RET:-8201}"
GPU_BASELINES="${GPU_BASELINES:-2}"
GPU_ULIP="${GPU_ULIP:-3}"

export GEMINI_SEED="${GEMINI_SEED:-20261111}"
export AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-gemini-3.8-flash}"
export JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.8-flash}"
export OPENCV_IO_ENABLE_OPENEXR=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"   # A6000 / 3090 = Ampere sm_86

mkdir -p "$RUN_DIR/logs"
# conda from eval/setup_box.sh (Miniforge in $HOME), and the venv's `modal` CLI
# on PATH (tools.py shells out to `modal volume get` to download results).
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
# shellcheck disable=SC1091
[ -f "$CONDA_DIR/etc/profile.d/conda.sh" ] && . "$CONDA_DIR/etc/profile.d/conda.sh"
export PATH="$PWD/.venv/bin:$PATH"
crun() { conda run --no-capture-output -n "$1" "${@:2}"; }
port_for() { [ "$1" = generated ] && echo "$PORT_GEN" || echo "$PORT_RET"; }

case "${1:-}" in
preflight)
  fail=0; say() { printf '%-28s %s\n' "$1" "$2"; }
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
    && say "GPUs" "ok" || { say "GPUs" "nvidia-smi FAILED"; fail=1; }
  [ -x "$PY" ] && say "eval venv" "$($PY --version)" || { say "eval venv" "missing $PY (python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt)"; fail=1; }
  [ -f .env ] && grep -q GEMINI_API_KEY .env && say ".env" "ok" || { say ".env" "missing GEMINI_API_KEY (copy .env from your Mac)"; fail=1; }
  if command -v conda >/dev/null; then
    for e in "$TRELLIS_ENV" "$HUNYUAN_ENV" "$PARTCRAFTER_ENV" "$ULIP_ENV"; do
      conda env list | awk '{print $1}' | grep -qx "$e" && say "conda env $e" "ok" || { say "conda env $e" "MISSING (bash eval/setup_box.sh ${e})"; fail=1; }
    done
  else
    say "conda" "MISSING (bash eval/setup_box.sh conda)"; fail=1
  fi
  for r in "$TRELLIS_REPO" "$HUNYUAN_REPO" "$PARTCRAFTER_REPO"; do
    [ -d "$r" ] && say "repo $(basename "$r")" "ok" || { say "repo $(basename "$r")" "MISSING ($r)"; fail=1; }
  done
  n=$(ls eval/dataset/images/*.png 2>/dev/null | wc -l)
  [ "$n" -ge 20 ] && say "input photos" "$n" || { say "input photos" "only $n/20 in eval/dataset/images"; fail=1; }
  ws=$(grep -E '^TRELLIS_WORKSPACE=' .env | cut -d= -f2 | tr -d '"')
  prof=$(modal profile current 2>/dev/null || true)
  if [ -z "$prof" ]; then say "modal" "not authed (bash eval/setup_box.sh modal)"; fail=1
  elif [ "$prof" != "$ws" ]; then say "modal" "logged in as $prof but .env expects $ws"; fail=1
  else say "modal" "authed as $prof"; fi
  for h in hunyuan3d-2-generator-web partcrafter-objectgenerator-web; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 600 "https://$ws--$h.modal.run/" || true)
    [ "$code" = 200 ] && say "modal $h" "up" || { say "modal $h" "HTTP $code (deployed? cold start can take minutes)"; fail=1; }
  done
  curl -s -o /dev/null -w '%{http_code}' https://generativelanguage.googleapis.com/ | grep -q '^[234]' \
    && say "Gemini reachable" "ok" || { say "Gemini reachable" "NO network to Google API"; fail=1; }
  curl -s -o /dev/null https://huggingface.co && say "HF reachable" "ok" || { say "HF reachable" "NO (weights download)"; fail=1; }
  [ -x "$CONDA_DIR/envs/cuda124/bin/nvcc" ] && say "build toolchain" "cuda124 env ok" \
    || say "build toolchain" "missing (bash eval/setup_box.sh buildtools) — only needed for setup"
  exit $fail
  ;;
servers)
  for pair in "$GPU_TRELLIS_GEN:$PORT_GEN" "$GPU_TRELLIS_RET:$PORT_RET"; do
    gpu=${pair%%:*}; port=${pair##*:}
    if curl -sf "http://127.0.0.1:$port/" >/dev/null; then echo "server :$port already up"; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu nohup conda run --no-capture-output -n "$TRELLIS_ENV" \
      python eval/local_trellis_server.py --repo "$TRELLIS_REPO" --port "$port" \
      > "$RUN_DIR/logs/trellis_$port.log" 2>&1 &
    echo "started TRELLIS on GPU$gpu :$port (log $RUN_DIR/logs/trellis_$port.log)"
  done
  echo "waiting for servers to load weights..."
  for port in "$PORT_GEN" "$PORT_RET"; do
    for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/" && echo && break; sleep 10; done
  done
  ;;
stop-servers)
  pkill -f local_trellis_server.py && echo stopped || echo "none running"
  ;;
smoke)
  export LOCAL_TRELLIS_URL="http://127.0.0.1:$PORT_RET" AGENT_CKPT_PATH="$PWD/eval/runs/smoke_ckpt.sqlite3"
  "$PY" eval/run_pilot.py --run-id "smoke_$(date +%H%M)" --input retrieved --limit 1 --conditions raw all_on
  echo "Inspect the CSV above: status=ok, condition_enforced=True, backbone_tool set, agent_model_resolved set."
  ;;
agent)
  mode="${2:?usage: agent generated|retrieved}"
  port=$(port_for "$mode")
  export LOCAL_TRELLIS_URL="http://127.0.0.1:$port"
  export AGENT_CKPT_PATH="$PWD/$RUN_DIR/ckpt_$mode.sqlite3"   # separate DB per parallel sweep
  curl -sf "$LOCAL_TRELLIS_URL/" >/dev/null || { echo "TRELLIS server not up at $LOCAL_TRELLIS_URL (run: servers)"; exit 1; }
  for pass in $(seq 1 "$MAX_PASSES"); do
    echo "=== $mode pass $pass ==="
    "$PY" eval/run_pilot.py --run-id "$RUN_ID" --input "$mode" 2>&1 | tee -a "$RUN_DIR/logs/agent_$mode.log"
  done
  ;;
baselines)
  mode="${2:?usage: baselines retrieved|generated}"
  imgs=eval/dataset/images
  if [ "$mode" = generated ]; then
    imgs="$RUN_DIR/concepts"
    n=$(ls "$imgs"/*.png 2>/dev/null | wc -l)
    [ "$n" -ge 20 ] || { echo "only $n/20 concept images in $imgs — finish 'agent generated' first"; exit 1; }
  fi
  export CUDA_VISIBLE_DEVICES=$GPU_BASELINES
  crun "$TRELLIS_ENV"     python eval/local_baseline_trellis.py     --repo "$TRELLIS_REPO"     --images "$imgs" --out "$RUN_DIR/baseline_trellis2_$mode"
  crun "$HUNYUAN_ENV"     python eval/local_baseline_hunyuan.py     --repo "$HUNYUAN_REPO"     --images "$imgs" --out "$RUN_DIR/baseline_hunyuan3d2_$mode"
  crun "$PARTCRAFTER_ENV" python eval/local_baseline_partcrafter.py --repo "$PARTCRAFTER_REPO" --images "$imgs" --out "$RUN_DIR/baseline_partcrafter_$mode"
  ;;
score)
  for mode in retrieved generated; do
    for bb in trellis2 hunyuan3d2 partcrafter; do
      d="$RUN_DIR/baseline_${bb}_$mode"; [ -d "$d" ] || { echo "skip $d (not generated yet)"; continue; }
      "$PY" eval/score_baseline.py "$d" "$RUN_DIR/results_baseline_${bb}_$mode.csv" \
        --backbone "$bb" --input-mode "$mode" --run-id "$RUN_ID"
    done
  done
  ;;
ulip)
  export CUDA_VISIBLE_DEVICES=$GPU_ULIP
  for csv in "$RUN_DIR"/results_*.csv; do
    name=$(basename "$csv" .csv)
    pcs="$RUN_DIR/pointclouds/$name"
    "$PY" eval/export_pointclouds.py "$csv" "$pcs"
    crun "$ULIP_ENV" python eval/local_ulip.py --pointclouds "$pcs" --out "$RUN_DIR/ulip_$name.csv"
    "$PY" eval/merge_ulip.py "$csv" "$RUN_DIR/ulip_$name.csv"
  done
  ;;
check)
  "$PY" - "$RUN_DIR" <<'EOF'
import csv, sys, glob, pathlib
run = pathlib.Path(sys.argv[1]); bad = 0
caps = [r["caption"] for r in csv.DictReader(open("eval/dataset/captions.csv"))]
expected = {"results_generated.csv", "results_retrieved.csv"} | {
    f"results_baseline_{b}_{m}.csv" for b in ("trellis2", "hunyuan3d2", "partcrafter")
    for m in ("retrieved", "generated")}
have = {pathlib.Path(f).name for f in glob.glob(str(run / "results_*.csv"))}
for f in sorted(expected - have):
    print("MISSING FILE", f); bad += 1
for f in sorted(have):
    rows = list(csv.DictReader(open(run / f)))
    ok = {(r["prompt"], r["condition"]) for r in rows if r["status"] == "ok"}
    conds = sorted({r["condition"] for r in rows})
    missing = [(p, c) for p in caps for c in conds if (p, c) not in ok]
    noulip = [r for r in rows if r["status"] == "ok" and not r["ulip_mean"]]
    unenf = [r for r in rows if r["status"] == "ok" and r.get("condition_enforced") != "True"]
    print(f"{f}: ok={len(ok)}/{len(caps)*len(conds)} missing={len(missing)} "
          f"no_ulip={len(noulip)} not_enforced={len(unenf)}")
    for p, c in missing[:10]: print("   MISSING", c, p)
    bad += len(missing) + len(noulip) + len(unenf)
sys.exit(1 if bad else 0)
EOF
  ;;
analyze)
  for csv in "$RUN_DIR"/results_*.csv; do echo "### $csv"; "$PY" eval/analyze.py "$csv"; done
  ;;
*) sed -n 2,26p "$0"; exit 1 ;;
esac
