#!/usr/bin/env bash
# A complete protocol for cifar10 / resnet18 that fits ~2 hours on 1x3090.
#
#   bash scripts/run_4h.sh 0            # phase 0: eta1 screening  (~0.3 h)
#   bash scripts/run_4h.sh 1 0.1        # phase 1: FW grid         (~1.9 h), ETA from phase 0
#
# Phase 0 has to finish and be READ before phase 1 starts: phase 1 needs the
# eta1 that phase 0 picks.  Everything is resumable -- `launch` skips any run
# that already has a summary.json, so re-running a phase costs nothing.
#
# Fixed for every run in phase 1: 20 epochs, seeds 1-3, batch 128.  One budget
# for every method is what makes the table a comparison; do not raise --epochs
# for one arm only.
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:?usage: run_4h.sh <0|1> [eta1]}"
W=3

if [ "$PHASE" = "0" ]; then
  # 0. eta1 screen.  eta1 is the ONE knob shared by all three schedules
  # (mlogm takes no k0, so k0 cannot be tuned without confounding the
  # comparison).  eta1 = 1 is excluded on purpose: gamma_1 = 1 sets
  # theta_1 = s_0 and kills the run -- that is what this screen exists to avoid.
  python run_ablation.py eta --dataset cifar10 --model resnet18 \
    --radii 5 --eta1s 0.3 0.1 0.01 --seeds 1 --epochs 10 --workers $W

  echo
  echo "=== phase 0 done.  Read it, then start phase 1: ==="
  python scripts/pick_winner.py --dataset cifar10 --model resnet18 \
    --radius 5 --groups eta || true
  echo "  bash scripts/run_4h.sh 1 <eta1>"
  exit 0
fi

ETA="${2:?phase 1 needs the eta1 chosen in phase 0: run_4h.sh 1 0.1}"
EP=20

# 1a. The FW grid: 3 schedules x R in {1, 5} x 3 seeds.  Two radii is the
# minimum that answers "why this R?" -- R = 1 puts theta_0 exactly on the
# boundary of its block's ball, R = 5 leaves it room.
python run_ablation.py ablation --dataset cifar10 --model resnet18 \
  --schedules harmonic power mlogm --radii 1 5 --eta1s "$ETA" \
  --seeds 3 --epochs $EP --workers $W

# 1b. The arm OUTSIDE the theorem: gamma_k = const, so gamma_k -> 0 fails.
# One eta1 is enough to show it does not converge; the grid over eta1 is an
# appendix, not a claim.
python run_ablation.py constlr --dataset cifar10 --model resnet18 \
  --radii 5 --eta1s 0.1 --fw-only --seeds 3 --epochs $EP --workers $W

echo
python scripts/pick_winner.py --dataset cifar10 --model resnet18 --radius 5
python run_ablation.py report --dataset cifar10 --model resnet18 --epochs $EP
