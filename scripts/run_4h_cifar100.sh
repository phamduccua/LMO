#!/usr/bin/env bash
# Phase 1 for CIFAR-100 (assumes eta1 already chosen from CIFAR-10 phase 0)
set -euo pipefail
cd "$(dirname "$0")/.."

ETA="${1:-0.1}"
W=3
EP=20

echo "Running phase 1 for CIFAR-100 with eta1=$ETA"

python run_ablation.py ablation --dataset cifar100 --model resnet18 \
  --schedules harmonic power mlogm --radii 1 5 --eta1s "$ETA" \
  --seeds 3 --epochs $EP --workers $W

python run_ablation.py constlr --dataset cifar100 --model resnet18 \
  --radii 5 --eta1s 0.1 --fw-only --seeds 3 --epochs $EP --workers $W

echo
python scripts/pick_winner.py --dataset cifar100 --model resnet18 --radius 5
python run_ablation.py report --dataset cifar100 --model resnet18 --epochs $EP
