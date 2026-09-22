#!/usr/bin/env bash
# Chạy cả CIFAR-10 và CIFAR-100 với eta1=0.1
set -euo pipefail
cd "$(dirname "$0")/.."

ETA="${1:-0.1}"
EP=20
W=1  # Reduced from 3 to avoid MemoryError

echo "=== Running CIFAR-10 with eta1=$ETA ==="
python run_ablation.py ablation --dataset cifar10 --model resnet18 \
  --schedules harmonic power mlogm --radii 1 5 --eta1s "$ETA" \
  --seeds 3 --epochs $EP --workers $W

python run_ablation.py constlr --dataset cifar10 --model resnet18 \
  --radii 5 --eta1s "$ETA" --fw-only --seeds 3 --epochs $EP --workers $W

echo
echo "=== Running CIFAR-100 with eta1=$ETA ==="
python run_ablation.py ablation --dataset cifar100 --model resnet18 \
  --schedules harmonic power mlogm --radii 1 5 --eta1s "$ETA" \
  --seeds 3 --epochs $EP --workers $W

python run_ablation.py constlr --dataset cifar100 --model resnet18 \
  --radii 5 --eta1s "$ETA" --fw-only --seeds 3 --epochs $EP --workers $W

echo
echo "=== CIFAR-10 Summary ==="
python scripts/pick_winner.py --dataset cifar10 --model resnet18 --radius 5

echo
echo "=== CIFAR-100 Summary ==="
python scripts/pick_winner.py --dataset cifar100 --model resnet18 --radius 5
