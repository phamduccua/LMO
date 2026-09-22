#!/usr/bin/env bash
# Chạy cả CIFAR-10 và CIFAR-100 với eta1=0.1
# (Thay thế kết quả cũ với eta1=1.0)
set -euo pipefail
cd "$(dirname "$0")/.."

W=1  # 1 worker để tránh MemoryError
EP=20
ETA=0.1  # Giá trị chuẩn từ phase 0 screening

echo "=== CIFAR-10: Ablation với eta1=$ETA ==="
python run_ablation.py ablation --dataset cifar10 --model resnet18 \
  --schedules harmonic power mlogm --radii 1 5 --eta1s $ETA \
  --seeds 3 --epochs $EP --workers $W

echo
echo "=== CIFAR-10: Constant với eta1=$ETA ==="
python run_ablation.py constlr --dataset cifar10 --model resnet18 \
  --radii 5 --eta1s $ETA --fw-only \
  --seeds 3 --epochs $EP --workers $W

echo
echo "=== CIFAR-100: Ablation với eta1=$ETA ==="
python run_ablation.py ablation --dataset cifar100 --model resnet18 \
  --schedules harmonic power mlogm --radii 1 5 --eta1s $ETA \
  --seeds 3 --epochs $EP --workers $W

echo
echo "=== CIFAR-100: Constant với eta1=$ETA ==="
python run_ablation.py constlr --dataset cifar100 --model resnet18 \
  --radii 5 --eta1s $ETA --fw-only \
  --seeds 3 --epochs $EP --workers $W

echo
echo "=== PGD baseline: CIFAR-10 ==="
python run_ablation.py baselines --dataset cifar10 --model resnet18 \
  --optimizers pgd --seeds 3 --epochs $EP --workers $W

echo
echo "=== PGD baseline: CIFAR-100 ==="
python run_ablation.py baselines --dataset cifar100 --model resnet18 \
  --optimizers pgd --seeds 3 --epochs $EP --workers $W

echo
echo "=== CIFAR-100: Thêm seeds 4-5 (eta1=$ETA) ==="
python run_ablation.py ablation --dataset cifar100 --model resnet18 \
  --schedules harmonic mlogm --radii 5 --eta1s $ETA \
  --seeds 5 --epochs $EP --workers $W

echo
echo "=== Hoàn thành! ==="
echo "CIFAR-10:  21 runs (18 ablation + 3 constant)"
echo "CIFAR-100: 25 runs (18 ablation + 3 constant + 4 seeds)"
echo "PGD:       6 runs (3 per dataset)"
echo "Tổng:      52 runs với eta1=$ETA"

echo
echo "=== Xem kết quả ==="
python scripts/pick_winner.py --dataset cifar10 --model resnet18 --radius 5 || true
echo
python scripts/pick_winner.py --dataset cifar100 --model resnet18 --radius 5 || true
