#!/usr/bin/env bash
# run_cifar100_all.sh  –  2 runs bắt buộc (seed 2,3) + 9 runs lưới eta1 (CIFAR-100)
# Chạy: bash scripts/run_cifar100_all.sh
# Dừng: Ctrl-C  (chạy lại để tiếp tục — bỏ qua run đã có summary.json)

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="results/runs"
PY="${PYTHON:-python}"
MAX_JOBS=3

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

# ─────────────────────────────────────────────────────────────────────────────
# Hàm chạy 1 run
# ─────────────────────────────────────────────────────────────────────────────
run_one() {
    local task="$1" group="$2" name="$3"; shift 3
    local dst="$RUNS/$group/$name"

    if [ -f "$dst/summary.json" ]; then
        echo -e "${YELLOW}[SKIP]${NC} [$task] $name"
        return 0
    fi

    echo -e "${GREEN}[START]${NC} [$task] $name"
    mkdir -p "$dst"

    if $PY train.py \
        --model resnet18 --epochs 20 --radius 5 \
        --out-dir "$RUNS/$group" --run-name "$name" \
        "$@" > "$dst/stdout.txt" 2>&1; then

        if [ -f "$dst/summary.json" ]; then
            echo -e "${CYAN}[DONE ]${NC} [$task] $name"
        else
            echo -e "${RED}[FAIL ]${NC} [$task] $name (Không có summary.json)"
        fi
    else
        echo -e "${RED}[FAIL ]${NC} [$task] $name (Lỗi! Xem $dst/stdout.txt)"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Bộ quản lý chạy song song (Semaphore)
# ─────────────────────────────────────────────────────────────────────────────
submit_job() {
    run_one "$@" &

    while [ $(jobs -pr | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 5
    done
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: 2 runs bắt buộc (cifar100 / power / eta1=0.01 / R=5 / seed 2,3)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo -e "${CYAN}  PHẦN 1: cifar100 / power / eta1=0.01 / R=5${NC}"
echo -e "${CYAN}  seed 2, 3  |  MAX_JOBS=$MAX_JOBS${NC}"
echo -e "${CYAN}════════════════════════════════════════════${NC}"

for seed in 2 3; do
    submit_job "T1_seed" ablation \
        "cifar100__resnet18__fw__eta10.01_radius5_schedulepower__e20__s${seed}" \
        --optimizer fw --dataset cifar100 --seed "$seed" \
        --schedule power --eta1 0.01 --alpha 0.5001
done

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: Lưới eta1 (0.3, 0.1, 0.03) x 3 lịch trình, CIFAR-100, seed 1
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${CYAN}════════════════════════════════════════════${NC}"
echo -e "${CYAN}  PHẦN 2: Lưới eta1 trên CIFAR-100${NC}"
echo -e "${CYAN}  9 runs  |  MAX_JOBS=$MAX_JOBS${NC}"
echo -e "${CYAN}════════════════════════════════════════════${NC}"

for sch in harmonic power mlogm; do
    for eta in 0.3 0.1 0.03; do
        submit_job "T2_grid" ablation \
            "cifar100__resnet18__fw__eta1${eta}_radius5_schedule${sch}__e20__s1" \
            --optimizer fw --dataset cifar100 --seed 1 \
            --schedule "$sch" --eta1 "$eta" --alpha 0.5001
    done
done

# Chờ tất cả 11 jobs xong
wait
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# In kết quả
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo -e "${CYAN}  KẾT QUẢ: eval_acc_final${NC}"
echo -e "${CYAN}════════════════════════════════════════════${NC}"

$PY - <<'PYEOF'
import json, os

print("--- 3 seeds cho power, eta1=0.01 ---")
for s in (1, 2, 3):
    f = f"results/runs/ablation/cifar100__resnet18__fw__eta10.01_radius5_schedulepower__e20__s{s}/summary.json"
    if os.path.isfile(f):
        d = json.load(open(f))
        print(f"  seed {s}:  {d['eval_acc_final']:.2f}%")
    else:
        print(f"  seed {s}:  NOT FOUND")

print("\n--- Lưới eta1 trên CIFAR-100 ---")
for sch in ("harmonic", "power", "mlogm"):
    for eta in ("0.3", "0.1", "0.03"):
        f = f"results/runs/ablation/cifar100__resnet18__fw__eta1{eta}_radius5_schedule{sch}__e20__s1/summary.json"
        if os.path.isfile(f):
            d = json.load(open(f))
            print(f"  {sch:8s} | eta1={eta:<4s} :  {d['eval_acc_final']:.2f}%")
PYEOF

echo ""
echo -e "${GREEN}HOÀN TẤT TOÀN BỘ.${NC}"
