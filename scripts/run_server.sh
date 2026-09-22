#!/usr/bin/env bash
# run_server.sh  –  43 runs (Chạy Song Song / Parallel)
# Tối ưu cho RTX 3090 (chạy 3 jobs cùng lúc)

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="results/runs"
PY="${PYTHON:-python}"
MAX_JOBS=3  # Số lượng tiến trình chạy song song

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

# ─────────────────────────────────────────────────────────────────────────────
# Hàm chạy 1 run (ghi log ra file, không in ra màn hình để tránh rối)
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

    # Chạy ngầm, chuyển toàn bộ output vào stdout.txt
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
    # Khởi chạy ngầm
    run_one "$@" &
    
    # Kiểm tra số lượng job đang chạy, nếu >= MAX_JOBS thì đợi
    # Dùng 'jobs -pr' để đếm số process đang running
    while [ $(jobs -pr | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 5
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Đọc eta1 tốt nhất
# ─────────────────────────────────────────────────────────────────────────────
best_eta1() {
    local sch="$1"
    $PY - "$RUNS" "$sch" <<'PYEOF'
import sys, glob, json, os
runs_dir, sch = sys.argv[1], sys.argv[2]
best_acc, best_eta = -1.0, None

for f in glob.glob(os.path.join(runs_dir, "*", "*", "summary.json")):
    with open(f) as fh:
        s = json.load(fh)
    c = s["config"]
    if (c["dataset"] != "cifar10" or c["optimizer"] != "fw"
            or abs(float(c["radius"]) - 5.0) > 1e-9
            or int(c["seed"]) != 1 or c["schedule"] != sch
            or s.get("diverged", False)):
        continue
    acc = float(s.get("eval_acc_final", -1))
    if acc > best_acc:
        best_acc, best_eta = acc, float(c["eta1"])

if best_eta is None:
    print("0.1")
else:
    print("%g" % best_eta)
PYEOF
}

# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${CYAN}════ TASK 1: lưới η₁ (Chạy 3 jobs song song) ════${NC}"
for sch in harmonic power mlogm; do
    for eta1 in 1.0 0.3 0.1 0.03 0.01; do
        submit_job T1 ablation \
            "cifar10__resnet18__fw__eta1${eta1}_radius5_schedule${sch}__e20__s1" \
            --optimizer fw --dataset cifar10 --seed 1 \
            --schedule "$sch" --eta1 "$eta1"
    done
done

echo -e "\n${YELLOW}>> Đang đợi toàn bộ Task 1 hoàn thành để đánh giá eta1...${NC}"
wait # Bắt buộc phải đợi xong Task 1 mới đi tiếp được

# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${CYAN}════ Chọn eta1 tốt nhất từ Task 1 ════${NC}"
ETA1_HARMONIC=$(best_eta1 harmonic)
ETA1_POWER=$(best_eta1 power)
ETA1_MLOGM=$(best_eta1 mlogm)

echo "  harmonic → eta1 = $ETA1_HARMONIC"
echo "  power    → eta1 = $ETA1_POWER"
echo "  mlogm    → eta1 = $ETA1_MLOGM"

# ═══════════════════════════════════════════════════════════════════════════════
echo -e "\n${CYAN}════ TASK 2 -> 7: Chạy song song tiếp tục ════${NC}"

# TASK 2
for seed in 2 3; do
    submit_job T2 ablation "cifar10__resnet18__fw__eta1${ETA1_HARMONIC}_radius5_scheduleharmonic__e20__s${seed}" \
        --optimizer fw --dataset cifar10 --seed "$seed" --schedule harmonic --eta1 "$ETA1_HARMONIC"
    submit_job T2 ablation "cifar10__resnet18__fw__eta1${ETA1_POWER}_radius5_schedulepower__e20__s${seed}" \
        --optimizer fw --dataset cifar10 --seed "$seed" --schedule power --eta1 "$ETA1_POWER"
    submit_job T2 ablation "cifar10__resnet18__fw__eta1${ETA1_MLOGM}_radius5_schedulemlogm__e20__s${seed}" \
        --optimizer fw --dataset cifar10 --seed "$seed" --schedule mlogm --eta1 "$ETA1_MLOGM"
done

# TASK 3
for eta1 in 0.3 0.1 0.03 0.01 0.003; do
    submit_job T3 constlr "cifar10__resnet18__fw__eta1${eta1}_radius5_scheduleconstant__e20__s1" \
        --optimizer fw --dataset cifar10 --seed 1 --schedule constant --eta1 "$eta1"
done

# TASK 4
submit_job T4 ablation "cifar100__resnet18__fw__eta1${ETA1_HARMONIC}_radius5_scheduleharmonic__e20__s1" \
    --optimizer fw --dataset cifar100 --seed 1 --schedule harmonic --eta1 "$ETA1_HARMONIC"
submit_job T4 ablation "cifar100__resnet18__fw__eta1${ETA1_POWER}_radius5_schedulepower__e20__s1" \
    --optimizer fw --dataset cifar100 --seed 1 --schedule power --eta1 "$ETA1_POWER"
submit_job T4 ablation "cifar100__resnet18__fw__eta1${ETA1_MLOGM}_radius5_schedulemlogm__e20__s1" \
    --optimizer fw --dataset cifar100 --seed 1 --schedule mlogm --eta1 "$ETA1_MLOGM"

# TASK 5
for ds in cifar10 cifar100; do
    for seed in 1 2 3; do
        submit_job T5 baselines "${ds}__resnet18__pgd__lr0.1_radius5__e20__s${seed}" \
            --optimizer pgd --lr 0.1 --dataset "$ds" --seed "$seed"
    done
done

# TASK 6
for sch in harmonic mlogm; do
    for seed in 4 5; do
        submit_job T6 ablation "cifar100__resnet18__fw__eta11_radius5_schedule${sch}__e20__s${seed}" \
            --optimizer fw --dataset cifar100 --seed "$seed" --schedule "$sch" --eta1 1.0
    done
done

# TASK 7
for ds in cifar10 cifar100; do
    for sch in harmonic power; do
        submit_job T7 block "${ds}__resnet18__fw__blockmodule_nobn_eta11_radius5_schedule${sch}__e20__s1" \
            --optimizer fw --dataset "$ds" --seed 1 --schedule "$sch" --eta1 1.0 --block module_nobn
    done
done

echo -e "\n${YELLOW}>> Đang đợi các job cuối cùng hoàn thành...${NC}"
wait

echo -e "\n${GREEN}════ XONG: tất cả 43 runs ════${NC}"
