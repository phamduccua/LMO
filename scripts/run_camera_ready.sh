#!/usr/bin/env bash
# run_camera_ready.sh <pha> [<pha> ...]      -- camera-ready SOICT 2026, paper 4555
#
#   check      moi truong + sanity_check + thu tinh xac dinh (2 run x 2 epoch)
#   val-grid   luoi (form x eta1) tren VALIDATION, 3 seed         (114 run, Tier 1)
#   val-pgd    chinh lr cua PGD tren VALIDATION, 3 seed           ( 24 run, Tier 1)
#   val-ell    ho ell/(k+ell), ell in {2,4,8,16}, VALIDATION       ( 24 run, Tier 2)
#   select     chon scale tren validation -> final_plan.tsv       (khong GPU)
#   final      chay TEST mot lan, 5 seed moi (11..15), theo plan   (~40-50 run)
#   bn         BatchNorm: module vs module_nobn, seed 2,3          ( 16 run, Tier 3)
#   status     dem run xong / dang do
#
# Bien moi truong: DRY_RUN=1 (chi in lenh) MAX_JOBS=3 DATA_WORKERS=4
#   DETERMINISTIC=1 (mac dinh; 0 de tat --torch-deterministic) PYTHON=python
#   DATASETS="cifar10 cifar100"
# Chay lai an toan: run nao da co summary.json se duoc BO QUA.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
MAX_JOBS="${MAX_JOBS:-3}"
DATA_WORKERS="${DATA_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-cifar10 cifar100}"
RUNS="results/runs"
CR="results/camera_ready"
MODEL=resnet18; EP=20; RADIUS=5; VAL_FRAC=0.1
VAL_SEEDS="${VAL_SEEDS:-1 2 3}"
TEST_SEEDS="${TEST_SEEDS:-11 12 13 14 15}"      # tach hang khoi seed dung de chon
DET_FLAG=(); [ "${DETERMINISTIC:-1}" = "1" ] && DET_FLAG=(--torch-deterministic)

ETAS_harmonic="10 3 1 0.3 0.1 0.03 0.01"
ETAS_power="1 0.3 0.1 0.03 0.01 0.003"
ETAS_mlogm="3 1 0.3 0.1 0.03 0.01"
ELLS="2 4 8 16"
PGD_LRS="0.01 0.03 0.1 0.3"

run_one() {
    local group="$1" name="$2"; shift 2
    local dst="$RUNS/$group/$name"
    if [ -f "$dst/summary.json" ]; then echo "[SKIP ] $group/$name"; return 0; fi
    local cmd=("$PY" train.py --model "$MODEL" --epochs "$EP" --radius "$RADIUS"
               --workers "$DATA_WORKERS" --out-dir "$RUNS/$group" --run-name "$name"
               ${DET_FLAG[@]+"${DET_FLAG[@]}"} "$@")
    if [ "$DRY_RUN" = "1" ]; then echo "[DRY  ] ${cmd[*]}"; return 0; fi
    mkdir -p "$dst"
    echo "[START] $group/$name"
    if "${cmd[@]}" > "$dst/stdout.txt" 2>&1 < /dev/null && [ -f "$dst/summary.json" ]; then
        echo "[DONE ] $group/$name"
    else
        echo "[FAIL ] $group/$name   (xem $dst/stdout.txt)"
    fi
}
submit_job() {
    run_one "$@" &
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do sleep 5; done
}
etas_of() { local v="ETAS_$1"; echo "${!v}"; }

# ---------------------------------------------------------------------------
phase_check() {
    mkdir -p "$CR"
    {
        date; echo "git: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || true
        "$PY" -c "import sys,torch,torchvision;print(sys.version.split()[0]);print('torch',torch.__version__,'torchvision',torchvision.__version__,'cuda',torch.version.cuda)" 2>&1
        "$PY" -m pip freeze 2>&1
    } > "$CR/env.txt" || true
    echo "ghi $CR/env.txt (GPU, phien ban torch, pip freeze, git hash)"
    "$PY" sanity_check.py
    echo "--- co CLI can dung (phai thay ca hai) ---"
    "$PY" train.py --help | grep -i -E "torch-deterministic|val-frac" || { echo "THIEU co --torch-deterministic/--val-frac: bao co"; exit 1; }
    echo "--- thu tinh xac dinh: 2 run giong het, 2 epoch ---"
    for t in a b; do
        d="$RUNS/cr_check/det_$t"
        if [ ! -f "$d/summary.json" ]; then
            "$PY" train.py --model "$MODEL" --dataset cifar10 --optimizer fw --schedule harmonic \
                --eta1 0.1 --radius "$RADIUS" --epochs 2 --seed 1 --workers "$DATA_WORKERS" \
                --out-dir "$RUNS/cr_check" --run-name "det_$t" ${DET_FLAG[@]+"${DET_FLAG[@]}"} \
                > /dev/null 2>&1 < /dev/null || true
        fi
    done
    "$PY" - <<'PYEOF'
import json, sys
try:
    a = json.load(open("results/runs/cr_check/det_a/summary.json"))
    b = json.load(open("results/runs/cr_check/det_b/summary.json"))
except OSError as e:
    print("khong doc duoc summary.json cua run thu:", e); sys.exit(1)
same = a["acc_history"] == b["acc_history"] and a["train_loss_final"] == b["train_loss_final"]
print("acc_history a:", a["acc_history"], "| b:", b["acc_history"])
print("train_loss_final a:", a["train_loss_final"], "| b:", b["train_loss_final"])
print("=> GIONG HET (xac dinh)" if same else
      "=> KHAC NHAU: --torch-deterministic chua du. Van chay duoc, nhung PHAI bao co va ghi vao bao cao.")
PYEOF
}

phase_val_grid() {
    for seed in $VAL_SEEDS; do for ds in $DATASETS; do for sch in harmonic power mlogm; do
        for eta in $(etas_of $sch); do
            submit_job cr_val "${ds}__${MODEL}__fw__eta1${eta}_radius${RADIUS}_schedule${sch}__e${EP}__s${seed}" \
                --optimizer fw --dataset "$ds" --seed "$seed" --schedule "$sch" --eta1 "$eta" \
                --val-frac "$VAL_FRAC"
        done
    done; done; done
    wait
}
phase_val_pgd() {
    for seed in $VAL_SEEDS; do for ds in $DATASETS; do for lr in $PGD_LRS; do
        submit_job cr_val "${ds}__${MODEL}__pgd__lr${lr}_radius${RADIUS}__e${EP}__s${seed}" \
            --optimizer pgd --lr "$lr" --dataset "$ds" --seed "$seed" --val-frac "$VAL_FRAC"
    done; done; done
    wait
}
phase_val_ell() {
    # ell/(k+ell)  ==  harmonic voi eta1 = k0 = ell
    for seed in $VAL_SEEDS; do for ds in $DATASETS; do for L in $ELLS; do
        submit_job cr_val "${ds}__${MODEL}__fw__ell${L}_radius${RADIUS}_scheduleharmonic__e${EP}__s${seed}" \
            --optimizer fw --dataset "$ds" --seed "$seed" --schedule harmonic --eta1 "$L" --k0 "$L" \
            --val-frac "$VAL_FRAC"
    done; done; done
    wait
}
phase_select() { "$PY" scripts/select_val.py; }
phase_final() {
    local plan="$CR/final_plan.tsv"
    [ -f "$plan" ] || { echo "chua co $plan -- chay pha 'select' truoc"; exit 1; }
    for seed in $TEST_SEEDS; do
        while IFS=$'\t' read -r ds fam scale opt sch eta k0 lr; do
            [ "$ds" = "dataset" ] && continue
            if [ "$opt" = "pgd" ]; then
                submit_job cr_test "${ds}__${MODEL}__pgd__final_lr${lr}_radius${RADIUS}__e${EP}__s${seed}" \
                    --optimizer pgd --lr "$lr" --dataset "$ds" --seed "$seed"
            elif [ "$fam" = "ell" ]; then
                submit_job cr_test "${ds}__${MODEL}__fw__final_ell${eta}_radius${RADIUS}_scheduleharmonic__e${EP}__s${seed}" \
                    --optimizer fw --dataset "$ds" --seed "$seed" --schedule harmonic --eta1 "$eta" --k0 "$k0"
            else
                submit_job cr_test "${ds}__${MODEL}__fw__final_eta1${eta}_radius${RADIUS}_schedule${sch}__e${EP}__s${seed}" \
                    --optimizer fw --dataset "$ds" --seed "$seed" --schedule "$sch" --eta1 "$eta"
            fi
        done < "$plan"
    done
    wait
}
phase_bn() {
    for seed in 2 3; do for ds in $DATASETS; do for sch in harmonic power; do for blk in module module_nobn; do
        submit_job cr_bn "${ds}__${MODEL}__fw__block${blk}_eta11_radius${RADIUS}_schedule${sch}__e${EP}__s${seed}" \
            --optimizer fw --dataset "$ds" --seed "$seed" --schedule "$sch" --eta1 1.0 --block "$blk"
    done; done; done; done
    wait
}
phase_status() {
    for g in cr_check cr_val cr_test cr_bn; do
        [ -d "$RUNS/$g" ] || continue
        local done_=0 todo=0
        for d in "$RUNS/$g"/*/; do
            if [ -f "${d}summary.json" ]; then done_=$((done_+1)); else todo=$((todo+1)); fi
        done
        echo "$g: xong $done_ | chua xong/loi $todo"
        for d in "$RUNS/$g"/*/; do [ -f "${d}summary.json" ] || echo "   chua xong: $(basename "$d")"; done
    done
}

[ $# -ge 1 ] || { sed -n 2,19p "$0"; exit 1; }
for ph in "$@"; do
    echo; echo "════ pha: $ph ════"
    case "$ph" in
        check) phase_check;;  val-grid) phase_val_grid;;  val-pgd) phase_val_pgd;;
        val-ell) phase_val_ell;;  select) phase_select;;  final) phase_final;;
        bn) phase_bn;;  status) phase_status;;
        *) echo "pha khong ro: $ph"; exit 1;;
    esac
done
echo; echo "xong: $*"
