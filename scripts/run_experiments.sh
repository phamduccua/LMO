#!/usr/bin/env bash
# Chay toan bo thuc nghiem CODE_CV theo ke hoach: quick | standard | full
#
#     bash scripts/run_experiments.sh quick
#     bash scripts/run_experiments.sh standard
#     bash scripts/run_experiments.sh full
#
# Moi stage deu RESUME: run nao da co summary.json thi bo qua. Tat may giua chung
# roi chay lai dung lenh cu -> chay tiep phan con thieu.
#
# NGAN SACH (uoc tinh tren RTX 3090, batch 128, AMP: resnet18 ~20s/epoch,
# resnet34 ~35s, wrn28_10 ~110s -- do lai bang `python bench_speed.py`):
#
#   quick     ~6 gio    resnet18 x 2 dataset, 1 seed, 20 epoch, luoi quick
#   standard  ~4 ngay   ca 3 model x 2 dataset; resnet 3 seed / 40 epoch,
#                       wrn28_10 1 seed / 30 epoch; luoi quick
#   full      ~2-3 tuan ca 6 o, luoi full, 3 seed, 50 epoch, them stage block + eta
#
# Muon biet chinh xac truoc khi chay:
#     python run_ablation.py all --dataset cifar10 --model wrn28_10 \
#            --grid quick --seeds 3 --epochs 40 --plan

set -euo pipefail
cd "$(dirname "$0")/.."

PLAN="${1:-quick}"
PY="${PYTHON:-python}"
DATA_WORKERS="${DATA_WORKERS:-8}"
BATCH="${BATCH:-128}"
DATASETS=(cifar10 cifar100)
# WORKERS = so RUN chay song song tren cung mot GPU.  Mac dinh 2 (WRN luon 1).
# Chi tang khi GPU CHUA bao hoa: xem `nvidia-smi` -- neu Pwr:Usage da gan Cap va
# util ~99% thi them run KHONG nhanh hon, chi lam moi run cham di.
#     WORKERS=3 bash scripts/run_experiments.sh standard

case "$PLAN" in
  quick)
    GRID=quick; TUNE_EPOCHS=5; WORKERS_DEFAULT=2
    MODELS=(resnet18)
    STAGES=(tune ablation baselines fwadam constlr report)
    seeds_of() { echo 1; }
    epochs_of() { echo 20; }
    ;;
  standard)
    GRID=quick; TUNE_EPOCHS=10; WORKERS_DEFAULT=2
    MODELS=(resnet18 resnet34 wrn28_10)
    STAGES=(tune ablation baselines fwadam constlr report)
    # WRN-28-10 dat gap ~5.5 lan resnet18 moi epoch -> it seed va it epoch hon
    seeds_of()  { [ "$1" = "wrn28_10" ] && echo 1 || echo 3; }
    epochs_of() { [ "$1" = "wrn28_10" ] && echo 30 || echo 40; }
    ;;
  full)
    GRID=full; TUNE_EPOCHS=10; WORKERS_DEFAULT=2
    MODELS=(resnet18 resnet34 wrn28_10)
    STAGES=(tune ablation baselines fwadam constlr block eta report)
    seeds_of()  { echo 3; }
    epochs_of() { echo 50; }
    ;;
  *) echo "ke hoach khong hop le: $PLAN  (quick | standard | full)"; exit 1 ;;
esac
WORKERS="${WORKERS:-$WORKERS_DEFAULT}"

echo "=============================================================="
echo " ke hoach   : $PLAN   (luoi: $GRID)"
echo " model      : ${MODELS[*]}"
echo " dataset    : ${DATASETS[*]}"
echo " stages     : ${STAGES[*]}"
echo " song song  : $WORKERS run   dataloader: $DATA_WORKERS worker   batch: $BATCH"
echo "=============================================================="

# --- 0. du lieu + kiem tra dung dan ---------------------------------------- #
$PY data.py
$PY sanity_check.py --quick

START=$(date +%s)
for DS in "${DATASETS[@]}"; do
  for MD in "${MODELS[@]}"; do
    SEEDS=$(seeds_of "$MD")
    EPOCHS=$(epochs_of "$MD")
    # WRN-28-10 chiem nhieu VRAM hon: chay tuan tu
    W=$WORKERS
    [ "$MD" = "wrn28_10" ] && W=1

    echo
    echo "##############################################################"
    echo "### $DS / $MD   (seeds 1..$SEEDS, $EPOCHS epoch, $W run song song)"
    echo "##############################################################"
    $PY run_ablation.py all --dataset "$DS" --model "$MD" --grid "$GRID" \
        --seeds "$SEEDS" --epochs "$EPOCHS" --tune-epochs "$TUNE_EPOCHS" \
        --workers "$W" --stages "${STAGES[@]}" --plan

    for ST in "${STAGES[@]}"; do
      # chi do lr tren resnet18; cac model lon dung lai ket qua do do
      if [ "$ST" = "tune" ] && [ "$MD" != "resnet18" ]; then
        echo "--- stage tune --- (bo qua: dung lai lr da do tren resnet18/$DS)"
        continue
      fi
      LR_FROM=()
      if [ "$ST" = "baselines" ] && [ "$MD" != "resnet18" ]; then
        LR_FROM=(--lr-from-model resnet18 --lr-from-dataset "$DS")
      fi
      echo "--- stage $ST ---"
      $PY run_ablation.py "$ST" \
        --dataset "$DS" --model "$MD" --grid "$GRID" \
        --seeds "$SEEDS" --epochs "$EPOCHS" --tune-epochs "$TUNE_EPOCHS" \
        --workers "$W" --data-workers "$DATA_WORKERS" --batch-size "$BATCH" \
        "${LR_FROM[@]}"
    done
  done
done

# --- bang LaTeX + hinh cho bao cao ----------------------------------------- #
MAXSEEDS=1
for MD in "${MODELS[@]}"; do S=$(seeds_of "$MD"); [ "$S" -gt "$MAXSEEDS" ] && MAXSEEDS=$S; done
echo
echo "--- sinh bang LaTeX + copy hinh -> ../paper_cv ---"
$PY make_paper_tables.py --seeds $(seq 1 "$MAXSEEDS")

ELAPSED=$(( $(date +%s) - START ))
echo
echo "XONG sau $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
echo "  bang text : results/tables_<dataset>_<model>.txt"
echo "  hinh      : results/figures/<dataset>_<model>/"
echo "  bang LaTeX: ../paper_cv/tables/   hinh: ../paper_cv/figures/"
echo "  dung bao cao: cd ../paper_cv && pdflatex main && bibtex main && pdflatex main && pdflatex main"
