# run_server.ps1  –  43 runs cho Tasks 1-7 (bỏ qua run đã có summary.json)
# Chạy: powershell -ExecutionPolicy Bypass -File scripts\run_server.ps1
# Dừng: Ctrl-C  (chạy lại để tiếp tục từ chỗ dừng)

$RUNS  = "results/runs"
$PY    = "python"

function Run-One {
    param($Task, $Group, $Name, [string[]]$Extra)
    $dst = "$RUNS/$Group/$Name"
    if (Test-Path "$dst/summary.json") {
        Write-Host "[SKIP] [$Task] $Name" -ForegroundColor Yellow
        return
    }
    Write-Host "[RUN ] [$Task] $Name" -ForegroundColor Green
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    & $PY train.py `
        --model resnet18 --epochs 20 --radius 5 `
        --out-dir "$RUNS/$Group" --run-name $Name `
        @Extra
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $Name (exit $LASTEXITCODE)" }
}

# ═══════════════════════════════════════════════════════════════════
# TASK 1 — Lưới η₁  (fw, cifar10, R=5, seed=1)
#          15 runs ≈ 4.0 GPU-giờ
# ═══════════════════════════════════════════════════════════════════
Write-Host "`n════ TASK 1: lưới η₁ ════" -ForegroundColor Cyan

foreach ($sch in @("harmonic","power","mlogm")) {
  foreach ($eta1 in @("1.0","0.3","0.1","0.03","0.01")) {
    Run-One T1 ablation `
      "cifar10__resnet18__fw__eta1${eta1}_radius5_schedule${sch}__e20__s1" `
      @("--optimizer","fw","--dataset","cifar10","--seed","1",
        "--schedule",$sch,"--eta1",$eta1)
  }
}

# ═══════════════════════════════════════════════════════════════════
# TASK 2 — Xác nhận seed 2,3  (eta1=0.1)
#          6 runs ≈ 1.6 GPU-giờ
# ═══════════════════════════════════════════════════════════════════
Write-Host "`n════ TASK 2: xác nhận seed 2,3 ════" -ForegroundColor Cyan

foreach ($sch in @("harmonic","power","mlogm")) {
  foreach ($seed in @(2,3)) {
    Run-One T2 ablation `
      "cifar10__resnet18__fw__eta10.1_radius5_schedule${sch}__e20__s${seed}" `
      @("--optimizer","fw","--dataset","cifar10","--seed","$seed",
        "--schedule",$sch,"--eta1","0.1")
  }
}

# ═══════════════════════════════════════════════════════════════════
# TASK 3 — Quét bước hằng  (constant, cifar10, R=5, seed=1)
#          5 runs ≈ 1.3 GPU-giờ
# ═══════════════════════════════════════════════════════════════════
Write-Host "`n════ TASK 3: quét bước hằng ════" -ForegroundColor Cyan

foreach ($eta1 in @("0.3","0.1","0.03","0.01","0.003")) {
  Run-One T3 constlr `
    "cifar10__resnet18__fw__eta1${eta1}_radius5_scheduleconstant__e20__s1" `
    @("--optimizer","fw","--dataset","cifar10","--seed","1",
      "--schedule","constant","--eta1",$eta1)
}

# ═══════════════════════════════════════════════════════════════════
# TASK 4 — Chéo CIFAR-100  (fw, cifar100, R=5, seed=1, eta1=0.1)
#          3 runs ≈ 0.8 GPU-giờ
# ═══════════════════════════════════════════════════════════════════
Write-Host "`n════ TASK 4: chéo CIFAR-100 ════" -ForegroundColor Cyan

foreach ($sch in @("harmonic","power","mlogm")) {
  Run-One T4 ablation `
    "cifar100__resnet18__fw__eta10.1_radius5_schedule${sch}__e20__s1" `
    @("--optimizer","fw","--dataset","cifar100","--seed","1",
      "--schedule",$sch,"--eta1","0.1")
}

# ═══════════════════════════════════════════════════════════════════
# TASK 5 — Baseline PGD  (pgd, lr=0.1, R=5, seed 1-3)
#          6 runs ≈ 1.6 GPU-giờ
# ═══════════════════════════════════════════════════════════════════
Write-Host "`n════ TASK 5: baseline PGD ════" -ForegroundColor Cyan

foreach ($ds in @("cifar10","cifar100")) {
  foreach ($seed in @(1,2,3)) {
    Run-One T5 baselines `
      "${ds}__resnet18__pgd__lr0.1_radius5__e20__s${seed}" `
      @("--optimizer","pgd","--lr","0.1","--dataset",$ds,"--seed","$seed")
  }
}

# ═══════════════════════════════════════════════════════════════════
# TASK 6 — Thêm seed  (fw, cifar100, eta1=1.0, harmonic+mlogm, seed 4,5)
#          4 runs ≈ 1.1 GPU-giờ
# ═══════════════════════════════════════════════════════════════════
Write-Host "`n════ TASK 6: thêm seed ════" -ForegroundColor Cyan

foreach ($sch in @("harmonic","mlogm")) {
  foreach ($seed in @(4,5)) {
    Run-One T6 ablation `
      "cifar100__resnet18__fw__eta11_radius5_schedule${sch}__e20__s${seed}" `
      @("--optimizer","fw","--dataset","cifar100","--seed","$seed",
        "--schedule",$sch,"--eta1","1.0")
  }
}

# ═══════════════════════════════════════════════════════════════════
# TASK 7 — BN ngoài ràng buộc  (block=module_nobn, eta1=1.0)
#          4 runs ≈ 1.1 GPU-giờ
# ═══════════════════════════════════════════════════════════════════
Write-Host "`n════ TASK 7: BN ngoài ràng buộc ════" -ForegroundColor Cyan

foreach ($ds in @("cifar10","cifar100")) {
  foreach ($sch in @("harmonic","power")) {
    Run-One T7 block `
      "${ds}__resnet18__fw__blockmodule_nobn_eta11_radius5_schedule${sch}__e20__s1" `
      @("--optimizer","fw","--dataset",$ds,"--seed","1",
        "--schedule",$sch,"--eta1","1.0","--block","module_nobn")
  }
}

Write-Host "`n════ XONG: tất cả 43 runs ════" -ForegroundColor Green
