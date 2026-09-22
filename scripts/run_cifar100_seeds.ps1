<#
.SYNOPSIS
  Chạy 2 run bắt buộc (seed 2, 3) + 9 run lưới eta1 trên CIFAR-100.
  Song song tối đa MAX_JOB = 3.

.NOTES
  Chạy từ thư mục CODE_CV:
    powershell -File scripts\run_cifar100_seeds.ps1

  Ước tính trên RTX 3090 (~44s/epoch):
    - 2 run seed:      ~2 × 15 phút = 30 phút (chạy song song → ~15 phút)
    - 9 run lưới eta1: ~9 × 15 phút = 3 đợt × 15 phút = 45 phút
    - Tổng:            ~1 giờ
#>

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$MAX_JOB = 3

# ── Dọn thư mục partial seed 2 nếu còn sót ──
$partial = "results\runs\ablation\cifar100__resnet18__fw__eta10.01_radius5_schedulepower__e20__s2"
if (Test-Path $partial) {
    Remove-Item -Recurse -Force $partial
    Write-Host "[cleanup] Removed partial $partial"
}

# ── Danh sách run ──
# Phần A: 2 run BẮT BUỘC (seed 2, 3)
$runs = @(
    @{name="cifar100__resnet18__fw__eta10.01_radius5_schedulepower__e20__s2"; schedule="power"; eta1="0.01"; seed="2"},
    @{name="cifar100__resnet18__fw__eta10.01_radius5_schedulepower__e20__s3"; schedule="power"; eta1="0.01"; seed="3"}
)

# Phần B: 9 run TÙY CHỌN – lưới eta1 trên CIFAR-100
foreach ($sched in @("harmonic", "power", "mlogm")) {
    foreach ($eta in @("0.3", "0.1", "0.03")) {
        $tag = "cifar100__resnet18__fw__eta1${eta}_radius5_schedule${sched}__e20__s1"
        $runs += @{name=$tag; schedule=$sched; eta1=$eta; seed="1"}
    }
}

Write-Host "============================================"
Write-Host "  $($runs.Count) runs, MAX_JOB=$MAX_JOB"
Write-Host "============================================"

# ── Hàm chạy song song ──
$jobs = @()
$done = 0

foreach ($r in $runs) {
    # Bỏ qua nếu đã có summary.json
    $outdir = "results\runs\ablation\$($r.name)"
    if (Test-Path "$outdir\summary.json") {
        Write-Host "[skip] $($r.name) — already done"
        $done++
        continue
    }

    # Chờ nếu đã đủ MAX_JOB
    while ($jobs.Count -ge $MAX_JOB) {
        $finished = $jobs | Where-Object { $_.HasExited }
        if ($finished) {
            foreach ($f in $finished) {
                $jobs = @($jobs | Where-Object { $_.Id -ne $f.Id })
                $done++
                Write-Host "[done $done/$($runs.Count)] PID $($f.Id) exited with code $($f.ExitCode)"
            }
        } else {
            Start-Sleep -Seconds 5
        }
    }

    $cmd = "python train.py --optimizer fw --dataset cifar100 --model resnet18 " +
           "--schedule $($r.schedule) --eta1 $($r.eta1) --alpha 0.5001 " +
           "--radius 5 --radius-mode rel --block module " +
           "--epochs 20 --batch-size 128 --seed $($r.seed) " +
           "--run-name $($r.name) --out-dir results/runs/ablation"

    Write-Host "[launch] $($r.name)"
    $proc = Start-Process -FilePath python -ArgumentList (
        "train.py",
        "--optimizer", "fw",
        "--dataset", "cifar100",
        "--model", "resnet18",
        "--schedule", $r.schedule,
        "--eta1", $r.eta1,
        "--alpha", "0.5001",
        "--radius", "5",
        "--radius-mode", "rel",
        "--block", "module",
        "--epochs", "20",
        "--batch-size", "128",
        "--seed", $r.seed,
        "--run-name", $r.name,
        "--out-dir", "results/runs/ablation"
    ) -PassThru -NoNewWindow -RedirectStandardOutput "$outdir.stdout.txt" -RedirectStandardError "$outdir.stderr.txt"

    $jobs += $proc
}

# Chờ các job cuối cùng
while ($jobs.Count -gt 0) {
    $finished = $jobs | Where-Object { $_.HasExited }
    if ($finished) {
        foreach ($f in $finished) {
            $jobs = @($jobs | Where-Object { $_.Id -ne $f.Id })
            $done++
            Write-Host "[done $done/$($runs.Count)] PID $($f.Id) exited with code $($f.ExitCode)"
        }
    } else {
        Start-Sleep -Seconds 5
    }
}

Write-Host ""
Write-Host "============================================"
Write-Host "  ALL $($runs.Count) RUNS FINISHED"
Write-Host "============================================"

# ── In kết quả seed 1/2/3 ──
Write-Host ""
Write-Host "=== eval_acc_final cho cifar100 / power / eta1=0.01 / R=5 ==="
foreach ($s in @(1, 2, 3)) {
    $f = 'results\runs\ablation\cifar100__resnet18__fw__eta10.01_radius5_schedulepower__e20__s' + $s + '\summary.json'
    if (Test-Path $f) {
        $j = Get-Content $f | ConvertFrom-Json
        Write-Host ('  seed ' + $s + ': eval_acc_final = ' + $j.eval_acc_final + '%')
    } else {
        Write-Host ('  seed ' + $s + ': NOT FOUND')
    }
}
