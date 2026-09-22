# run_cifar100_power_seeds.ps1  –  2 runs: seed 2,3 cho ô cifar100/power/eta1=0.01/R=5
# Chạy: powershell -ExecutionPolicy Bypass -File scripts\run_cifar100_power_seeds.ps1
# Dừng: Ctrl-C  (chạy lại để tiếp tục từ chỗ dừng — bỏ qua run đã có summary.json)

$MAX_JOB = 3
$RUNS    = "results/runs"
$PY      = "python"

# ── Hàm queue job (song song, tối đa $MAX_JOB) ──
$script:jobs = @()
$script:done = 0
$script:total = 0

function Wait-Slot {
    while ($script:jobs.Count -ge $MAX_JOB) {
        $finished = @($script:jobs | Where-Object { $_.HasExited })
        if ($finished.Count -gt 0) {
            foreach ($f in $finished) {
                $script:done++
                $rc = $f.ExitCode
                $color = if ($rc -eq 0) { "Green" } else { "Red" }
                Write-Host "[DONE $($script:done)/$($script:total)] PID $($f.Id) exit=$rc" -ForegroundColor $color
                $script:jobs = @($script:jobs | Where-Object { $_.Id -ne $f.Id })
            }
        } else {
            Start-Sleep -Seconds 5
        }
    }
}

function Wait-All {
    while ($script:jobs.Count -gt 0) {
        $finished = @($script:jobs | Where-Object { $_.HasExited })
        if ($finished.Count -gt 0) {
            foreach ($f in $finished) {
                $script:done++
                $rc = $f.ExitCode
                $color = if ($rc -eq 0) { "Green" } else { "Red" }
                Write-Host "[DONE $($script:done)/$($script:total)] PID $($f.Id) exit=$rc" -ForegroundColor $color
                $script:jobs = @($script:jobs | Where-Object { $_.Id -ne $f.Id })
            }
        } else {
            Start-Sleep -Seconds 5
        }
    }
}

function Queue-Run {
    param($Task, $Group, $Name, [string[]]$Extra)
    $script:total++
    $dst = "$RUNS/$Group/$Name"
    if (Test-Path "$dst/summary.json") {
        Write-Host "[SKIP] [$Task] $Name" -ForegroundColor Yellow
        $script:done++
        return
    }
    Wait-Slot
    Write-Host "[RUN ] [$Task] $Name  (slot $($script:jobs.Count+1)/$MAX_JOB)" -ForegroundColor Green
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    $logOut = "$dst/stdout.txt"
    $logErr = "$dst/stderr.txt"
    $proc = Start-Process -FilePath $PY -ArgumentList (
        @("train.py",
          "--model", "resnet18", "--epochs", "20", "--radius", "5",
          "--out-dir", "$RUNS/$Group", "--run-name", $Name) + $Extra
    ) -PassThru -NoNewWindow -RedirectStandardOutput $logOut -RedirectStandardError $logErr
    $script:jobs += $proc
}

# ═══════════════════════════════════════════════════════════════════
# 2 runs bắt buộc: cifar100 / power / eta1=0.01 / R=5 / seed 2,3
# Ước tính trên RTX 3090: ~15 phút (2 run song song)
# ═══════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  cifar100 / power / eta1=0.01 / R=5"       -ForegroundColor Cyan
Write-Host "  seed 2, 3  |  MAX_JOB=$MAX_JOB"           -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

foreach ($seed in @(2, 3)) {
    Queue-Run "seed" ablation `
        "cifar100__resnet18__fw__eta10.01_radius5_schedulepower__e20__s$seed" `
        @("--optimizer","fw","--dataset","cifar100","--seed","$seed",
          "--schedule","power","--eta1","0.01","--alpha","0.5001")
}

Wait-All

# ═══════════════════════════════════════════════════════════════════
# In kết quả 3 seed
# ═══════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  eval_acc_final  (cifar100/power/eta1=0.01)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

foreach ($s in @(1, 2, 3)) {
    $f = 'results/runs/ablation/cifar100__resnet18__fw__eta10.01_radius5_schedulepower__e20__s' + $s + '/summary.json'
    if (Test-Path $f) {
        $j = Get-Content $f -Raw | ConvertFrom-Json
        Write-Host ('  seed ' + $s + ':  eval_acc_final = ' + [math]::Round($j.eval_acc_final, 2) + '%') -ForegroundColor White
    } else {
        Write-Host ('  seed ' + $s + ':  NOT FOUND') -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "XONG." -ForegroundColor Green
