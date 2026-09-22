# Chay toan bo thuc nghiem CODE_CV theo ke hoach: quick | standard | full
#
#     powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 quick
#     powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 standard
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
#     python run_ablation.py all --dataset cifar10 --model wrn28_10 `
#            --grid quick --seeds 3 --epochs 40 --plan

param(
    [ValidateSet("quick", "standard", "full")]
    [string]$Plan = "quick",
    [int]$Workers = 0,          # 0 = mac dinh cua ke hoach; chi tang khi GPU chua bao hoa
    [int]$DataWorkers = 8,
    [int]$Batch = 128,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

switch ($Plan) {
    "quick" {
        $Grid = "quick"; $TuneEpochs = 5; $WorkersDefault = 2
        $Models = @("resnet18")
        $Stages = @("tune", "ablation", "baselines", "fwadam", "constlr", "report")
        $SeedsOf  = { param($m) 1 }
        $EpochsOf = { param($m) 20 }
    }
    "standard" {
        $Grid = "quick"; $TuneEpochs = 10; $WorkersDefault = 2
        $Models = @("resnet18", "resnet34", "wrn28_10")
        $Stages = @("tune", "ablation", "baselines", "fwadam", "constlr", "report")
        # WRN-28-10 dat gap ~5.5 lan resnet18 moi epoch -> it seed va it epoch hon
        $SeedsOf  = { param($m) if ($m -eq "wrn28_10") { 1 } else { 3 } }
        $EpochsOf = { param($m) if ($m -eq "wrn28_10") { 30 } else { 40 } }
    }
    "full" {
        $Grid = "full"; $TuneEpochs = 10; $WorkersDefault = 2
        $Models = @("resnet18", "resnet34", "wrn28_10")
        $Stages = @("tune", "ablation", "baselines", "fwadam", "constlr",
                    "block", "eta", "report")
        $SeedsOf  = { param($m) 3 }
        $EpochsOf = { param($m) 50 }
    }
}
if ($Workers -le 0) { $Workers = $WorkersDefault }
$Datasets = @("cifar10", "cifar100")

Write-Host "=============================================================="
Write-Host " ke hoach   : $Plan   (luoi: $Grid)"
Write-Host " model      : $($Models -join ' ')"
Write-Host " dataset    : $($Datasets -join ' ')"
Write-Host " stages     : $($Stages -join ' ')"
Write-Host " song song  : $Workers run   dataloader: $DataWorkers worker   batch: $Batch"
Write-Host "=============================================================="

# --- 0. du lieu + kiem tra dung dan ---------------------------------------- #
& $Python data.py
& $Python sanity_check.py --quick

$Start = Get-Date
foreach ($ds in $Datasets) {
    foreach ($md in $Models) {
        $seeds  = & $SeedsOf $md
        $epochs = & $EpochsOf $md
        # WRN-28-10 chiem nhieu VRAM hon: chay tuan tu
        $w = $Workers
        if ($md -eq "wrn28_10") { $w = 1 }

        Write-Host ""
        Write-Host "##############################################################"
        Write-Host "### $ds / $md   (seeds 1..$seeds, $epochs epoch, $w run song song)"
        Write-Host "##############################################################"
        & $Python run_ablation.py all --dataset $ds --model $md --grid $Grid `
            --seeds $seeds --epochs $epochs --tune-epochs $TuneEpochs `
            --workers $w --stages @Stages --plan

        foreach ($st in $Stages) {
            # chi do lr tren resnet18; cac model lon dung lai ket qua do do
            if ($st -eq "tune" -and $md -ne "resnet18") {
                Write-Host "--- stage tune --- (bo qua: dung lai lr da do tren resnet18/$ds)"
                continue
            }
            $extra = @()
            if ($st -eq "baselines" -and $md -ne "resnet18") {
                $extra = @("--lr-from-model", "resnet18", "--lr-from-dataset", $ds)
            }
            Write-Host "--- stage $st ---"
            & $Python run_ablation.py $st `
                --dataset $ds --model $md --grid $Grid `
                --seeds $seeds --epochs $epochs --tune-epochs $TuneEpochs `
                --workers $w --data-workers $DataWorkers --batch-size $Batch `
                @extra
        }
    }
}

# --- bang LaTeX + hinh cho bao cao ----------------------------------------- #
$maxSeeds = 1
foreach ($md in $Models) { $s = & $SeedsOf $md; if ($s -gt $maxSeeds) { $maxSeeds = $s } }
Write-Host ""
Write-Host "--- sinh bang LaTeX + copy hinh -> ..\paper_cv ---"
& $Python make_paper_tables.py --seeds (1..$maxSeeds)

$El = (Get-Date) - $Start
Write-Host ""
Write-Host ("XONG sau {0}h {1}m" -f [int]$El.TotalHours, $El.Minutes)
Write-Host "  bang text : results\tables_<dataset>_<model>.txt"
Write-Host "  hinh      : results\figures\<dataset>_<model>\"
Write-Host "  bang LaTeX: ..\paper_cv\tables\   hinh: ..\paper_cv\figures\"
Write-Host "  dung bao cao: cd ..\paper_cv; pdflatex main; bibtex main; pdflatex main; pdflatex main"
