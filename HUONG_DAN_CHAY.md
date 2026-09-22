# Hướng dẫn chạy thực nghiệm CODE_CV (máy RTX 3090)

Toàn bộ quy trình: cài thư viện → tải CIFAR-10 + CIFAR-100 → kiểm tra đúng đắn → chạy thực nghiệm
→ sinh bảng/hình → dựng báo cáo LaTeX.

Mọi lệnh chạy từ thư mục `CODE_CV`:

```bash
cd D:\Flank_Wolfe\CODE_CV        # Windows
cd /path/to/Flank_Wolfe/CODE_CV  # Linux
```

---

## 1. Cài thư viện

**Bước 1 — cài PyTorch bản CUDA** (RTX 3090 = Ampere sm_86, cần CUDA ≥ 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Bước 2 — cài phần còn lại:**

```bash
pip install -r requirements.txt
```

`requirements.txt` gồm: `torch`, `torchvision`, **`pytorchcv`** (nguồn của WRN-28-10 — torchvision
không có kiến trúc này), `numpy`, `tyro`, `pandas`, `matplotlib`.

**Bước 3 — kiểm tra GPU nhận đúng:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Phải in ra `True` và `NVIDIA GeForce RTX 3090`.

---

## 2. Tải 2 bộ dữ liệu (một lần duy nhất)

```bash
python data.py
```

Lệnh này tải **CIFAR-10** (170 MB) và **CIFAR-100** (169 MB) vào `CODE_CV/data/` rồi in
`CIFAR-10 + CIFAR-100 ready in ...`. Sau đó mọi lần chạy đều dùng lại, không tải lại nữa.

Muốn để dữ liệu ở nơi khác (ví dụ ổ SSD chung cho nhiều dự án):

```bash
export CIFAR_ROOT=/data/cifar          # Linux
$env:CIFAR_ROOT = "D:\data\cifar"      # PowerShell
python data.py
# hoặc truyền trực tiếp:  python train.py --data-root /data/cifar ...
```

Kiểm tra đã có đủ:

```bash
python -c "from data import make_loaders; [print(d, make_loaders(d, workers=0)[2], 'classes') for d in ('cifar10','cifar100')]"
```

---

## 3. Kiểm tra đúng đắn trước khi tốn giờ GPU (~3 phút, chạy CPU)

```bash
python sanity_check.py
```

Phải kết thúc bằng `all checks passed`. Bộ kiểm tra này xác nhận: LMO đúng là nghiệm của
`argmin <F,s>`, các schedule thoả điều kiện của định lý, **θ_k luôn nằm trong C ở mọi bước**
(tuyên bố projection-free), FW hội tụ đúng trên bài toán lồi có định lý, và cả 12 optimizer đều
lái được ResNet-18 thật mà không NaN.

**Đo tốc độ máy bạn** (quyết định chọn kế hoạch nào ở mục 4):

```bash
python bench_speed.py --steps 30
```

In ra `s/step` và `min/epoch` cho từng (model × optimizer × amp × channels_last). Trên 3090 dự kiến
khoảng: ResNet-18 ~0.4 phút/epoch, ResNet-34 ~0.7, WRN-28-10 ~2.0 phút/epoch.

> **Lưu ý:** số ước lượng thời gian ở mục 4 dựa trên giả định này. Hãy chạy `bench_speed.py` trước
> và nhân lại theo số đo thật của máy bạn.

---

## 4. Chạy thực nghiệm

Có sẵn script cho cả hai hệ điều hành, chọn kế hoạch bằng tham số:

```bash
bash scripts/run_experiments.sh quick        # Linux / Git Bash
powershell -File scripts\run_experiments.ps1 quick   # Windows
```

| Kế hoạch | Nội dung | Ước tính trên 3090 |
|---|---|---|
| `quick` | ResNet-18 × 2 dataset, 1 seed, 20 epoch, lưới rút gọn (R∈{1,2,5}, 4 baseline) | **~6 giờ** |
| `standard` | **cả 6 ô** (3 model × 2 dataset), lưới rút gọn; ResNet 3 seed/40 epoch, WRN-28-10 1 seed/30 epoch | **~4 ngày** |
| `full` | cả 6 ô, lưới đầy đủ (R∈{0.5,1,2,3,5,10}, 10 baseline), 3 seed, 50 epoch, thêm stage `block` và `eta` | **~2–3 tuần** |

Khuyến nghị: chạy `quick` trước (6 giờ) để có bảng/hình đầu tiên và kiểm tra mọi thứ thông suốt,
rồi mới chạy `standard`.

### Biết chính xác chi phí TRƯỚC khi chạy

```bash
python run_ablation.py all --dataset cifar10 --model wrn28_10 --grid quick --seeds 3 --epochs 40 --plan
```

`--plan` đếm số run từng stage và quy ra số giờ, **không chạy gì cả**:

```
stage        runs     epochs     hours @110s/ep
tune         12       120        3.7
ablation     18       720        22.0
baselines    12       480        14.7
fwadam       18       720        22.0
TOTAL        60                  62.4
```

Hệ số giây/epoch lấy từ `SEC_PER_EPOCH` trong `run_ablation.py` (đo trên 3090). Chạy
`python bench_speed.py` trên máy bạn rồi sửa lại ba con số đó nếu muốn ước tính chính xác hơn.
Các script `run_experiments.*` tự in bảng `--plan` của từng ô trước khi chạy ô đó.

### Mọi stage đều **resume được**

Script bỏ qua mọi run đã có `summary.json`. Tắt máy giữa chừng, chạy lại đúng lệnh cũ → nó chạy tiếp
phần còn thiếu, không làm lại từ đầu.

### Chạy tay từng phần (nếu muốn kiểm soát chi tiết)

```bash
# một ô = một (dataset, model)
python run_ablation.py tune      --dataset cifar10 --model resnet18 --tune-epochs 10
python run_ablation.py ablation  --dataset cifar10 --model resnet18 --epochs 50 --seeds 3
python run_ablation.py baselines --dataset cifar10 --model resnet18 --epochs 50 --seeds 3
python run_ablation.py fwadam    --dataset cifar10 --model resnet18 --epochs 50 --seeds 3
python run_ablation.py report    --dataset cifar10 --model resnet18 --seeds 3

# hoặc tất cả các stage của một ô
python run_ablation.py all --dataset cifar100 --model wrn28_10 --epochs 50 --seeds 3

# hoặc quét cả 6 ô
python run_ablation.py sweep --stages ablation baselines report --epochs 50 --seeds 3
```

Các cờ hữu ích:

| Cờ | Ý nghĩa |
|---|---|
| `--epochs 50` | số epoch mỗi run |
| `--seeds 3` | seed 1..3 (seed dò lr là 100, tách hẳn) |
| `--workers 2` | **số run chạy song song**; 3090 24GB chạy được 2–3 run ResNet-18 cùng lúc, WRN-28-10 nên để 1 |
| `--data-workers 8` | số worker của DataLoader mỗi run |
| `--batch-size 128` | 3090 có thể tăng lên 256 (nhớ ghi vào báo cáo vì nó đổi số bước FW mỗi epoch) |
| `--radii 1 2 5` | tập R cho stage ablation |
| `--grid quick` | lưới rút gọn: R∈{1,2,5}, 4 baseline (gda/pgd/sgdm/adam) × 3 lr, bỏ phần quét `eta1` của FW-Adam |
| `--amp-dtype auto` | `auto` = **bf16** trên Ampere trở lên (3090 ✓) — không cần loss scaling; `fp16` dùng `GradScaler`; `--no-amp` chạy fp32 |
| `--plan` | chỉ in số run + số giờ ước tính rồi thoát, không chạy |
| `--lr-from-model resnet18 --lr-from-dataset cifar10` | dùng lại lr đã dò của ô khác, đỡ phải dò lại |

### Một run đơn lẻ

```bash
python train.py --optimizer fw      --model resnet18 --dataset cifar10  --radius 2 --epochs 50
python train.py --optimizer fw_adam --model wrn28_10 --dataset cifar100 --radius 2 --epochs 50
python train.py --optimizer sgdm --lr 0.1 --model resnet34 --dataset cifar100 --epochs 50
python train.py --help                  # xem toàn bộ cờ
```

---

## 5. Kết quả nằm ở đâu

```
results/
  runs/<stage>/<ten_run>/log.csv        # 1 dòng / epoch: train+test loss, acc, top5, ECE,
  runs/<stage>/<ten_run>/summary.json   #   macro-F1, gen_gap, gamma_k, fw_gap, |theta|, ...
  runs/<stage>/<ten_run>/stdout.txt
  tables_<dataset>_<model>.txt          # bảng A..H dạng text (stage report)
  figures/<dataset>_<model>/*.png       # curves.png, acc_vs_R.png, diagnostics.png
```

Xem nhanh một run:

```bash
python -c "import json;d=json.load(open('results/runs/baselines/cifar10__resnet18__sgdm__lr0.1__s1/summary.json'));print(d['eval_acc_final'], d['train_acc_final'], d['sec_per_epoch'])"
```

---

## 6. Sinh bảng LaTeX và dựng báo cáo

```bash
python make_paper_tables.py --seeds 1 2 3     # -> ../paper_cv/tables/*.tex
cd ../paper_cv
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`make_paper_tables.py` đọc thẳng `summary.json`, nên **mọi con số trong báo cáo đều sinh tự động từ
kết quả chạy** — văn bản và thí nghiệm không thể lệch nhau. Thống kê: trung vị [IQR] khi có ≥3 seed,
trung bình ± độ lệch chuẩn khi có 2 seed, giá trị trần khi chỉ có 1 seed.

---

## 7. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `Unsupported model: wrn28_10_cifar10` | thiếu `pytorchcv`: `pip install pytorchcv` |
| CUDA out of memory (WRN-28-10) | giảm `--batch-size 96`, hoặc `--workers 1` (giảm số run song song) |
| Chạy rất chậm, GPU ~30% | tăng `--data-workers 8`, hoặc để data trên SSD (`CIFAR_ROOT`) |
| DataLoader treo trên Windows | chạy `--data-workers 0` (chậm hơn nhưng luôn chạy được) |
| `error code: <1455>` / `The paging file is too small` (Windows) | Hết bộ nhớ ảo: giảm `--data-workers` (ví dụ 2), giảm `--workers` xuống 1, hoặc tăng paging file (System → Advanced → Performance → Virtual memory) lên ≥ 32 GB |
| Muốn chạy lại sạch một stage | xoá thư mục `results/runs/<stage>/` rồi chạy lại |
| Kết quả không tái lập bit-by-bit | đặt `--torch-deterministic` (chậm ~15%); mặc định bật `cudnn.benchmark` |
