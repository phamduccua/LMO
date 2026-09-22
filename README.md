# Stochastic Frank-Wolfe VI optimizer cho Computer Vision (CIFAR-10/100, backbone có sẵn)

Cùng bộ optimizer và cùng giao thức thực nghiệm như `../CODE_MARL`, nhưng thay MAPPO/PettingZoo
bằng bài toán phân loại ảnh có giám sát trên **backbone lấy sẵn từ thư viện, khởi tạo ngẫu nhiên**
(mục tiêu là so sánh **tốc độ hội tụ của optimizer**, không phải fine-tune trọng số pretrained).

> **Chạy thực nghiệm lần đầu?** Đọc [`HUONG_DAN_CHAY.md`](HUONG_DAN_CHAY.md) — hướng dẫn từng bước:
> cài thư viện, tải dữ liệu, ước lượng thời gian, ba kế hoạch chạy (`quick` / `standard` / `full`)
> qua `scripts/run_experiments.sh` (hoặc `.ps1`), và cách dựng báo cáo LaTeX.

```
pip install -r requirements.txt                          # torch cài riêng theo bản CUDA
python sanity_check.py                                   # bậc 1: kiểm tra đúng đắn (~3 phút, CPU)
python data.py                                           # tải CIFAR-10 + CIFAR-100 một lần
python bench_speed.py                                    # đo giây/epoch trên GPU của bạn
python train.py --optimizer fw --model resnet18 --dataset cifar10 --radius 2
python train.py --optimizer fw_adam --model wrn28_10 --dataset cifar100
python train.py --optimizer sgdm --lr 0.1 --model resnet34 --dataset cifar100
python run_ablation.py all --dataset cifar10 --model resnet18 --seeds 3 --epochs 50
python run_ablation.py sweep --stages ablation baselines report   # cả 6 ô (dataset x model)
python make_paper_tables.py                              # bảng LaTeX -> ../paper_cv/tables
```

Yêu cầu: Python 3.10+, torch, torchvision, **pytorchcv** (cho WRN-28-10), numpy, tyro, pandas,
matplotlib.

## 1. Backbone: lấy sẵn, không viết lại, không dùng trọng số pretrained

| model | nguồn | tham số (C10 / C100) |
|---|---|---|
| `resnet18` | `torchvision.models.resnet18(weights=None)` | 11.17M / 11.22M |
| `resnet34` | `torchvision.models.resnet34(weights=None)` | 21.28M / 21.33M |
| `wrn28_10` | `pytorchcv` → `wrn28_10_cifar{10,100}`, `pretrained=False` | 36.48M / 36.54M |

* **Vì sao WRN lấy từ `pytorchcv`?** torchvision **không có** WRN-28-10; `wide_resnet50_2` /
  `wide_resnet101_2` của nó là mạng bottleneck cho ImageNet (69M / 127M tham số), kiến trúc khác hẳn.
  `pytorchcv` (osmr/imgclsmob) có đúng WRN-28-10 chuẩn của Zagoruyko & Komodakis (2016).
* **Vì sao ResNet phải đổi stem?** ResNet của torchvision là bản ImageNet: stem 7×7 stride 2 + max-pool,
  làm ảnh 32×32 co còn 8×8 trước block đầu tiên (mất ~5 điểm accuracy). `--cifar-stem` (mặc định bật)
  thay bằng 3×3 stride 1, bỏ max-pool — đúng biến thể mà mọi số CIFAR trong tài liệu dùng.
* **Luôn `pretrained=False`.** Nạp trọng số ImageNet/CIFAR sẽ khiến mọi phương pháp xuất phát gần
  nghiệm và ta đo fine-tuning chứ không đo hội tụ.

## 2. Cấu trúc

| File | Nội dung |
|---|---|
| `optimizers/frank_wolfe.py` | **`StochasticFrankWolfe`** — mỗi `param_group` là một khối của C |
| `optimizers/lmo.py` | LMO: `L2Ball` (s = −R F̂/‖F̂‖), `LinfBall`, `Simplex` |
| `optimizers/schedules.py` | γ_k: `harmonic`, `power` (trong định lý); `constant`, `mlogm` (ablation) |
| `optimizers/gda.py` | Baseline 0: **GDA** (= SGD thuần) và **PGD** (chiếu lên đúng tập C của FW) |
| `optimizers/ogda.py`, `extragradient.py`, `lookahead.py` | OGDA / EG / Lookahead (+ bản Adam) |
| `optimizers/__init__.py` | `make_optimizer(...)` — một factory cho mọi phương pháp, thêm `sgdm` |
| `models/__init__.py` | 3 backbone có sẵn + `make_groups` (các khối của C) |
| `data.py` | CIFAR-10/100, augmentation chuẩn, tách val cố định (seed 0) để dò lr |
| `metrics.py` | loss, acc, err, top-5, ECE, macro-F1, confidence — cho **cả train lẫn test** |
| `train.py` | Vòng huấn luyện một file, `--optimizer` chọn optimizer |
| `sanity_check.py` | Bậc 1: LMO, schedule, bất biến θ_k ∈ C, bài toán lồi có định lý, plumbing |
| `run_ablation.py` | Giao thức: dò lr → ablation γ×R → baseline × seed → bảng + hình |
| `make_paper_tables.py` | Sinh bảng LaTeX + copy hình cho `../paper_cv` thẳng từ `summary.json` |
| `bench_speed.py` | Đo giây/bước và phút/epoch cho từng (model × optimizer × amp × channels_last) |
| `scripts/run_experiments.{sh,ps1}` | Chạy trọn bộ theo kế hoạch `quick` / `standard` / `full`, resume được |
| `HUONG_DAN_CHAY.md` | Hướng dẫn chạy chi tiết (cài đặt → dữ liệu → thực nghiệm → báo cáo) |
| `results/` | `tables_<dataset>_<model>.txt`, `figures/`, `runs/<group>/<run>/{log.csv,summary.json}` |

## 3. Ánh xạ paper → code

| Paper / note | Code |
|---|---|
| x ∈ C ⊂ ℝⁿ | θ = toàn bộ trọng số; C = ∏_g C_g, C_g = {‖θ_g‖₂ ≤ R_g} — mỗi khối một `param_group` |
| khối của C | `--block all` (một quả cầu duy nhất) / `module` (mỗi lớp conv/bn/linear) / `param` (mỗi tensor) |
| bán kính R | `--radius-mode rel`: R_g = `radius`·‖θ_g⁰‖ — thang đo duy nhất dùng chung được cho 11M và 36M tham số |
| F(θ) | gradient cross-entropy của **một minibatch**; sau `loss.backward()` thì `p.grad` **chính là** F̂_k |
| s_k ∈ β(F(x_k)) | `lmo(grads, params)` — L2: s = −R F̂/‖F̂‖ |
| x_{k+1} = x_k + γ_{k+1}(s_k − x_k) | `p.add_(s - p, alpha=gamma)`; không bao giờ cần phép chiếu |
| x₀ ∈ C | `project_init=True`: chiếu θ₀ lên C **một lần** lúc khởi tạo |
| Frank-Wolfe gap V(x) | log `fw_gap` = ⟨F̂,θ⟩ + R‖F̂‖ và `gap_rel` = V̂/(R‖F̂‖) ∈ [0,2] |
| một vòng lặp k | một minibatch = một bước optimizer (391 bước FW mỗi epoch với batch 128) |

BatchNorm running mean/var là **buffer**, không phải tham số, nên ràng buộc không đụng tới chúng.

## 4. Metrics — ghi mỗi epoch, cho **cả hai** split

`metrics.py` tính đầy đủ cho train (cộng dồn trực tiếp từ minibatch, không tốn thêm forward) và
test/val (một lượt sạch với `model.eval()`):

`loss`, `acc` (top-1), `err`, `top5`, `ece` (15 bin, Guo et al. 2017), `macro_f1`, `conf`
— cộng với `gen_gap` = train_acc − test_acc, `theta_norm`, `grad_norm`, `img_per_s`, `epoch_s`,
và các chẩn đoán FW: `gamma`, `fw_gap`, `gap_rel`, `cos_ts`, `cos_prev`, `cos_adam`, `in_C`.

`summary.json` bổ sung: accuracy cuối / tốt nhất / trung bình 10% cuối, `ep_to_<target>` (epoch đầu
tiên đạt mốc accuracy — trục **tốc độ hội tụ**), accuracy từng lớp, và bán kính từng khối.

## 5. Những chỗ cố ý rời khỏi paper — phải ghi vào báo cáo

1. **Toán tử ngẫu nhiên**: paper dùng F(x_k) chính xác; ở đây là F̂_k từ một minibatch.
2. **F không đơn điệu**: phân loại ảnh là bài toán **cực tiểu hoá không lồi**, F = ∇L là trường
   gradient nhưng không đơn điệu — giống hệt tình trạng ở `CODE_MARL`.
3. **BatchNorm**: buffer không nằm trong C, nên iterate "trong C" chỉ nói về tham số.
4. **AMP**: `GradScaler` được `unscale_` trước `step()`, nên `p.grad` mà LMO nhìn thấy là F̂ thật.
5. **`sgdm` không thuộc thang VI**: nó là công thức thực dụng của CIFAR (momentum 0.9, weight decay
   5e-4, cosine). Vẫn phải có, vì mọi số ResNet/WRN đã công bố đều từ nó; so với SGD thuần sẽ tâng bốc FW.
6. **Đính chính `mlogm`**: docstring ở `CODE_MARL/optimizers/schedules.py` nói γ_k ~ 1/(k log k) là
   **khả tổng** — sai. Chuỗi ∑1/(m log m) **phân kỳ** (Cauchy condensation), nên `mlogm` vẫn thoả cả ba
   điều kiện của (FW). Điều đáng nói là **tốc độ**: tổng riêng chỉ tăng như log log k (đo được: +1.32
   mỗi decade ở 1e4→1e5, +1.05 ở 1e5→1e6, so với đúng +2.30 mỗi decade của `harmonic`). Xem
   `optimizers/schedules.py` và test 2 của `sanity_check.py`.
