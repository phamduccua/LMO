# Stochastic Frank-Wolfe VI optimizer for Computer Vision (CIFAR-10/100, off-the-shelf backbones)

Evaluation of the **Stochastic Frank-Wolfe** optimizer for variational inequalities (VI) on
supervised image classification, using **off-the-shelf backbones from libraries, randomly
initialised** (the goal is to compare the **convergence speed of optimizers**, not to fine-tune
pretrained weights). Every method — FW, FW-Adam, GDA, PGD, OGDA, EG, Lookahead, Adam, SGDM — runs
inside the same training loop, on the same data, with the same seeds and the same set of metrics.

> **First time running the experiments?** Read [`HUONG_DAN_CHAY.md`](HUONG_DAN_CHAY.md) — a
> step-by-step guide (in Vietnamese): installing the libraries, downloading the data, estimating
> runtime, the three run plans (`quick` / `standard` / `full`) via `scripts/run_experiments.sh`
> (or `.ps1`), and how to build the LaTeX report.

```
pip install -r requirements.txt                          # install torch separately for your CUDA build
python sanity_check.py                                   # tier 1: correctness checks (~3 min, CPU)
python data.py                                           # download CIFAR-10 + CIFAR-100 once
python bench_speed.py                                    # measure seconds/epoch on your GPU
python train.py --optimizer fw --model resnet18 --dataset cifar10 --radius 2
python train.py --optimizer fw_adam --model wrn28_10 --dataset cifar100
python train.py --optimizer sgdm --lr 0.1 --model resnet34 --dataset cifar100
python run_ablation.py all --dataset cifar10 --model resnet18 --seeds 3 --epochs 50
python run_ablation.py sweep --stages ablation baselines report   # all 6 cells (dataset x model)
python make_paper_tables.py                              # LaTeX tables -> ../paper_cv/tables
```

Requirements: Python 3.10+, torch, torchvision, **pytorchcv** (for WRN-28-10), numpy, tyro, pandas,
matplotlib.

## 1. Backbones: off-the-shelf, not reimplemented, never pretrained

| model | source | parameters (C10 / C100) |
|---|---|---|
| `resnet18` | `torchvision.models.resnet18(weights=None)` | 11.17M / 11.22M |
| `resnet34` | `torchvision.models.resnet34(weights=None)` | 21.28M / 21.33M |
| `wrn28_10` | `pytorchcv` → `wrn28_10_cifar{10,100}`, `pretrained=False` | 36.48M / 36.54M |

* **Why is WRN taken from `pytorchcv`?** torchvision **does not have** WRN-28-10; its
  `wide_resnet50_2` / `wide_resnet101_2` are bottleneck networks for ImageNet (69M / 127M
  parameters), a completely different architecture. `pytorchcv` (osmr/imgclsmob) provides exactly
  the standard WRN-28-10 of Zagoruyko & Komodakis (2016).
* **Why must the ResNet stem be changed?** torchvision's ResNet is the ImageNet variant: a 7×7
  stride-2 stem plus max-pool, which shrinks a 32×32 image down to 8×8 before the first block
  (costing ~5 accuracy points). `--cifar-stem` (on by default) replaces it with 3×3 stride 1 and
  drops the max-pool — exactly the variant behind every CIFAR number in the literature.
* **Always `pretrained=False`.** Loading ImageNet/CIFAR weights would start every method close to a
  solution, so we would be measuring fine-tuning rather than convergence.

## 2. Layout

| File | Contents |
|---|---|
| `optimizers/frank_wolfe.py` | **`StochasticFrankWolfe`** — each `param_group` is one block of C |
| `optimizers/lmo.py` | LMO: `L2Ball` (s = −R F̂/‖F̂‖), `LinfBall`, `Simplex` |
| `optimizers/schedules.py` | γ_k: `harmonic`, `power` (inside the theorem); `constant`, `mlogm` (ablation) |
| `optimizers/gda.py` | Baseline 0: **GDA** (= plain SGD) and **PGD** (projection onto FW's exact set C) |
| `optimizers/ogda.py`, `extragradient.py`, `lookahead.py` | OGDA / EG / Lookahead (+ Adam variants) |
| `optimizers/__init__.py` | `make_optimizer(...)` — one factory for every method, plus `sgdm` |
| `models/__init__.py` | the 3 off-the-shelf backbones + `make_groups` (the blocks of C) |
| `data.py` | CIFAR-10/100, standard augmentation, fixed val split (seed 0) for lr tuning |
| `metrics.py` | loss, acc, err, top-5, ECE, macro-F1, confidence — for **both train and test** |
| `train.py` | Single-file training loop; `--optimizer` selects the optimizer |
| `sanity_check.py` | Tier 1: LMO, schedules, the θ_k ∈ C invariant, a convex problem covered by the theorem, plumbing |
| `run_ablation.py` | Protocol: lr tuning → γ×R ablation → baselines × seeds → tables + figures |
| `make_paper_tables.py` | Generates LaTeX tables + copies figures for `../paper_cv` straight from `summary.json` |
| `bench_speed.py` | Measures seconds/step and minutes/epoch per (model × optimizer × amp × channels_last) |
| `scripts/run_experiments.{sh,ps1}` | Runs the whole suite under the `quick` / `standard` / `full` plan, resumable |
| `HUONG_DAN_CHAY.md` | Detailed run guide (setup → data → experiments → report) |
| `results/` | `tables_<dataset>_<model>.txt`, `figures/`, `runs/<group>/<run>/{log.csv,summary.json}` |

## 3. Paper → code mapping

| Paper / note | Code |
|---|---|
| x ∈ C ⊂ ℝⁿ | θ = all weights; C = ∏_g C_g, C_g = {‖θ_g‖₂ ≤ R_g} — one `param_group` per block |
| blocks of C | `--block all` (a single ball) / `module` (each conv/bn/linear layer, default) / `module_nobn` (`module` minus BatchNorm) / `param` (each tensor) |
| radius R | `--radius-mode rel`: R_g = `radius`·‖θ_g⁰‖ — one scale usable for both 11M and 36M parameters |
| F(θ) | cross-entropy gradient of **a single minibatch**; after `loss.backward()`, `p.grad` **is** F̂_k |
| s_k ∈ β(F(x_k)) | `lmo(grads, params)` — L2: s = −R F̂/‖F̂‖ |
| x_{k+1} = x_k + γ_{k+1}(s_k − x_k) | `p.add_(s - p, alpha=gamma)`; no projection is ever needed |
| x₀ ∈ C | `project_init=True`: project θ₀ onto C **once** at initialisation |
| Frank-Wolfe gap V(x) | logs `fw_gap` = ⟨F̂,θ⟩ + R‖F̂‖ and `gap_rel` = V̂/(R‖F̂‖) ∈ [0,2] |
| one iteration k | one minibatch = one optimizer step (391 FW steps per epoch at batch 128) |

BatchNorm running mean/var are **buffers**, not parameters, so the constraint never touches them.

### 3.1 `--block module_nobn` — the constraint covers only conv/linear

`module_nobn` is `module` with every BatchNorm layer removed from the product set: the blocks are
built from the leaf modules, but any `nn.modules.batchnorm._NormBase` (BatchNorm1d/2d/3d,
SyncBatchNorm) is skipped. On `resnet18`/CIFAR-10 that is 41 blocks → **21 blocks**, and the 9 600
BatchNorm affine parameters (γ, β) fall outside C.

**Those parameters are then left out of `param_groups` entirely, so the optimizer never updates
them** — they stay frozen at their initial values (γ = 1, β = 0) for the whole run. BN is therefore
a pure normalisation layer with no learnable affine part; the running mean/var buffers still update
as usual, since they are buffers, not parameters. (This is a stronger statement than "unconstrained":
unconstrained would mean free, here it means fixed.)

Why it exists, and when to use it:

* **Motivation.** A BatchNorm block holds only ~2·C parameters against ~C·C·9 in a conv layer, so
  `R_g = radius·‖θ_g⁰‖` gives it a ball that is tiny in absolute terms but enormous relative to the
  gradient signal — a per-layer L2 ball is a poor model of the constraint for BN. `module_nobn` takes
  those degenerate blocks out instead of tuning around them.
* **Use it** as an ablation: it isolates *"does constraining the BN scale/bias matter?"* from the
  effect of constraining the conv/linear weights, and it is the setting that matches the deep-FW
  papers that constrain weight matrices only.
* **Cost.** Freezing γ, β typically costs a little accuracy relative to `module`; run it against
  `--block module` on the same seeds before reading anything into the numbers.
* **Reporting.** `n_blocks` in `summary.json` and the "`N` blocks of C" line printed at startup count
  the *constrained* blocks only (21, not 41) — ‖θ‖ and `in_C` diagnostics likewise ignore the BN
  parameters.

```bash
python train.py --optimizer fw --model resnet18 --dataset cifar10 --block module_nobn --radius 2
python train.py --optimizer fw --model resnet18 --dataset cifar10 --block module      --radius 2  # control
```

## 4. Metrics — logged every epoch, for **both** splits

`metrics.py` computes the full set for train (accumulated directly from minibatches, no extra
forward pass) and for test/val (one clean pass under `model.eval()`):

`loss`, `acc` (top-1), `err`, `top5`, `ece` (15 bins, Guo et al. 2017), `macro_f1`, `conf`
— plus `gen_gap` = train_acc − test_acc, `theta_norm`, `grad_norm`, `img_per_s`, `epoch_s`, and the
FW diagnostics: `gamma`, `fw_gap`, `gap_rel`, `cos_ts`, `cos_prev`, `cos_adam`, `in_C`.

`summary.json` adds: final / best / last-10% mean accuracy, `ep_to_<target>` (the first epoch that
reaches an accuracy milestone — the **convergence-speed** axis), per-class accuracy, and per-block
radii.

## 5. Deliberate departures from the paper — must be stated in the report

1. **Stochastic operator**: the paper uses the exact F(x_k); here it is F̂_k from a single minibatch.
2. **F is not monotone**: image classification is a **non-convex minimisation** problem; F = ∇L is a
   gradient field but not a monotone operator.
3. **BatchNorm**: buffers lie outside C, so "the iterate stays in C" refers to the parameters only.
4. **AMP**: `GradScaler` is `unscale_`d before `step()`, so the `p.grad` the LMO sees is the true F̂.
5. **`sgdm` is not on the VI scale**: it is the pragmatic CIFAR recipe (momentum 0.9, weight decay
   5e-4, cosine). It still has to be included, because every published ResNet/WRN number comes from
   it; comparing against plain SGD alone would flatter FW.
6. **A correction about `mlogm`**: it is commonly assumed that γ_k ~ 1/(k log k) is **summable** —
   that is wrong. The series ∑1/(m log m) **diverges** (Cauchy condensation), so `mlogm` does satisfy
   all three (FW) conditions. What matters is the **rate**: the partial sums only grow like log log k
   (measured: +1.32 per decade over 1e4→1e5, +1.05 over 1e5→1e6, against exactly +2.30 per decade for
   `harmonic`). See `optimizers/schedules.py` and test 2 in `sanity_check.py`.

## 6. License

Released under the **MIT License** — see [`LICENSE`](LICENSE). The third-party dependencies keep
their own licenses: PyTorch / torchvision (BSD-3-Clause) and `pytorchcv` (MIT); the CIFAR-10/100
datasets are distributed by their authors under their own terms and are not redistributed here.
