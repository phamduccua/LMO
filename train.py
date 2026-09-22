"""CIFAR classification with a swappable optimizer -- the CV twin of `mappo.py`.

Single file, one `Args` dataclass parsed by tyro, everything visible in the
loop.  Backbones are taken OFF THE SHELF and left UNTRAINED (torchvision
ResNet18 / ResNet34, pytorchcv WRN-28-10; see `models/__init__.py`), datasets
are CIFAR-10 / CIFAR-100.  What varies between runs is only the optimizer:

    fw       StochasticFrankWolfe (Hough 2026, stochastic version)
    fw_adam  same, but s_k = LMO(F_Adam): Adam transforms F_hat before the oracle
    gda pgd ogda eg la_gda sgdm adam oadam eadam la_adam    (optimizers/__init__.py)

HOW THE VI OF THE PAPER MAPS ONTO SUPERVISED TRAINING
    theta   = all weights of the network
    C       = prod_g C_g,  C_g = {||theta_g||_2 <= R_g}   one param_group per block
              (`--block all|module|param`; `--radius-mode rel` sets
               R_g = radius * ||theta_g^0||, the only scale that transfers
               across an 11M ResNet18 and a 36M WRN-28-10)
    F_hat_k = grad of the cross-entropy on ONE minibatch = p.grad after backward
    one optimizer step = one FW iteration k

Classification is a MINIMISATION problem, so F = grad L is a gradient field and
the paper's monotonicity assumption does not hold for a deep network.  The run
is therefore outside the theorem in the same way the MARL runs are, and for the
same reason it is still the honest test: the optimizer code path is identical.

BatchNorm running means/variances are buffers, not parameters, so the constraint
never touches them -- only weights and biases live in C.

METRICS.  Every epoch logs the FULL metric set of `metrics.py` on BOTH splits:
loss, top-1 accuracy, error, top-5, ECE, macro-F1, mean confidence, plus the
generalisation gap, throughput, ||theta||, ||F_hat||, and the Frank-Wolfe
diagnostics (gamma_k, the gap estimate, cos(theta, s), cos of consecutive
oracle inputs).  Train metrics are accumulated from the minibatches themselves,
so they cost nothing.

    python train.py --optimizer fw --model resnet18 --dataset cifar10 --radius 2
    python train.py --optimizer fw_adam --model wrn28_10 --dataset cifar100
    python train.py --optimizer sgdm --lr 0.1 --model resnet34 --dataset cifar100
    python train.py --help

Outputs, under `results/runs/<group>/<run_name>/`:
    log.csv        one row per epoch (all metrics above)
    summary.json   config + final metrics (read by run_ablation.py)
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import make_loaders
from metrics import METRIC_KEYS, RunningClassification, epochs_to, evaluate
from models import MODELS, make_groups, make_model, model_source, n_params, param_norm
from optimizers import FW_FAMILY, make_optimizer, n_grad_evals


@dataclass
class Args:
    # ---- experiment ----
    exp_name: str = "cv"
    run_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = False        # cudnn.deterministic: ~15% slower
    cuda: bool = True
    out_dir: str = "results/runs"
    num_threads: int = 4
    amp: bool = True                         # mixed precision autocast
    amp_dtype: str = "auto"                  # auto | bf16 | fp16 | off
                                             # auto = bf16 on Ampere+ (no loss
                                             # scaling needed), else fp16

    # ---- data / model ----
    dataset: str = "cifar10"                 # cifar10 | cifar100
    model: str = "resnet18"                  # resnet18 | resnet34 | wrn28_10
    cifar_stem: bool = True                  # resnet*: 3x3 s1 stem, no max-pool
    data_root: Optional[str] = None
    batch_size: int = 128
    eval_batch_size: int = 512
    workers: int = 4
    val_frac: float = 0.0                    # >0: select on a held-out split

    # ---- schedule ----
    epochs: int = 100
    lr_schedule: str = "cosine"              # cosine | step | none  (baselines only)
    warmup_epochs: float = 0.0
    label_smoothing: float = 0.0

    # ---- optimizer ----
    optimizer: str = "fw"
    lr: float = 0.1                          # baselines only (FW has no lr)
    momentum: float = 0.9                    # sgdm only
    weight_decay: float = 5e-4               # sgdm only
    nesterov: bool = True                    # sgdm only
    max_grad_norm: float = 0.0               # baselines only; 0 = off
    # Frank-Wolfe (Hough 2026)
    radius: float = 2.0                      # R (abs) or R_g/||theta_g^0|| (rel)
    radius_mode: str = "rel"                 # rel | abs
    block: str = "module"                    # all | module | param  -> blocks of C
    lmo: str = "l2"                          # l2 | linf
    schedule: str = "harmonic"               # harmonic | power | constant | mlogm
    eta1: float = 1.0                        # gamma_k = eta1 / (k + k0)^alpha
    alpha: float = 0.5001
    k0: float = 0.0
    project_init: bool = True                # paper: x_0 in C
    fw_momentum: float = 0.0                 # 0 = paper; >0 = averaged F_hat
    fw_diag_every: int = 10                  # cos_prev costs a copy of the net
    # FW-Adam (--optimizer fw_adam)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    # Lookahead (la_gda, la_adam)
    la_k: int = 5
    la_alpha: float = 0.5

    # ---- logging ----
    eval_every: int = 1                      # epochs between evaluation passes
    log_every: int = 1
    save_ckpt: bool = False                  # final weights -> model.pt


# --------------------------------------------------------------------------- #
def lr_factor(args: Args, t: float) -> float:
    """Multiplier on `--lr` at training progress t in [0, 1] (baselines only).

    FW ignores this entirely: its step size is gamma_k from the schedule, which
    is the whole point of the method -- there is no learning rate to decay.
    """
    w = args.warmup_epochs / max(args.epochs, 1)
    if w > 0 and t < w:
        return t / w
    u = (t - w) / max(1e-12, 1.0 - w)
    if args.lr_schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * min(u, 1.0)))
    if args.lr_schedule == "step":                    # the classic CIFAR staircase
        return 0.1 ** sum(u >= s for s in (0.5, 0.75))
    if args.lr_schedule == "none":
        return 1.0
    raise KeyError("unknown lr_schedule %r; have ['cosine', 'step', 'none']"
                   % args.lr_schedule)


def _prefixed(m: Dict[str, float], p: str) -> Dict[str, float]:
    return {"%s_%s" % (p, k): v for k, v in m.items() if k in METRIC_KEYS}


def pick_amp_dtype(name: str, device) -> Optional[torch.dtype]:
    """Which autocast dtype to use; None means full fp32.

    `auto` picks bfloat16 wherever the card supports it (Ampere and newer, e.g.
    RTX 3090 / A100).  bf16 has fp32's exponent range, so it needs no loss
    scaling at all -- which removes a whole class of problems for extragradient,
    the only method here that backpropagates twice per iteration.  Older cards
    (Pascal, Turing) fall back to fp16 + GradScaler.
    """
    if device.type != "cuda" or name == "off":
        return None
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise KeyError("unknown amp dtype %r; have ['auto', 'bf16', 'fp16', 'off']" % name)


@torch.no_grad()
def unscale_grads_(model: nn.Module, inv_scale: float) -> bool:
    """Divide every gradient by the loss scale; False if any is inf/nan.

    The manual half of `GradScaler` -- used only for extragradient's FIRST
    gradient evaluation, where the scaler's own `unscale_` is not available yet
    (it is reserved for the second one, which feeds `step()`).
    """
    finite = True
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(inv_scale)
            finite &= bool(torch.isfinite(p.grad).all())
    return finite


# --------------------------------------------------------------------------- #
def main(args: Args) -> dict:
    run_name = args.run_name or "%s__%s__%s__%s__s%d__%d" % (
        args.dataset, args.model, args.exp_name, args.optimizer, args.seed,
        int(time.time()))
    out = os.path.join(args.out_dir, run_name)
    os.makedirs(out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic
    torch.set_num_threads(args.num_threads)
    # Fail fast instead of falling back.  A silent drop to CPU does not break a
    # run, it just makes it 30-50x slower, so a whole protocol can finish on the
    # wrong device and nobody notices until the wall-clock is compared against
    # SEC_PER_EPOCH.  Asking for the CPU is fine -- it just has to be explicit.
    if args.cuda and not torch.cuda.is_available():
        raise SystemExit(
            "torch.cuda.is_available() is False: this run would silently train on "
            "the CPU (30-50x slower).\n"
            "  torch %s, cuda build %s\n"
            "  -> install a CUDA build of torch, or pass --no-cuda if you really "
            "mean to use the CPU." % (torch.__version__, torch.version.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    amp_dtype = pick_amp_dtype(args.amp_dtype if args.amp else "off", device)
    amp = amp_dtype is not None
    use_scaler = amp and amp_dtype is torch.float16

    train_loader, eval_loader, n_classes = make_loaders(
        args.dataset, root=args.data_root, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, workers=args.workers,
        seed=args.seed, val_frac=args.val_frac)
    eval_split = "val" if args.val_frac > 0 else "test"

    model = make_model(args.model, n_classes, cifar_stem=args.cifar_stem).to(device)
    model = model.to(memory_format=torch.channels_last)

    groups = make_groups(model, args.block, args.radius, args.radius_mode)
    theta0_norm = param_norm(model.parameters())
    is_fw = args.optimizer in FW_FAMILY
    opt = make_optimizer(
        args.optimizer, groups, lr=args.lr, radius=args.radius, lmo=args.lmo,
        schedule=args.schedule, eta1=args.eta1, alpha=args.alpha, k0=args.k0,
        project_init=args.project_init, fw_momentum=args.fw_momentum,
        adam_betas=(args.adam_beta1, args.adam_beta2), adam_eps=args.adam_eps,
        la_k=args.la_k, la_alpha=args.la_alpha, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=args.nesterov,
        diag_every=args.fw_diag_every)
    evals_per_step = n_grad_evals(opt)
    is_eg = hasattr(opt, "extrapolation")
    # FW on the L2 ball reads only the DIRECTION of F_hat, so clipping by norm
    # would be a no-op there; it stays a baseline-only knob.
    use_clip = (not is_fw) and args.max_grad_norm > 0
    # fp16 needs loss scaling; bf16 does not (same exponent range as fp32)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    print("%s\n  %s (%s, untrained) %.2fM params | %s | %d classes | opt=%s | "
          "%d blocks of C | %s | amp=%s"
          % (run_name, args.model, model_source(args.model), n_params(model) / 1e6,
             args.dataset, n_classes, args.optimizer, len(groups), device,
             str(amp_dtype).replace("torch.", "") if amp else "off"),
          flush=True)
    if is_fw:
        radii = [float(g["lmo"].R) for g in opt.param_groups]
        print("  C: %s ball, R in [%.3g, %.3g] (%s, radius=%g), "
              "||theta_0|| %.2f -> %.2f after projection"
              % (args.lmo, min(radii), max(radii), args.radius_mode, args.radius,
                 theta0_norm, param_norm(model.parameters())), flush=True)

    # one row per epoch: train_* and eval_* carry the full metric set
    fields = (["epoch", "global_step", "grad_evals", "time_s", "epoch_s",
               "img_per_s", "lr"]
              + ["train_%s" % k for k in METRIC_KEYS]
              + ["eval_%s" % k for k in METRIC_KEYS]
              + ["gen_gap", "theta_norm", "grad_norm", "update_norm", "gamma",
                 "fw_gap", "gap_rel", "cos_ts", "cos_prev", "cos_adam", "in_C"])
    fh = open(os.path.join(out, "log.csv"), "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=fields)
    wr.writeheader()

    t0 = time.time()
    global_step, grad_evals, diverged = 0, 0, False
    steps_per_epoch = max(1, len(train_loader))
    best = {"acc": -1.0, "epoch": 0}
    acc_hist: List[float] = []
    rows: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        run = RunningClassification(n_classes, device=device)
        stats: Dict[str, List[float]] = {k: [] for k in
                                         ("gamma", "fw_gap", "gap_rel", "grad_norm",
                                          "cos_ts", "cos_prev", "cos_adam",
                                          "update_norm")}
        cur_lr = float("nan")
        for it, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            if not is_fw:                       # FW's step size is gamma_k, not lr
                cur_lr = args.lr * lr_factor(
                    args, (epoch - 1 + it / steps_per_epoch) / max(args.epochs, 1))
                for g in opt.param_groups:
                    g["lr"] = cur_lr

            def forward():
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                    o = model(x)
                    return crit(o, y), o

            def clip():
                if use_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            th_before = param_norm(model.parameters()) if it % 50 == 0 else None
            opt.zero_grad(set_to_none=True)
            loss, logits = forward()
            scaler.scale(loss).backward()

            if not is_eg:
                scaler.unscale_(opt)            # p.grad IS F_hat_k, in true units
                clip()
                scaler.step(opt)                # skipped by the scaler on inf/nan
                scaler.update()
            else:
                # Extragradient needs TWO gradient evaluations per iteration, but
                # `scaler.unscale_` may be called at most once per optimizer per
                # update.  So the first pass is unscaled by hand (and checked for
                # overflow), and the scaler's own bookkeeping is spent on the
                # second -- the one that actually feeds `step()`.
                ok = unscale_grads_(model, 1.0 / scaler.get_scale()) if use_scaler \
                    else True
                if not ok:                      # fp16 overflow: skip the iteration
                    scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                else:
                    clip()
                    opt.extrapolation()         # on the SAME minibatch (common RN)
                    opt.zero_grad(set_to_none=True)
                    loss2, _ = forward()
                    scaler.scale(loss2).backward()
                    scaler.unscale_(opt)
                    clip()
                    scaler.step(opt)
                    scaler.update()
                    if getattr(opt, "_extrapolated", False):
                        # the scaler skipped step(), so theta is still at v_k
                        opt.restore()
            if th_before is not None:
                stats["update_norm"].append(
                    abs(param_norm(model.parameters()) - th_before))
            global_step += 1
            grad_evals += evals_per_step

            run.update(logits, y, loss=float(loss))
            info = getattr(opt, "last_info", {}) or {}
            if info:
                stats["gamma"].append(info["gamma"])
                stats["fw_gap"].append(info["fw_gap"])
                stats["gap_rel"].append(info["gap_rel"])
                for key in ("grad_norm", "cos_ts", "cos_prev", "cos_adam"):
                    v = [w for k, w in info.items()
                         if k.startswith(key + "/") and np.isfinite(w)]
                    if v:
                        stats[key].append(float(np.mean(v)))

        tr = run.compute()
        th = param_norm(model.parameters())
        if not np.isfinite(th) or th > 1e6 or not np.isfinite(tr["loss"]):
            diverged = True

        do_eval = (epoch % args.eval_every == 0 or epoch in (1, args.epochs)
                   or diverged)
        ev = (evaluate(model, eval_loader, device, amp, n_classes,
                       dtype=amp_dtype) if do_eval
              else {k: float("nan") for k in METRIC_KEYS})
        if do_eval and np.isfinite(ev["acc"]):
            acc_hist.append(ev["acc"])
            if ev["acc"] > best["acc"]:
                best = dict(ev, epoch=epoch)

        def m(key):
            v = stats[key]
            return float(np.mean(v)) if v else float("nan")

        ep_s = time.time() - t_ep
        row = dict(epoch=epoch, global_step=global_step, grad_evals=grad_evals,
                   time_s=round(time.time() - t0, 2), epoch_s=round(ep_s, 2),
                   img_per_s=round(run.n / max(ep_s, 1e-9)), lr=cur_lr,
                   gen_gap=tr["acc"] - ev["acc"], theta_norm=th,
                   grad_norm=m("grad_norm"), update_norm=m("update_norm"),
                   gamma=stats["gamma"][-1] if stats["gamma"] else float("nan"),
                   fw_gap=m("fw_gap"), gap_rel=m("gap_rel"), cos_ts=m("cos_ts"),
                   cos_prev=m("cos_prev"), cos_adam=m("cos_adam"),
                   in_C=float(opt.in_C()) if is_fw else float("nan"),
                   **_prefixed(tr, "train"), **_prefixed(ev, "eval"))
        wr.writerow(row)
        rows.append(row)
        if epoch % args.log_every == 0 or epoch == args.epochs:
            fh.flush()
            print("[%s] ep %3d/%d | train loss %.4f acc %5.2f%% top5 %5.2f%% | "
                  "%s loss %.4f acc %5.2f%% top5 %5.2f%% ece %.2f (best %5.2f%%) | "
                  "gap %5.2f | |th| %7.1f | %s | %.0fs"
                  % (args.optimizer, epoch, args.epochs, tr["loss"], tr["acc"],
                     tr["top5"], eval_split, ev["loss"], ev["acc"], ev["top5"],
                     ev["ece"], best["acc"], row["gen_gap"], th,
                     ("gamma %.2e gap_rel %.3f" % (row["gamma"], row["gap_rel"]))
                     if is_fw else ("lr %.4g" % cur_lr), row["time_s"]), flush=True)
        if diverged:
            print("diverged at epoch %d" % epoch, flush=True)
            break
    fh.close()

    final = evaluate(model, eval_loader, device, amp, n_classes,
                     return_per_class=True, dtype=amp_dtype)
    per_class = final.pop("per_class_acc")
    n_last = max(1, len(acc_hist) // 10)
    summary = dict(
        config=asdict(args), run_name=run_name, diverged=diverged,
        eval_split=eval_split, n_classes=n_classes, n_params=n_params(model),
        model_source=model_source(args.model), pretrained=False,
        epochs_done=epoch, global_step=global_step, grad_evals=grad_evals,
        time_s=time.time() - t0,
        # ---- headline metrics (higher is better except *loss / *err / ece) ----
        eval_acc_final=final["acc"], eval_acc_best=best["acc"],
        eval_acc_best_epoch=best["epoch"],
        eval_acc_last10=float(np.mean(acc_hist[-n_last:])) if acc_hist else float("nan"),
        eval_err_final=final["err"], eval_top5_final=final["top5"],
        eval_loss_final=final["loss"], eval_ece_final=final["ece"],
        eval_macro_f1_final=final["macro_f1"], eval_conf_final=final["conf"],
        train_acc_final=rows[-1]["train_acc"], train_loss_final=rows[-1]["train_loss"],
        train_top5_final=rows[-1]["train_top5"],
        train_ece_final=rows[-1]["train_ece"],
        train_macro_f1_final=rows[-1]["train_macro_f1"],
        gen_gap_final=rows[-1]["train_acc"] - final["acc"],
        best_train_loss=float(np.nanmin([r["train_loss"] for r in rows])),
        # ---- convergence speed ----
        **epochs_to(acc_hist, args.dataset),
        img_per_s=float(np.median([r["img_per_s"] for r in rows])),
        sec_per_epoch=float(np.median([r["epoch_s"] for r in rows])),
        # ---- geometry / FW diagnostics ----
        theta0_norm=theta0_norm, theta_final_norm=param_norm(model.parameters()),
        block_radii=[float(g["lmo"].R) for g in opt.param_groups] if is_fw else None,
        n_blocks=len(groups), fw_k=getattr(opt, "k", None),
        in_C=bool(opt.in_C()) if is_fw else None,
        per_class_acc=per_class,
        acc_history=acc_hist,
    )
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if args.save_ckpt:
        torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    print("final: %s acc %.2f%% (top5 %.2f, err %.2f, ece %.2f, macroF1 %.2f) | "
          "best %.2f%% @ep%d | train acc %.2f%% loss %.4f | gap %.2f | %.0fs"
          % (eval_split, final["acc"], final["top5"], final["err"], final["ece"],
             final["macro_f1"], best["acc"], best["epoch"],
             summary["train_acc_final"], summary["train_loss_final"],
             summary["gen_gap_final"], summary["time_s"]), flush=True)
    return summary


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
