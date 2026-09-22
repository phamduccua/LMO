"""Classification metrics -- everything the tables and figures are built on.

The CV twin of `../CODE_MARL/metrics.py`.  Every run logs, EVERY epoch, on BOTH
splits (train and eval), so that a curve can be drawn for any of them:

    loss        mean cross-entropy (the quantity actually minimised)
    acc         top-1 accuracy [%]           <- the headline metric
    err         100 - acc, the error rate reported in the CIFAR literature
    top5        top-5 accuracy [%]           (near-saturated on CIFAR-10;
                                              the informative one on CIFAR-100)
    ece         expected calibration error [%] over 15 equal-width confidence
                bins -- Guo et al. 2017.  Worth logging here because Frank-Wolfe
                keeps ||theta|| bounded, which is a regulariser, and a method
                can buy accuracy while ruining calibration.
    macro_f1    unweighted mean per-class F1 [%] -- on CIFAR-100 a method can
                sit at a decent top-1 while collapsing a handful of classes,
                and macro-F1 is what shows it.
    conf        mean max-softmax confidence [%], the companion of ece

and, derived from the two splits,

    gen_gap     train_acc - eval_acc, the overfitting axis

`train_*` metrics are accumulated ON THE FLY from the training minibatches, so
they cost nothing but are measured under augmentation and under a moving
theta (they are an average over the epoch, not a snapshot).  `eval_*` metrics
are a clean pass over the held-out split with `model.eval()`.

Cheap running versions (`RunningClassification`) are used inside the train loop;
the full versions (`classification_metrics`) need the stacked logits of a whole
split and are used by `evaluate`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

METRIC_KEYS = ("loss", "acc", "err", "top5", "ece", "macro_f1", "conf")


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> int:
    k = min(k, logits.shape[1])
    return int((logits.topk(k, dim=1).indices == target[:, None]).any(1).sum())


def expected_calibration_error(probs: torch.Tensor, target: torch.Tensor,
                               n_bins: int = 15) -> float:
    """ECE over equal-width confidence bins (Guo et al. 2017), in percent."""
    conf, pred = probs.max(1)
    acc = (pred == target).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return 100.0 * float(ece)


def macro_f1(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> float:
    """Unweighted mean per-class F1, in percent (classes never predicted count 0)."""
    f1 = []
    for c in range(n_classes):
        tp = int(((pred == c) & (target == c)).sum())
        fp = int(((pred == c) & (target != c)).sum())
        fn = int(((pred != c) & (target == c)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return 100.0 * float(np.mean(f1))


def per_class_acc(pred: torch.Tensor, target: torch.Tensor,
                  n_classes: int) -> List[float]:
    out = []
    for c in range(n_classes):
        m = target == c
        out.append(100.0 * float((pred[m] == c).float().mean()) if m.any()
                   else float("nan"))
    return out


def classification_metrics(logits: torch.Tensor, target: torch.Tensor,
                           n_classes: Optional[int] = None,
                           label_smoothing: float = 0.0) -> Dict[str, float]:
    """The full metric set from the stacked logits of one split."""
    logits = logits.float()
    n_classes = n_classes or logits.shape[1]
    probs = logits.softmax(1)
    pred = probs.argmax(1)
    acc = 100.0 * float((pred == target).float().mean())
    return {
        "loss": float(F.cross_entropy(logits, target,
                                      label_smoothing=label_smoothing)),
        "acc": acc,
        "err": 100.0 - acc,
        "top5": 100.0 * topk_correct(logits, target, 5) / target.numel(),
        "ece": expected_calibration_error(probs, target),
        "macro_f1": macro_f1(pred, target, n_classes),
        "conf": 100.0 * float(probs.max(1).values.mean()),
    }


class RunningClassification:
    """Streaming train-split metrics: no logits are kept, only sums.

    `ece` and `macro_f1` need the whole split, so on the train split they are
    accumulated from per-batch sufficient statistics: a 15-bin confidence
    histogram for ECE and a per-class TP/FP/FN table for macro-F1.  Both are
    exact for the epoch's sequence of minibatches (i.e. under augmentation and
    a moving theta), which is exactly what a "training metric" means.
    """

    def __init__(self, n_classes: int, n_bins: int = 15, device=None):
        self.n_classes, self.n_bins = n_classes, n_bins
        d = device
        self.loss_sum = 0.0
        self.n = 0
        self.correct = 0
        self.top5 = 0
        self.conf_sum = 0.0
        self.bin_cnt = torch.zeros(n_bins, device=d)
        self.bin_conf = torch.zeros(n_bins, device=d)
        self.bin_acc = torch.zeros(n_bins, device=d)
        self.tp = torch.zeros(n_classes, device=d)
        self.fp = torch.zeros(n_classes, device=d)
        self.fn = torch.zeros(n_classes, device=d)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor,
               loss: Optional[float] = None) -> None:
        logits = logits.detach().float()
        probs = logits.softmax(1)
        conf, pred = probs.max(1)
        ok = (pred == target).float()
        b = target.numel()
        self.n += b
        self.correct += int(ok.sum())
        self.top5 += topk_correct(logits, target, 5)
        self.conf_sum += float(conf.sum())
        self.loss_sum += (float(loss) * b if loss is not None
                          else float(F.cross_entropy(logits, target,
                                                     reduction="sum")))
        idx = torch.clamp((conf * self.n_bins).long(), max=self.n_bins - 1)
        self.bin_cnt.index_add_(0, idx, torch.ones_like(conf))
        self.bin_conf.index_add_(0, idx, conf)
        self.bin_acc.index_add_(0, idx, ok)
        one = torch.ones_like(ok)
        self.tp.index_add_(0, target, ok)
        self.fp.index_add_(0, pred, one - ok)
        self.fn.index_add_(0, target, one - ok)

    def compute(self) -> Dict[str, float]:
        n = max(self.n, 1)
        acc = 100.0 * self.correct / n
        nz = self.bin_cnt > 0
        ece = float(((self.bin_cnt[nz] / n)
                     * (self.bin_acc[nz] / self.bin_cnt[nz]
                        - self.bin_conf[nz] / self.bin_cnt[nz]).abs()).sum())
        den = 2 * self.tp + self.fp + self.fn
        f1 = torch.where(den > 0, 2 * self.tp / den.clamp(min=1),
                         torch.zeros_like(den))
        return {"loss": self.loss_sum / n, "acc": acc, "err": 100.0 - acc,
                "top5": 100.0 * self.top5 / n, "ece": 100.0 * ece,
                "macro_f1": 100.0 * float(f1.mean()),
                "conf": 100.0 * self.conf_sum / n}


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = False,
             n_classes: Optional[int] = None,
             return_per_class: bool = False,
             dtype: Optional["torch.dtype"] = None) -> Dict[str, float]:
    """One clean pass over a split -> the full metric set.

    `dtype` is the autocast dtype of the training loop (bf16 or fp16), so that
    evaluation runs in the same precision the weights were trained in; None
    keeps autocast's default.
    """
    was_training = model.training
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype,
                            enabled=amp and device.type == "cuda"):
            o = model(x)
        logits.append(o.float().cpu())
        targets.append(y.cpu())
    model.train(was_training)
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    m = classification_metrics(logits, targets, n_classes)
    if return_per_class:
        m["per_class_acc"] = per_class_acc(logits.argmax(1), targets,
                                           n_classes or logits.shape[1])
    return m


# Accuracy thresholds used by `epochs_to`: "how many epochs to get to a level
# that a tuned SGD+momentum run reaches comfortably".  Speed, not final quality.
TARGETS = {"cifar10": (80.0, 90.0, 93.0), "cifar100": (50.0, 65.0, 72.0)}


def epochs_to(history: List[float], dataset: str) -> Dict[str, Optional[int]]:
    """First epoch (1-indexed) at which eval accuracy reached each target."""
    out: Dict[str, Optional[int]] = {}
    for t in TARGETS.get(dataset, ()):
        hit = next((i + 1 for i, v in enumerate(history)
                    if np.isfinite(v) and v >= t), None)
        out["ep_to_%g" % t] = hit
    return out
