"""Optimizers for the CV experiments, one file per method, one factory.

Same ladder as `../../CODE_MARL/optimizers`, re-pointed at supervised image
classification (ResNet18 / ResNet34 / WRN-28-10 on CIFAR-10 / CIFAR-100).  The
optimizer code itself is method-generic -- it only ever sees `p.grad` of a
quantity that is being MINIMISED -- so the files are the same implementations;
only the training loop that drives them (`train.py` instead of `mappo.py`) and
the constraint blocks differ.

The method:
    frank_wolfe.py   StochasticFrankWolfe  -- Hough (2026) (FW), stochastic F_hat
                     "fw_adam": same class, adam=True -- s_k = LMO(F_Adam), Adam
                     transforms the gradient before the oracle (extension)
    lmo.py           L2Ball / LinfBall / Simplex linear minimization oracles
    schedules.py     gamma_k: harmonic, power (inside the theorem); constant,
                     mlogm (ablations outside it)

Baselines -- the same ladder, plus the CIFAR practitioner default:
    #  name       F-evals/iter  file               question it answers
    0  gda        1             gda.py             is any structure needed? (= SGD)
    0' pgd        1             gda.py             same set C as FW, by projection
    1  ogda       1             ogda.py            is one (optimistic) eval enough?
    2  eg         2             extragradient.py   do two evals pay for themselves?
    -  la_gda     1             lookahead.py       is weight averaging alone enough?
    -  adam       1             torch              the default of the field
    -  sgdm       1             torch              THE CIFAR recipe: SGD + momentum
                                                   0.9 + weight decay 5e-4 (+ cosine,
                                                   applied by train.py)
       oadam/eadam/la_adam: Adam-preconditioned OGDA / EG / Lookahead

`sgdm` is the honest reference point for this benchmark: every published
ResNet/WRN CIFAR number comes from it, and a comparison against unmomentumed
SGD alone would flatter Frank-Wolfe.  It is NOT part of the VI ladder, which is
why it is listed separately here.

Every optimizer is driven the same way (`loss.backward(); opt.step()`), except
extragradient, which exposes `extrapolation()` and needs a second backward --
the training script checks `hasattr(opt, "extrapolation")`.  The honest cost
axis is therefore gradient evaluations, counted by the training script.

CONSTRAINT BLOCKS.  `make_optimizer` accepts param_group dicts that already
carry their own `radius` (train.py sets one per block when
`--radius-mode rel`, i.e. R_g = radius * ||theta_g^0||).  A group's `radius`
overrides the global one, so C is a product of balls with DIFFERENT radii --
still a product set, still a separable LMO, so nothing in the method changes.
"""

from __future__ import annotations

from typing import Iterable, List

import torch

from .extragradient import Extragradient
from .frank_wolfe import StochasticFrankWolfe
from .gda import GDA
from .lmo import LMO, L2Ball, LinfBall, Simplex, make_lmo
from .lookahead import Lookahead
from .ogda import OGDA
from .schedules import Constant, Harmonic, MLogM, Power, Schedule, make_schedule

__all__ = [
    "StochasticFrankWolfe", "LMO", "L2Ball", "LinfBall", "Simplex", "make_lmo",
    "Schedule", "Harmonic", "Power", "Constant", "MLogM", "make_schedule",
    "GDA", "OGDA", "Extragradient", "Lookahead", "make_optimizer", "OPTIMIZERS",
    "FW_FAMILY", "BASELINE_OPTIMIZERS", "n_grad_evals", "uses_lr",
]

FW_FAMILY = ("fw", "fw_adam")
BASELINE_OPTIMIZERS = ("gda", "pgd", "ogda", "eg", "la_gda", "sgdm",
                       "adam", "oadam", "eadam", "la_adam")
OPTIMIZERS = FW_FAMILY + BASELINE_OPTIMIZERS


def _groups_with_lmo(params: Iterable, lmo: str, radius: float) -> List[dict]:
    """Give every group its own oracle, honouring a per-group `radius` key."""
    out = []
    for g in params:
        g = dict(g) if isinstance(g, dict) else {"params": list(g)}
        R = float(g.pop("radius", radius))
        g.setdefault("lmo", make_lmo(lmo, R))
        out.append(g)
    return out


def make_optimizer(name: str, params: Iterable, *, lr: float = 0.1,
                   radius: float = 10.0, lmo: str = "l2",
                   schedule: str = "harmonic", eta1: float = 1.0,
                   alpha: float = 0.5001, k0: float = 0.0,
                   project_init: bool = True, la_k: int = 5,
                   la_alpha: float = 0.5, fw_momentum: float = 0.0,
                   adam_betas=(0.9, 0.999), adam_eps: float = 1e-8,
                   momentum: float = 0.9, weight_decay: float = 5e-4,
                   nesterov: bool = True, diag_every: int = 1):
    """Build any optimizer of `OPTIMIZERS` from one set of flags.

    `params` may be param_group dicts; for "fw", "fw_adam" and "pgd" each group
    is one block of the product constraint set C = prod_g {||theta_g|| <= R_g},
    and a group may set its own `radius`.
    """
    params = list(params)
    if name in FW_FAMILY:
        kw = {"eta1": eta1}
        if schedule in ("harmonic", "power"):
            kw["k0"] = k0
        if schedule == "power":
            kw["alpha"] = alpha
        return StochasticFrankWolfe(_groups_with_lmo(params, lmo, radius),
                                    lmo=make_lmo(lmo, radius),
                                    schedule=make_schedule(schedule, **kw),
                                    project_init=project_init,
                                    momentum=fw_momentum,
                                    adam=(name == "fw_adam"),
                                    betas=adam_betas, eps=adam_eps,
                                    diag_every=diag_every)
    if name == "gda":
        return GDA(params, lr=lr)
    if name == "pgd":
        gs = _groups_with_lmo(params, lmo, radius)
        return GDA(gs, lr=lr, lmo=make_lmo(lmo, radius), project_init=project_init)
    if name == "ogda":
        return OGDA(params, lr=lr)
    if name == "oadam":
        return OGDA(params, lr=lr, adam=True, betas=adam_betas, eps=adam_eps)
    if name == "eg":
        return Extragradient(params, lr=lr)
    if name == "eadam":
        return Extragradient(params, lr=lr, adam=True, betas=adam_betas, eps=adam_eps)
    if name == "la_gda":
        return Lookahead(GDA(params, lr=lr), k=la_k, alpha=la_alpha)
    if name == "sgdm":
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=nesterov)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=adam_betas, eps=adam_eps)
    if name == "la_adam":
        return Lookahead(torch.optim.Adam(params, lr=lr, betas=adam_betas,
                                          eps=adam_eps), k=la_k, alpha=la_alpha)
    raise KeyError("unknown optimizer %r; have %s" % (name, list(OPTIMIZERS)))


def n_grad_evals(opt) -> int:
    """Gradient evaluations per optimizer step (the honest cost axis)."""
    return 2 if hasattr(opt, "extrapolation") else 1


def uses_lr(name: str) -> bool:
    """FW has no learning rate: its knobs are (schedule, eta1, R)."""
    return name not in FW_FAMILY
