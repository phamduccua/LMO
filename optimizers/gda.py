"""Baseline 0 -- GDA / simultaneous policy gradient, and its projected variant.

    theta_{k+1} = theta_k - lr * F_hat(theta_k)                       (GDA)
    theta_{k+1} = P_C( theta_k - lr * F_hat(theta_k) )                (PGD)

GDA is the bottom rung of the ladder, exactly as in `phase1/optimizers/gda.py`:
it answers "is any structure needed at all?".  With F_hat = p.grad of the loss
it is plain SGD, i.e. naive simultaneous policy gradient for every agent.

PGD is the baseline that makes the Frank-Wolfe comparison FAIR on geometry: it
lives on the very same compact convex set C = {||theta|| <= R} (one ball per
param_group) and differs from FW only in how it stays feasible -- a projection
after an unconstrained step, versus a convex combination with an LMO vertex.
That pair (PGD vs FW at equal R) isolates "projection vs projection-free";
GDA vs PGD isolates the effect of the constraint itself.

CV MAPPING (CODE_CV): `gda` IS plain SGD on the cross-entropy loss (no momentum,
no weight decay) and `pgd` is that same step followed by the projection onto the
very same C that FW uses.  The practitioner's CIFAR default -- SGD with momentum
0.9, weight decay 5e-4 and a cosine schedule -- is a separate baseline, `sgdm`,
built from `torch.optim.SGD` in `__init__.py`.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer

from .lmo import LMO


class GDA(Optimizer):
    """SGD on F_hat; with `lmo` given, followed by the projection onto C."""

    def __init__(self, params: Iterable, lr: float, lmo: Optional[LMO] = None,
                 project_init: bool = True):
        super().__init__(params, dict(lr=lr, lmo=lmo))
        self.last_info = {}
        if lmo is not None and project_init:
            with torch.no_grad():
                for g in self.param_groups:
                    g["lmo"].project(g["params"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-g["lr"])
            if g["lmo"] is not None:
                g["lmo"].project(g["params"])
        return loss
