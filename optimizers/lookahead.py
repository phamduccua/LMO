"""Baseline 3 -- Lookahead (Zhang et al. NeurIPS 2019; Chavdarova et al. ICLR 2021).

    fast weights: k steps of any inner optimizer
    slow weights: phi <- phi + alpha (theta_fast - phi);  theta_fast <- phi

Wraps any optimizer here.  Worth having next to Frank-Wolfe for a specific
reason: with gamma_k = 1/k, FW makes theta_k the running AVERAGE of the LMO
vertices, i.e. it is also a trajectory-averaging method.  Lookahead answers
"is averaging alone enough?" without the LMO.
"""

from __future__ import annotations

from typing import Dict

import torch


class Lookahead:
    """Duck-typed optimizer: zero_grad / step / param_groups (+ extrapolation)."""

    def __init__(self, inner: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.inner = inner
        self.k, self.alpha = int(k), float(alpha)
        self.n = 0
        self.last_info: Dict[str, float] = {}
        # share groups/state with the inner optimizer
        self.param_groups = inner.param_groups
        self.defaults = inner.defaults
        self.state = inner.state
        self.slow = [[p.detach().clone() for p in g["params"]]
                     for g in inner.param_groups]
        if hasattr(inner, "extrapolation"):
            self.extrapolation = inner.extrapolation

    def zero_grad(self, set_to_none: bool = True):
        self.inner.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.inner.step(closure)
        self.n += 1
        if self.n % self.k == 0:
            for g, slow in zip(self.inner.param_groups, self.slow):
                for p, s in zip(g["params"], slow):
                    s.add_(p - s, alpha=self.alpha)
                    p.copy_(s)
        return loss

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, sd):
        self.inner.load_state_dict(sd)
