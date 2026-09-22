"""Baseline 1 -- optimistic gradient (OGDA), Daskalakis et al. ICLR 2018.

    theta_{k+1} = theta_k - lr * ( 2 F_hat_k - F_hat_{k-1} )      1 F-eval / iter

One gradient per iteration, reusing the previous one as a predictor of the
next -- the "is ONE evaluation enough?" rung of `phase1`'s ladder.  With
`adam=True` the optimistic correction is applied to Adam's preconditioned
direction instead (Optimistic Adam, Daskalakis et al. Alg. 1), which is how the
method is actually used on networks.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


class OGDA(Optimizer):
    def __init__(self, params: Iterable, lr: float, adam: bool = False,
                 betas=(0.9, 0.999), eps: float = 1e-5):
        super().__init__(params, dict(lr=lr, adam=adam, betas=betas, eps=eps))
        self.last_info = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if g["adam"]:
                    if not st:
                        st["t"] = 0
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                    st["t"] += 1
                    st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                    st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    mh = st["m"] / (1 - b1 ** st["t"])
                    vh = st["v"] / (1 - b2 ** st["t"])
                    d = mh / (vh.sqrt() + g["eps"])
                else:
                    d = p.grad.clone()
                prev = st.get("prev")
                if prev is None:
                    p.add_(d, alpha=-g["lr"])
                else:
                    p.add_(2 * d - prev, alpha=-g["lr"])
                st["prev"] = d.clone()
        return loss
