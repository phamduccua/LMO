"""Baseline 2 -- stochastic extragradient (Korpelevich 1976; Gidel et al. ICLR 2019).

    v_k     = theta_k - lr * F_hat(theta_k)          extrapolation
    theta_+ = theta_k - lr * F_hat(v_k)              update, from theta_k      2 F-evals / iter

The closest VI baseline to Frank-Wolfe: same operator view, but a gradient
(projection) method with two oracle calls.  Implemented in the two-call style of
Gidel et al.'s ExtraSGD/ExtraAdam so it fits a normal training loop:

    loss(theta).backward();  opt.extrapolation()     # parameters now at v_k
    opt.zero_grad()
    loss(v).backward();      opt.step()              # back to theta_k, then update

The training script detects `hasattr(opt, "extrapolation")` and evaluates the
loss twice on the SAME minibatch (common random numbers -- phase1/sampling.py
explains the trade-off; here the minibatch is fixed data, so the only thing
that changes between the two calls is the policy, which is the point).

`adam=True` gives ExtraAdam: both half-steps use Adam's preconditioned direction,
with one shared moment estimate updated at the second call only.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


class Extragradient(Optimizer):
    def __init__(self, params: Iterable, lr: float, adam: bool = False,
                 betas=(0.9, 0.999), eps: float = 1e-5):
        super().__init__(params, dict(lr=lr, adam=adam, betas=betas, eps=eps))
        self.last_info = {}
        self._extrapolated = False

    def _direction(self, p, g, commit: bool):
        if not g["adam"]:
            return p.grad
        st = self.state[p]
        if "t" not in st:
            st["t"] = 0
            st["m"] = torch.zeros_like(p)
            st["v"] = torch.zeros_like(p)
        b1, b2 = g["betas"]
        t = st["t"] + 1
        m = st["m"] * b1 + p.grad * (1 - b1)
        v = st["v"] * b2 + p.grad * p.grad * (1 - b2)
        if commit:
            st["t"], st["m"], st["v"] = t, m, v
        return (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + g["eps"])

    @torch.no_grad()
    def extrapolation(self):
        """theta_k -> v_k = theta_k - lr F_hat(theta_k); remembers theta_k."""
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                self.state[p]["backup"] = p.detach().clone()
                p.add_(self._direction(p, g, commit=False), alpha=-g["lr"])
        self._extrapolated = True

    @torch.no_grad()
    def restore(self):
        """Undo `extrapolation()` without updating: parameters go back to theta_k.

        Needed under fp16 mixed precision: when the second gradient evaluation
        overflows, the update must be skipped -- but the parameters are then
        sitting at the extrapolated point v_k, not at theta_k.  Skipping without
        this would silently leave the iterate half a step ahead.
        """
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state[p]
                if "backup" in st:
                    p.copy_(st.pop("backup"))
        self._extrapolated = False

    @torch.no_grad()
    def step(self, closure=None):
        """Must follow `extrapolation()`: grads are F_hat(v_k)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state[p]
                if "backup" in st:
                    p.copy_(st.pop("backup"))
                if p.grad is None:
                    continue
                p.add_(self._direction(p, g, commit=True), alpha=-g["lr"])
        self._extrapolated = False
        return loss
