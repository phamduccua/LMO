"""Stochastic Frank-Wolfe VI optimizer (Hough 2026, eq. (FW), stochastic oracle).

Paper, deterministic:

    x_{k+1} = x_k + gamma_{k+1} (s_k - x_k),     s_k in beta(F(x_k)),   x_0 in C

MARL mapping (design note sections 2-12):

    x          = theta, the network parameters (one block per param_group)
    F(theta)   = -grad_theta J(theta)          (J = expected return, maximised)
    F_hat_k    = -g_hat_k                      (a stochastic policy gradient)

In PyTorch terms the training script minimises a loss L = -J_hat, so after
`loss.backward()` the tensor `p.grad` already IS F_hat_k.  No sign flip is done
here; the only contract is "call backward() on a quantity to MINIMISE".

One call to `step()` performs, for every param_group g (every block of C):

    Step 4   s_k     = LMO_{C_g}(F_hat_k)          e.g. -R F_hat / ||F_hat|| on the L2 ball
    Step 5   d_k     = s_k - theta_k
    Step 6   theta  <- theta_k + gamma_{k+1} d_k   (a convex combination: stays in C)

The iterate never leaves C, so no projection is ever needed AFTER x_0.  The
paper does require x_0 in C, and a network initialisation generally is not in a
small ball, so the constructor projects once (`project_init=True`).  Turn it off
and a theta_0 outside C is pulled into C geometrically -- ||theta_k|| is a
contraction towards C at rate prod(1 - gamma_j) -- but that is outside the paper.

Optional extension, OFF by default (`momentum=0` is exactly the paper):
  momentum = rho in (0, 1] feeds the LMO an averaged operator estimate
      D_k = (1 - rho) D_{k-1} + rho F_hat_k,      s_k = LMO(D_k)
  as in stochastic conditional gradient (Mokhtari, Hassani, Karbasi 2018).
  This is the "approximate / stochastic LMO" direction of design note sec. 20,
  NOT part of Hough (2026); no theorem here covers it.

Optional extension "FW-Adam", OFF by default (`adam=False` is exactly the paper):
  Adam transforms the stochastic gradient BEFORE the LMO; the FW update is kept.

      g_hat_k  = -F_hat_k                                  (= p.grad with a minus)
      m_k      = b1 m_{k-1} + (1 - b1) g_hat_k
      v_k      = b2 v_{k-1} + (1 - b2) g_hat_k^2
      m_hat_k  = m_k / (1 - b1^k),     v_hat_k = v_k / (1 - b2^k)
      g_Adam_k = m_hat_k / (sqrt(v_hat_k) + eps)
      F_Adam_k = -g_Adam_k
      s_k      = LMO_C(F_Adam_k)          L2 ball: s_k = R g_Adam_k / ||g_Adam_k||
      theta   <- theta_k + gamma_{k+1} (s_k - theta_k)

      g_hat -> Adam -> g_Adam -> F_Adam -> LMO -> s_k -> FW update

  The iterate still stays in C (a convex combination of points of C), so the
  method stays projection-free; only the oracle input changes.  Implemented on
  p.grad = F_hat directly: m of F_hat is -m of g_hat and v is sign-free, so the
  tensor fed to the LMO is F_Adam without any explicit sign flip.  NOT part of
  Hough (2026); no theorem here covers it.  With the Linf ball the LMO only sees
  sign(F_Adam) = sign(m_hat), so the preconditioner has no effect there -- use L2.
  `momentum` and `adam` are exclusive (Adam's b1 already is a momentum).
  In this mode the diagnostics fw_gap / gap_rel / grad_norm are still computed
  on the RAW F_hat_k (the operator of the VI, not the preconditioned one), and
  `cos_adam` = cos(F_hat_k, F_Adam_k) records how far Adam rotates the direction.

What is NOT here, on purpose:
  * no momentum, no variance reduction, no gradient clipping.  Only the direction
    of F_hat reaches the L2 oracle, so clipping by norm is a no-op anyway.
  * no weight decay: with an L2 ball, the step already shrinks theta by
    (1 - gamma) every iteration -- that IS the regulariser of this method.

Diagnostics (`self.last_info`, one entry per group plus the scalars):

  gamma      gamma_{k+1} actually used
  fw_gap     V_hat(theta_k) = <F_hat, theta_k> - min_{s in C} <F_hat, s>  >= 0
             (stochastic estimate of the Frank-Wolfe gap of Lemma 2.2; the
             theorem says the TRUE gap -> 0, the estimate has a noise floor)
  gap_rel    fw_gap / (R ||F_hat||) for the L2 ball: in [0, 2], scale free,
             = 1 - cos(theta, s)*||theta||/R
  theta_norm ||theta_{k+1}|| per group
  grad_norm  ||F_hat_k|| per group
  cos_ts     cosine between theta_k and s_k: 1 means theta already points at
             the LMO vertex
  cos_adam   (adam=True only) cosine between F_hat_k and F_Adam_k

CV MAPPING (CODE_CV, image classification)
------------------------------------------
    x          = theta, the weights of ResNet18 / ResNet34 / WRN-28-10
    C          = prod_g {||theta_g||_2 <= R_g}, one block g per param_group
                 (`--block all|module|param` in train.py builds the groups; with
                 `--radius-mode rel` each R_g = radius * ||theta_g^0||, the
                 layer-wise radius of Pokutta et al.'s deep FW experiments)
    F(theta)   = grad_theta L(theta),  L = cross-entropy on a minibatch
    F_hat_k    = p.grad after `loss.backward()` on one minibatch

Classification is a MINIMISATION problem, not a saddle point, so F = grad L is
already a gradient field (a monotone operator iff L is convex, which it is not
for a deep net).  The code is unchanged: the contract "call backward() on a
quantity to MINIMISE" is exactly what a supervised loop does, so `p.grad` IS
F_hat_k here with no sign flip, just as it was in the MARL loop.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch.optim import Optimizer

from .lmo import LMO, _dot, _norm2
from .schedules import Schedule


class StochasticFrankWolfe(Optimizer):
    """Frank-Wolfe for monotone VIs, driven by a stochastic operator estimate.

    Args:
        params:    iterable of tensors or of param_group dicts.  Each group is
                   one block of C; a group dict may carry its own `lmo`.
        lmo:       default oracle (e.g. `L2Ball(R)`) for groups without one.
        schedule:  gamma_k, a `Schedule`.  One counter k is shared by all groups
                   (all players move on the same clock, as in the paper).
        project_init: project theta_0 onto C at construction (paper: x_0 in C).
        momentum:  rho of the averaged-operator extension (0 = paper).
        adam:      FW-Adam extension: LMO(F_Adam) instead of LMO(F_hat).
        betas, eps: Adam's (b1, b2) and epsilon, used only when adam=True.
    """

    def __init__(self, params: Iterable, lmo: LMO, schedule: Schedule,
                 project_init: bool = True, momentum: float = 0.0,
                 adam: bool = False, betas=(0.9, 0.999), eps: float = 1e-5,
                 diag_every: int = 1):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must lie in [0, 1]")
        if adam and momentum > 0.0:
            raise ValueError("momentum and adam are exclusive")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must lie in [0, 1)")
        defaults = dict(lmo=lmo)
        super().__init__(params, defaults)
        self.schedule = schedule
        self.momentum = float(momentum)
        self.adam = bool(adam)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        # cos_prev keeps a flat copy of every block's operator estimate, i.e. a
        # second copy of the whole network per step.  On an 11M-36M parameter
        # backbone that is real time, so it is computed every `diag_every`
        # steps; the FW update itself is never affected.
        self.diag_every = max(1, int(diag_every))
        self._prev: Dict[int, torch.Tensor] = {}
        self.k = 0
        self.last_info: Dict[str, float] = {}
        if project_init:
            with torch.no_grad():
                for g in self.param_groups:
                    g["lmo"].project(g["params"])
        self.init_in_C = self.in_C()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        """num/den, or nan where den <= 0 -- decided on the device, not the host."""
        ok = den > 0
        return torch.where(ok, num / torch.where(ok, den, torch.ones_like(den)),
                           torch.full_like(num, float("nan")))

    # ------------------------------------------------------------------ #
    def in_C(self) -> bool:
        with torch.no_grad():
            return all(g["lmo"].contains(g["params"]) for g in self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        gamma = float(self.schedule(self.k + 1))           # gamma_{k+1}
        info: Dict[str, float] = {"gamma": gamma, "k": float(self.k + 1)}
        # Every diagnostic below stays a 0-dim device tensor and is parked in
        # `pend`; the whole dict crosses to the host in ONE transfer at the end
        # of the step.  The old code read each value with float() as it went --
        # ~7 syncs per block per step here, plus ~8 more inside lmo.py -- which
        # serialised the GPU behind the host and made `fw` 4.4x slower than
        # `gda` (1034 vs 4586 img/s measured) even though it does less work.
        # The VALUES logged are unchanged; only when they are read is.
        pend: Dict[str, torch.Tensor] = {}
        gap_tot: Optional[torch.Tensor] = None
        gap_rel_acc: Optional[torch.Tensor] = None
        n_blocks: Optional[torch.Tensor] = None
        for gi, group in enumerate(self.param_groups):
            params: List[torch.Tensor] = group["params"]
            lmo: LMO = group["lmo"]
            # F_hat_k: missing grads are zero components of the operator
            grads = [p.grad if p.grad is not None else torch.zeros_like(p)
                     for p in params]
            if self.momentum > 0.0:                         # extension, see docstring
                avg = []
                for p, gr in zip(params, grads):
                    st = self.state[p]
                    if "D" not in st:
                        st["D"] = gr.detach().clone()
                    else:
                        st["D"].mul_(1.0 - self.momentum).add_(gr, alpha=self.momentum)
                    avg.append(st["D"])
                grads = avg

            # FW-Adam (extension, see docstring): the LMO gets F_Adam, while the
            # gap / grad_norm diagnostics below keep the raw operator F_hat
            dirs = grads
            if self.adam:
                b1, b2 = self.betas
                t = self.k + 1                              # shared clock, as gamma
                dirs = []
                for p, gr in zip(params, grads):
                    st = self.state[p]
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                    st["m"].mul_(b1).add_(gr, alpha=1 - b1)          # = -m_k of g_hat
                    st["v"].mul_(b2).addcmul_(gr, gr, value=1 - b2)  # = v_k
                    mh = st["m"] / (1 - b1 ** t)
                    vh = st["v"] / (1 - b2 ** t)
                    dirs.append(mh / (vh.sqrt() + self.eps))         # = F_Adam_k
                fa = _norm2(dirs) * _norm2(grads)
                pend["cos_adam/%d" % gi] = self._safe_div(_dot(dirs, grads), fa)

            # Step 4: s_k = argmin_{s in C} <F, s>,  F = F_hat (paper) or F_Adam
            s = lmo(dirs, params)

            # Frank-Wolfe gap estimate at theta_k (before the move).
            # lmo.support(grads) is evaluated ONCE and reused for gap_rel
            # below; the old code called it a second time down there.
            support = lmo.support(grads)
            g_norm = _norm2(grads)
            gap = _dot(grads, params) - support
            t_norm0 = _norm2(params)
            s_norm = _norm2(s)
            cos_ts = self._safe_div(_dot(params, s), t_norm0 * s_norm)

            # coherence of the oracle: cos(F_hat_k, F_hat_{k-1}).  Near 0 means
            # consecutive LMO vertices are ~orthogonal, so their gamma-average
            # shrinks: ||theta|| ~ R sqrt(gamma/2) for the power schedule.
            if (self.k + 1) % self.diag_every == 0:
                flat = torch.cat([gr.reshape(-1) for gr in dirs])
                prev = self._prev.get(gi)
                pend["cos_prev/%d" % gi] = (
                    torch.full((), float("nan"), device=flat.device,
                               dtype=torch.float32) if prev is None
                    else self._safe_div(flat @ prev, flat.norm() * prev.norm()))
                self._prev[gi] = flat.clone()

            # Steps 5-6: theta <- theta + gamma (s - theta)
            for p, sp in zip(params, s):
                p.add_(sp - p, alpha=gamma)

            pend["grad_norm/%d" % gi] = g_norm
            pend["theta_norm/%d" % gi] = _norm2(params)   # after the move
            pend["cos_ts/%d" % gi] = cos_ts
            pend["fw_gap/%d" % gi] = gap
            gap_tot = gap if gap_tot is None else gap_tot + gap
            denom = -support
            ok = denom > 0
            share = torch.where(
                ok, gap / torch.where(ok, denom, torch.ones_like(denom)),
                torch.zeros_like(gap))
            gap_rel_acc = share if gap_rel_acc is None else gap_rel_acc + share
            cnt = ok.to(gap.dtype)
            n_blocks = cnt if n_blocks is None else n_blocks + cnt

        if gap_tot is None:                       # no param groups
            info["fw_gap"], info["gap_rel"] = 0.0, 0.0
        else:
            pend["fw_gap"] = gap_tot
            pend["gap_rel"] = gap_rel_acc / n_blocks.clamp_min(1.0)
        if pend:
            # the one and only device->host transfer of the step
            keys = list(pend)
            vals = torch.stack([pend[k].detach().reshape(()).float()
                                for k in keys]).tolist()
            info.update(zip(keys, vals))
        self.k += 1
        self.last_info = info
        return loss

    # ------------------------------------------------------------------ #
    def state_dict(self):
        sd = super().state_dict()
        sd["fw_k"] = self.k
        return sd

    def load_state_dict(self, state_dict):
        self.k = int(state_dict.pop("fw_k", 0))
        super().load_state_dict(state_dict)
