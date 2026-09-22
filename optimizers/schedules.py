"""Step-size schedules gamma_k for the Frank-Wolfe iteration.

Hough (2026), eq. (FW), requires

    gamma_k in (0, 1],     gamma_k -> 0,     sum_k gamma_k = infinity.

Three of the four schedules below satisfy all three conditions; `constant` is
kept on purpose as an ABLATION that sits outside the theorem (see
`stochastic_frank_wolfe_vi_marl_optimizer.md` section 11):

    name        gamma_k                          (FW) conditions
    ---------   ------------------------------   ----------------------------
    harmonic    eta1 / (k + k0)                  all three   (GFP when eta1=1)
    power       eta1 / (k + k0)^alpha            all three   (0 < alpha <= 1)
    constant    eta1                             fails gamma_k -> 0
    mlogm       eta1 / (m log m), m=k//5 + 2     all three

k counts FW iterations starting at 1, i.e. the update x_{k} -> x_{k+1} uses
gamma_{k+1}, exactly as written in the paper.  `k0 >= 0` is an index offset:
it changes neither limit nor summability, so it keeps the schedule inside the
theorem, but it lets gamma_1 start below 1 -- with k0 = 0 and eta1 = 1 the first
step is x_1 = s_0, which throws the network initialisation away entirely.
"""

from __future__ import annotations

import math
from typing import Callable, Dict


class Schedule:
    """gamma(k) for k = 1, 2, ...  Values are clipped into (0, 1]."""

    name: str = "schedule"
    satisfies_theorem: bool = False

    def __call__(self, k: int) -> float:
        raise NotImplementedError

    def config(self) -> Dict[str, float]:
        return {}

    def __repr__(self) -> str:
        cfg = ", ".join("%s=%g" % (k, v) for k, v in self.config().items())
        return "%s(%s)" % (self.name, cfg)


class Harmonic(Schedule):
    name = "harmonic"
    satisfies_theorem = True

    def __init__(self, eta1: float = 1.0, k0: float = 0.0):
        _check_eta(eta1)
        self.eta1, self.k0 = float(eta1), float(k0)

    def __call__(self, k: int) -> float:
        return min(1.0, self.eta1 / (k + self.k0))

    def config(self):
        return {"eta1": self.eta1, "k0": self.k0}


class Power(Schedule):
    name = "power"
    satisfies_theorem = True

    def __init__(self, eta1: float = 1.0, alpha: float = 0.5001, k0: float = 0.0):
        _check_eta(eta1)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1] to keep sum gamma_k = inf")
        self.eta1, self.alpha, self.k0 = float(eta1), float(alpha), float(k0)

    def __call__(self, k: int) -> float:
        return min(1.0, self.eta1 / (k + self.k0) ** self.alpha)

    def config(self):
        return {"eta1": self.eta1, "alpha": self.alpha, "k0": self.k0}


class Constant(Schedule):
    """Ablation only: gamma_k does not vanish, so the theorem does not apply."""

    name = "constant"
    satisfies_theorem = False

    def __init__(self, eta1: float = 0.01):
        _check_eta(eta1)
        self.eta1 = float(eta1)

    def __call__(self, k: int) -> float:
        return self.eta1

    def config(self):
        return {"eta1": self.eta1}


class MLogM(Schedule):
    """Slowest admissible decay: gamma_k = eta1 / (m log m), m = k//5 + 2.

    sum_m 1/(m log m) DIVERGES (Cauchy condensation: 2^n a_{2^n} ~ 1/(n log 2)),
    so the schedule satisfies all three conditions of (FW) and is covered by the
    theorem.  What makes it a distinct ablation is the SHAPE of the decay.  The
    factor 5 from m = k//5 + 2 makes it the LARGEST of the three early on
    (gamma_1 = 0.72, gamma_10 = 0.18 vs 0.1 for harmonic) and the SMALLEST late
    (gamma_8000 = 8.5e-5 vs 1.25e-4), the two curves crossing near k = 740.  Its
    partial sums then grow like log log k: measured here, sum gamma_k gains 1.33
    between 1e4 and 1e5 steps and 1.04 between 1e5 and 1e6, versus a constant
    2.30 per decade for harmonic.  So over the 8000 FW steps of a run it spends
    a LARGER total budget than harmonic (13.3 vs 9.6) but concentrates it in the
    first few hundred steps, and afterwards the iterate barely moves.

    The practical consequence, measured in the experiments: because gamma_1 is
    0.72 rather than 1, the first step does not push theta_1 = s_0 all the way
    onto the boundary of C, so the last-layer logits do not saturate.  That
    makes this schedule far more robust to a large radius R than `harmonic`.
    """

    name = "mlogm"
    satisfies_theorem = True        # divergent sum, but log log slow

    def __init__(self, eta1: float = 1.0):
        _check_eta(eta1)
        self.eta1 = float(eta1)

    def __call__(self, k: int) -> float:
        m = k // 5 + 2
        return min(1.0, self.eta1 / (m * math.log(m)))

    def config(self):
        return {"eta1": self.eta1}


def _check_eta(eta1: float) -> None:
    if not 0.0 < eta1 <= 1.0:
        raise ValueError("eta1 must lie in (0, 1] so that gamma_k <= 1")


_TABLE: Dict[str, Callable[..., Schedule]] = {
    "harmonic": Harmonic,
    "power": Power,
    "constant": Constant,
    "mlogm": MLogM,
}


def make_schedule(name: str, **kw) -> Schedule:
    if name not in _TABLE:
        raise KeyError("unknown schedule %r; have %s" % (name, sorted(_TABLE)))
    return _TABLE[name](**kw)
