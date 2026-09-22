"""Linear minimization oracles  beta(pi) = argmin_{s in C} <pi, s>.

Every oracle acts on ONE block: a list of tensors that together form the vector
the constraint set is defined on (for the optimizer, one torch param_group).
A product set C = C_1 x ... x C_N has a separable LMO, so a multi-agent problem
with theta = (theta_1, ..., theta_N) and C = prod_i C_i is handled by giving
each agent its own param_group -- no cross-block coupling is needed.

    L2Ball(R)     C = {x : ||x||_2 <= R}      s = -R pi / ||pi||_2
    LinfBall(R)   C = {x : ||x||_inf <= R}    s_i = -R sign(pi_i)
    Simplex()     C = Delta (per tensor row)  s = e_{argmin pi}

`Simplex` is not used for networks.  It is here so that `sanity_check.py` can
run the SAME optimizer class in the exact setting of the paper -- a matrix game
on a product of simplices, where (FW) with gamma_k = 1/k IS fictitious play --
and verify the implementation where the theorem really applies.

Each oracle also provides

    project(xs)          in-place map into C (used once, on x_0, because the
                         paper requires x_0 in C; a network initialisation is
                         generally not)
    support(pi)          min_{s in C} <pi, s>, so that the Frank-Wolfe gap
                         V(x) = <F(x), x> - min_s <F(x), s> is computed without
                         materialising s twice

When pi = 0 every point of C is a minimiser.  Following section 7 of the design
note we return s = x (the current point), i.e. the step does nothing.

CV MAPPING (CODE_CV): one block = one param_group of the network, chosen by
`--block`: `all` (one ball for the whole weight vector), `module` (one ball per
leaf module: conv, bn, linear) or `param` (one ball per tensor).  `Simplex` is
not used on networks; it is kept so `sanity_check.py` can run the SAME optimizer
class where the theorem actually applies.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

Tensors = Sequence[torch.Tensor]


def _dot(a: Tensors, b: Tensors) -> torch.Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))


# Below this, float32 squares of the entries start to underflow (x*x < 1e-38 for
# |x| < 1e-19), so sqrt(sum x^2) UNDER-estimates ||x|| and -R x/||x|| leaves the
# ball.  Observed: FW + momentum on rps, deterministic policy, D_k ~ 1e-25,
# ||theta|| reached 1.64 R.  Normal-sized vectors keep the plain formula, so
# runs that never hit the regime are bit-for-bit unchanged.
_SAFE_NORM = 1e-15
# Smallest scale the rescaled branch may divide by.  Only guards 0/0; any real
# operator norm is many orders of magnitude above it.
_TINY = 1e-30


def _norm2(a: Tensors) -> torch.Tensor:
    # The guard above used to be a host-side `if`, which meant a device->host
    # sync on every call -- and this runs ~6x per block per step, i.e. ~250
    # syncs per optimizer step on a 41-block ResNet18.  The branch is therefore
    # evaluated on the device instead: the rescaled value is computed
    # unconditionally and `torch.where` picks it only in the underflow regime,
    # so a run that never enters that regime returns exactly the same `n` as
    # before and stays bit-for-bit identical.
    a = list(a)
    if not a:
        return torch.zeros(())
    n = torch.sqrt(sum((x * x).sum() for x in a))
    m = torch.stack([x.abs().max() for x in a]).max().clamp_min(_TINY)
    safe = m * torch.sqrt(sum(((x / m) * (x / m)).sum() for x in a))
    return torch.where(n >= _SAFE_NORM, n, safe)


class LMO:
    name = "lmo"

    def __call__(self, grads: Tensors, xs: Tensors) -> List[torch.Tensor]:
        raise NotImplementedError

    def project(self, xs: Tensors) -> None:
        raise NotImplementedError

    def support(self, grads: Tensors) -> torch.Tensor:
        raise NotImplementedError

    def contains(self, xs: Tensors, tol: float = 1e-5) -> bool:
        raise NotImplementedError


class L2Ball(LMO):
    name = "l2"

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("radius must be > 0")
        self.R = float(radius)

    def __call__(self, grads, xs):
        g = _norm2(grads)
        # `g` is underflow-safe already (see _norm2), so the old three-way
        # host-side branch collapses to one expression plus a device-side mask
        # for the pi = 0 case, where every point of C is a minimiser and the
        # design note says to return x (the step then does nothing).  No sync.
        pos = g > 0
        gs = torch.where(pos, g, torch.ones_like(g))
        return [torch.where(pos, -self.R * gr / gs, x.detach())
                for gr, x in zip(grads, xs)]

    def project(self, xs):
        n = float(_norm2(xs))
        if n > self.R:
            for x in xs:
                x.mul_(self.R / n)

    def support(self, grads):
        return -self.R * _norm2(grads)

    def contains(self, xs, tol=1e-5):
        return float(_norm2(xs)) <= self.R * (1.0 + tol)


class LinfBall(LMO):
    name = "linf"

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("radius must be > 0")
        self.R = float(radius)

    def __call__(self, grads, xs):
        # sign(0) = 0 would put a coordinate at the centre; keep x there instead
        out = []
        for gr, x in zip(grads, xs):
            s = -self.R * torch.sign(gr)
            out.append(torch.where(gr == 0, x.detach(), s))
        return out

    def project(self, xs):
        for x in xs:
            x.clamp_(-self.R, self.R)

    def support(self, grads):
        return -self.R * sum(gr.abs().sum() for gr in grads)

    def contains(self, xs, tol=1e-5):
        return all(float(x.abs().max()) <= self.R * (1.0 + tol) for x in xs)


class Simplex(LMO):
    """Each tensor is a probability vector (1-D) or a stack of them (rows)."""

    name = "simplex"

    def __call__(self, grads, xs):
        out = []
        for gr in grads:
            g2 = gr.reshape(-1, gr.shape[-1])
            s = torch.zeros_like(g2)
            s[torch.arange(g2.shape[0]), g2.argmin(dim=-1)] = 1.0   # lexicographic ties
            out.append(s.reshape(gr.shape))
        return out

    def project(self, xs):
        for x in xs:
            x2 = x.reshape(-1, x.shape[-1])
            u, _ = torch.sort(x2, dim=-1, descending=True)
            css = u.cumsum(-1) - 1.0
            idx = torch.arange(1, x2.shape[-1] + 1, dtype=x.dtype, device=x.device)
            rho = ((u - css / idx) > 0).sum(-1, keepdim=True)
            theta = css.gather(-1, rho - 1) / rho.to(x.dtype)
            x.copy_(torch.clamp(x2 - theta, min=0.0).reshape(x.shape))

    def support(self, grads):
        return sum(gr.reshape(-1, gr.shape[-1]).min(-1).values.sum() for gr in grads)

    def contains(self, xs, tol=1e-5):
        for x in xs:
            x2 = x.reshape(-1, x.shape[-1])
            if float(x2.min()) < -tol or float((x2.sum(-1) - 1).abs().max()) > tol:
                return False
        return True


def make_lmo(name: str, radius: float = 1.0) -> LMO:
    if name == "l2":
        return L2Ball(radius)
    if name == "linf":
        return LinfBall(radius)
    if name == "simplex":
        return Simplex()
    raise KeyError("unknown lmo %r; have ['l2', 'linf', 'simplex']" % name)
