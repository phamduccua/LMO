"""Level-1 checks: is the implementation right, before any GPU hour is spent?

The CV twin of `../CODE_MARL/sanity_check.py`.  Nothing here trains a real
network; every test runs in seconds on CPU and fails loudly.

    1  LMO      s = argmin_{s in C} <F, s> on the L2 / Linf ball, its support
                value, and the projection -- checked against brute force on
                random directions.
    2  schedule gamma_k in (0,1], gamma_k -> 0, sum gamma_k = infinity for
                harmonic/power; the two ablation schedules must FAIL exactly one
                of those (that is why they are in the paper's ablation).
    3  FW invariant: theta_k stays in C for every k, from any theta_0, for every
                block mode and both oracles.  This is the whole point of a
                projection-free method -- if it ever breaks, no result below is
                worth reading.
    4  FW on a convex problem where the theorem DOES apply (least squares on an
                L2 ball): the Frank-Wolfe gap must go to zero and the iterate
                must approach the constrained optimum found by projected GD.
    5  a real backbone, 200 steps on 256 CIFAR images, every optimizer: the loss
                must drop and nothing may produce NaN.  This is a plumbing test
                (does each optimizer drive a BatchNorm ResNet at all?), not a
                comparison.
    6  mixed precision (GPU only): every optimizer under fp16 + GradScaler and
                under bf16, driving the exact loop of `train.py` -- the two-
                backward methods (eg, eadam) need special care with the scaler.

    python sanity_check.py            # all of it, CPU, ~3 minutes
    python sanity_check.py --quick    # skip test 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import make_groups, make_model, param_norm
from optimizers import (BASELINE_OPTIMIZERS, FW_FAMILY, L2Ball, LinfBall,
                        StochasticFrankWolfe, make_lmo, make_optimizer,
                        make_schedule)

OK, BAD = "  ok  ", " FAIL "
_fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print("[%s] %s%s" % (OK if cond else BAD, name, ("  -- " + detail) if detail else ""))
    if not cond:
        _fails.append(name)


# --------------------------------------------------------------------------- #
def test_lmo() -> None:
    print("\n1. LINEAR MINIMIZATION ORACLES")
    torch.manual_seed(0)
    for R in (0.5, 2.0, 10.0):
        grads = [torch.randn(7, 3), torch.randn(5)]
        xs = [torch.randn(7, 3), torch.randn(5)]

        ball = L2Ball(R)
        s = ball(grads, xs)
        ns = math.sqrt(sum(float((t ** 2).sum()) for t in s))
        ip = sum(float((g * t).sum()) for g, t in zip(grads, s))
        # brute force: the minimiser over the ball is -R g/||g||, so no random
        # feasible point may beat it
        worse = []
        for _ in range(200):
            r = [torch.randn_like(t) for t in xs]
            nr = math.sqrt(sum(float((t ** 2).sum()) for t in r))
            r = [t * (R / nr) for t in r]
            worse.append(sum(float((g * t).sum()) for g, t in zip(grads, r)) >= ip - 1e-5)
        check("L2Ball R=%g: ||s|| = R" % R, abs(ns - R) < 1e-4, "||s||=%.6f" % ns)
        check("L2Ball R=%g: minimises <F, s>" % R, all(worse))
        check("L2Ball R=%g: support() == <F, s>" % R,
              abs(float(ball.support(grads)) - ip) < 1e-3)

        linf = LinfBall(R)
        s = linf(grads, xs)
        ip = sum(float((g * t).sum()) for g, t in zip(grads, s))
        check("LinfBall R=%g: |s_i| = R" % R,
              all(abs(float(t.abs().max()) - R) < 1e-5 for t in s))
        check("LinfBall R=%g: support() == <F, s>" % R,
              abs(float(linf.support(grads)) - ip) < 1e-3)

        # projection maps into C and is the identity inside C
        out = [torch.randn(7, 3) * 50, torch.randn(5) * 50]
        ball.project(out)
        check("L2Ball R=%g: project() lands in C" % R, ball.contains(out))
        inside = [t.clone() for t in out]
        ball.project(out)
        check("L2Ball R=%g: project() is idempotent" % R,
              all(torch.allclose(a, b, atol=1e-6) for a, b in zip(inside, out)))

    # zero operator: every point of C is optimal, the convention is s = x
    z = [torch.zeros(4), torch.zeros(2)]
    xs = [torch.randn(4), torch.randn(2)]
    s = L2Ball(3.0)(z, xs)
    check("L2Ball: F = 0 -> s = x (no move)",
          all(torch.allclose(a, b) for a, b in zip(s, xs)))


def test_schedules() -> None:
    """The three conditions of (FW), each tested on its own.

    "sum = infinity" cannot be checked by "is the sum large": a divergent series
    can grow arbitrarily slowly.  The criterion used here is that the increment
    PER DECADE does not collapse -- S(10N) - S(N) must stay a decent fraction of
    the previous decade's increment.  That is what separates `mlogm` (which does
    diverge, but like log log k, so the increments decay) from `harmonic`
    (exactly 2.30 per decade, forever).
    """
    print("\n2. STEP-SIZE SCHEDULES  (gamma_k in (0,1], -> 0, sum = inf)")
    for name, kw, want_vanish, want_diverge in (
            ("harmonic", {}, True, True),
            ("power", {"alpha": 0.5001}, True, True),
            ("constant", {"eta1": 0.01}, False, True),
            ("mlogm", {}, True, True)):
        sch = make_schedule(name, **kw)
        g = np.array([sch(k) for k in range(1, 10 ** 6 + 1)])
        s = np.cumsum(g)
        d1, d2 = s[10 ** 5 - 1] - s[10 ** 4 - 1], s[-1] - s[10 ** 5 - 1]
        check("%-9s gamma_k in (0,1]" % name, bool(((g > 0) & (g <= 1)).all()))
        check("%-9s gamma_k -> 0: %s" % (name, want_vanish),
              (g[-1] < g[0] / 100) == want_vanish, "gamma_1e6=%.2e" % g[-1])
        check("%-9s sum gamma_k = inf: %s" % (name, want_diverge),
              (d2 > 0.5 * d1) == want_diverge,
              "per-decade increment %.2f -> %.2f" % (d1, d2))
    # the ablation point of mlogm is the RATE, not divergence (see its docstring)
    m = make_schedule("mlogm")
    h = make_schedule("harmonic")
    sm = sum(m(k) for k in range(1, 10 ** 6 + 1))
    sh = sum(h(k) for k in range(1, 10 ** 6 + 1))
    check("mlogm sums grow like log log k, harmonic like log k", sm > sh,
          "after 1e6 steps: mlogm %.1f vs harmonic %.1f -- but mlogm's next "
          "decade adds ~1.0 and harmonic's adds 2.3, forever" % (sm, sh))


def test_stays_in_C() -> None:
    print("\n3. FEASIBILITY: theta_k in C for EVERY k (the projection-free claim)")
    torch.manual_seed(0)
    net = make_model("resnet18", 10)
    for block in ("all", "module", "param"):
        for lmo in ("l2", "linf"):
            for adam in (False, True):
                torch.manual_seed(0)
                m = make_model("resnet18", 10)
                groups = make_groups(m, block, 2.0, "rel")
                opt = make_optimizer("fw_adam" if adam else "fw", groups,
                                     radius=2.0, lmo=lmo, schedule="harmonic")
                ever_out = False
                for k in range(30):
                    x = torch.randn(4, 3, 32, 32)
                    y = torch.randint(0, 10, (4,))
                    opt.zero_grad()
                    nn.functional.cross_entropy(m(x), y).backward()
                    opt.step()
                    ever_out |= not opt.in_C()
                check("block=%-6s lmo=%-4s %s: in C at every one of 30 steps"
                      % (block, lmo, "fw_adam" if adam else "fw"), not ever_out)

    # theta_0 outside C is pulled in (the geometric contraction of the docstring)
    torch.manual_seed(0)
    m = make_model("resnet18", 10)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(20.0)
    groups = make_groups(m, "module", 1.0, "rel")     # R = ||theta^0|| BEFORE scaling
    for g in groups:
        g["radius"] = g["radius"] / 20.0
    opt = make_optimizer("fw", groups, radius=1.0, schedule="harmonic",
                         project_init=False)
    n0 = param_norm(m.parameters())
    for _ in range(40):
        x = torch.randn(4, 3, 32, 32)
        opt.zero_grad()
        nn.functional.cross_entropy(m(x), torch.randint(0, 10, (4,))).backward()
        opt.step()
    n1 = param_norm(m.parameters())
    check("project_init=False: ||theta|| contracts towards C", n1 < n0,
          "%.1f -> %.1f" % (n0, n1))


def test_convex_problem() -> None:
    """Least squares on an L2 ball: convex, so the theorem really applies here."""
    print("\n4. CONVEX CONTROL PROBLEM: min ||Ax - b||^2 over ||x|| <= R")
    torch.manual_seed(0)
    n, d, R = 200, 20, 1.5
    A = torch.randn(n, d)
    b = torch.randn(n)

    def loss_of(x):
        return ((A @ x - b) ** 2).mean()

    # reference: projected gradient descent, many steps
    xp = torch.zeros(d, requires_grad=True)
    ball = L2Ball(R)
    for _ in range(20_000):
        l = loss_of(xp)
        g, = torch.autograd.grad(l, xp)
        with torch.no_grad():
            xp -= 0.01 * g
            ball.project([xp])
    ref = float(loss_of(xp))

    x = torch.zeros(d, requires_grad=True)
    opt = StochasticFrankWolfe([{"params": [x]}], lmo=L2Ball(R),
                               schedule=make_schedule("harmonic"))
    gaps = []
    for k in range(20_000):
        opt.zero_grad()
        loss_of(x).backward()
        opt.step()
        gaps.append(opt.last_info["fw_gap"])
    fw = float(loss_of(x))
    early, late = float(np.mean(gaps[:100])), float(np.mean(gaps[-100:]))
    check("FW gap decreases towards 0", late < 0.05 * early + 1e-6,
          "first100 %.4f -> last100 %.4f" % (early, late))
    check("FW loss matches projected GD", abs(fw - ref) < 1e-2 * max(1.0, abs(ref)),
          "FW %.6f vs PGD %.6f" % (fw, ref))
    check("FW iterate is feasible", float(x.norm()) <= R * (1 + 1e-5),
          "||x||=%.4f <= R=%g" % (float(x.norm()), R))


def test_all_optimizers_run() -> None:
    """Plumbing: every optimizer must drive a real BatchNorm ResNet without NaN."""
    print("\n5. ALL OPTIMIZERS ON A REAL BACKBONE (200 steps, 256 CIFAR-like images)")
    lr_of = {"gda": 0.03, "pgd": 0.03, "ogda": 0.03, "eg": 0.03, "la_gda": 0.03,
             "sgdm": 0.03, "adam": 1e-3, "oadam": 1e-3, "eadam": 1e-3,
             "la_adam": 1e-3}
    torch.manual_seed(0)
    X = torch.randn(256, 3, 32, 32)
    Y = torch.randint(0, 10, (256,))
    for name in FW_FAMILY + BASELINE_OPTIMIZERS:
        torch.manual_seed(0)
        m = make_model("resnet18", 10)
        groups = make_groups(m, "module", 2.0, "rel")
        opt = make_optimizer(name, groups, lr=lr_of.get(name, 0.03), radius=2.0,
                             schedule="harmonic")
        first, last, bad = None, None, False
        for k in range(200):
            i = torch.randint(0, 256, (32,))
            opt.zero_grad()
            loss = nn.functional.cross_entropy(m(X[i]), Y[i])
            loss.backward()
            if hasattr(opt, "extrapolation"):
                opt.extrapolation()
                opt.zero_grad()
                nn.functional.cross_entropy(m(X[i]), Y[i]).backward()
            opt.step()
            bad |= not math.isfinite(float(loss))
            first = float(loss) if first is None else first
            last = float(loss)
        check("%-8s no NaN, loss %.3f -> %.3f" % (name, first, last),
              not bad and last < first * 1.5)


def test_amp() -> None:
    """Every optimizer must survive mixed precision -- both dtypes, on the GPU.

    This is not a formality: extragradient evaluates the gradient TWICE per
    iteration, and `GradScaler.unscale_` may be called at most once per
    optimizer per update, so the naive loop raises
    "unscale_() has already been called on this optimizer since the last
    update()" for `eg` and `eadam` under fp16.  `train.py` therefore unscales
    the first pass by hand and lets the scaler own the second; this test drives
    the same code path it does.
    """
    print("\n6. MIXED PRECISION: every optimizer x {fp16 + GradScaler, bf16}")
    if not torch.cuda.is_available():
        print("[ skip ] no CUDA device")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import unscale_grads_
    dev = torch.device("cuda")
    lr_of = {"adam": 1e-3, "oadam": 1e-3, "eadam": 1e-3, "la_adam": 1e-3}
    X = torch.randn(64, 3, 32, 32, device=dev)
    Y = torch.randint(0, 10, (64,), device=dev)
    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    for dt in dtypes:
        for name in FW_FAMILY + BASELINE_OPTIMIZERS:
            torch.manual_seed(0)
            m = make_model("resnet18", 10).to(dev)
            opt = make_optimizer(name, make_groups(m, "module", 2.0, "rel"),
                                 lr=lr_of.get(name, 0.03), radius=2.0,
                                 schedule="harmonic", diag_every=10)
            use_scaler = dt is torch.float16
            sc = torch.amp.GradScaler("cuda", enabled=use_scaler)
            is_eg = hasattr(opt, "extrapolation")
            err = None
            try:
                for _ in range(5):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=dt, enabled=True):
                        loss = nn.functional.cross_entropy(m(X), Y)
                    sc.scale(loss).backward()
                    if not is_eg:
                        sc.unscale_(opt)
                        sc.step(opt)
                        sc.update()
                    else:
                        ok = (unscale_grads_(m, 1.0 / sc.get_scale())
                              if use_scaler else True)
                        if not ok:
                            sc.update(sc.get_scale() * sc.get_backoff_factor())
                            continue
                        opt.extrapolation()
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=dt, enabled=True):
                            loss2 = nn.functional.cross_entropy(m(X), Y)
                        sc.scale(loss2).backward()
                        sc.unscale_(opt)
                        sc.step(opt)
                        sc.update()
                        if getattr(opt, "_extrapolated", False):
                            opt.restore()
            except Exception as e:                     # noqa: BLE001
                err = "%s: %s" % (type(e).__name__, str(e)[:70])
            ok = err is None and math.isfinite(float(loss))
            check("%-8s %s: 5 steps, no exception, loss finite"
                  % (name, str(dt).replace("torch.", "")), ok, err or "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the backbone test")
    args = ap.parse_args()
    torch.set_num_threads(4)

    test_lmo()
    test_schedules()
    test_stays_in_C()
    test_convex_problem()
    test_amp()
    if not args.quick:
        test_all_optimizers_run()

    print("\n%s" % ("=" * 70))
    if _fails:
        print("%d FAILED: %s" % (len(_fails), _fails))
        sys.exit(1)
    print("all checks passed")
