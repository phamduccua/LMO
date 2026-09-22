"""Chung minh ban va khong doi ket qua: params + moi khoa chan doan, 60 step,
ba che do (plain / adam / momentum), so voi optimizers/*.py.bak.

    python scripts/check_patch_equiv.py
"""
"""Bit-exactness: old (.bak) vs patched FW, identical seeds, identical stream."""
import importlib.util, sys, torch, shutil, os, copy

def load(tag, lmo_src, fw_src):
    for name, src in (("lmo_%s"%tag, lmo_src), ("fw_%s"%tag, fw_src)):
        spec = importlib.util.spec_from_file_location(name, src)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m
        if name.startswith("fw_"):
            # fw imports `from .lmo import ...`; give it the matching lmo
            sys.modules[name.replace("fw_","lmo_")] = sys.modules["lmo_%s"%tag]
        spec.loader.exec_module(m)
    return sys.modules["lmo_%s"%tag], sys.modules["fw_%s"%tag]

os.makedirs("/tmp/oldpkg", exist_ok=True)
shutil.copy("optimizers/lmo.py.bak", "/tmp/oldpkg/lmo.py")
shutil.copy("optimizers/frank_wolfe.py.bak", "/tmp/oldpkg/frank_wolfe.py")
shutil.copy("optimizers/schedules.py", "/tmp/oldpkg/schedules.py")
open("/tmp/oldpkg/__init__.py","w").write("")
sys.path.insert(0, "/tmp"); sys.path.insert(0, os.getcwd())

from oldpkg.lmo import L2Ball as OldBall
from oldpkg.frank_wolfe import StochasticFrankWolfe as OldFW
from optimizers.lmo import L2Ball as NewBall
from optimizers.frank_wolfe import StochasticFrankWolfe as NewFW
from optimizers.schedules import make_schedule

def run(Ball, FW, steps=60, adam=False, mom=0.0):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(20, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 5))
    groups = [{"params": list(m.parameters()), "lmo": Ball(2.0)}
              for m in net if isinstance(m, torch.nn.Linear)]
    opt = FW(groups, lmo=Ball(2.0), schedule=make_schedule("harmonic"),
             adam=adam, momentum=mom, diag_every=3)
    torch.manual_seed(1)
    infos = []
    for _ in range(steps):
        x = torch.randn(16, 20); y = torch.randint(0, 5, (16,))
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(x), y).backward()
        opt.step()
        infos.append(dict(opt.last_info))
    return [p.detach().clone() for p in net.parameters()], infos

for tag, kw in (("plain", {}), ("adam", {"adam": True}), ("momentum", {"mom": 0.3})):
    po, io_ = run(OldBall, OldFW, **kw)
    pn, in_ = run(NewBall, NewFW, **kw)
    dp = max(float((a - b).abs().max()) for a, b in zip(po, pn))
    keys = set(io_[-1]) | set(in_[-1])
    bad = []
    for k in sorted(keys):
        for a, b in zip(io_, in_):
            va, vb = a.get(k), b.get(k)
            if va is None and vb is None: continue        # non-diag_every step
            if (va is None) != (vb is None):
                bad.append((k, "present in only one")); break
            import math
            if math.isnan(va) and math.isnan(vb): continue
            if abs(va - vb) > 1e-6 * max(1.0, abs(va)): bad.append((k, va, vb)); break
    print("%-9s params max|delta| = %.3e | diagnostics mismatched: %s"
          % (tag, dp, bad if bad else "none (%d keys x %d steps)" % (len(keys), len(io_))))
