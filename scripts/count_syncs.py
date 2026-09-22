"""Dem so lan sync device->host trong MOT opt.step(), truoc/sau ban va.

    python scripts/count_syncs.py

Doc lap voi GPU: do co che, khong do toc do.  Do duoc 656 -> 1 tren resnet18.
"""
import sys, os, shutil, warnings, torch
sys.path.insert(0, "/tmp"); sys.path.insert(0, os.getcwd())
os.makedirs("/tmp/oldpkg", exist_ok=True)
for f in ("lmo", "frank_wolfe"):
    shutil.copy("optimizers/%s.py.bak" % f, "/tmp/oldpkg/%s.py" % f)
shutil.copy("optimizers/schedules.py", "/tmp/oldpkg/schedules.py")
open("/tmp/oldpkg/__init__.py", "w").write("")
from optimizers.schedules import make_schedule
from models import make_model, make_groups

def count(Ball, FW, steps=3):
    torch.manual_seed(0)
    net = make_model("resnet18", 10, cifar_stem=True).cuda()
    groups = [{"params": g["params"], "lmo": Ball(2.0)} for g in make_groups(net, "module")]
    opt = FW(groups, lmo=Ball(2.0), schedule=make_schedule("harmonic"), diag_every=10)
    x = torch.randn(32, 3, 32, 32, device="cuda"); y = torch.randint(0, 10, (32,), device="cuda")
    # warm up outside the counter
    opt.zero_grad(); torch.nn.functional.cross_entropy(net(x), y).backward(); opt.step()
    n = 0
    torch.cuda.set_sync_debug_mode("warn")
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(x), y).backward()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opt.step()
            n += sum(1 for r in w if "sync" in str(r.message).lower()
                     or "called a synchronizing" in str(r.message))
    torch.cuda.set_sync_debug_mode("default")
    return n / steps

from oldpkg.lmo import L2Ball as OB; from oldpkg.frank_wolfe import StochasticFrankWolfe as OF
from optimizers.lmo import L2Ball as NB; from optimizers.frank_wolfe import StochasticFrankWolfe as NF
o = count(OB, OF); n = count(NB, NF)
print("sync device->host trong MOT opt.step(), resnet18, 41 block:")
print("  cu  : %6.1f" % o)
print("  moi : %6.1f" % n)
print("  giam %.0fx" % (o / max(n, 1e-9)))
