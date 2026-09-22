"""Do ms/step cua FW truoc/sau ban va bo sync (can optimizers/*.py.bak).

    python scripts/bench_fw_patch.py

Chay TRUOC KHI dot GPU-gio: neu tang toc < 1.2x thi ban va khong dang giu va
nut that nam o kernel launch chu khong phai sync.
"""
import sys, os, time, shutil, torch
sys.path.insert(0, "/tmp"); sys.path.insert(0, os.getcwd())
os.makedirs("/tmp/oldpkg", exist_ok=True)
for f in ("lmo", "frank_wolfe"):
    shutil.copy("optimizers/%s.py.bak" % f, "/tmp/oldpkg/%s.py" % f)
shutil.copy("optimizers/schedules.py", "/tmp/oldpkg/schedules.py")
open("/tmp/oldpkg/__init__.py", "w").write("")

from optimizers.schedules import make_schedule
from models import make_model, make_groups

dev = "cuda"
def bench(Ball, FW, steps=40, warmup=8):
    torch.manual_seed(0)
    net = make_model("resnet18", 10, cifar_stem=True).to(dev).to(memory_format=torch.channels_last)
    groups = [{"params": g["params"], "lmo": Ball(2.0)} for g in make_groups(net, "module")]
    opt = FW(groups, lmo=Ball(2.0), schedule=make_schedule("harmonic"), diag_every=10)
    x = torch.randn(128, 3, 32, 32, device=dev).to(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (128,), device=dev)
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(net(x), y).backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000

from oldpkg.lmo import L2Ball as OB; from oldpkg.frank_wolfe import StochasticFrankWolfe as OF
from optimizers.lmo import L2Ball as NB; from optimizers.frank_wolfe import StochasticFrankWolfe as NF
print("GPU:", torch.cuda.get_device_name(0), "| %d blocks of C" % 41)
o = bench(OB, OF); n = bench(NB, NF)
print("cu   : %7.1f ms/step" % o)
print("moi  : %7.1f ms/step   -> %.2fx nhanh hon" % (n, o / n))
print("tiet kiem %.1f ms/step = %.1f s/epoch (391 step)" % (o - n, (o - n) * 391 / 1000))
