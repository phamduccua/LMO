"""How many seconds does one epoch cost, on THIS GPU, for each configuration?

The protocol in `run_ablation.py` is priced in epochs, so before spending days
of GPU time it is worth measuring what an epoch actually costs here and which
of the "free" speed switches actually help.  Pascal-class cards (GTX 10xx,
Quadro P-series) have no tensor cores, so fp16 autocast and channels_last are
NOT automatic wins the way they are on Ampere -- they have to be measured.

    python bench_speed.py                       # all models, all switches
    python bench_speed.py --steps 40 --models resnet18

Reports seconds per optimizer step and the implied minutes per CIFAR epoch
(391 steps at batch 128), for each (model, optimizer, amp, channels_last).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MODELS, make_groups, make_model
from optimizers import make_optimizer

STEPS_PER_EPOCH = 50_000 // 128


def bench(model_name: str, opt_name: str, amp: bool, chlast: bool, steps: int,
          batch: int, device, n_classes: int = 10) -> float:
    torch.manual_seed(0)
    net = make_model(model_name, n_classes).to(device)
    if chlast:
        net = net.to(memory_format=torch.channels_last)
    groups = make_groups(net, "module", 2.0, "rel")
    opt = make_optimizer(opt_name, groups, lr=0.1, radius=2.0,
                         schedule="harmonic", diag_every=10)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, 32, 32, device=device)
    if chlast:
        x = x.to(memory_format=torch.channels_last)
    y = torch.randint(0, n_classes, (batch,), device=device)

    for i in range(steps + 5):
        if i == 5:                                  # warm-up: cudnn autotune, alloc
            torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            loss = crit(net(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--optimizers", nargs="*", default=["sgdm", "fw", "fw_adam"])
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    device = torch.device("cuda")
    print("GPU: %s | torch %s | batch %d | %d steps/epoch"
          % (torch.cuda.get_device_name(0), torch.__version__, a.batch,
             STEPS_PER_EPOCH))
    print("%-10s %-8s %-6s %-9s %-10s %-10s"
          % ("model", "opt", "amp", "chlast", "s/step", "min/epoch"))
    for md in a.models:
        for op in a.optimizers:
            for amp in (True, False):
                for chlast in (True, False):
                    try:
                        s = bench(md, op, amp, chlast, a.steps, a.batch, device)
                    except RuntimeError as e:        # OOM on a 4 GB card
                        print("%-10s %-8s %-6s %-9s %s" % (md, op, amp, chlast,
                                                           str(e)[:40]))
                        torch.cuda.empty_cache()
                        continue
                    print("%-10s %-8s %-6s %-9s %-10.3f %-10.1f"
                          % (md, op, amp, chlast, s, s * STEPS_PER_EPOCH / 60))
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
