"""Rank the gamma_k schedules of one cell -- which arm wins B3.

`make_paper_tables.py` and `run_ablation.py report` both hardcode the columns
("harmonic", "power"), so a third schedule such as `mlogm` never reaches the
table even when its runs are on disk.  This reads the run directories directly
and ranks EVERY schedule it finds, so the B3 decision is made on all the arms.

    python scripts/pick_winner.py --dataset cifar10 --model resnet18 --radius 5

Decision rule, in order:
  1. drop any arm with a diverged seed or a seed at chance level -- an arm that
     fails once is not a winner, whatever its median is;
  2. rank the survivors by MEDIAN test top-1 over seeds;
  3. the lead only counts if it also holds SEED BY SEED (same seed = same init
     and same data order, so the comparison is paired).  3 seeds is too few for
     a significance test; a 3/3 paired sweep is the honest substitute.
An arm that wins on median but not seed-by-seed is reported as a TIE.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "results", "runs")
METRIC = "eval_acc_final"


def load(group: str) -> List[dict]:
    out = []
    for f in glob.glob(os.path.join(RUNS, group, "*", "summary.json")):
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        s["_dir"] = os.path.basename(os.path.dirname(f))
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=None,
                    help="epoch budget to report (default: the largest on disk)")
    ap.add_argument("--groups", nargs="*", default=["ablation", "constlr"])
    a = ap.parse_args()

    runs = [s for g in a.groups for s in load(g)]
    runs = [s for s in runs if s["config"]["dataset"] == a.dataset
            and s["config"]["model"] == a.model
            and s["config"]["optimizer"] == "fw"
            and abs(float(s["config"]["radius"]) - a.radius) < 1e-9]
    if not runs:
        raise SystemExit("no fw runs for %s/%s at R=%g under %s"
                         % (a.dataset, a.model, a.radius, RUNS))

    budgets = sorted({int(s["config"]["epochs"]) for s in runs})
    ep = a.epochs if a.epochs in budgets else budgets[-1]
    runs = [s for s in runs if int(s["config"]["epochs"]) == ep]
    chance = 100.0 / (100 if a.dataset == "cifar100" else 10)

    # arm = schedule, plus eta1 when it is not the default (the constant arm)
    arms: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for s in runs:
        c = s["config"]
        name = c["schedule"]
        if name == "constant":
            name = "constant(eta1=%g)" % float(c["eta1"])
        elif float(c.get("eta1", 1.0)) != 1.0:
            name = "%s(eta1=%g)" % (name, float(c["eta1"]))
        arms[name][int(c["seed"])] = s

    print("cell %s / %s | R=%g | %d epochs | %d arms | chance %.2f%%"
          % (a.dataset, a.model, a.radius, ep, len(arms), chance))
    print("metric = %s (test top-1, higher is better)\n" % METRIC)
    print("%-22s %-26s %-8s %-9s %-7s %s"
          % ("arm", "per seed", "median", "mean+-sd", "|th|/R", "flag"))

    stats = {}
    for name in sorted(arms):
        by_seed = arms[name]
        seeds = sorted(by_seed)
        v = np.array([by_seed[k][METRIC] for k in seeds], float)
        tn = np.array([by_seed[k]["theta_final_norm"]
                       / max(by_seed[k]["theta0_norm"] * a.radius, 1e-9)
                       for k in seeds], float)
        bad = []
        if any(by_seed[k]["diverged"] for k in seeds):
            bad.append("DIVERGED")
        if (v <= chance + 0.5).any():
            bad.append("AT-CHANCE")
        if len(seeds) < 3:
            bad.append("only %d seeds" % len(seeds))
        stats[name] = dict(seeds=seeds, v=v, med=float(np.median(v)), bad=bad)
        print("%-22s %-26s %-8.2f %-9s %-7.3f %s"
              % (name, " ".join("%.2f" % x for x in v), np.median(v),
                 "%.2f+-%.2f" % (v.mean(), v.std(ddof=1)) if v.size > 1 else "-",
                 float(np.median(tn)), ",".join(bad)))

    ok = {k: s for k, s in stats.items() if not s["bad"]}
    print("\n" + "-" * 78)
    if not ok:
        raise SystemExit("every arm is flagged -- no winner; fix the failures first")

    order = sorted(ok, key=lambda k: -ok[k]["med"])
    win, second = order[0], (order[1] if len(order) > 1 else None)
    print("best median: %s (%.2f%%)" % (win, ok[win]["med"]))
    if second is None:
        print("only one clean arm -- nothing to compare against")
        return

    # paired check: same seed = same init and same data order
    shared = sorted(set(ok[win]["seeds"]) & set(ok[second]["seeds"]))
    if not shared:
        print("no shared seeds with %s -- cannot pair" % second)
        return
    d = np.array([arms[win][k][METRIC] - arms[second][k][METRIC] for k in shared])
    print("paired vs runner-up %s (%.2f%%): per-seed delta %s | wins %d/%d"
          % (second, ok[second]["med"], " ".join("%+.2f" % x for x in d),
             int((d > 0).sum()), len(d)))
    if (d > 0).all():
        print("\n==> WINNER: %s  (leads on median AND on every shared seed)" % win)
        print("    use it for B4:  --schedules %s" % win.split("(")[0])
    else:
        print("\n==> TIE between %s and %s: the median lead does not hold seed by "
              "seed.\n    Pick on a secondary criterion (report both, or take the "
              "one\n    with the smaller spread / the cleaner |theta|/R geometry)."
              % (win, second))


if __name__ == "__main__":
    main()
