#!/usr/bin/env python3
"""Tong hop ket qua cuoi (test, 1 lan) va nhom BatchNorm.  Khong chon gi o day.

    python scripts/summarize_camera_ready.py

Ghi results/camera_ready/final_test.csv (+ bn_seeds.csv neu co nhom cr_bn).
Chi so: eval_acc_final = accuracy test o EPOCH CUOI; delta = peak - final.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def q(v, p):
    return float(np.quantile(np.asarray(v, dtype=float), p))


def load(d):
    out = []
    for f in glob.glob(os.path.join(d, "*", "summary.json")):
        s = json.load(open(f))
        s["_dir"] = os.path.dirname(f)
        out.append(s)
    return out


def key(c):
    if c["optimizer"] == "pgd":
        return "pgd", float(c["lr"])
    k0 = float(c.get("k0", 0.0) or 0.0)
    if c["schedule"] == "harmonic" and k0 > 0:
        return "ell", float(c["eta1"])
    return c["schedule"], float(c["eta1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="results/runs/cr_test")
    ap.add_argument("--bn", default="results/runs/cr_bn")
    ap.add_argument("--out", default="results/camera_ready")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = load(a.test)
    bad = [s["_dir"] for s in runs if s.get("eval_split") != "test"]
    if bad:
        print("LOI: %d run trong %s khong danh gia tren TEST (val_frac > 0?):" % (len(bad), a.test))
        for b in bad[:5]:
            print("   ", b)
        return 1
    cells = defaultdict(list)
    for s in runs:
        c = s["config"]
        cells[(c["dataset"], *key(c))].append(s)
    rows = []
    for (ds, fam, sc), ss in sorted(cells.items()):
        acc = [s["eval_acc_final"] for s in ss]
        gap = [s["eval_acc_best"] - s["eval_acc_final"] for s in ss]
        rows.append(dict(dataset=ds, family=fam, scale=sc, n=len(ss),
                         seeds=" ".join(str(s["config"]["seed"]) for s in sorted(ss, key=lambda s: s["config"]["seed"])),
                         median=q(acc, .5), q1=q(acc, .25), q3=q(acc, .75),
                         min=min(acc), max=max(acc), mean=float(np.mean(acc)),
                         instability_median=q(gap, .5), n_diverged=sum(bool(s["diverged"]) for s in ss),
                         time_s_median=q([s["time_s"] for s in ss], .5)))
    if rows:
        with open(os.path.join(a.out, "final_test.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("%-9s %-9s %-7s %2s  %-26s %-15s %6s" % ("dataset", "family", "scale", "n",
                                                       "median [IQR]", "(min, max)", "peak-fin"))
        for r in rows:
            print("%-9s %-9s %-7g %2d  %6.2f [%6.2f, %6.2f]   (%5.2f, %5.2f) %6.2f"
                  % (r["dataset"], r["family"], r["scale"], r["n"], r["median"], r["q1"], r["q3"],
                     r["min"], r["max"], r["instability_median"]))
    else:
        print("chua co run nao trong %s" % a.test)

    bn = load(a.bn)
    if bn:
        cells = defaultdict(list)
        for s in bn:
            c = s["config"]
            cells[(c["dataset"], c["schedule"], c.get("block", "module"))].append(s)
        with open(os.path.join(a.out, "bn_seeds.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "schedule", "block", "n", "seeds", "acc_final_list", "median"])
            print("\nBatchNorm (test, epoch cuoi):")
            for (ds, sch, blk), ss in sorted(cells.items()):
                ss = sorted(ss, key=lambda s: s["config"]["seed"])
                acc = [s["eval_acc_final"] for s in ss]
                w.writerow([ds, sch, blk, len(ss), " ".join(str(s["config"]["seed"]) for s in ss),
                            " ".join("%.2f" % x for x in acc), "%.2f" % q(acc, .5)])
                print("  %-9s %-9s %-13s n=%d  %s" % (ds, sch, blk, len(ss), " ".join("%.2f" % x for x in acc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
