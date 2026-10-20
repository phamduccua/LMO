#!/usr/bin/env python3
"""Chon scale (eta1 / ell / lr) CHI tren validation, ghi ke hoach chay test cuoi.

Doc  results/runs/cr_val/*/summary.json  (cac run co --val-frac > 0, nen
`eval_acc_final` o day la accuracy tren TAP VALIDATION, khong phai test).

Quy tac chon (co dinh truoc khi thay so lieu):
  * moi o (dataset, ho schedule, scale) can du --min-seeds seed (mac dinh 3);
  * diem cua o = TRUNG BINH eval_acc_final (epoch cuoi, khong dung peak) tren cac seed;
  * run diverged tinh la ket qua that (khong bi loai), accuracy = chance;
  * chon o co diem trung binh cao nhat; hoa thi chon o co min-seed cao hon;
  * bao cao o duoc chon co nam o BIEN luoi hay khong.

Dau ra:
  results/camera_ready/val_grid.csv     moi o: n, mean, std, min, max, selected, boundary
  results/camera_ready/final_plan.tsv   cac cau hinh duoc chon (de run_camera_ready.sh final doc)
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def family(c):
    """-> (ten ho, scale) hoac None neu khong thuoc thiet ke cua camera-ready."""
    opt = c["optimizer"]
    if opt == "pgd":
        return "pgd", float(c["lr"])
    if opt == "fw":
        sch, eta = c["schedule"], float(c["eta1"])
        k0 = float(c.get("k0", 0.0) or 0.0)
        if sch == "harmonic" and k0 > 0:
            # ell/(k+ell) <=> harmonic voi eta1 = k0 = ell
            return ("ell", eta) if abs(eta - k0) < 1e-9 else None
        if sch in ("harmonic", "power", "mlogm") and k0 == 0:
            return sch, eta
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/cr_val")
    ap.add_argument("--out", default="results/camera_ready")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cells = defaultdict(list)
    skipped = 0
    for f in glob.glob(os.path.join(a.runs, "*", "summary.json")):
        s = json.load(open(f))
        c = s["config"]
        if s.get("eval_split") != "val" or float(c.get("val_frac", 0)) <= 0:
            skipped += 1          # khong phai run validation -> khong dung de chon
            continue
        if int(c["epochs"]) != a.epochs or c.get("block", "module") != "module":
            skipped += 1
            continue
        fam = family(c)
        if fam is None:
            skipped += 1
            continue
        cells[(c["dataset"], fam[0], fam[1])].append(
            (int(c["seed"]), float(s["eval_acc_final"]), bool(s.get("diverged", False))))
    if not cells:
        print("khong thay run validation nao trong %s" % a.runs)
        return 1
    if skipped:
        print("bo qua %d run khong hop le cho viec chon (khong phai val / sai epochs / sai ho)" % skipped)

    rows, best = [], {}
    by_fam = defaultdict(list)
    for (ds, fam, sc), v in cells.items():
        seeds = sorted({x[0] for x in v})
        accs = [x[1] for x in sorted(v)]
        rows.append(dict(dataset=ds, family=fam, scale=sc, n=len(seeds),
                         mean=float(np.mean(accs)), std=float(np.std(accs)),
                         min=float(np.min(accs)), max=float(np.max(accs)),
                         n_diverged=sum(x[2] for x in v), selected=0, boundary=""))
        by_fam[(ds, fam)].append(rows[-1])
    for (ds, fam), rs in sorted(by_fam.items()):
        scales = sorted(r["scale"] for r in rs)
        ok = [r for r in rs if r["n"] >= a.min_seeds]
        short = [r for r in rs if r["n"] < a.min_seeds]
        if short:
            print("CANH BAO %s/%s: %d o chua du %d seed: %s" %
                  (ds, fam, len(short), a.min_seeds, sorted(r["scale"] for r in short)))
        if not ok:
            print("KHONG chon duoc %s/%s (khong o nao du seed)" % (ds, fam))
            continue
        b = max(ok, key=lambda r: (round(r["mean"], 6), r["min"]))
        b["selected"] = 1
        b["boundary"] = ("lowest" if b["scale"] == scales[0] else
                         "highest" if b["scale"] == scales[-1] else "interior")
        best[(ds, fam)] = b

    rows.sort(key=lambda r: (r["dataset"], r["family"], r["scale"]))
    with open(os.path.join(a.out, "val_grid.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n%-9s %-9s %-8s %2s %8s %6s %7s %7s  %s" %
          ("dataset", "family", "scale", "n", "mean", "std", "min", "max", ""))
    for r in rows:
        flag = ("<== CHON (%s)" % r["boundary"]) if r["selected"] else ""
        if r["selected"] and r["boundary"] != "interior":
            flag += "  !! o BIEN luoi: noi ro trong bai, khong goi la 'optimum'"
        print("%-9s %-9s %-8g %2d %8.2f %6.2f %7.2f %7.2f  %s" %
              (r["dataset"], r["family"], r["scale"], r["n"], r["mean"], r["std"],
               r["min"], r["max"], flag))

    with open(os.path.join(a.out, "final_plan.tsv"), "w") as fh:
        fh.write("dataset\tfamily\tscale\toptimizer\tschedule\teta1\tk0\tlr\n")
        for (ds, fam), b in sorted(best.items()):
            sc = b["scale"]
            if fam == "pgd":
                fh.write("%s\t%s\t%g\tpgd\t-\t-\t-\t%g\n" % (ds, fam, sc, sc))
            elif fam == "ell":
                fh.write("%s\t%s\t%g\tfw\tharmonic\t%g\t%g\t-\n" % (ds, fam, sc, sc, sc))
            else:
                fh.write("%s\t%s\t%g\tfw\t%s\t%g\t0\t-\n" % (ds, fam, sc, fam, sc))
    print("\nghi: %s/val_grid.csv va %s/final_plan.tsv (%d cau hinh duoc chon)"
          % (a.out, a.out, len(best)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
