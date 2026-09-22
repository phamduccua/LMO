"""Generate the LaTeX tables of ../paper_cv directly from results/runs/*/summary.json.

Every number in the report comes from here, so text, tables and runs cannot
drift apart.  Statistic: median [IQR] over the report seeds when there are 3+,
mean +- sample std when there are 2, the bare value when there is 1 -- the
caption of each table says which, via `\\statnote`.

    python make_paper_tables.py                  # -> ../paper_cv/tables/*.tex
    python make_paper_tables.py --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "runs")
OUT = os.path.join(HERE, "..", "paper_cv", "tables")
sys.path.insert(0, HERE)

GROUPS = ("ablation", "baselines", "eta", "fwadam", "constlr", "block",
          "tune")
DATASETS = ("cifar10", "cifar100")
MODELS = ("resnet18", "resnet34", "wrn28_10")
METRIC = "eval_acc_final"

BASE_ORDER = ["gda", "pgd", "ogda", "eg", "la_gda", "sgdm", "adam", "oadam",
              "eadam", "la_adam"]
BASE_NAME = {"gda": "GDA (SGD)", "pgd": r"PGD ($R{=}2$)", "ogda": "OGDA",
             "eg": "EG", "la_gda": "Lookahead-GDA", "sgdm": "SGD + momentum",
             "adam": "Adam", "oadam": "Optimistic Adam", "eadam": "Extra-Adam",
             "la_adam": "Lookahead-Adam"}
MODEL_NAME = {"resnet18": "ResNet-18", "resnet34": "ResNet-34",
              "wrn28_10": "WRN-28-10"}
DATA_NAME = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}

SEEDS: Sequence[int] = (1, 2, 3)


def load_all() -> List[dict]:
    out = []
    for g in GROUPS:
        for f in glob.glob(os.path.join(RUNS, g, "*", "summary.json")):
            try:
                with open(f) as fh:
                    s = json.load(fh)
            except json.JSONDecodeError:           # a run still being written
                continue
            s["_group"], s["_dir"] = g, os.path.dirname(f)
            out.append(s)
    return out


ALL: List[dict] = []


def cell_epochs(dataset: str, model: str) -> Optional[int]:
    """The epoch budget reported for this cell: the largest one on disk.

    Runs of different budgets must never be averaged together (a 20-epoch
    `quick` run and a 40-epoch `standard` run of the same configuration are
    different experiments), so every table filters on this.  The tuning group is
    exempt -- it runs short on purpose.
    """
    have = {int(s["config"]["epochs"]) for s in ALL
            if s["_group"] != "tune" and s["config"]["dataset"] == dataset
            and s["config"]["model"] == model}
    return max(have) if have else None


def sel(group: str, dataset: str, model: str, **cfg) -> List[dict]:
    res = []
    ep = cell_epochs(dataset, model)
    for s in ALL:
        c = s["config"]
        if (s["_group"] != group or c["dataset"] != dataset
                or c["model"] != model or c["seed"] not in SEEDS):
            continue
        if group != "tune" and ep is not None and int(c["epochs"]) != ep:
            continue
        ok = True
        for k, v in cfg.items():
            cv = c.get(k)
            if isinstance(v, float):
                ok &= cv is not None and abs(float(cv) - v) < 1e-9
            else:
                ok &= cv == v
        if ok:
            res.append(s)
    return res


# --------------------------------------------------------------------------- #
def stat(vals: Sequence[float], fmt: str = "%.2f") -> str:
    """median [IQR] for 3+ seeds, mean +- std for 2, the value itself for 1."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return "--"
    if v.size == 1:
        return fmt % v[0]
    if v.size == 2:
        return (r"%s\,$\pm$\,%s" % (fmt, fmt)) % (v.mean(), v.std(ddof=1))
    return (r"%s\,\tiny{[%s, %s]}" % (fmt, fmt, fmt)) % (
        np.median(v), np.quantile(v, .25), np.quantile(v, .75))


def best_of(cands: Dict) -> Optional[tuple]:
    """The key of the cell with the best median metric, or None if all empty."""
    live = {k: v for k, v in cands.items() if v}
    if not live:
        return None
    return max(live, key=lambda k: np.median([s[METRIC] for s in live[k]]))


def write(name: str, text: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(text)
    print("wrote %s (%d lines)" % (os.path.join("tables", name),
                                   text.count("\n")))


def fw_cells(dataset: str, model: str, group: str = "ablation",
             opt: str = "fw") -> Dict[tuple, List[dict]]:
    cells: Dict[tuple, List[dict]] = {}
    ep = cell_epochs(dataset, model)
    for s in ALL:
        c = s["config"]
        if (s["_group"] != group or c["dataset"] != dataset
                or c["model"] != model or c["optimizer"] != opt
                or c["seed"] not in SEEDS):
            continue
        if ep is not None and int(c["epochs"]) != ep:     # never mix budgets
            continue
        cells.setdefault((c["schedule"], float(c["eta1"]),
                          float(c["radius"])), []).append(s)
    return cells


# --------------------------------------------------------------------------- #
def table_main() -> None:
    """One row per method, one column pair per (dataset, model) cell."""
    lines = []
    for opt in BASE_ORDER:
        cells = []
        for ds in DATASETS:
            for md in MODELS:
                ss = sel("baselines", ds, md, optimizer=opt)
                cells.append(stat([s[METRIC] for s in ss]))
        if any(c != "--" for c in cells):
            lines.append("%s & %s \\\\" % (BASE_NAME[opt], " & ".join(cells)))
    lines.append(r"\midrule")
    for label, grp, opt in (("FW (harmonic)", "ablation", "fw"),
                            ("FW (power)", "ablation", "fw"),
                            ("FW-Adam (harmonic)", "fwadam", "fw_adam"),
                            ("FW-Adam (power)", "fwadam", "fw_adam")):
        sch = "harmonic" if "harmonic" in label else "power"
        cells = []
        for ds in DATASETS:
            for md in MODELS:
                cd = {k: v for k, v in fw_cells(ds, md, grp, opt).items()
                      if k[0] == sch}
                kb = best_of(cd)
                cells.append(stat([s[METRIC] for s in cd[kb]]) if kb else "--")
        if any(c != "--" for c in cells):
            lines.append("%s & %s \\\\" % (label, " & ".join(cells)))
    write("main.tex", "\n".join(lines) + "\n")


def table_ablation(dataset: str, model: str, fname: str) -> None:
    """The FW grid: schedule x R, with the FW-Adam twin of each cell."""
    fw = fw_cells(dataset, model)
    fwa = fw_cells(dataset, model, "fwadam", "fw_adam")
    radii = sorted({k[2] for k in fw} | {k[2] for k in fwa})
    lines = []
    for R in radii:
        cells = []
        for grp in (fw, fwa):
            for sch in ("harmonic", "power"):
                ss = grp.get((sch, 1.0, R), [])
                cells.append(stat([s[METRIC] for s in ss]))
        # geometry: ||theta_final|| / R||theta_0|| for the harmonic FW cell
        ss = fw.get(("harmonic", 1.0, R), [])
        cells.append(stat([s["theta_final_norm"] / max(s["theta0_norm"] * R, 1e-9)
                           for s in ss], "%.3f"))
        lines.append("$%g$ & %s \\\\" % (R, " & ".join(cells)))
    write(fname, "\n".join(lines) + "\n")


def table_full_metrics(dataset: str, model: str, fname: str) -> None:
    """Train and test metrics side by side for the leaders of one cell."""
    rows: List[tuple] = []
    for opt in BASE_ORDER:
        ss = sel("baselines", dataset, model, optimizer=opt)
        if ss:
            rows.append((BASE_NAME[opt], ss))
    for grp, opt, label in (("ablation", "fw", "FW"),
                            ("fwadam", "fw_adam", "FW-Adam")):
        cd = fw_cells(dataset, model, grp, opt)
        for sch in ("harmonic", "power"):
            kb = best_of({k: v for k, v in cd.items() if k[0] == sch})
            if kb:
                rows.append(("%s %s ($R{=}%g$)" % (label, sch, kb[2]), cd[kb]))
    lines = []
    for label, ss in rows:
        lines.append("%s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            label,
            stat([s["train_loss_final"] for s in ss], "%.3f"),
            stat([s["train_acc_final"] for s in ss]),
            stat([s["eval_loss_final"] for s in ss], "%.3f"),
            stat([s[METRIC] for s in ss]),
            stat([s["eval_top5_final"] for s in ss]),
            stat([s["eval_ece_final"] for s in ss]),
            stat([s["gen_gap_final"] for s in ss])))
    write(fname, "\n".join(lines) + "\n")


def table_speed(dataset: str, model: str, fname: str) -> None:
    """Convergence speed: epochs to each accuracy target, and wall clock."""
    from metrics import TARGETS
    tg = TARGETS[dataset]
    rows: List[tuple] = []
    for opt in BASE_ORDER:
        ss = sel("baselines", dataset, model, optimizer=opt)
        if ss:
            rows.append((BASE_NAME[opt], ss))
    for grp, opt, label in (("ablation", "fw", "FW"),
                            ("fwadam", "fw_adam", "FW-Adam")):
        cd = fw_cells(dataset, model, grp, opt)
        kb = best_of(cd)
        if kb:
            rows.append(("%s %s ($R{=}%g$)" % (label, kb[0], kb[2]), cd[kb]))
    lines = []
    for label, ss in rows:
        cells = []
        for t in tg:
            v = [s.get("ep_to_%g" % t) for s in ss]
            v = [x for x in v if x is not None]
            cells.append(("%.0f" % np.median(v)) if v else r"--")
        lines.append("%s & %s & %s & %s \\\\" % (
            label, " & ".join(cells),
            stat([s["sec_per_epoch"] for s in ss], "%.0f"),
            stat([float(s["grad_evals"]) / 1000 for s in ss], "%.0f")))
    write(fname, "\n".join(lines) + "\n")


def table_eta() -> None:
    """gamma_1 = eta1 sweep, one block per (dataset, model) that has runs."""
    lines = []
    for ds in DATASETS:
        for md in MODELS:
            cells_any = False
            block = []
            for sch in ("harmonic", "power"):
                for e in (1.0, 0.1, 0.01, 0.001):
                    row = []
                    for R in (1.0, 2.0, 5.0):
                        grp = "ablation" if e == 1.0 else "eta"
                        ss = sel(grp, ds, md, optimizer="fw", schedule=sch,
                                 eta1=e, radius=R)
                        row.append(stat([s[METRIC] for s in ss]))
                    if any(c != "--" for c in row):
                        cells_any = True
                        block.append("%s & $%g$ & %s \\\\" % (sch, e, " & ".join(row)))
            if cells_any:
                lines.append(r"\multicolumn{5}{l}{\textit{%s / %s}} \\"
                             % (DATA_NAME[ds], MODEL_NAME[md]))
                lines += block
                lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    write("eta.tex", "\n".join(lines) + "\n")


def table_constlr() -> None:
    """Constant step gamma_k = gamma, for FW and FW-Adam, each against its own
    harmonic cell at the same R.

    Paired down to the seed: same initialisation, same data order, same R, same
    block structure.  The only difference is that gamma_k stops decaying, i.e.
    exactly the condition of (FW) that is being dropped.
    """
    lines = []
    for ds in DATASETS:
        for md in MODELS:
            block = []
            for opt, label in (("fw", "FW"), ("fw_adam", "FW-Adam")):
                for e in (1.0, 0.1, 0.01, 0.001):
                    row, any_cell = [], False
                    for R in (1.0, 2.0, 5.0):
                        c = stat([s[METRIC] for s in
                                  sel("constlr", ds, md, optimizer=opt,
                                      schedule="constant", eta1=e, radius=R)])
                        any_cell = any_cell or c != "--"
                        row.append(c)
                    if any_cell:
                        block.append(r"%s & $\gamma{=}%g$ & %s \\"
                                     % (label, e, " & ".join(row)))
                grp = "ablation" if opt == "fw" else "fwadam"
                ref = [stat([s[METRIC] for s in
                             sel(grp, ds, md, optimizer=opt, schedule="harmonic",
                                 eta1=1.0, radius=R)])
                       for R in (1.0, 2.0, 5.0)]
                if any(v != "--" for v in ref):
                    block.append(r"%s & harmonic & %s \\" % (label, " & ".join(ref)))
            if block:
                lines.append(r"\multicolumn{5}{l}{\textit{%s / %s}} \\"
                             % (DATA_NAME[ds], MODEL_NAME[md]))
                lines += block
                lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    write("constlr.tex", "\n".join(lines) + "\n")


def table_blocks() -> None:
    """How finely C is split: one ball, per module, or per tensor."""
    lines = []
    for ds in DATASETS:
        for md in MODELS:
            row = []
            for b in ("all", "module", "param"):
                grp = "block" if b != "module" else "ablation"
                ss = sel(grp, ds, md, optimizer="fw", block=b, radius=2.0,
                         schedule="harmonic")
                row.append(stat([s[METRIC] for s in ss]))
            nb = [s["n_blocks"] for s in sel("ablation", ds, md, optimizer="fw",
                                             block="module")]
            if any(c != "--" for c in row):
                lines.append("%s & %s & %s & %s \\\\"
                             % (DATA_NAME[ds], MODEL_NAME[md],
                                "%d" % nb[0] if nb else "--", " & ".join(row)))
    write("blocks.tex", "\n".join(lines) + "\n")


def table_lrs() -> None:
    """The tuning stage: the lr grid and the winner, per cell."""
    lines = []
    for ds in DATASETS:
        for md in MODELS:
            for opt in BASE_ORDER:
                ss = [s for s in ALL if s["_group"] == "tune"
                      and s["config"]["dataset"] == ds
                      and s["config"]["model"] == md
                      and s["config"]["optimizer"] == opt]
                if not ss:
                    continue
                ss.sort(key=lambda s: s["config"]["lr"])
                best = max(ss, key=lambda s: -np.inf if s["diverged"] else s[METRIC])
                lines.append("%s & %s & %s & %s & $%g$ \\\\" % (
                    DATA_NAME[ds], MODEL_NAME[md], BASE_NAME[opt],
                    ", ".join("%g:%.1f%s" % (s["config"]["lr"], s[METRIC],
                                             "$^*$" if s["diverged"] else "")
                              for s in ss),
                    best["config"]["lr"]))
    write("lrs.tex", "\n".join(lines) + "\n")


def counts() -> None:
    """Run counts + the feasibility claim, as LaTeX macros the text can cite."""
    n = {g: sum(1 for s in ALL if s["_group"] == g) for g in GROUPS}
    fw = [s for s in ALL if s["config"]["optimizer"] in ("fw", "fw_adam")]
    in_c = [s.get("in_C") for s in fw]
    cells = {(s["config"]["dataset"], s["config"]["model"]) for s in ALL}
    gpu_h = sum(s.get("time_s", 0.0) for s in ALL) / 3600.0
    # main.tex defines all six with \newcommand (so it compiles before any run
    # exists) and then \@input's this file, which overrides them.
    txt = [r"\renewcommand{\nruns}{%d}" % len(ALL),
           r"\renewcommand{\nfw}{%d}" % len(fw),
           r"\renewcommand{\nfwinC}{%d}" % sum(bool(x) for x in in_c),
           r"\renewcommand{\ncells}{%d}" % len(cells),
           r"\renewcommand{\gpuhours}{%.1f}" % gpu_h,
           r"\renewcommand{\nseeds}{%d}" % len(SEEDS),
           "%% per group: %s" % n]
    write("counts.tex", "\n".join(txt) + "\n")


# figures/<cell>/<name>.png  ->  ../paper_cv/figures/<short>_<name>.png, the
# names main.tex refers to.  Keeping the copy here means one command refreshes
# every number AND every picture in the report.
FIG_SHORT = {("cifar10", "resnet18"): "c10_r18", ("cifar10", "resnet34"): "c10_r34",
             ("cifar10", "wrn28_10"): "c10_wrn", ("cifar100", "resnet18"): "c100_r18",
             ("cifar100", "resnet34"): "c100_r34", ("cifar100", "wrn28_10"): "c100_wrn"}


def copy_figures() -> None:
    import shutil
    src_root = os.path.join(HERE, "results", "figures")
    dst = os.path.join(HERE, "..", "paper_cv", "figures")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for (ds, md), short in FIG_SHORT.items():
        d = os.path.join(src_root, "%s_%s" % (ds, md))
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".png"):
                shutil.copyfile(os.path.join(d, f),
                                os.path.join(dst, "%s_%s" % (short, f)))
                n += 1
    print("copied %d figures -> paper_cv/figures" % n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    a = ap.parse_args()
    global SEEDS, ALL
    SEEDS = tuple(a.seeds)
    ALL = load_all()
    if not ALL:
        raise SystemExit("no runs found under %s -- run run_ablation.py first" % RUNS)
    print("%d runs loaded, seeds %s" % (len(ALL), list(SEEDS)))

    table_main()
    for ds in DATASETS:
        for md in MODELS:
            if sel("ablation", ds, md) or sel("baselines", ds, md):
                tag = "%s_%s" % (ds, md)
                table_ablation(ds, md, "abl_%s.tex" % tag)
                table_full_metrics(ds, md, "metrics_%s.tex" % tag)
                table_speed(ds, md, "speed_%s.tex" % tag)
    table_eta()
    table_constlr()
    table_blocks()
    table_lrs()
    counts()
    copy_figures()


if __name__ == "__main__":
    main()
