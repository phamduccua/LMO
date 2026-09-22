"""What is already on disk, per (group, dataset, model, epoch budget).

Run this BEFORE resuming a stage: the resume logic keys on the run directory
name, which contains the epoch budget and every swept hyperparameter, so a
command with a different --epochs or --grid does not continue the old runs, it
starts a parallel set beside them.  This prints the budgets that exist so the
resume command can be given the matching ones.

    python scripts/show_runs.py              # everything
    python scripts/show_runs.py ablation     # one group

"done" counts runs with summary.json (finished, will be SKIPPED on resume);
"partial" counts run directories without one (killed mid-run: they will be
re-run from scratch and their directory overwritten, which is the intent --
a partial run is not usable).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "results", "runs")


def main() -> None:
    want = sys.argv[1:]
    if not os.path.isdir(RUNS):
        raise SystemExit("no runs yet under %s" % RUNS)

    # (group, dataset, model, epochs) -> {"done": n, "partial": n, "seeds": set}
    tally: dict = defaultdict(lambda: {"done": 0, "partial": 0, "seeds": set()})
    for d in sorted(glob.glob(os.path.join(RUNS, "*", "*"))):
        if not os.path.isdir(d):
            continue
        group = os.path.basename(os.path.dirname(d))
        if want and group not in want:
            continue
        f = os.path.join(d, "summary.json")
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    c = json.load(fh)["config"]
            except (json.JSONDecodeError, KeyError):
                continue
            k = (group, c["dataset"], c["model"], int(c["epochs"]))
            tally[k]["done"] += 1
            tally[k]["seeds"].add(int(c["seed"]))
        else:
            # no summary.json: parse the name, which carries the same fields
            # (dataset__model__opt__tag__e<epochs>__s<seed>)
            p = os.path.basename(d).split("__")
            if len(p) < 6 or not p[-2].startswith("e"):
                continue
            k = (group, p[0], p[1], int(p[-2][1:]))
            tally[k]["partial"] += 1

    if not tally:
        raise SystemExit("nothing matched%s" % (" %s" % want if want else ""))

    print("%-10s %-9s %-10s %-8s %-6s %-8s %s"
          % ("group", "dataset", "model", "epochs", "done", "partial", "seeds"))
    for k in sorted(tally):
        v = tally[k]
        print("%-10s %-9s %-10s %-8d %-6d %-8d %s"
              % (k[0], k[1], k[2], k[3], v["done"], v["partial"],
                 ",".join(str(s) for s in sorted(v["seeds"])) or "-"))
    tot = sum(v["done"] for v in tally.values())
    par = sum(v["partial"] for v in tally.values())
    print("\n%d finished, %d partial (partial ones get re-run and overwritten)"
          % (tot, par))


if __name__ == "__main__":
    main()
