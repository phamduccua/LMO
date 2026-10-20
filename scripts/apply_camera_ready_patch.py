#!/usr/bin/env python3
"""Cho phep eta1 > 1 o ba schedule giam dan (harmonic, power, mlogm).

gamma_k VAN bi cat o <= 1 (min(1.0, ...) co san trong code), nen khong co gi
vo nghia xay ra; chi la rao chan `eta1 <= 1` trong _check_eta bi go cho ba
lop nay.  `constant` van giu rao chan cu (eta1 <= 1).

Idempotent (chay 2 lan khong sao), giu nguyen kieu xuong dong (CRLF/LF), tao
ban sao luu optimizers/schedules.py.pre_camera_ready.

    python scripts/apply_camera_ready_patch.py            # tu thu muc goc cua repo
    python scripts/apply_camera_ready_patch.py --repo /duong/dan/LMO
"""
import argparse
import pathlib
import shutil
import sys

OLD_FN = (
    "def _check_eta(eta1: float) -> None:\n"
    "    if not 0.0 < eta1 <= 1.0:\n"
    "        raise ValueError(\"eta1 must lie in (0, 1] so that gamma_k <= 1\")\n"
)
NEW_FN = (
    "# camera-ready: eta1 > 1 is allowed for the diminishing schedules; gamma_k is\n"
    "# still clipped to <= 1 inside __call__.  `constant` keeps the old bound.\n"
    "ETA1_MAX = 1000.0\n"
    "\n"
    "\n"
    "def _check_eta(eta1: float, upper: float = 1.0) -> None:\n"
    "    if not 0.0 < eta1 <= upper:\n"
    "        raise ValueError(\"eta1 must lie in (0, %g]\" % upper)\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    p = pathlib.Path(a.repo) / "optimizers" / "schedules.py"
    if not p.exists():
        print("khong thay %s -- chay tu thu muc goc repo hoac dung --repo" % p)
        return 2
    raw = p.read_bytes().decode("utf-8")
    if "ETA1_MAX" in raw:
        print("da patch roi: %s" % p)
        return 0
    crlf = "\r\n" in raw
    t = raw.replace("\r\n", "\n")
    if t.count(OLD_FN) != 1:
        print("KHONG khop ham _check_eta ban dau (%d lan) -- dung lai, bao co."
              % t.count(OLD_FN))
        return 3
    t = t.replace(OLD_FN, NEW_FN)
    parts = t.split("\nclass ")
    changed = 0
    for i, seg in enumerate(parts):
        if seg.startswith(("Harmonic(", "Power(", "MLogM(")):
            n = seg.count("_check_eta(eta1)")
            changed += n
            parts[i] = seg.replace("_check_eta(eta1)", "_check_eta(eta1, upper=ETA1_MAX)")
    if changed != 3:
        print("mong doi 3 cho goi _check_eta (Harmonic/Power/MLogM), thay %d -- dung lai." % changed)
        return 4
    t = "\nclass ".join(parts)
    if crlf:
        t = t.replace("\n", "\r\n")
    shutil.copyfile(p, str(p) + ".pre_camera_ready")
    p.write_bytes(t.encode("utf-8"))
    print("OK: da sua %s (3 cho goi), ban sao luu: %s.pre_camera_ready" % (p, p.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
