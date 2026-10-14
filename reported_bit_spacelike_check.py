#!/usr/bin/env python3
"""Which spacelike Z logicals can flip the reported bit of the two-block measurement?

For one [[4,2,2]]-input block of the two-block circuits (Supplemental Secs.
two-block and two-block-scaling) this computes, exactly with CP-SAT, the
lightest Z-type operator that commutes with every X check and anticommutes
with a chosen X logical:

  X1        the measured logical (a Z error of this class flips the reported bit),
  X2        the spectator,
  X1 X2     either one but not both.

The block's Z distance d_Z = 2s is attained by operators that flip the
spectator only; every operator that flips the reported bit is much heavier.
So the reported bit's circuit fault distance is set by the number of
deformed rounds, as the fault distances in fault_distance.csv show.

Writes numerical/scaling_campaign/reported_bit_spacelike.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scaling_campaign" / "campaign"))
import two_block_family as F  # noqa: E402
from min_weight_logical import min_weight_logical  # noqa: E402


def lightest(tb, target, time_limit=600.0):
    """min |z| subject to HX z = 0 and <z, target> = 1."""
    HX = tb.lc.HX.tocsr()
    cols = [[] for _ in range(tb.lc.n)]
    for r in range(HX.shape[0]):
        for q in HX.indices[HX.indptr[r]:HX.indptr[r + 1]]:
            cols[q].append(f"D{r}")
    for q in np.nonzero(target)[0]:
        cols[q].append("L0")
    dem = stim.DetectorErrorModel("\n".join(f"error(0.01) {' '.join(c)}" for c in cols if c))
    return min_weight_logical(dem, 0, time_limit, workers=8)


def main():
    out = ROOT / "numerical" / "scaling_campaign" / "reported_bit_spacelike.csv"
    rows = []
    for s in (2, 3):
        tb = F.TwoBlock(k=2, s=s, interface="thick")
        for lab, x in (("X1", tb.Xs[0]), ("X2", tb.Xs[1]), ("X1X2", (tb.Xs[0] + tb.Xs[1]) % 2)):
            r = lightest(tb, x)
            row = dict(s=s, n=tb.n, d_Z=2 * s, anticommutes_with=lab, weight=r["distance"],
                       lower_bound=r["lower_bound"], status=r["status"])
            rows.append(row)
            print(row, flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
