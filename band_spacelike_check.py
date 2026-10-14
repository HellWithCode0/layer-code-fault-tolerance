#!/usr/bin/env python3
"""Spacelike Z distances per logical class of the band_m2 block (routed-campaign input).

The routed campaign (Supplemental Sec. two-block-routed) uses a second input, the
6-qubit 'band' code with checks 111100 and 001111 for both X and Z (K=2, d_in=2).
Its spacing-2 realization has n=720, L_x=L_z=6.  The routed X-basis witnesses of
weight 6 flip the measured logical X1 (input class of 101010, slab weight 29, which
matches the campaign's X1_weight column).  An X-basis observable is flipped by Z-type
errors, so the relevant code-capacity quantity is the lightest Z-type operator that
commutes with every X check and anticommutes with X1.  This script computes it, and
the same for the other two nontrivial X classes, exactly with CP-SAT.

Writes numerical/routed_campaign/band_spacelike.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from layer_code_explicit import build_layer_code, x_slab  # noqa: E402
from min_weight_logical import min_weight_logical  # noqa: E402

H = np.array([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]], np.uint8)
CLASSES = {
    "X1 (class of 101010)": np.array([1, 0, 1, 0, 1, 0], np.uint8),
    "class of 110000": np.array([1, 1, 0, 0, 0, 0], np.uint8),
    "class of 011010": np.array([0, 1, 1, 0, 1, 0], np.uint8),
}


def lightest(HX, n, target, time_limit):
    """min |z| subject to HX z = 0 and <z, target> = 1."""
    HX = HX.tocsr()
    cols = [[] for _ in range(n)]
    for r in range(HX.shape[0]):
        for q in HX.indices[HX.indptr[r]:HX.indptr[r + 1]]:
            cols[q].append(f"D{r}")
    for q in np.nonzero(target)[0]:
        cols[q].append("L0")
    dem = stim.DetectorErrorModel("\n".join(f"error(0.01) {' '.join(c)}" for c in cols if c))
    return min_weight_logical(dem, 0, time_limit, workers=8)


def main():
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
    lc = build_layer_code(H, H, spacing=2)
    out = ROOT / "numerical" / "routed_campaign" / "band_spacelike.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for lab, xin in CLASSES.items():
        x = x_slab(lc, xin, 0)
        r = lightest(lc.HX, lc.n, x, tl)
        row = dict(input="band_m2", spacing=2, n=lc.n, x_in="".join(map(str, xin)),
                   slab_weight=int(x.sum()), anticommutes_with=lab,
                   weight=r["distance"], lower_bound=r["lower_bound"], status=r["status"])
        rows.append(row)
        print(row, flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
