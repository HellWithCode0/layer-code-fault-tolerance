#!/usr/bin/env python3
"""Code-capacity distances of small explicit layer codes, bracketed from both sides.

d_X = min |v| over v in ker H_Z not in rowspace H_X.  The upper bound is the
weight of an explicit undetectable logical found by stim's
search_for_undetectable_logical_errors; the lower bound is proved by CP-SAT
(min_weight_logical.py), minimizing |v| subject to H_Z v = 0 and <v, Zbar_k> = 1
for each basis observable k; d_Z dually.  The distance is exact when the two
bounds meet.  Output: explicit_layer_code_distance.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from layer_code_explicit import build_layer_code, x_slab, z_slab  # noqa: E402
from min_weight_logical import min_weight_logical  # noqa: E402
from slab_gluing_certificate import nullspace  # noqa: E402


def dem_for(checks, logicals, n):
    """Code-capacity DEM: one mechanism per qubit."""
    cols = [[] for _ in range(n)]
    for r in range(checks.shape[0]):
        for q in checks.indices[checks.indptr[r]:checks.indptr[r + 1]]:
            cols[q].append(f"D{r}")
    for k, v in enumerate(logicals):
        for q in np.nonzero(v)[0]:
            cols[q].append(f"L{k}")
    return stim.DetectorErrorModel("\n".join(f"error(0.01) {' '.join(c)}" for c in cols if c))


def stim_upper(det, logs, n, P, R, Err):
    """Weight of an explicit undetectable logical found by stim's search."""
    c = stim.Circuit()
    c.append(R, range(n))
    c.append(Err, range(n), 0.01)
    for r in range(det.shape[0]):
        t = []
        for q in det.indices[det.indptr[r]:det.indptr[r + 1]]:
            t += [stim.target_pauli(int(q), P), stim.target_combiner()]
        c.append("MPP", t[:-1])
        c.append("DETECTOR", [stim.target_rec(-1)])
    for k, v in enumerate(logs):
        t = []
        for q in np.nonzero(v)[0]:
            t += [stim.target_pauli(int(q), P), stim.target_combiner()]
        c.append("MPP", t[:-1])
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], k)
    return len(c.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=False))


def distances(lc, time_limit=180):
    HX, HZ = lc.HX.tocsr(), lc.HZ.tocsr()
    zlog = [z_slab(lc, z, 0) for z in nullspace(lc.HX_in)]
    xlog = [x_slab(lc, x, 0) for x in nullspace(lc.HZ_in)]
    out = {}
    for name, det, logs, P, R, Err in (("dX", HZ, zlog, "Z", "R", "X_ERROR"),
                                       ("dZ", HX, xlog, "X", "RX", "Z_ERROR")):
        ub = stim_upper(det, logs, lc.n, P, R, Err)
        dem = dem_for(det, logs, lc.n)
        lb = None
        for k in range(len(logs)):
            r = min_weight_logical(dem, k, time_limit)
            v = r["distance"] if r["status"] == "OPTIMAL" else r["lower_bound"]
            lb = v if lb is None else min(lb, v)
        out[name] = ub
        out[name + "_lower"] = lb
        out[name + "_exact"] = (lb == ub)
    return out


def main():
    rows = []
    cases = [("[[4,2,2]]", np.ones((1, 4), np.uint8), (2, 3, 4, 5)),
             ("[[6,4,2]]", np.ones((1, 6), np.uint8), (2, 3, 4))]
    for name, H, spacings in cases:
        for s in spacings:
            lc = build_layer_code(H, H, spacing=s)
            d = distances(lc)
            row = dict(input=name, spacing=s, n=lc.n, Lx=lc.geometry["Lx"], Lz=lc.geometry["Lz"], **d)
            rows.append(row)
            print(row, flush=True)
            with open(ROOT / "explicit_layer_code_distance.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
