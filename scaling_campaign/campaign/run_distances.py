#!/usr/bin/env python3
"""Deterministic companions to the Monte Carlo campaign (CP-SAT, OR-Tools).

  python campaign/run_distances.py fault --spacings 2 3 --time-limit 3600
  python campaign/run_distances.py code  --spacings 2   --time-limit 7200

fault: circuit-level fault distance of the joint measurement for the thin
       bridges (w = 1, 2, 3) and the thick grid, at each spacing.  For the
       measured pair Z1(a)Z1(b) the paper proves: thin bridge w -> exactly w,
       thick grid -> the block distance.  The paper computed s = 2 only.
       Output: results/fault_distance.csv
code:  code distances of the explicit realization of the Steane code
       [[7,1,3]] (d_in = 3, check weight 4), compared with the distance
       theorem d_X >= (L_x+1) ceil(d_in/eta), d_Z >= L_z ceil(d_in/eta),
       eta = floor(4/2) = 2.  The paper's exact comparisons cover d_in = 2 only.
       Output: results/code_distance.csv

Each value is bracketed: "distance" is the weight of the best undetectable
logical fault found (an upper bound), "lower_bound" is proved by CP-SAT; the
status is OPTIMAL when they meet.  Every row is written as soon as it is done,
and rows already in the CSV are skipped, so the script can be restarted.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import stim  # noqa: E402


def append(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def done_keys(path, cols):
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {tuple(r[c] for c in cols) for r in csv.DictReader(fh)}


def fault(args):
    import two_block_family as F
    from min_weight_logical import min_weight_logical
    out = RES / "fault_distance.csv"
    seen = done_keys(out, ["s", "interface", "observable"])
    configs = [("bridge", 1, 1), ("bridge", 2, 1), ("bridge", 3, 1), ("thick", 0, 1)]
    for s in args.spacings:
        for iface, w, m in configs:
            lab = f"bridge w={w}" if iface == "bridge" else "thick"
            tb = F.TwoBlock(k=2, s=s, interface=iface, width=max(1, w), mult=m)
            R = 2 * s + 1
            for basis, nm, sec in (("Z", "Z1aZ1b", {1, 3}), ("X", "m", {0, 2})):
                if (str(s), lab, nm) in seen:
                    continue
                circ, names = F.build_circuit(tb, 0.001, basis, R, R, R)
                dem = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
                coords = dem.get_detector_coordinates()
                keep = {d for d, c in coords.items() if int(c[2]) in sec}
                t0 = time.time()
                r = min_weight_logical(dem, names.index(nm), args.time_limit, workers=args.threads,
                                       keep_detectors=keep)
                # expected: Z1aZ1b -> min(w, d_X) (bridge) or d_X = 2s+1 (thick); m -> d_Z = 2s
                expect = (min(w, 2 * s + 1) if iface == "bridge" else 2 * s + 1) if nm == "Z1aZ1b" else 2 * s
                row = dict(s=s, d_X=2 * s + 1, d_Z=2 * s, interface=lab, interface_cut=tb.cut_size(),
                           rounds=R, observable=nm, expected=expect, distance=r.get("distance"),
                           lower_bound=r.get("lower_bound"), status=r["status"],
                           mechanisms=r.get("mechanisms"), seconds=round(time.time() - t0, 1))
                append(out, row)
                print(row, flush=True)


def hamming(r):
    cols = [[(c >> b) & 1 for b in range(r)] for c in range(1, 2 ** r)]
    return np.array(cols, dtype=np.uint8).T


def nullspace(H):
    import scipy.sparse as sp
    from ldpc import mod2
    K = mod2.nullspace(sp.csr_matrix(np.asarray(H, np.uint8)))
    return np.asarray(K.todense() if sp.issparse(K) else K, dtype=np.uint8)


def theorem_check(upper, lower, status, bound):
    if upper is not None and upper < bound:
        return "VIOLATED"            # an explicit logical below the proved bound
    if status == "OPTIMAL" and upper == bound:
        return "tight"
    if status == "OPTIMAL":
        return "consistent (distance above the bound)"
    return "consistent" if (lower is not None and lower >= bound) else "undetermined (raise --time-limit)"


def nontrivial(V, S):
    """Rows of V independent modulo rowspace(S): a basis of nontrivial classes."""
    from layer_code_explicit import gf2_rank
    out, cur = [], np.asarray(S, np.uint8)
    r = gf2_rank(cur)
    for v in V:
        trial = np.vstack([cur, v])
        if gf2_rank(trial) > r:
            out.append(v)
            cur, r = trial, r + 1
    return out


def code(args):
    from layer_code_explicit import build_layer_code, x_slab, z_slab
    from min_weight_logical import min_weight_logical
    out = RES / "code_distance.csv"
    seen = done_keys(out, ["input", "spacing", "sector"])
    H = hamming(3)
    for s in args.spacings:
        lc = build_layer_code(H, H, spacing=s)
        g = lc.geometry
        HX, HZ = lc.HX.tocsr(), lc.HZ.tocsr()
        # one representative per nontrivial class (K = 1 for the Steane code)
        zlog = [z_slab(lc, z, 0) for z in nontrivial(nullspace(lc.HX_in), lc.HZ_in)]
        xlog = [x_slab(lc, x, 0) for x in nontrivial(nullspace(lc.HZ_in), lc.HX_in)]
        bound = {"dX": (g["Lx"] + 1) * 2, "dZ": g["Lz"] * 2}     # ceil(3/2) = 2
        for sector, det, logs in (("dX", HZ, zlog), ("dZ", HX, xlog)):
            if ("[[7,1,3]]", str(s), sector) in seen:
                continue
            cols = [[] for _ in range(lc.n)]
            for r in range(det.shape[0]):
                for q in det.indices[det.indptr[r]:det.indptr[r + 1]]:
                    cols[q].append(f"D{r}")
            for kk, v in enumerate(logs):
                for q in np.nonzero(v)[0]:
                    cols[q].append(f"L{kk}")
            dem = stim.DetectorErrorModel("\n".join(f"error(0.01) {' '.join(c)}" for c in cols if c))
            t0 = time.time()
            best, lb, status = None, None, "OPTIMAL"
            for kk in range(len(logs)):
                r = min_weight_logical(dem, kk, args.time_limit, workers=args.threads)
                if r.get("distance") is not None:
                    best = r["distance"] if best is None else min(best, r["distance"])
                v = r["lower_bound"]
                lb = v if lb is None else min(lb, v)
                if r["status"] != "OPTIMAL":
                    status = r["status"]
            row = dict(input="[[7,1,3]]", spacing=s, n=lc.n, Lx=g["Lx"], Lz=g["Lz"], sector=sector,
                       theorem_bound=bound[sector], distance_upper=best, lower_bound=lb, status=status,
                       theorem_check=theorem_check(best, lb, status, bound[sector]),
                       seconds=round(time.time() - t0, 1))
            append(out, row)
            print(row, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["fault", "code"])
    ap.add_argument("--spacings", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--time-limit", type=float, default=3600.0, help="CP-SAT seconds per observable")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    {"fault": fault, "code": code}[args.part](args)


if __name__ == "__main__":
    main()
