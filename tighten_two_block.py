#!/usr/bin/env python3
"""Longer CP-SAT pass for selected two-block fault distances.

Recomputes the named observable of the named interfaces with a longer time
limit and writes the tighter bracket (the larger proven lower bound and the
smaller found pattern) back into two_block_distance.csv.

    python tighten_two_block.py --configs thick:0:1,thick:0:2 --observable Z1aZ1b --time-limit 1200
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from two_block_measurement import TwoBlock, build_circuit  # noqa: E402
from min_weight_logical import circuit_distances  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="thick:0:1,thick:0:2")
    ap.add_argument("--observable", default="Z1aZ1b")
    ap.add_argument("--time-limit", type=float, default=1200.0)
    ap.add_argument("--csv", default=str(ROOT / "two_block_distance.csv"))
    args = ap.parse_args()

    basis = "Z" if args.observable.startswith("Z") else "X"
    rows = list(csv.DictReader(open(args.csv)))
    for spec in args.configs.split(","):
        interface, w, m = spec.split(":")
        tb = TwoBlock(s=2, interface=interface, width=int(w), mult=int(m))
        lab = f"bridge w={w}" if interface == "bridge" else f"thick x{m}"
        circ, names = build_circuit(tb, 0.002, basis, 4, 4, 4)
        d = circuit_distances(circ, names, time_limit=args.time_limit, sector_coord=2,
                              sector_values={0, 2} if basis == "X" else {1, 3},
                              only={args.observable})[args.observable]
        for r in rows:
            if r["interface"] != lab:
                continue
            k = f"fd_{args.observable}"
            old_d, old_lo = int(r[k]), int(r[k + "_lower"])
            new_d = min(old_d, d["distance"]) if d["distance"] is not None else old_d
            new_lo = max(old_lo, d["lower_bound"])
            r[k], r[k + "_lower"] = str(new_d), str(new_lo)
            r[k + "_status"] = "OPTIMAL" if new_d == new_lo else "FEASIBLE"
            print(lab, args.observable, "bracket", new_lo, new_d, flush=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
