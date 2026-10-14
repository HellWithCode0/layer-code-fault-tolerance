#!/usr/bin/env python3
"""Merge chunked two-block Monte Carlo runs and add Wilson intervals.

two_block_measurement.py --skip-code-distance --shots S --seed B --out F
writes, for every interface, failure counts of the reported bit and of every
logical frame in the X- and Z-basis experiments, together with the integer
seed of each (interface, basis) run.  The archived campaign ran four chunks of
250 shots per interface with base seeds 20260926 + 1000 k (k = 0..3), in
parallel.  This script sums the counts over chunks, checks that no seed is
used twice, and writes two_block_mc.csv with the pooled failure rates and
two-sided 95% Wilson score intervals.

Usage: python merge_two_block_mc.py chunk1.csv chunk2.csv ... [--out two_block_mc.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "two_block_mc.csv"))
    args = ap.parse_args()
    pooled = OrderedDict()
    seeds = set()
    for f in args.chunks:
        for r in csv.DictReader(open(f)):
            lab = r["interface"]
            P = pooled.setdefault(lab, dict(interface=lab, interface_cut=r["interface_cut"],
                                            edge_qubits=r["edge_qubits"], p=r["p"], shots=0,
                                            counts={}, seeds=[]))
            shots = int(r["shots"])
            P["shots"] += shots
            for basis in ("X", "Z"):
                sd = int(r[f"seed_{basis}"])
                assert sd not in seeds, f"seed {sd} used twice"
                seeds.add(sd)
                P["seeds"].append(sd)
            for k, v in r.items():
                if k.startswith("nfail_"):
                    P["counts"][k[6:]] = P["counts"].get(k[6:], 0) + int(v)
    rows = []
    for lab, P in pooled.items():
        row = dict(interface=lab, interface_cut=P["interface_cut"], edge_qubits=P["edge_qubits"],
                   p=P["p"], shots_per_basis=P["shots"],
                   seeds=" ".join(map(str, P["seeds"])))
        for name, k in P["counts"].items():
            lo, hi = wilson(k, P["shots"])
            row[f"nfail_{name}"] = k
            row[f"rate_{name}"] = round(k / P["shots"], 5)
            row[f"wilson95_{name}"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["interface"], r["shots_per_basis"], {k: r[k] for k in r if k.startswith("nfail_")})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
