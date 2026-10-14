#!/usr/bin/env python3
"""Paired decoder check: the paper's full-model BP+LSD versus the per-sector
BP+LSD used by the campaign, on IDENTICAL shots (smallest size, s = 2).

    python campaign/decoder_check.py --shots 2000

For each point the same detector samples are decoded twice.  Reported: the
failures of each decoder and the discordant shots (only one decoder wrong),
with an exact McNemar test.  Points are chosen where failures are not rare.
Output: results/decoder_check.csv (one row per chunk, resumable).
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

POINTS = [("thick", 1, "X", 0.003), ("thick", 1, "Z", 0.003), ("bridge", 1, "Z", 0.002)]
CHUNK = 100
FIELDS = ["point", "interface", "width", "basis", "p", "chunk", "seed", "shots",
          "fail_full", "fail_sector", "only_full", "only_sector", "sec_full", "sec_sector"]


def run_chunk(args):
    sys.path.insert(0, str(HERE))
    import numpy as np
    import two_block_family as F
    from decoders import BpLsd
    (iface, w, basis, p), chunk = args
    tb = F.TwoBlock(k=2, s=2, interface=iface, width=w)
    circ, names = F.build_circuit(tb, p, basis, 5, 5, 5)
    dem = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    key = f"{iface}_w{w}_{basis}_p{p:g}"
    seed = (20260927 + zlib.crc32(f"decoder_check|{key}|{chunk}".encode())) % (2 ** 63)
    dets, obs = circ.compile_detector_sampler(seed=seed).sample(CHUNK, separate_observables=True,
                                                                bit_packed=True)
    out = {}
    for label, sec in (("full", None), ("sector", {0, 2} if basis == "X" else {1, 3})):
        comp = BpLsd(sec).compile_decoder_for_dem(dem=dem)
        t0 = time.time()
        pred = comp.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
        bad = np.unpackbits(pred ^ obs, axis=1, bitorder="little")[:, :len(names)].any(axis=1)
        out[label] = (bad, time.time() - t0)
    bf, bs = out["full"][0], out["sector"][0]
    return dict(point=key, interface=iface, width=w, basis=basis, p=p, chunk=chunk, seed=seed,
                shots=CHUNK, fail_full=int(bf.sum()), fail_sector=int(bs.sum()),
                only_full=int((bf & ~bs).sum()), only_sector=int((bs & ~bf).sum()),
                sec_full=round(out["full"][1], 2), sec_sector=round(out["sector"][1], 2))


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value for b, c discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=2000, help="shots per point")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    out = res / "decoder_check.csv"
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["point"], int(r["chunk"])))
    jobs = []
    for pt in POINTS:
        key = f"{pt[0]}_w{pt[1]}_{pt[2]}_p{pt[3]:g}"
        for c in range(args.shots // CHUNK):
            if (key, c) not in done:
                jobs.append((pt, c))
    print(f"decoder check: {len(jobs)} chunks of {CHUNK} shots to run", flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        if jobs:
            with mp.get_context("spawn").Pool(args.workers) as pool:
                for row in pool.imap_unordered(run_chunk, jobs):
                    w.writerow(row)
                    fh.flush()
    agg = {}
    with open(out, newline="") as fh:
        for r in csv.DictReader(fh):
            a = agg.setdefault(r["point"], [0, 0, 0, 0, 0])
            for i, k in enumerate(("shots", "fail_full", "fail_sector", "only_full", "only_sector")):
                a[i] += int(r[k])
    for pt, (n, ff, fs, of, os_) in agg.items():
        print(f"{pt}: shots {n}, full {ff}, sector {fs}, only-full {of}, only-sector {os_}, "
              f"McNemar p = {mcnemar_p(of, os_):.3g}")


if __name__ == "__main__":
    main()
