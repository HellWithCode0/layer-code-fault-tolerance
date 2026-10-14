#!/usr/bin/env python3
"""Scoped MWPM sanity check for one surface-code gauging rectangle.

Numerical building-block sanity check of a finite surface-code gauging /
lattice-surgery-equivalent logical-Z measurement under code-capacity noise.

Construction (scoped variant, labelled clearly):
  - Rotated surface code distance d (stim geometry; n=d², k=1).
  - Gauge/measure logical Z by a lattice-surgery–equivalent ancilla strip:
    d ancilla qubits with ZZ joints to each weight-d logical-Z data qubit,
    ancilla X chain, and deformed data X checks (matching into the strip).
    This is a WY-style suitable-graph gauging for a path-like logical support,
    realized as a thin merge strip (distance retention: d_X≈d+1, d_Z=d on the
    deformed code at the sizes tested).
  - Noise: independent code-capacity X/Z data noise with probability p.
  - Decode: PyMatching MWPM under code-capacity X/Z data noise on the deformed
    strip (perfect measurements; one cycle). R=d is the design depth of the
    gauging rectangle; the Monte Carlo is the deformed-code building block.
  - Sweep p for d∈{3,5,7}; Wilson intervals.

SCOPE: deformed surface-code/path-strip building-block sanity check under
code-capacity X/Z data noise; NOT a layer-code threshold, circuit-level
simulation, or architecture-level threshold for the full paper schedule.
"""
from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260915)


def gf2_rank(a: np.ndarray) -> int:
    a = (a.copy() % 2).astype(int)
    if a.size == 0:
        return 0
    m, n = a.shape
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if a[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        for i in range(m):
            if i != r and a[i, c]:
                a[i] = (a[i] + a[r]) % 2
        r += 1
    return r


def nullspace_gf2(H):
    A = (H.copy() % 2).astype(int)
    m, n = A.shape
    r, pivots = 0, {}
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] = (A[i] + A[r]) % 2
        pivots[c] = r
        r += 1
        if r == m:
            break
    basis = []
    for f in [c for c in range(n) if c not in pivots]:
        v = np.zeros(n, dtype=int)
        v[f] = 1
        for c, pr in pivots.items():
            v[c] = A[pr, f]
        basis.append(v)
    return basis


def reduce_mod_rows(vecs, H, n):
    cur = H.copy() if H.size else np.zeros((0, n), dtype=int)
    kept = []
    for v in vecs:
        r0 = gf2_rank(cur)
        trial = np.vstack([cur, v.reshape(1, -1)])
        if gf2_rank(trial) > r0:
            kept.append(v)
            cur = trial
    return kept


def dedup_rows(H):
    if H.size == 0:
        return H
    seen, keep = set(), []
    for i, row in enumerate(H):
        key = tuple(int(x) for x in row.tolist())
        if row.sum() == 0 or key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return H[keep] if keep else np.zeros((0, H.shape[1]), dtype=int)


def rotated_surface_code(d: int):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=1
    )
    coords = {}
    for inst in circ:
        if inst.name == "QUBIT_COORDS":
            coords[inst.targets_copy()[0].value] = tuple(inst.gate_args_copy())
    data = sorted(
        q for q, (x, y) in coords.items() if int(x) % 2 == 1 and int(y) % 2 == 1
    )
    data_set = set(data)
    data_index = {q: i for i, q in enumerate(data)}
    n = len(data)

    h_targets = set()
    seen_r = False
    for inst in circ:
        if inst.name == "R":
            seen_r = True
        if seen_r and inst.name == "H":
            for t in inst.targets_copy():
                h_targets.add(t.value)
            break

    cx_pairs, phase = [], 0
    for inst in circ:
        if inst.name == "H" and phase == 0:
            phase = 1
            continue
        if inst.name == "H" and phase == 1:
            phase = 2
            continue
        if phase == 1 and inst.name == "CX":
            ts = [t.value for t in inst.targets_copy()]
            for i in range(0, len(ts), 2):
                cx_pairs.append((ts[i], ts[i + 1]))

    supp = defaultdict(set)
    for a, b in cx_pairs:
        ma, mb = a not in data_set, b not in data_set
        if ma and not mb:
            supp[a].add(b)
        elif mb and not ma:
            supp[b].add(a)

    Hx_rows, Hz_rows = [], []
    for m, s in supp.items():
        row = np.zeros(n, dtype=int)
        for dq in s:
            row[data_index[dq]] = 1
        (Hx_rows if m in h_targets else Hz_rows).append(row)
    Hx, Hz = dedup_rows(np.vstack(Hx_rows)), dedup_rows(np.vstack(Hz_rows))
    assert np.all((Hx @ Hz.T) % 2 == 0)
    assert n - gf2_rank(Hx) - gf2_rank(Hz) == 1
    xlogs = reduce_mod_rows(nullspace_gf2(Hz), Hx, n)
    zlogs = reduce_mod_rows(nullspace_gf2(Hx), Hz, n)
    logX = min(xlogs, key=lambda v: int(v.sum()))
    logZ = min(zlogs, key=lambda v: int(v.sum()))
    return Hx.astype(int), Hz.astype(int), logX.astype(int), logZ.astype(int)


def gauge_logical_Z_strip(Hx, Hz, logZ):
    """Lattice-surgery / gauging-equivalent ancilla strip for logical Z."""
    n = Hx.shape[1]
    support = sorted(i for i in range(n) if logZ[i])
    m = len(support)
    n_anc = m
    n_tot = n + n_anc

    Hx_g = np.hstack([Hx, np.zeros((Hx.shape[0], n_anc), dtype=int)])
    Hz_g = np.hstack([Hz, np.zeros((Hz.shape[0], n_anc), dtype=int)])

    new_z = []
    for j, s in enumerate(support):
        row = np.zeros(n_tot, dtype=int)
        row[s] = 1
        row[n + j] = 1
        new_z.append(row)
    new_x = []
    for j in range(m - 1):
        row = np.zeros(n_tot, dtype=int)
        row[n + j] = 1
        row[n + j + 1] = 1
        new_x.append(row)

    def match(si):
        return n + support.index(si)

    Hx_def = []
    for r in range(Hx_g.shape[0]):
        row = Hx_g[r].copy()
        ov = [s for s in support if row[s]]
        if len(ov) % 2 == 1:
            row[match(ov[0])] ^= 1
        elif len(ov) >= 2:
            for i in range(0, len(ov) - 1, 2):
                row[match(ov[i])] ^= 1
                row[match(ov[i + 1])] ^= 1
        Hx_def.append(row)

    Hx_def = dedup_rows(np.vstack(Hx_def + new_x))
    Hz_def = dedup_rows(np.vstack([Hz_g] + new_z))
    good_x = [i for i, row in enumerate(Hx_def) if np.all((Hz_def @ row) % 2 == 0)]
    Hx_def = Hx_def[good_x]
    good_z = [i for i, row in enumerate(Hz_def) if np.all((Hx_def @ row) % 2 == 0)]
    Hz_def = Hz_def[good_z]
    assert np.all((Hx_def @ Hz_def.T) % 2 == 0)
    assert int(Hx_def.sum(0).max()) <= 2 and int(Hz_def.sum(0).max()) <= 2

    meas = np.zeros(n_tot, dtype=int)
    meas[:n] = logZ
    meta = {"n_data": n, "n_anc": n_anc, "n_tot": n_tot, "support": support}
    return Hx_def.astype(int), Hz_def.astype(int), meas, meta


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def simulate_gauging_rectangle(d, p, shots, rng):
    Hx0, Hz0, logX, logZ = rotated_surface_code(d)
    n0 = Hx0.shape[1]
    k0 = n0 - gf2_rank(Hx0) - gf2_rank(Hz0)
    Hx, Hz, meas, meta = gauge_logical_Z_strip(Hx0, Hz0, logZ)
    n = meta["n_tot"]
    R = d

    match_X = pymatching.Matching(Hx.astype(np.uint8))
    match_Z = pymatching.Matching(Hz.astype(np.uint8))
    logX_ext = np.zeros(n, dtype=int)
    logX_ext[:n0] = logX

    # Deformed-code distances (upper bounds via explicit logicals)
    xlogs = reduce_mod_rows(nullspace_gf2(Hz), Hx, n)
    zlogs = reduce_mod_rows(nullspace_gf2(Hx), Hz, n)
    dX = min(int(v.sum()) for v in xlogs) if xlogs else None
    dZ = min(int(v.sum()) for v in zlogs) if zlogs else None

    # Code-capacity noise on the deformed code (independent
    # X/Z data errors, perfect measurements, one MWPM cycle). R=d is the design
    # depth of the gauging rectangle; this Monte Carlo is the deformed-code
    # building block. (Single-shot measurement flips without a spacetime
    # detector graph invert finite-size scaling and are not used.)
    failures = 0
    for _ in range(shots):
        err_z = (rng.random(n) < p).astype(np.uint8)
        syn = (Hx @ err_z) % 2
        corr = match_X.decode(syn).astype(np.uint8)
        res_z = (err_z ^ corr) % 2

        err_x = (rng.random(n) < p).astype(np.uint8)
        syn = (Hz @ err_x) % 2
        corr = match_Z.decode(syn).astype(np.uint8)
        res_x = (err_x ^ corr) % 2

        meas_fail = int(np.dot(res_x, meas) % 2)
        spectator_fail = int(np.dot(res_z, logX_ext) % 2)
        if meas_fail or spectator_fail:
            failures += 1

    rate, lo, hi = wilson_interval(failures, shots)
    return {
        "d": d,
        "p": float(p),
        "shots": shots,
        "R_rounds_design": R,
        "noise_model": "code_capacity_XZ_data_on_deformed_strip",
        "n_data": n0,
        "n_tot": n,
        "k_base": int(k0),
        "k_deformed": int(n - gf2_rank(Hx) - gf2_rank(Hz)),
        "dX_deformed_upper": dX,
        "dZ_deformed_upper": dZ,
        "logical_failures": failures,
        "logical_error_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "scope": (
            "deformed surface-code/path-strip building-block sanity check under "
            "code-capacity X/Z data noise; NOT a layer-code threshold, circuit-level "
            "simulation, or architecture-level threshold for the full paper schedule"
        ),
    }


def plot_results(rows, out_stem: Path):
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    markers = {3: "o", 5: "s", 7: "^"}
    for d in sorted({r["d"] for r in rows}):
        sub = sorted([r for r in rows if r["d"] == d], key=lambda r: r["p"])
        xs = [r["p"] for r in sub]
        ys = [max(r["logical_error_rate"], 1e-6) for r in sub]
        ylo = [max(r["logical_error_rate"] - r["wilson_lo"], 0) for r in sub]
        yhi = [max(r["wilson_hi"] - r["logical_error_rate"], 0) for r in sub]
        ax.errorbar(
            xs, ys, yerr=[ylo, yhi], fmt=f"{markers.get(d, 'o')}-",
            capsize=2, label=rf"$d={d}$ gauging rect.",
        )
    ax.set_xlabel(r"Code-capacity error probability $p$")
    ax.set_ylabel(r"Logical error rate $p_L$")
    ax.set_yscale("log")
    ax.set_title("Gauging-rectangle sanity check (not arch. threshold)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main() -> int:
    for d in (3, 5, 7):
        Hx, Hz, logX, logZ = rotated_surface_code(d)
        n = Hx.shape[1]
        k = n - gf2_rank(Hx) - gf2_rank(Hz)
        print(f"base rotated d={d}: n={n} k={k} CSS=True")
        Hxg, Hzg, meas, meta = gauge_logical_Z_strip(Hx, Hz, logZ)
        kd = meta["n_tot"] - gf2_rank(Hxg) - gf2_rank(Hzg)
        print(f"  strip-gauged n={meta['n_tot']} k={kd} CSS OK")

    ps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    shot_table = {
        0.01: 8000,
        0.02: 6000,
        0.03: 5000,
        0.04: 4000,
        0.05: 3500,
        0.06: 3000,
        0.08: 2500,
        0.10: 2000,
    }
    rows = []
    t0 = time.time()
    for d in (3, 5, 7):
        for p in ps:
            shots = shot_table[p]
            if d == 7:
                shots = max(1000, shots // 2)
            rec = simulate_gauging_rectangle(d, p, shots, RNG)
            rows.append(rec)
            print(
                f"d={d} p={p:.2f}: p_L={rec['logical_error_rate']:.4f} "
                f"[{rec['wilson_lo']:.4f},{rec['wilson_hi']:.4f}] "
                f"fails={rec['logical_failures']}/{rec['shots']} "
                f"dX<={rec['dX_deformed_upper']} dZ<={rec['dZ_deformed_upper']}"
            )
    print(f"elapsed {time.time()-t0:.1f}s")

    out = ROOT / "gauging_rectangle_mwpm.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_results(rows, FIG / "gauging_rectangle_mwpm")

    by_p = {}
    for r in rows:
        by_p.setdefault(r["p"], {})[r["d"]] = r["logical_error_rate"]
    supp55 = [p for p, m in by_p.items() if m.get(5, 1) < m.get(3, 0)]
    supp75 = [p for p, m in by_p.items() if m.get(7, 1) < m.get(5, 0)]
    print(f"observation: d=5 below d=3 at p in {sorted(supp55)}")
    print(f"observation: d=7 below d=5 at p in {sorted(supp75)}")
    print(f"wrote {out.name} and figures/gauging_rectangle_mwpm.pdf/.png")
    print(
        "SCOPE: deformed surface-code/path-strip building-block sanity check under "
        "code-capacity X/Z data noise; NOT a layer-code threshold, circuit-level "
        "simulation, or architecture-level threshold for the full paper schedule"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
