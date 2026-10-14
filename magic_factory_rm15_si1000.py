#!/usr/bin/env python3
"""Gap 8: Steane + RM15 factory stack under SI1000-like Stim noise.

Codes
  Steane [[7,1,3]] with cyclic Hamming Hx=Hz
  RM15 / quantum Hamming [[15,7,3]] with Hx=Hz = 4x15 simplex (all nonzero F2^4
  columns), tracking the all-ones logical — matches factory_map_audit.py BK
  weight classes {0,8}/{7,15}.

Circuit protocol (Clifford Stim build — no native T):
  feedback-projected codespace encode of |0_L> -> transversal S_DAG ->
  noisy syndrome + LUT -> Z_L readout.  Noiseless p_L=0 required.
  RM15 uses the same SI1000 projection+S_DAG SE schedule as a Clifford factory
  skeleton; non-Clifford |A>/T content is covered by factory_map_audit +
  code-capacity curves on the same code/logical.

Multi-level: Steane SI1000 p_L(p) fed as effective error into RM15 capacity.
Optional: OR-combine with d=3 CAS consume proxy.

SCOPE: Steane+RM15 noiseless+noisy curves + two-level stack; NOT full routing
into literal layer-code / CAD artwork.  T unitary absent in this Stim build.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stim

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SEED = 20260918
P_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01]
SHOTS = {
    0.0: 4000,
    0.0005: 16000,
    0.001: 12000,
    0.002: 10000,
    0.003: 8000,
    0.005: 6000,
    0.008: 5000,
    0.01: 4000,
}
SCOPE = (
    "Steane+RM15 factory under SI1000-like Stim noise with feedback-projected "
    "|0_L> + S_DAG + LUT (noiseless p_L=0) and multi-level stack; T via "
    "factory_map_audit+capacity (Stim build has no T); NOT layer-code routing"
)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def si1000(p):
    return dict(p2=p, p1=p / 10.0, p_meas=5 * p, p_reset=2 * p)


def rm15_H():
    cols = [[(i >> b) & 1 for b in range(4)] for i in range(1, 16)]
    return np.array(cols, dtype=np.uint8).T


HX7 = np.array(
    [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]],
    dtype=np.uint8,
)
HZ7 = HX7.copy()
LZ7 = np.ones(7, dtype=np.uint8)
HX15 = rm15_H()
HZ15 = HX15.copy()
LZ15 = np.ones(15, dtype=np.uint8)


def lut(syn, H):
    s = tuple(int(x) for x in syn)
    m = H.shape[0]
    n = H.shape[1]
    if s == tuple(0 for _ in range(m)):
        return np.zeros(n, dtype=np.uint8), True
    for j in range(n):
        if tuple(int(H[i, j]) for i in range(m)) == s:
            e = np.zeros(n, dtype=np.uint8)
            e[j] = 1
            return e, True
    return None, False


def append_noise1(c, qs, p):
    if p > 0:
        c.append("DEPOLARIZE1", qs if isinstance(qs, list) else [qs], p)


def append_noise2(c, a, b, p):
    if p > 0:
        c.append("DEPOLARIZE2", [a, b], p)


def append_project_X(c, data, H_x, ax, pr, feedback=True):
    """Measure X-stabs; optional CZ feedback onto first support qubit to force +1."""
    for i, row in enumerate(H_x):
        a = ax[i]
        c.append("R", [a])
        if pr["p_reset"] > 0:
            c.append("X_ERROR", [a], pr["p_reset"])
        c.append("H", [a])
        append_noise1(c, a, pr["p1"])
        support = [int(q) for q in np.flatnonzero(row)]
        for q in support:
            c.append("CX", [a, data[q]])
            append_noise2(c, a, data[q], pr["p2"])
        c.append("H", [a])
        append_noise1(c, a, pr["p1"])
        if pr["p_meas"] > 0:
            c.append("X_ERROR", [a], pr["p_meas"])
        c.append("MR", [a])
        if feedback and support:
            # Force +1 eigenspace with Z on first support qubit
            c.append("CZ", [stim.target_rec(-1), data[support[0]]])
    c.append("TICK")


def append_syndrome_no_fb(c, data, H_x, H_z, ax, az, pr):
    # Z stabs
    for i, row in enumerate(H_z):
        a = az[i]
        c.append("R", [a])
        if pr["p_reset"] > 0:
            c.append("X_ERROR", [a], pr["p_reset"])
        for q in np.flatnonzero(row):
            q = int(q)
            c.append("CX", [data[q], a])
            append_noise2(c, data[q], a, pr["p2"])
        if pr["p_meas"] > 0:
            c.append("X_ERROR", [a], pr["p_meas"])
        c.append("MR", [a])
    c.append("TICK")
    # X stabs (no feedback — soft info for LUT)
    for i, row in enumerate(H_x):
        a = ax[i]
        c.append("R", [a])
        if pr["p_reset"] > 0:
            c.append("X_ERROR", [a], pr["p_reset"])
        c.append("H", [a])
        append_noise1(c, a, pr["p1"])
        for q in np.flatnonzero(row):
            q = int(q)
            c.append("CX", [a, data[q]])
            append_noise2(c, a, data[q], pr["p2"])
        c.append("H", [a])
        append_noise1(c, a, pr["p1"])
        if pr["p_meas"] > 0:
            c.append("X_ERROR", [a], pr["p_meas"])
        c.append("MR", [a])
    c.append("TICK")


def build_factory_circuit(n, H_x, H_z, p):
    """ |0>^n --X-project+feedback--> |0_L>-like --S_DAG--> syndrome --M data."""
    pr = si1000(p)
    m = H_x.shape[0]
    assert H_z.shape[0] == m
    c = stim.Circuit()
    data = list(range(n))
    ax = list(range(n, n + m))
    az = list(range(n + m, n + 2 * m))
    c.append("R", data)
    if pr["p_reset"] > 0:
        c.append("X_ERROR", data, pr["p_reset"])
    c.append("TICK")
    # Project into X-stab +1 (Z corrections via feedback); |0> already Z-stab +1 / Z_L=0
    append_project_X(c, data, H_x, ax, pr, feedback=True)
    # Transversal S_DAG (logical S_DAG on CSS Hamming-type codes for |0_L> eigenstate)
    c.append("S_DAG", data)
    append_noise1(c, data, pr["p1"])
    c.append("TICK")
    append_syndrome_no_fb(c, data, H_x, H_z, ax, az, pr)
    if pr["p_meas"] > 0:
        c.append("X_ERROR", data, pr["p_meas"])
    c.append("M", data)
    return c, m


def simulate_factory_circuit(n, H_x, H_z, Lz, p, shots, seed, name):
    circ, m = build_factory_circuit(n, H_x, H_z, p)
    raw = circ.compile_sampler(seed=seed).sample(shots)
    # measurements: m (project X) + m (Z syn) + m (X syn) + n data
    accept = fail = 0
    for row in raw:
        # skip project outcomes (already feedback-applied)
        syn_z = row[m : 2 * m]
        syn_x = row[2 * m : 3 * m]
        data_m = np.asarray(row[3 * m : 3 * m + n], dtype=np.uint8)
        e_x, okx = lut(syn_z, H_z)
        e_z, okz = lut(syn_x, H_x)
        if not (okx and okz):
            continue
        accept += 1
        bits = (data_m ^ e_z) % 2
        if int((bits @ Lz) % 2) != 0:
            fail += 1
    rate, lo, hi = wilson(fail, accept) if accept else (1.0, 0.0, 1.0)
    return dict(
        p=float(p),
        shots=shots,
        accepted=accept,
        accept_rate=accept / shots if shots else 0.0,
        logical_failures=fail,
        logical_error_rate=rate,
        wilson_lo=lo,
        wilson_hi=hi,
        factory=name,
        noise_model="si1000_like_circuit_project_Sdag_LUT",
        scope=SCOPE,
        seed=seed,
    )


def simulate_capacity(n, H_x, H_z, Lz, p, shots, rng, name):
    accept = fail = 0
    for _ in range(shots):
        e_x = (rng.random(n) < p).astype(np.uint8)
        e_z = (rng.random(n) < p).astype(np.uint8)
        cx, okx = lut((H_z @ e_x) % 2, H_z)
        cz, okz = lut((H_x @ e_z) % 2, H_x)
        if not (okx and okz):
            continue
        accept += 1
        res_z = (e_z ^ cz) % 2
        res_x = (e_x ^ cx) % 2
        if int((res_z @ Lz) % 2) or int((res_x @ Lz) % 2):
            fail += 1
    rate, lo, hi = wilson(fail, accept) if accept else (1.0, 0.0, 1.0)
    return dict(
        p=float(p),
        shots=shots,
        accepted=accept,
        accept_rate=accept / shots if shots else 0.0,
        logical_failures=fail,
        logical_error_rate=rate,
        wilson_lo=lo,
        wilson_hi=hi,
        factory=name,
        noise_model="code_capacity_independent_XZ",
        scope=SCOPE,
        seed=int(rng.integers(1e9)),
    )


def multilevel(steane_cir_rows, p_grid, shots, rng):
    xs = np.array([r["p"] for r in steane_cir_rows], dtype=float)
    ys = np.array([r["logical_error_rate"] for r in steane_cir_rows], dtype=float)
    rows = []
    for p in p_grid:
        peff = float(np.interp(p, xs, ys))
        rec = simulate_capacity(15, HX15, HZ15, LZ15, peff, shots, rng, "MultiLevel_Steane_then_RM15_capacity")
        rec = dict(rec)
        rec["p"] = float(p)
        rec["p_eff_stage1"] = peff
        rec["noise_model"] = "stage1_si1000_pL_into_RM15_capacity"
        rows.append(rec)
    return rows


def cas_consume_proxy(curve, shots_per, seed0):
    try:
        import architecture_e2e_schedule_sim as cas
    except Exception as ex:
        return [
            dict(
                p=r["p"],
                shots=0,
                logical_error_rate=float("nan"),
                note=str(ex)[:120],
                factory="Factory_plus_CAS_consume_d3_proxy",
                scope=SCOPE,
            )
            for r in curve
        ]
    rows = []
    for i, r in enumerate(curve):
        p = float(r["p"])
        rec = cas.simulate_cas(3, max(p, 1e-9), shots_per, "si1000", seed0 + i)
        pf = float(r["logical_error_rate"])
        pc = float(rec["logical_error_rate"])
        p_or = pf + pc - pf * pc
        lo = max(0.0, p_or - 1.96 * math.sqrt(max(p_or * (1 - p_or), 0) / shots_per))
        hi = min(1.0, p_or + 1.96 * math.sqrt(max(p_or * (1 - p_or), 0) / shots_per))
        rows.append(
            dict(
                p=p,
                shots=shots_per,
                accepted=shots_per,
                accept_rate=1.0,
                logical_failures=int(round(p_or * shots_per)),
                logical_error_rate=p_or,
                wilson_lo=lo,
                wilson_hi=hi,
                factory="Factory_plus_CAS_consume_d3_proxy",
                p_factory=pf,
                p_consume=pc,
                noise_model="independent_or_combine",
                scope=SCOPE,
                seed=seed0 + i,
            )
        )
    return rows


def plot_all(rows, stem):
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    styles = {
        "Steane7_code_capacity_LUT": ("o", "-"),
        "Steane7_project_Sdag_LUT": ("s", "-"),
        "RM15_allones_code_capacity_LUT": ("^", "-"),
        "RM15_project_Sdag_LUT": ("D", "-"),
        "MultiLevel_Steane_then_RM15_capacity": ("v", "--"),
        "Factory_plus_CAS_consume_d3_proxy": ("P", ":"),
    }
    by = {}
    for r in rows:
        by.setdefault(r["factory"], []).append(r)
    for name, sub in by.items():
        sub = sorted(sub, key=lambda r: r["p"])
        xs = [r["p"] for r in sub]
        ys = [max(float(r["logical_error_rate"]), 1e-6) for r in sub]
        mk, ls = styles.get(name, ("o", "-"))
        ax.plot(xs, ys, marker=mk, linestyle=ls, ms=4, label=name.replace("_", " ")[:40])
    ax.set_xlabel(r"physical / SI1000-like $p$")
    ax.set_ylabel(r"$p_L$ (accepted)")
    ax.set_yscale("log")
    ax.set_title("Magic factory stack (Steane + RM15)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"SEED={SEED}\nSCOPE={SCOPE}")
    print("RM15 unique single-Z syndromes:", len({tuple(HX15[:, j].tolist()) for j in range(15)}) == 15)

    for p in P_GRID:
        sh = SHOTS[p]
        r = simulate_capacity(7, HX7, HZ7, LZ7, p, sh, rng, "Steane7_code_capacity_LUT")
        rows.append(r)
        print(f"Steane cap p={p}: acc={r['accept_rate']:.3g} p_L={r['logical_error_rate']:.4g}")
        r = simulate_factory_circuit(7, HX7, HZ7, LZ7, p, sh, SEED + int(p * 1e6) + 1, "Steane7_project_Sdag_LUT")
        rows.append(r)
        print(f"Steane cir p={p}: acc={r['accept_rate']:.3g} p_L={r['logical_error_rate']:.4g}")
        r = simulate_capacity(15, HX15, HZ15, LZ15, p, sh, rng, "RM15_allones_code_capacity_LUT")
        rows.append(r)
        print(f"RM15  cap p={p}: acc={r['accept_rate']:.3g} p_L={r['logical_error_rate']:.4g}")
        r = simulate_factory_circuit(15, HX15, HZ15, LZ15, p, sh, SEED + int(p * 1e6) + 2, "RM15_project_Sdag_LUT")
        rows.append(r)
        print(f"RM15  cir p={p}: acc={r['accept_rate']:.3g} p_L={r['logical_error_rate']:.4g}")

    steane_cir = [r for r in rows if r["factory"] == "Steane7_project_Sdag_LUT"]
    ml = multilevel(steane_cir, P_GRID, 8000, rng)
    rows.extend(ml)
    for r in ml:
        print(f"Multi  p={r['p']}: p_eff={r['p_eff_stage1']:.4g} p_L={r['logical_error_rate']:.4g}")

    cas_rows = cas_consume_proxy(ml, 2000, SEED + 99)
    rows.extend(cas_rows)
    for r in cas_rows:
        print(f"CAS+F p={r['p']}: p_L={r['logical_error_rate']:.4g}")

    for name in (
        "Steane7_code_capacity_LUT",
        "Steane7_project_Sdag_LUT",
        "RM15_allones_code_capacity_LUT",
        "RM15_project_Sdag_LUT",
    ):
        z = next(r for r in rows if r["factory"] == name and r["p"] == 0.0)
        print(f"SANITY {name}: p=0 p_L={z['logical_error_rate']}")
        if z["logical_error_rate"] > 0.05:
            print(f"WARNING noiseless high for {name}")

    out = ROOT / "magic_factory_rm15_si1000.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    plot_all(rows, FIG / "magic_factory_rm15_si1000")
    plot_all(rows, FIG / "magic_factory_stack_si1000")

    report = ["Gap 8 factory stack sanity"]
    by = {}
    for r in rows:
        by.setdefault(r["factory"], []).append(r)
    for name, sub in sorted(by.items()):
        z = next((r for r in sub if r["p"] == 0.0), None)
        mid = next((r for r in sub if abs(r["p"] - 0.001) < 1e-12), None)
        report.append(
            f"  {name}: noiseless_p_L={None if z is None else z['logical_error_rate']}; "
            f"p=1e-3 p_L={None if mid is None else mid['logical_error_rate']}"
        )
    (ROOT / "magic_factory_rm15_si1000_crossing.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"elapsed {time.time()-t0:.1f}s; wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
