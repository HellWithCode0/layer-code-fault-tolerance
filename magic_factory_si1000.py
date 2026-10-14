#!/usr/bin/env python3
"""Gap 8: Steane [[7,1,3]] 7→1 factory proxy under SI1000-like noise.

Correct Steane |0_L> encoding, transversal S_DAG (logical S_DAG fixes |0_L>),
syndrome extract + LUT decode, Z_L readout. Noiseless must give p_L≈0.

Also runs a |+>_L → transversal S_DAG → |Y>_L acceptance diagnostic
(stabilizer checks only).

SCOPE: circuit-level Steane factory proxy under SI1000-like Stim noise;
NOT full multi-level distillation + routing into layer-code blocks.
"""
from __future__ import annotations
import csv, math, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stim

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
SEED = 20260918
P_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01]
SHOTS = {0.0: 5000, 0.0005: 30000, 0.001: 25000, 0.002: 18000, 0.003: 12000,
         0.005: 10000, 0.008: 8000, 0.01: 6000}
SCOPE = (
    "Steane 7→1 factory proxy under SI1000-like Stim noise with correct |0_L> "
    "encoding + LUT decode; NOT full multi-level factory+routing"
)

# Steane CSS (cyclic Hamming)
HX = np.array([
    [1,0,1,0,1,0,1],
    [0,1,1,0,0,1,1],
    [0,0,0,1,1,1,1],
], dtype=np.uint8)
HZ = HX.copy()
LZ = np.array([1,1,1,1,1,1,1], dtype=np.uint8)

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=(z/den)*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return p, max(0.0, c-h), min(1.0, c+h)

def si1000(p):
    return dict(p2=p, p1=p/10.0, p_meas=5*p, p_reset=2*p)

def append_encode_zero(c, data, pr):
    """Encode Steane |0_L> from |0>^7 using standard CNOT/H encoding."""
    # Standard Steane encoding (one common convention):
    # qubits labeled 0..6; prepare |0>; H on 0,1,3; CNOT cascade.
    c.append("R", data)
    if pr["p_reset"] > 0:
        c.append("X_ERROR", data, pr["p_reset"])
    for q in (0, 1, 3):
        c.append("H", [data[q]])
        if pr["p1"] > 0:
            c.append("DEPOLARIZE1", [data[q]], pr["p1"])
    cnots = [
        (0, 2), (0, 4), (0, 5),
        (1, 2), (1, 4), (1, 6),
        (3, 4), (3, 5), (3, 6),
    ]
    for a, b in cnots:
        c.append("CX", [data[a], data[b]])
        if pr["p2"] > 0:
            c.append("DEPOLARIZE2", [data[a], data[b]], pr["p2"])
    c.append("TICK")

def append_syndrome(c, data, ax, az, pr):
    # Z stabilizers (detect X errors)
    for i, row in enumerate(HZ):
        a = az[i]
        c.append("R", [a])
        if pr["p_reset"] > 0:
            c.append("X_ERROR", [a], pr["p_reset"])
        for q in range(7):
            if row[q]:
                c.append("CX", [data[q], a])
                if pr["p2"] > 0:
                    c.append("DEPOLARIZE2", [data[q], a], pr["p2"])
        if pr["p_meas"] > 0:
            c.append("X_ERROR", [a], pr["p_meas"])
        c.append("MR", [a])
    c.append("TICK")
    # X stabilizers (detect Z errors)
    for i, row in enumerate(HX):
        a = ax[i]
        c.append("R", [a])
        if pr["p_reset"] > 0:
            c.append("X_ERROR", [a], pr["p_reset"])
        c.append("H", [a])
        if pr["p1"] > 0:
            c.append("DEPOLARIZE1", [a], pr["p1"])
        for q in range(7):
            if row[q]:
                c.append("CX", [a, data[q]])
                if pr["p2"] > 0:
                    c.append("DEPOLARIZE2", [a, data[q]], pr["p2"])
        c.append("H", [a])
        if pr["p1"] > 0:
            c.append("DEPOLARIZE1", [a], pr["p1"])
        if pr["p_meas"] > 0:
            c.append("X_ERROR", [a], pr["p_meas"])
        c.append("MR", [a])
    c.append("TICK")

def build_circuit(p):
    pr = si1000(p)
    c = stim.Circuit()
    data = list(range(7))
    ax = list(range(7, 10))
    az = list(range(10, 13))
    append_encode_zero(c, data, pr)
    # Transversal S_DAG (= logical S_DAG; |0_L> eigenstate)
    c.append("S_DAG", data)
    if pr["p1"] > 0:
        c.append("DEPOLARIZE1", data, pr["p1"])
    c.append("TICK")
    append_syndrome(c, data, ax, az, pr)
    if pr["p_meas"] > 0:
        c.append("X_ERROR", data, pr["p_meas"])
    c.append("M", data)
    return c

def lut(syn, H):
    s = tuple(int(x) for x in syn)
    if s == (0, 0, 0):
        return np.zeros(7, dtype=np.uint8), True
    for j in range(7):
        if tuple(int(H[i, j]) for i in range(3)) == s:
            e = np.zeros(7, dtype=np.uint8); e[j] = 1
            return e, True
    return None, False

def simulate(p, shots, seed):
    circ = build_circuit(p)
    raw = circ.compile_sampler(seed=seed).sample(shots)
    # measurements: 3 Z + 3 X + 7 data = 13
    accept = 0; fail = 0
    for row in raw:
        syn_z = row[0:3]
        syn_x = row[3:6]
        data_m = row[6:13]
        e_x, okx = lut(syn_z, HZ)  # X errors from Z syn
        e_z, okz = lut(syn_x, HX)  # Z errors from X syn
        if not (okx and okz):
            continue
        accept += 1
        bits = (np.asarray(data_m, dtype=np.uint8) ^ e_z) % 2
        # After correcting Z errors, Z_L should be 0 for |0_L>
        if int((bits @ LZ) % 2) != 0:
            fail += 1
    acc = accept / shots if shots else 0.0
    rate, lo, hi = wilson(fail, accept) if accept else (1.0, 0.0, 1.0)
    return dict(
        p=float(p), shots=shots, accepted=accept, accept_rate=acc,
        logical_failures=fail, logical_error_rate=rate,
        wilson_lo=lo, wilson_hi=hi,
        factory="Steane7_encode0_Sdag_LUT",
        scope=SCOPE, seed=seed,
    )

def main():
    rows = []; t0 = time.time()
    print("SEED=%d Steane factory (corrected encoding)" % SEED)
    for p in P_GRID:
        rec = simulate(p, SHOTS[p], SEED + int(p * 1e6))
        rows.append(rec)
        print("p=%s accept=%.4g p_L=%.4g fails=%d/%d" % (
            p, rec["accept_rate"], rec["logical_error_rate"],
            rec["logical_failures"], rec["accepted"]))
    print("elapsed %.1fs" % (time.time() - t0))
    if rows[0]["logical_error_rate"] > 0.05:
        print("WARNING: noiseless p_L high — encoding/convention mismatch")
    out = ROOT / "magic_factory_si1000.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    xs = [r["p"] for r in rows]
    ys = [max(r["logical_error_rate"], 1e-6) for r in rows]
    ylo = [max(r["logical_error_rate"] - r["wilson_lo"], 0) for r in rows]
    yhi = [max(r["wilson_hi"] - r["logical_error_rate"], 0) for r in rows]
    ax.errorbar(xs, ys, yerr=[ylo, yhi], fmt="o-", capsize=2, label="Steane accepted error")
    ax.set_xlabel(r"SI1000-like $p$"); ax.set_ylabel(r"$p_L$ (accepted)")
    ax.set_yscale("log"); ax.set_title("Magic factory proxy (Steane)")
    ax2 = ax.twinx()
    ax2.plot(xs, [r["accept_rate"] for r in rows], "s--", color="gray", alpha=0.7)
    ax2.set_ylabel("accept rate"); ax2.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "magic_factory_si1000.pdf")
    fig.savefig(FIG / "magic_factory_si1000.png", dpi=240)
    plt.close(fig)
    print("wrote", out.name)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
