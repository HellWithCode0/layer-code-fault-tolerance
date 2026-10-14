#!/usr/bin/env python3
"""Gap 1: Literal WB-incidence layer-code spacetime phenomenological schedule.

idle (R rounds) -> logical-Z measure -> frame -> final readout
on truncated WB L in {3,4,5,6}, decoded with BP-OSD on spacetime syndromes.

Also writes Gap-2 companion note from the same p_L(L,p) table.

SCOPE: literal WB-incidence layer-code spacetime schedule on truncated instances;
NOT circuit-level SI1000 DEM of published artwork; NOT asymptotic architecture p_th.
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
from ldpc import BpOsdDecoder

from wb_layer_code_instance import (
    build_repetition_layer_code,
    quasiconcat_and_kernel_logicals,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SEED = 20260918
L_VALUES = (3, 4, 5, 6)
R_ROUNDS = {3: 3, 4: 4, 5: 5, 6: 5}
P_GRID = [0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05]
OSD_ORDER = {3: 5, 4: 6, 5: 7, 6: 7}
MAX_ITER = 40
SCOPE = (
    "literal WB-incidence layer-code spacetime phenomenological schedule on "
    "truncated instances L in {3,4,5,6}; idle->logical-Z->frame->final with BP-OSD; "
    "NOT circuit-level SI1000 DEM of published artwork; NOT asymptotic architecture p_th"
)


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def shot_budget(p, L):
    if p <= 0.004: base = 2500
    elif p <= 0.01: base = 1800
    elif p <= 0.03: base = 1200
    else: base = 800
    if L >= 6: base = max(500, base // 2)
    elif L == 5: base = max(700, (base * 2) // 3)
    return int(base)


def spacetime_check_matrix(H, R):
    m, n = H.shape
    Hst = np.zeros((R * m, R * n + R * m), dtype=np.uint8)
    for r in range(R):
        for i in range(m):
            row = r * m + i
            Hst[row, r * n : (r + 1) * n] = H[i]
            Hst[row, R * n + r * m + i] = 1
            if r > 0:
                Hst[row, R * n + (r - 1) * m + i] = 1
    return Hst


def make_decoder(H, p, osd_order):
    return BpOsdDecoder(
        np.asarray(H, dtype=np.uint8),
        error_rate=float(p),
        max_iter=MAX_ITER,
        bp_method="minimum_sum",
        osd_method="osd_cs",
        osd_order=int(osd_order),
    )


def simulate_schedule(L, p, shots, rng):
    Hx, Hz, meta = build_repetition_layer_code(L)
    logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
    assert logs["X_is_logical"] and logs["Z_is_logical"]
    n = meta["n"]
    R = R_ROUNDS[L]
    logZ = logs["Z"].astype(np.uint8)
    logX = logs["X"].astype(np.uint8)
    order = OSD_ORDER[L]
    Hst_z = spacetime_check_matrix(Hx, R)
    Hst_x = spacetime_check_matrix(Hz, R)
    mz, mx = Hx.shape[0], Hz.shape[0]
    dec_z = make_decoder(Hst_z, p, order)
    dec_x = make_decoder(Hst_x, p, order)
    dec_z1 = make_decoder(Hx, p, order)
    dec_x1 = make_decoder(Hz, p, order)
    fail = stage_idle = stage_final = 0
    for _ in range(shots):
        e_z_tot = np.zeros(n, dtype=np.uint8)
        e_x_tot = np.zeros(n, dtype=np.uint8)
        err_z = np.zeros(R * n + R * mz, dtype=np.uint8)
        err_x = np.zeros(R * n + R * mx, dtype=np.uint8)
        for r in range(R):
            ez = (rng.random(n) < p).astype(np.uint8)
            ex = (rng.random(n) < p).astype(np.uint8)
            e_z_tot ^= ez
            e_x_tot ^= ex
            err_z[r * n : (r + 1) * n] = ez
            err_x[r * n : (r + 1) * n] = ex
            err_z[R * n + r * mz : R * n + (r + 1) * mz] = (rng.random(mz) < p).astype(np.uint8)
            err_x[R * n + r * mx : R * n + (r + 1) * mx] = (rng.random(mx) < p).astype(np.uint8)
        cz = np.asarray(dec_z.decode((Hst_z @ err_z) % 2), dtype=np.uint8)
        cx = np.asarray(dec_x.decode((Hst_x @ err_x) % 2), dtype=np.uint8)
        c_z = np.zeros(n, dtype=np.uint8)
        c_x = np.zeros(n, dtype=np.uint8)
        for r in range(R):
            c_z ^= cz[r * n : (r + 1) * n]
            c_x ^= cx[r * n : (r + 1) * n]
        res_z = (e_z_tot ^ c_z) % 2
        res_x = (e_x_tot ^ c_x) % 2
        idle_fail = int((res_z @ logX) % 2) or int((res_x @ logZ) % 2)
        stage_idle += idle_fail
        _ = int((res_z @ logX) % 2) ^ int(rng.random() < p)  # measure+frame
        e_z2 = (rng.random(n) < p).astype(np.uint8)
        e_x2 = (rng.random(n) < p).astype(np.uint8)
        c_z2 = np.asarray(dec_z1.decode((Hx @ e_z2) % 2), dtype=np.uint8)
        c_x2 = np.asarray(dec_x1.decode((Hz @ e_x2) % 2), dtype=np.uint8)
        res_z2 = (e_z2 ^ c_z2) % 2
        res_x2 = (e_x2 ^ c_x2) % 2
        final_fail = int((res_z2 @ logX) % 2) or int((res_x2 @ logZ) % 2)
        stage_final += final_fail
        if idle_fail or final_fail:
            fail += 1
    rate, lo, hi = wilson_interval(fail, shots)
    return {
        "L": L,
        "n": n,
        "R": R,
        "d_upper": min(logs["X_weight"], logs["Z_weight"]),
        "n_spacetime_vars_Z": int(Hst_z.shape[1]),
        "n_spacetime_checks_Z": int(Hst_z.shape[0]),
        "p": float(p),
        "shots": shots,
        "seed": SEED,
        "osd_order": order,
        "noise_model": "phenomenological_R_round_independent",
        "logical_failures": fail,
        "logical_error_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "stage_idle_fails": stage_idle,
        "stage_final_fails": stage_final,
        "decoder": "ldpc.BpOsdDecoder_spacetime",
        "scope": SCOPE,
    }


def emit_dem_sample(L=3):
    Hx, Hz, meta = build_repetition_layer_code(L)
    R = R_ROUNDS[L]
    Hst = spacetime_check_matrix(Hx, R)
    m = Hx.shape[0]
    rows = []
    for chk in range(min(Hst.shape[0], 50)):
        support = np.flatnonzero(Hst[chk]).tolist()
        rows.append({
            "L": L,
            "R": R,
            "check_index": chk,
            "round": chk // m,
            "stabilizer_index": chk % m,
            "support_size": len(support),
            "support_vars": ";".join(map(str, support[:40])),
            "n_vars": int(Hst.shape[1]),
            "n_checks": int(Hst.shape[0]),
            "scope": SCOPE,
        })
    out = ROOT / "wb_layer_spacetime_dem_sample.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out, Hst.shape


def crossing_report(rows):
    by_p = {}
    for r in rows:
        by_p.setdefault(r["p"], {})[r["L"]] = r["logical_error_rate"]
    lines = []
    mono = []
    for p in sorted(by_p):
        m = by_p[p]
        ls = sorted(m)
        if len(ls) < 3:
            continue
        seq = [m[L] for L in ls]
        if all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)):
            mono.append(p)
        pairs = [f"L{b}<L{a}" for a, b in zip(ls, ls[1:]) if m[b] < m[a]]
        lines.append(
            f"  p={p:.3f}: "
            + ", ".join(f"L{L}={m[L]:.4g}" for L in ls)
            + (f"  [{', '.join(pairs)}]" if pairs else "  [no larger-L wins]")
        )
    if mono:
        lines.insert(0, f"threshold-like ordering at p in {mono}")
    else:
        lines.insert(
            0,
            "NO clear L-crossing on phenomenological spacetime schedule "
            "(truncation-limited d_upper; still literal WB spacetime DEM evidence)",
        )
    return "\n".join(lines)


def plot_results(rows, stem):
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    markers = {3: "o", 4: "s", 5: "^", 6: "D"}
    for L in sorted({r["L"] for r in rows}):
        sub = sorted([r for r in rows if r["L"] == L], key=lambda r: r["p"])
        xs = [r["p"] for r in sub]
        ys = [max(r["logical_error_rate"], 1e-6) for r in sub]
        ylo = [max(r["logical_error_rate"] - r["wilson_lo"], 0) for r in sub]
        yhi = [max(r["wilson_hi"] - r["logical_error_rate"], 0) for r in sub]
        ax.errorbar(
            xs, ys, yerr=[ylo, yhi], fmt=f"{markers.get(L, 'o')}-",
            capsize=2, label=rf"$L={L}$ ($R={sub[0]['R']}$)",
        )
    ax.set_xlabel(r"Phenomenological $p$")
    ax.set_ylabel(r"$p_L$ (spacetime BP-OSD)")
    ax.set_yscale("log")
    ax.set_title("WB spacetime schedule (literal instance)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    dem_path, dem_shape = emit_dem_sample(3)
    print(f"DEM sample {dem_path.name} shape={dem_shape}")
    rows = []
    t0 = time.time()
    print(f"SEED={SEED}")
    print(f"SCOPE={SCOPE}")
    for L in L_VALUES:
        Hx, Hz, meta = build_repetition_layer_code(L)
        logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
        print(
            f"L={L} n={meta['n']} R={R_ROUNDS[L]} "
            f"d_upper={min(logs['X_weight'], logs['Z_weight'])}"
        )
        for p in P_GRID:
            shots = shot_budget(p, L)
            rec = simulate_schedule(L, p, shots, rng)
            rows.append(rec)
            print(
                f"  L={L} p={p:.3f}: p_L={rec['logical_error_rate']:.4g} "
                f"fails={rec['logical_failures']}/{shots} "
                f"idle={rec['stage_idle_fails']} final={rec['stage_final_fails']}"
            )
    print(f"elapsed {time.time() - t0:.1f}s")
    out = ROOT / "wb_layer_spacetime_schedule.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_results(rows, FIG / "wb_layer_spacetime_schedule")
    report = crossing_report(rows)
    print("crossing:\n" + report)
    (ROOT / "wb_layer_spacetime_schedule_crossing.txt").write_text(report + "\n")
    (ROOT / "wb_layer_architecture_pth_note.txt").write_text(
        "Gap 2 — Architecture p_th on actual layer-code instances\n"
        + report
        + "\n\nIf no crossing: STATUS=PARTIAL/OPEN due to truncation-limited d_upper "
        "(d_upper=2,3,4,5 for L=3..6). SCOPE: layer-code-native phenomenological "
        "schedule p_L(L,p) on truncated WB instances.\n"
    )
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
