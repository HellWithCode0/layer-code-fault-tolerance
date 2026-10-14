#!/usr/bin/env python3
"""Gap 3: Deeper BP-OSD search for WB memory threshold crossing.

Denser low-p grid, more shots, higher OSD order, plus phenomenological R-round
spacetime BP-OSD. Reports whether a finite-size crossing appears.

SCOPE: deepened finite WB BP-OSD memory / phenomenological search;
NOT architecture schedule p_th.
"""
from __future__ import annotations
import csv, math, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ldpc import BpOsdDecoder
from wb_layer_code_instance import build_repetition_layer_code, quasiconcat_and_kernel_logicals
from wb_layer_spacetime_schedule import spacetime_check_matrix

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
SEED = 20260918
L_VALUES = (3, 4, 5, 6)
P_GRID = [0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05]
OSD_ORDER = {3: 6, 4: 8, 5: 10, 6: 10}
R_PHENO = {3: 3, 4: 4, 5: 4, 6: 5}
MAX_ITER = 50
SCOPE = (
    "deepened WB BP-OSD code-capacity + phenomenological R-round search on "
    "L in {3,4,5,6}; NOT architecture schedule threshold"
)

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = (z/den)*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, max(0.0, c-h), min(1.0, c+h)

def shots(p, L, mode):
    if p <= 0.002: base = 8000
    elif p <= 0.005: base = 5000
    elif p <= 0.01: base = 3000
    elif p <= 0.03: base = 2000
    else: base = 1200
    if mode == "pheno": base = max(600, base // 3)
    if L >= 6: base = max(800, (base * 2) // 3)
    elif L == 5: base = max(1000, (base * 4) // 5)
    return int(base)

def make_dec(H, p, order):
    return BpOsdDecoder(np.asarray(H, dtype=np.uint8), error_rate=float(p),
                        max_iter=MAX_ITER, bp_method="minimum_sum",
                        osd_method="osd_cs", osd_order=int(order))

def sim_capacity(L, p, nshots, rng):
    Hx, Hz, meta = build_repetition_layer_code(L)
    logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
    n = meta["n"]; order = OSD_ORDER[L]
    logX, logZ = logs["X"].astype(np.uint8), logs["Z"].astype(np.uint8)
    dz, dx = make_dec(Hx, p, order), make_dec(Hz, p, order)
    fail = 0
    for _ in range(nshots):
        ez = (rng.random(n) < p).astype(np.uint8)
        ex = (rng.random(n) < p).astype(np.uint8)
        cz = np.asarray(dz.decode((Hx @ ez) % 2), dtype=np.uint8)
        cx = np.asarray(dx.decode((Hz @ ex) % 2), dtype=np.uint8)
        fail += int((((ez ^ cz) @ logX) % 2) or (((ex ^ cx) @ logZ) % 2))
    rate, lo, hi = wilson(fail, nshots)
    return fail, rate, lo, hi, n, min(logs["X_weight"], logs["Z_weight"]), order

def sim_pheno(L, p, nshots, rng):
    Hx, Hz, meta = build_repetition_layer_code(L)
    logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
    n = meta["n"]; R = R_PHENO[L]; order = OSD_ORDER[L]
    logX, logZ = logs["X"].astype(np.uint8), logs["Z"].astype(np.uint8)
    Hst_z = spacetime_check_matrix(Hx, R)
    Hst_x = spacetime_check_matrix(Hz, R)
    mz, mx = Hx.shape[0], Hz.shape[0]
    dz, dx = make_dec(Hst_z, p, order), make_dec(Hst_x, p, order)
    fail = 0
    for _ in range(nshots):
        ez_tot = np.zeros(n, dtype=np.uint8); ex_tot = np.zeros(n, dtype=np.uint8)
        err_z = np.zeros(R*n + R*mz, dtype=np.uint8)
        err_x = np.zeros(R*n + R*mx, dtype=np.uint8)
        for r in range(R):
            ez = (rng.random(n) < p).astype(np.uint8); ex = (rng.random(n) < p).astype(np.uint8)
            ez_tot ^= ez; ex_tot ^= ex
            err_z[r*n:(r+1)*n] = ez; err_x[r*n:(r+1)*n] = ex
            err_z[R*n+r*mz:R*n+(r+1)*mz] = (rng.random(mz) < p).astype(np.uint8)
            err_x[R*n+r*mx:R*n+(r+1)*mx] = (rng.random(mx) < p).astype(np.uint8)
        cz = np.asarray(dz.decode((Hst_z @ err_z) % 2), dtype=np.uint8)
        cx = np.asarray(dx.decode((Hst_x @ err_x) % 2), dtype=np.uint8)
        c_z = np.zeros(n, dtype=np.uint8); c_x = np.zeros(n, dtype=np.uint8)
        for r in range(R):
            c_z ^= cz[r*n:(r+1)*n]; c_x ^= cx[r*n:(r+1)*n]
        fail += int((((ez_tot ^ c_z) @ logX) % 2) or (((ex_tot ^ c_x) @ logZ) % 2))
    rate, lo, hi = wilson(fail, nshots)
    return fail, rate, lo, hi, n, min(logs["X_weight"], logs["Z_weight"]), order, R

def crossing(rows, mode):
    sub = [r for r in rows if r["noise_model"] == mode]
    by_p = {}
    for r in sub: by_p.setdefault(r["p"], {})[r["L"]] = r["logical_error_rate"]
    lines = [f"mode={mode}"]; mono = []
    for p in sorted(by_p):
        m = by_p[p]
        if not all(L in m for L in L_VALUES): continue
        seq = [m[L] for L in L_VALUES]
        if all(seq[i] > seq[i+1] for i in range(len(seq)-1)): mono.append(p)
        pairs = [f"L{b}<L{a}" for a,b in ((3,4),(4,5),(5,6),(3,6)) if a in m and b in m and m[b] < m[a]]
        lines.append(f"  p={p:.3f}: " + ", ".join(f"L{L}={m[L]:.4g}" for L in L_VALUES)
                     + (f"  [{', '.join(pairs)}]" if pairs else "  [no larger-L wins]"))
    if mono:
        lines.insert(1, f"threshold-like L3>L4>L5>L6 at p in {mono}")
    else:
        p0 = min(by_p) if by_p else None
        if p0 and all(L in by_p[p0] for L in L_VALUES) and by_p[p0][6] < by_p[p0][3]:
            lines.insert(1, f"PARTIAL signal at p={p0}: L6={by_p[p0][6]:.4g}<L3={by_p[p0][3]:.4g}; no full crossing")
        else:
            lines.insert(1, "NO clear threshold-like crossing after deepened search")
    return "\n".join(lines)

def plot_mode(rows, mode, stem):
    sub = [r for r in rows if r["noise_model"] == mode]
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    mk = {3:"o",4:"s",5:"^",6:"D"}
    for L in L_VALUES:
        s = sorted([r for r in sub if r["L"]==L], key=lambda r: r["p"])
        if not s: continue
        xs=[r["p"] for r in s]; ys=[max(r["logical_error_rate"],1e-6) for r in s]
        ylo=[max(r["logical_error_rate"]-r["wilson_lo"],0) for r in s]
        yhi=[max(r["wilson_hi"]-r["logical_error_rate"],0) for r in s]
        ax.errorbar(xs, ys, yerr=[ylo,yhi], fmt=f"{mk[L]}-", capsize=2, label=rf"$L={L}$")
    ax.set_xlabel(r"$p$"); ax.set_ylabel(r"$p_L$"); ax.set_yscale("log")
    ax.set_title(f"WB deeper BP-OSD ({mode})"); ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=7); fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".png"), dpi=240); plt.close(fig)

def main():
    rng = np.random.default_rng(SEED)
    rows = []; t0 = time.time()
    print(f"SEED={SEED} deeper BP-OSD")
    for mode in ("code_capacity", "phenomenological_R"):
        for L in L_VALUES:
            print(f"=== {mode} L={L} ===")
            for p in P_GRID:
                nsh = shots(p, L, "pheno" if mode.startswith("pheno") else "cap")
                if mode == "code_capacity":
                    fail, rate, lo, hi, n, d_up, order = sim_capacity(L, p, nsh, rng); R=1
                else:
                    fail, rate, lo, hi, n, d_up, order, R = sim_pheno(L, p, nsh, rng)
                rows.append(dict(L=L, n=n, R=R, d_upper=d_up, p=float(p), shots=nsh, seed=SEED,
                                 osd_order=order, max_iter=MAX_ITER, noise_model=mode,
                                 logical_failures=fail, logical_error_rate=rate,
                                 wilson_lo=lo, wilson_hi=hi, decoder="ldpc.BpOsdDecoder", scope=SCOPE))
                print(f"  p={p:.3f}: p_L={rate:.4g} fails={fail}/{nsh}")
    print(f"elapsed {time.time()-t0:.1f}s")
    out = ROOT / "wb_bposd_threshold_deeper.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot_mode(rows, "code_capacity", FIG/"wb_bposd_threshold_deeper_capacity")
    plot_mode(rows, "phenomenological_R", FIG/"wb_bposd_threshold_deeper_pheno")
    rep = crossing(rows, "code_capacity") + "\n\n" + crossing(rows, "phenomenological_R")
    print(rep)
    (ROOT/"wb_bposd_threshold_deeper_crossing.txt").write_text(rep+"\n")
    print(f"wrote {out.name}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
