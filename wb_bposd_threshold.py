#!/usr/bin/env python3
"""BP-OSD code-capacity threshold sweep on finite WB-incidence CSS instances.

Builds truncated WB-incidence 3-repetition layer-code instances for L in
{3,4,5,6} via wb_layer_code_instance.build_repetition_layer_code (no geometry
duplication).  Independent code-capacity X and Z noise is decoded separately
with ldpc.BpOsdDecoder on the CSS sectors (Hx for Z-errors vs X-logical; Hz for
X-errors vs Z-logical (CSS symplectic)).  Logical failure is the OR of the two sectors.

SCOPE: finite WB-incidence CSS memory under code-capacity noise with BP-OSD;
NOT a full architecture schedule threshold.
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
    gf2_rank,
    quasiconcat_and_kernel_logicals,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SEED = 20260918
L_VALUES = (3, 4, 5, 6)
# Grid chosen to straddle expected finite-size crossover for small d_upper.
P_GRID = [0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
OSD_ORDER = {3: 4, 4: 5, 5: 6, 6: 6}
MAX_ITER = 30
BP_METHOD = "minimum_sum"
OSD_METHOD = "osd_cs"
SCOPE = (
    "finite WB-incidence CSS memory under code-capacity noise with BP-OSD; "
    "NOT a full architecture schedule threshold"
)


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def shot_budget(p: float, L: int) -> int:
    """More shots at low p / larger L for tighter Wilson intervals."""
    if p <= 0.004:
        base = 20000
    elif p <= 0.008:
        base = 12000
    elif p <= 0.04:
        base = 8000
    elif p <= 0.08:
        base = 5000
    else:
        base = 3000
    if L >= 6:
        base = max(2000, (base * 3) // 4)
    if L == 5:
        base = max(2500, (base * 7) // 8)
    return int(base)


def make_decoder(H: np.ndarray, p: float, osd_order: int) -> BpOsdDecoder:
    return BpOsdDecoder(
        np.asarray(H, dtype=np.uint8),
        error_rate=float(p),
        max_iter=MAX_ITER,
        bp_method=BP_METHOD,
        osd_method=OSD_METHOD,
        osd_order=int(osd_order),
    )


def simulate_instance(L: int, p: float, shots: int, rng: np.random.Generator) -> dict:
    Hx, Hz, meta = build_repetition_layer_code(L)
    logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
    assert logs["X_is_logical"] and logs["Z_is_logical"]
    n = meta["n"]
    rx, rz = gf2_rank(Hx), gf2_rank(Hz)
    k = n - rx - rz
    assert k == 1
    logX = logs["X"].astype(np.uint8)
    logZ = logs["Z"].astype(np.uint8)
    order = OSD_ORDER[L]

    # Z-errors -> Hx syndrome -> residual vs logX (CSS)
    dec_z = make_decoder(Hx, p, order)
    # X-errors -> Hz syndrome -> residual vs logZ (CSS)
    dec_x = make_decoder(Hz, p, order)

    fail_z = fail_x = fail_any = 0
    for _ in range(shots):
        e_z = (rng.random(n) < p).astype(np.uint8)
        syn_z = (Hx @ e_z) % 2
        c_z = np.asarray(dec_z.decode(syn_z), dtype=np.uint8)
        res_z = (e_z ^ c_z) % 2
        fz = int((res_z @ logX) % 2)

        e_x = (rng.random(n) < p).astype(np.uint8)
        syn_x = (Hz @ e_x) % 2
        c_x = np.asarray(dec_x.decode(syn_x), dtype=np.uint8)
        res_x = (e_x ^ c_x) % 2
        fx = int((res_x @ logZ) % 2)

        fail_z += fz
        fail_x += fx
        fail_any += int(fz or fx)

    rate, lo, hi = wilson_interval(fail_any, shots)
    rate_z, lo_z, hi_z = wilson_interval(fail_z, shots)
    rate_x, lo_x, hi_x = wilson_interval(fail_x, shots)
    return {
        "L": L,
        "n": n,
        "k": k,
        "rank_Hx": rx,
        "rank_Hz": rz,
        "d_upper": min(logs["X_weight"], logs["Z_weight"]),
        "X_logical_weight": logs["X_weight"],
        "Z_logical_weight": logs["Z_weight"],
        "p": float(p),
        "shots": shots,
        "seed": SEED,
        "osd_order": order,
        "bp_method": BP_METHOD,
        "osd_method": OSD_METHOD,
        "max_iter": MAX_ITER,
        "noise_model": "code_capacity_independent_XZ",
        "logical_failures": fail_any,
        "logical_error_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "fail_Z_sector": fail_z,
        "p_L_Z_sector": rate_z,
        "wilson_lo_Z": lo_z,
        "wilson_hi_Z": hi_z,
        "fail_X_sector": fail_x,
        "p_L_X_sector": rate_x,
        "wilson_lo_X": lo_x,
        "wilson_hi_X": hi_x,
        "decoder": "ldpc.BpOsdDecoder",
        "scope": SCOPE,
    }


def plot_results(rows, out_stem: Path):
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    markers = {3: "o", 4: "s", 5: "^", 6: "D"}
    for L in sorted({r["L"] for r in rows}):
        sub = sorted([r for r in rows if r["L"] == L], key=lambda r: r["p"])
        xs = [r["p"] for r in sub]
        ys = [max(r["logical_error_rate"], 1e-6) for r in sub]
        ylo = [max(r["logical_error_rate"] - r["wilson_lo"], 0.0) for r in sub]
        yhi = [max(r["wilson_hi"] - r["logical_error_rate"], 0.0) for r in sub]
        ax.errorbar(
            xs,
            ys,
            yerr=[ylo, yhi],
            fmt=f"{markers.get(L, 'o')}-",
            capsize=2,
            label=rf"$L={L}$ ($n={sub[0]['n']}$)",
        )
    ax.set_xlabel(r"Code-capacity error probability $p$")
    ax.set_ylabel(r"Logical error rate $p_L$ (BP-OSD)")
    ax.set_yscale("log")
    ax.set_title("WB-instance memory (BP-OSD; not arch. $p_{th}$)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def crossing_report(rows) -> str:
    """Honest finite-size crossing scan: larger L lower p_L at low p."""
    by_p = {}
    for r in rows:
        by_p.setdefault(r["p"], {})[r["L"]] = r["logical_error_rate"]
    lines = []
    mono_low = []
    for p in sorted(by_p):
        m = by_p[p]
        if not all(L in m for L in L_VALUES):
            continue
        seq = [m[L] for L in L_VALUES]
        # Strict monotonic improvement with L
        if all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)):
            mono_low.append(p)
        # Pairwise larger-beats-smaller
        pairs = []
        for a, b in ((3, 4), (4, 5), (5, 6), (3, 5), (3, 6), (4, 6)):
            if m[b] < m[a]:
                pairs.append(f"L{b}<L{a}")
        lines.append(f"  p={p:.2f}: " + ", ".join(f"L{L}={m[L]:.4g}" for L in L_VALUES)
                     + (f"  [{', '.join(pairs)}]" if pairs else "  [no larger-L wins]"))
    if mono_low:
        lines.insert(
            0,
            f"threshold-like finite-size ordering (L3>L4>L5>L6) at p in {mono_low}",
        )
    else:
        # Weaker: does larger L win at the lowest simulated p?
        p0 = min(by_p)
        m = by_p[p0]
        if all(L in m for L in L_VALUES) and m[6] < m[3]:
            lines.insert(
                0,
                f"partial finite-size signal at lowest p={p0}: L6 p_L={m[6]:.4g} < L3 p_L={m[3]:.4g}; "
                "no full monotonic crossing across the whole low-p window",
            )
        else:
            lines.insert(
                0,
                "NO clear threshold-like crossing: larger L does not systematically "
                "outperform smaller L at low p on this grid (still valuable finite-size BP-OSD data)",
            )
    return "\n".join(lines)


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    print(f"SEED={SEED} decoder=BpOsdDecoder osd_method={OSD_METHOD}")
    print(f"SCOPE: {SCOPE}")
    for L in L_VALUES:
        Hx, Hz, meta = build_repetition_layer_code(L)
        logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
        print(
            f"instance L={L}: n={meta['n']} k=1 "
            f"d_upper={min(logs['X_weight'], logs['Z_weight'])} "
            f"osd_order={OSD_ORDER[L]}"
        )
        for p in P_GRID:
            shots = shot_budget(p, L)
            rec = simulate_instance(L, p, shots, rng)
            rows.append(rec)
            print(
                f"  L={L} p={p:.2f}: p_L={rec['logical_error_rate']:.4g} "
                f"[{rec['wilson_lo']:.4g},{rec['wilson_hi']:.4g}] "
                f"fails={rec['logical_failures']}/{shots} "
                f"(Z={rec['fail_Z_sector']}, X={rec['fail_X_sector']})"
            )
    elapsed = time.time() - t0
    print(f"elapsed {elapsed:.1f}s")

    out = ROOT / "wb_bposd_threshold.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_results(rows, FIG / "wb_bposd_threshold")
    report = crossing_report(rows)
    print("crossing scan:\n" + report)
    print(f"wrote {out.name} and figures/wb_bposd_threshold.pdf/.png")
    print(f"SCOPE: {SCOPE}")
    # Persist a short report line for NUMERICAL_RUN.log consumers
    (ROOT / "wb_bposd_threshold_crossing.txt").write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
