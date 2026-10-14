#!/usr/bin/env python3
"""Finite WB-instance family numerics: sweep L and record honest parameters.

Uses the truncated WB-incidence-derived 3-repetition layer-code builder in
wb_layer_code_instance.py. Every plotted/table number is produced by this run.

SCOPE: truncated finite test-instance family numerics; not a layer-code
threshold, not a metric calibration of the published lattice artwork, and not
a claim of architecture-level p_th. BP-OSD decoding of this family is provided separately by wb_bposd_threshold.py.
"""
from __future__ import annotations

import csv
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wb_layer_code_instance import (
    build_repetition_layer_code,
    gf2_rank,
    quasiconcat_and_kernel_logicals,
    region_prefix_y,
    region_yz_slab,
    slab_rank_identity,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# L=2 is unsupported by the interior-junction construction (requires L>=3).
L_CANDIDATES = (2, 3, 4, 5, 6)
TIME_BUDGET_S = 120.0  # stop gracefully if a single L exceeds this


def evaluate_L(L: int) -> dict:
    t0 = time.time()
    if L < 3:
        return {
            "L": L,
            "status": "unsupported",
            "n": "",
            "k": "",
            "rank_Hx": "",
            "rank_Hz": "",
            "css_ok": "",
            "n_junction_checks": "",
            "n_condensed_boundary": "",
            "X_logical_weight_upper": "",
            "Z_logical_weight_upper": "",
            "d_upper": "",
            "X_logical_ok": "",
            "Z_logical_ok": "",
            "regions_tested": "",
            "regions_both_pass": "",
            "regions_pass_X": "",
            "regions_pass_Z": "",
            "slab_pass_summary": "L<3 unsupported (need interior junction stars)",
            "runtime_s": round(time.time() - t0, 4),
            "scope": (
                "construction requires L>=3 for interior catalog junction stars; "
                "not attempted"
            ),
        }

    Hx, Hz, meta = build_repetition_layer_code(L)
    n = meta["n"]
    rx, rz = gf2_rank(Hx), gf2_rank(Hz)
    k = n - rx - rz
    css = bool(np.all((Hx @ Hz.T) % 2 == 0))
    logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)

    regions = [("all_qubits", list(range(n)))]
    for ci in (0, 1):
        regions.append((f"yz_slab_c{ci}", region_yz_slab(meta, ci)))
    for y_cut in meta["Y_DATA"]:
        regions.append((f"prefix_y_le_{y_cut}", region_prefix_y(meta, y_cut)))

    both, pass_x, pass_z = [], [], []
    for tag, S in regions:
        lx, _, px = slab_rank_identity(Hx, Hz, S, "X")
        lz, _, pz = slab_rank_identity(Hx, Hz, S, "Z")
        if px:
            pass_x.append(tag)
        if pz:
            pass_z.append(tag)
        if px and pz:
            both.append(tag)

    d_up = min(logs["X_weight"], logs["Z_weight"]) if (
        logs["X_is_logical"] and logs["Z_is_logical"]
    ) else ""
    dt = time.time() - t0
    return {
        "L": L,
        "status": "ok" if css and k >= 1 and logs["X_is_logical"] and logs["Z_is_logical"] else "partial",
        "n": n,
        "k": k,
        "rank_Hx": rx,
        "rank_Hz": rz,
        "css_ok": int(css),
        "n_junction_checks": meta["n_junction_checks"],
        "n_condensed_boundary": meta["n_condensed_boundary"],
        "X_logical_weight_upper": logs["X_weight"],
        "Z_logical_weight_upper": logs["Z_weight"],
        "d_upper": d_up,
        "X_logical_ok": int(logs["X_is_logical"]),
        "Z_logical_ok": int(logs["Z_is_logical"]),
        "regions_tested": len(regions),
        "regions_both_pass": len(both),
        "regions_pass_X": len(pass_x),
        "regions_pass_Z": len(pass_z),
        "slab_pass_summary": ";".join(both) if both else "(none)",
        "runtime_s": round(dt, 4),
        "scope": meta["scope"],
    }


def plot_family(rows: list[dict], stem: Path) -> None:
    ok = [r for r in rows if r["status"] in ("ok", "partial") and r["n"] != ""]
    if not ok:
        return
    Ls = [int(r["L"]) for r in ok]
    ns = [int(r["n"]) for r in ok]
    dups = [int(r["d_upper"]) if r["d_upper"] != "" else np.nan for r in ok]
    both = [int(r["regions_both_pass"]) for r in ok]
    tested = [int(r["regions_tested"]) for r in ok]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))

    ax = axes[0]
    ax.plot(Ls, ns, "o-", color="#1f4e79")
    ax.set_xlabel(r"Instance size $L$")
    ax.set_ylabel(r"Number of qubits $n$")
    ax.set_title("WB family: $n$ vs $L$")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(Ls)

    ax = axes[1]
    ax.plot(Ls, dups, "s-", color="#a0312c")
    ax.set_xlabel(r"Instance size $L$")
    ax.set_ylabel(r"$d_{\mathrm{upper}}=\min(w_X,w_Z)$")
    ax.set_title("Explicit-logical distance upper bound")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(Ls)

    ax = axes[2]
    width = 0.35
    x = np.arange(len(Ls))
    ax.bar(x - width / 2, both, width, label="both-pass", color="#2e7d32")
    ax.bar(x + width / 2, tested, width, label="tested", color="#90caf9")
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel(r"Instance size $L$")
    ax.set_ylabel("Region count")
    ax.set_title("Slab-rank region pass summary")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Finite WB-incidence family numerics (truncated test instances; not arch. $p_{th}$)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main() -> int:
    rows = []
    for L in L_CANDIDATES:
        print(f"--- evaluating L={L} ---")
        t0 = time.time()
        try:
            rec = evaluate_L(L)
            # soft time budget after successful build
            if rec.get("runtime_s", 0) and float(rec["runtime_s"]) > TIME_BUDGET_S:
                rec["status"] = "too_slow"
            rows.append(rec)
            print(
                f"L={L}: status={rec['status']} n={rec['n']} k={rec['k']} "
                f"d_upper={rec['d_upper']} both_pass={rec['regions_both_pass']}/"
                f"{rec['regions_tested']} t={rec['runtime_s']}s"
            )
            print(f"  slab_pass: {rec['slab_pass_summary']}")
        except MemoryError:
            rows.append({
                "L": L, "status": "oom", "n": "", "k": "", "rank_Hx": "", "rank_Hz": "",
                "css_ok": "", "n_junction_checks": "", "n_condensed_boundary": "",
                "X_logical_weight_upper": "", "Z_logical_weight_upper": "", "d_upper": "",
                "X_logical_ok": "", "Z_logical_ok": "", "regions_tested": "",
                "regions_both_pass": "", "regions_pass_X": "", "regions_pass_Z": "",
                "slab_pass_summary": "OOM", "runtime_s": round(time.time() - t0, 4),
                "scope": "stopped due to memory error",
            })
            print(f"L={L}: OOM — stopping larger sizes")
            break
        except Exception as e:
            rows.append({
                "L": L, "status": f"fail:{type(e).__name__}", "n": "", "k": "",
                "rank_Hx": "", "rank_Hz": "", "css_ok": "", "n_junction_checks": "",
                "n_condensed_boundary": "", "X_logical_weight_upper": "",
                "Z_logical_weight_upper": "", "d_upper": "", "X_logical_ok": "",
                "Z_logical_ok": "", "regions_tested": "", "regions_both_pass": "",
                "regions_pass_X": "", "regions_pass_Z": "",
                "slab_pass_summary": str(e)[:120], "runtime_s": round(time.time() - t0, 4),
                "scope": "exception during build",
            })
            print(f"L={L}: FAIL {e}")
            traceback.print_exc()

    out = ROOT / "numerical_wb_family.csv"
    fields = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    plot_family(rows, FIG / "numerical_wb_family")
    print(f"wrote {out.name} and figures/numerical_wb_family.pdf/.png")
    print(
        "NOTE: BP-OSD / general CSS decoding stretch skipped "
        "(ldpc package not installed); no fake thresholds reported."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
