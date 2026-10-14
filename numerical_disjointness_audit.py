#!/usr/bin/env python3
"""Numerical disjointness / slab-mobility audit on explicit finite families.

For stacked-toric CSS codes and the truncated WB-incidence family, enumerate
candidate slab-like regions, evaluate the finite slab-rank criterion, and count
how many pairwise-disjoint candidates pass (lhs >= 1 for a given polarization)
as system size grows.

SCOPE: finite-instance numerical support for the uniform-gap / disjointness
narrative in the manuscript. This is NOT a proof of Theta(N) disjointness,
NOT a substitute for the analytic uniform-gap theorem, and NOT an
architecture-level threshold claim.
"""
from __future__ import annotations

import csv
import itertools
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from operational_anchor import slab_rank_identity, stack_codes, toric_code_2d
from wb_layer_code_instance import (
    build_repetition_layer_code,
    gf2_rank,
    region_prefix_y,
    region_yz_slab,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def max_pairwise_disjoint(regions: list[tuple[str, list[int]]]) -> tuple[int, list[str]]:
    """Exact max packing of pairwise-disjoint regions (small candidate sets)."""
    sets = [(tag, set(S)) for tag, S in regions if S]
    m = len(sets)
    best_size, best_tags = 0, []
    # Enumerate non-empty subsets; m is kept small (<= 20)
    for r in range(1, m + 1):
        found = False
        for comb in itertools.combinations(range(m), r):
            ok = True
            for a, b in itertools.combinations(comb, 2):
                if sets[a][1] & sets[b][1]:
                    ok = False
                    break
            if ok:
                best_size = r
                best_tags = [sets[i][0] for i in comb]
                found = True
                break  # greedily accept first of this size; continue to larger r
        if not found and r > best_size + 1:
            break
    # Better: full search for maximum
    best_size, best_tags = 0, []
    for r in range(m, 0, -1):
        for comb in itertools.combinations(range(m), r):
            ok = True
            for a, b in itertools.combinations(comb, 2):
                if sets[a][1] & sets[b][1]:
                    ok = False
                    break
            if ok:
                return r, [sets[i][0] for i in comb]
    return best_size, best_tags


def greedy_disjoint(regions: list[tuple[str, list[int]]]) -> tuple[int, list[str]]:
    """Greedy packing sorted by region size (fallback for large candidate lists)."""
    ordered = sorted(
        [(tag, set(S)) for tag, S in regions if S],
        key=lambda x: len(x[1]),
    )
    chosen, used = [], set()
    for tag, S in ordered:
        if S.isdisjoint(used):
            chosen.append(tag)
            used |= S
    return len(chosen), chosen


def pack_disjoint(regions: list[tuple[str, list[int]]]) -> tuple[int, list[str], str]:
    if len(regions) <= 18:
        n, tags = max_pairwise_disjoint(regions)
        return n, tags, "exact"
    n, tags = greedy_disjoint(regions)
    return n, tags, "greedy"


def audit_stacked_toric(L: int, M: int) -> dict:
    """Stacked M copies of LxL toric code; candidates = individual layers & prefixes."""
    t0 = time.time()
    Hx, Hz, meta = stack_codes([toric_code_2d(L) for _ in range(M)])
    n = meta["n"]
    k = n - gf2_rank(Hx) - gf2_rank(Hz)
    layer_n = n // M

    candidates = []
    for m in range(M):
        S = list(range(m * layer_n, (m + 1) * layer_n))
        candidates.append((f"layer_{m}", S))
    # Contiguous layer-prefix blocks (for complete-support comparison)
    for t in range(1, M + 1):
        S = list(range(0, t * layer_n))
        candidates.append((f"prefix_layers_1..{t}", S))

    pass_X, pass_Z, pass_both_complete = [], [], []
    detail = []
    for tag, S in candidates:
        lx, _, px_full = slab_rank_identity(Hx, Hz, S, "X")
        lz, _, pz_full = slab_rank_identity(Hx, Hz, S, "Z")
        px = lx >= 1
        pz = lz >= 1
        if px:
            pass_X.append((tag, S))
        if pz:
            pass_Z.append((tag, S))
        if px_full and pz_full:
            pass_both_complete.append(tag)
        detail.append((tag, len(S), lx, lz, int(px), int(pz), int(px_full and pz_full)))

    # Disjointness among single-layer candidates only (the mobility narrative)
    single_X = [(t, S) for t, S in pass_X if t.startswith("layer_")]
    single_Z = [(t, S) for t, S in pass_Z if t.startswith("layer_")]
    nX, tagsX, howX = pack_disjoint(single_X)
    nZ, tagsZ, howZ = pack_disjoint(single_Z)

    return {
        "family": "stacked_toric",
        "L": L,
        "M_or_size": M,
        "n": n,
        "k": k,
        "n_candidates": len(candidates),
        "n_pass_X_partial": len(pass_X),
        "n_pass_Z_partial": len(pass_Z),
        "n_pass_both_complete": len(pass_both_complete),
        "max_disjoint_X": nX,
        "max_disjoint_Z": nZ,
        "disjoint_X_tags": ";".join(tagsX),
        "disjoint_Z_tags": ";".join(tagsZ),
        "pack_method_X": howX,
        "pack_method_Z": howZ,
        "runtime_s": round(time.time() - t0, 4),
        "scope": (
            "finite stacked-toric slab-rank disjointness audit; "
            "NOT a proof of Theta(N) layer-code disjointness"
        ),
    }


def audit_wb_family(L: int) -> dict:
    """WB truncated instance: yz-slab corridors, y-prefixes, and z-slices."""
    t0 = time.time()
    Hx, Hz, meta = build_repetition_layer_code(L)
    n = meta["n"]
    k = n - gf2_rank(Hx) - gf2_rank(Hz)
    edges = meta["edges"]

    candidates = []
    for ci in (0, 1):
        candidates.append((f"yz_slab_c{ci}", region_yz_slab(meta, ci)))
    for y_cut in meta["Y_DATA"]:
        candidates.append((f"prefix_y_le_{y_cut}", region_prefix_y(meta, y_cut)))

    # z-slices: qubits whose both endpoints have z in [z0, z0+1] (thin slabs along z)
    for z0 in range(L):
        S = []
        for i, (a, b) in enumerate(edges):
            zs = {a[2], b[2]}
            if zs <= {z0, z0 + 1}:
                S.append(i)
        candidates.append((f"z_slice_{z0}", S))

    # x-slabs away from junctions
    XJ = set(meta["XJ"])
    for x0 in range(L + 1):
        if x0 in XJ:
            continue
        S = []
        for i, (a, b) in enumerate(edges):
            if a[0] == x0 or b[0] == x0:
                S.append(i)
        if S:
            candidates.append((f"x_column_{x0}", sorted(set(S))))

    pass_X, pass_Z = [], []
    n_complete = 0
    for tag, S in candidates:
        if not S:
            continue
        lx, _, px_full = slab_rank_identity(Hx, Hz, S, "X")
        lz, _, pz_full = slab_rank_identity(Hx, Hz, S, "Z")
        if lx >= 1:
            pass_X.append((tag, S))
        if lz >= 1:
            pass_Z.append((tag, S))
        if px_full and pz_full:
            n_complete += 1

    # Prefer packing among geometrically parallel families
    def filter_pref(passing, prefixes):
        return [(t, S) for t, S in passing if any(t.startswith(p) for p in prefixes)]

    # z-slices are the natural "gap mobility" family for this truncated geometry
    zX = filter_pref(pass_X, ("z_slice_",))
    zZ = filter_pref(pass_Z, ("z_slice_",))
    # also report among all passing
    nX_z, tagsX_z, howX = pack_disjoint(zX if zX else pass_X)
    nZ_z, tagsZ_z, howZ = pack_disjoint(zZ if zZ else pass_Z)
    nX_all, tagsX_all, _ = pack_disjoint(pass_X)
    nZ_all, tagsZ_all, _ = pack_disjoint(pass_Z)

    return {
        "family": "wb_rep3",
        "L": L,
        "M_or_size": L,
        "n": n,
        "k": k,
        "n_candidates": len(candidates),
        "n_pass_X_partial": len(pass_X),
        "n_pass_Z_partial": len(pass_Z),
        "n_pass_both_complete": n_complete,
        "max_disjoint_X": nX_z,
        "max_disjoint_Z": nZ_z,
        "max_disjoint_X_all": nX_all,
        "max_disjoint_Z_all": nZ_all,
        "disjoint_X_tags": ";".join(tagsX_z),
        "disjoint_Z_tags": ";".join(tagsZ_z),
        "pack_method_X": howX,
        "pack_method_Z": howZ,
        "runtime_s": round(time.time() - t0, 4),
        "scope": (
            "finite WB-incidence slab-mobility / disjointness audit; "
            "NOT a proof of the analytic uniform-gap Theta(N) theorem"
        ),
    }


def plot_audit(rows: list[dict], stem: Path) -> None:
    stacked = [r for r in rows if r["family"] == "stacked_toric"]
    wb = [r for r in rows if r["family"] == "wb_rep3"]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))

    ax = axes[0]
    if stacked:
        # Group by L; x-axis = M (number of layers)
        for L in sorted({int(r["L"]) for r in stacked}):
            sub = sorted(
                [r for r in stacked if int(r["L"]) == L],
                key=lambda r: int(r["M_or_size"]),
            )
            Ms = [int(r["M_or_size"]) for r in sub]
            dX = [int(r["max_disjoint_X"]) for r in sub]
            ax.plot(Ms, dX, "o-", label=rf"stacked toric $L={L}$ (X)")
        ax.set_xlabel(r"Number of stacked layers $M$")
        ax.set_ylabel(r"Max pairwise-disjoint passing slabs")
        ax.set_title("Stacked-toric layer disjointness")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    if wb:
        Ls = [int(r["L"]) for r in wb]
        dX = [int(r["max_disjoint_X"]) for r in wb]
        dZ = [int(r["max_disjoint_Z"]) for r in wb]
        npass = [int(r["n_pass_X_partial"]) for r in wb]
        ax.plot(Ls, dX, "s-", label=r"WB max-disjoint X (z-slices)")
        ax.plot(Ls, dZ, "^-", label=r"WB max-disjoint Z (z-slices)")
        ax.plot(Ls, npass, "x--", label=r"WB \# X-pass candidates", alpha=0.7)
        ax.set_xlabel(r"WB instance size $L$")
        ax.set_ylabel("Count")
        ax.set_title("WB finite slab-mobility audit")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
        ax.set_xticks(Ls)

    fig.suptitle(
        "Numerical disjointness audit (finite support; not a proof)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main() -> int:
    rows = []
    print("=== stacked toric family ===")
    for L in (3, 4):
        for M in range(1, 9):
            rec = audit_stacked_toric(L, M)
            # flatten extra keys for CSV uniformity
            rec.setdefault("max_disjoint_X_all", rec["max_disjoint_X"])
            rec.setdefault("max_disjoint_Z_all", rec["max_disjoint_Z"])
            rows.append(rec)
            print(
                f"stacked L={L} M={M}: n={rec['n']} k={rec['k']} "
                f"disjoint_X={rec['max_disjoint_X']} disjoint_Z={rec['max_disjoint_Z']} "
                f"complete_pass={rec['n_pass_both_complete']} t={rec['runtime_s']}s"
            )

    print("=== WB family ===")
    for L in (3, 4, 5, 6):
        rec = audit_wb_family(L)
        rows.append(rec)
        print(
            f"wb L={L}: n={rec['n']} k={rec['k']} "
            f"passX={rec['n_pass_X_partial']} passZ={rec['n_pass_Z_partial']} "
            f"disjoint_X={rec['max_disjoint_X']} (all={rec['max_disjoint_X_all']}) "
            f"disjoint_Z={rec['max_disjoint_Z']} (all={rec['max_disjoint_Z_all']}) "
            f"t={rec['runtime_s']}s"
        )
        print(f"  X tags: {rec['disjoint_X_tags']}")
        print(f"  Z tags: {rec['disjoint_Z_tags']}")

    # Unified CSV columns
    fieldnames = [
        "family", "L", "M_or_size", "n", "k", "n_candidates",
        "n_pass_X_partial", "n_pass_Z_partial", "n_pass_both_complete",
        "max_disjoint_X", "max_disjoint_Z",
        "max_disjoint_X_all", "max_disjoint_Z_all",
        "disjoint_X_tags", "disjoint_Z_tags",
        "pack_method_X", "pack_method_Z", "runtime_s", "scope",
    ]
    out = ROOT / "numerical_disjointness.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r.setdefault("max_disjoint_X_all", "")
            r.setdefault("max_disjoint_Z_all", "")
            w.writerow(r)

    plot_audit(rows, FIG / "numerical_disjointness")
    print(f"wrote {out.name} and figures/numerical_disjointness.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
