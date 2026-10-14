#!/usr/bin/env python3
"""PRIORITY F — Discrete CAD integer embedding + concrete t0.

From wb_junction_support_catalog + truncated WB layer geometry, embed all
qubits/checks on Z^3 with unit edge length and compute instance-exact
t0 = max L^infty radius of cells after unit embedding (integer).

Also reports combinatorial slab t0 from slab_rank_identity for cross-check.

SCOPE: instance-exact integer t0 on truncated WB construction; asymptotic
theorem still uses only t0=O(1). NOT metric calibration of published artwork.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wb_layer_code_instance import (
    FIRST_Z_SX,
    LAST_Z_SX,
    _EDGE_DELTA,
    build_repetition_layer_code,
    quasiconcat_and_kernel_logicals,
    slab_rank_identity,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
CATALOG = ROOT / "wb_junction_support_catalog.csv"

SCOPE = (
    "instance-exact integer t0 from unit Z^3 embedding of truncated WB cells; "
    "asymptotic remains O(1); NOT CAD artwork metric calibration"
)


def catalog_cell_radii():
    """Max L^inf radius of each catalog cell about its origin (unit embedding)."""
    rows = list(csv.DictReader(CATALOG.open()))
    out = []
    for row in rows:
        # support endpoints listed as label:(x,y,z)->(x,y,z)
        pts = []
        for part in row["support_endpoints"].split(";"):
            part = part.strip()
            if not part:
                continue
            # xm:(-1, 0, 0)->(0, 0, 0)
            _, rest = part.split(":", 1)
            a_s, b_s = rest.split("->")
            a = tuple(int(x) for x in a_s.strip().strip("()").split(","))
            b = tuple(int(x) for x in b_s.strip().strip("()").split(","))
            pts.extend([a, b])
        if not pts:
            continue
        # radius about origin (0,0,0) in L^inf
        r = max(max(abs(c) for c in p) for p in pts)
        out.append(
            dict(
                cell=row["cell"],
                check=row["check"],
                pauli=row["pauli"],
                weight=row["weight"],
                r_inf=r,
                n_points=len(set(pts)),
            )
        )
    return out


def embed_instance(L: int):
    """Integer embedding of all edge-qubits and check supports for truncated WB."""
    Hx, Hz, meta = build_repetition_layer_code(L)
    edges = meta["edges"]
    # qubit coords = edge midpoints * 2 so integers
    q_coords = []
    for a, b in edges:
        mx = a[0] + b[0]
        my = a[1] + b[1]
        mz = a[2] + b[2]
        q_coords.append((mx, my, mz))  # already integer on doubled lattice

    def row_radius(H):
        radii = []
        for row in H:
            idxs = np.flatnonzero(row)
            if len(idxs) == 0:
                continue
            pts = [q_coords[i] for i in idxs]
            # centroid (integer-rounded) then L^inf radius in doubled units
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cz = sum(p[2] for p in pts) / len(pts)
            # convert to original unit lattice: divide by 2
            r = max(
                max(abs(p[0] / 2 - cx / 2), abs(p[1] / 2 - cy / 2), abs(p[2] / 2 - cz / 2))
                for p in pts
            )
            radii.append(r)
        return radii

    rx = row_radius(Hx)
    rz = row_radius(Hz)
    # catalog cell radius (unit)
    cat = catalog_cell_radii()
    r_cat = max(c["r_inf"] for c in cat) if cat else 0
    # instance embedding: max over check supports and catalog
    r_checks = max(rx + rz) if (rx or rz) else 0.0
    # t0_cad = ceil of max radius; catalog stars have r_inf=1
    t0_float = max(float(r_cat), float(r_checks))
    t0_int = int(np.ceil(t0_float - 1e-12))
    # combinatorial slab cross-check at midplanes
    z0 = L / 2.0
    y0 = float(meta["Y_DATA"][1])

    def min_slab_t(axis, plane):
        for step in range(0, 2 * L + 3):
            t = 0.5 * step
            if axis == "z":
                S = [
                    i
                    for i, (a, b) in enumerate(edges)
                    if abs((a[2] + b[2]) / 2 - plane) <= t + 1e-9
                ]
            else:
                S = [
                    i
                    for i, (a, b) in enumerate(edges)
                    if abs((a[1] + b[1]) / 2 - plane) <= t + 1e-9
                ]
            _, _, px = slab_rank_identity(Hx, Hz, S, "X")
            _, _, pz = slab_rank_identity(Hx, Hz, S, "Z")
            if px and pz:
                return t
        return None

    tz = min_slab_t("z", z0)
    ty = min_slab_t("y", y0)
    logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
    return dict(
        L=L,
        n=meta["n"],
        d_upper=min(logs["X_weight"], logs["Z_weight"]),
        catalog_r_inf_max=r_cat,
        check_support_r_inf_max=round(r_checks, 6),
        t0_float=round(t0_float, 6),
        t0_integer=t0_int,
        t0_slab_z=tz,
        t0_slab_y=ty,
        t0_instance_recommended=max(t0_int, int(np.ceil((tz or 0) - 1e-12)), int(np.ceil((ty or 0) - 1e-12))),
        n_X_checks=int(Hx.shape[0]),
        n_Z_checks=int(Hz.shape[0]),
        scope=SCOPE,
    )


def main():
    cat = catalog_cell_radii()
    cat_path = ROOT / "discrete_cad_catalog_radii.csv"
    with cat_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cat[0].keys()))
        w.writeheader()
        w.writerows(cat)

    rows = []
    for L in (3, 4, 5, 6):
        rec = embed_instance(L)
        rows.append(rec)
        print(
            f"L={L}: t0_integer={rec['t0_integer']} "
            f"recommended={rec['t0_instance_recommended']} "
            f"slab_z={rec['t0_slab_z']} slab_y={rec['t0_slab_y']} "
            f"catalog_r={rec['catalog_r_inf_max']}"
        )

    out = ROOT / "discrete_cad_t0.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # concrete global t0: max over instances of recommended
    t0_global = max(r["t0_instance_recommended"] for r in rows)
    # catalog alone gives 1; slab needs up to 2; use max
    summary = (
        f"concrete_integer_t0={t0_global}\n"
        f"catalog_max_r_inf={max(c['r_inf'] for c in cat)}\n"
        f"per_L="
        + ",".join(f"L{r['L']}:{r['t0_instance_recommended']}" for r in rows)
        + "\n"
        f"SCOPE={SCOPE}\n"
        "Theorem II.3 discussion: instance-exact t0 is this integer; asymptotic remains O(1).\n"
    )
    (ROOT / "discrete_cad_t0_summary.txt").write_text(summary)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    Ls = [r["L"] for r in rows]
    ax.plot(Ls, [r["t0_integer"] for r in rows], "o-", label=r"CAD $\lceil r_\infty\rceil$")
    ax.plot(Ls, [r["t0_instance_recommended"] for r in rows], "s--", label=r"recommended $t_0$")
    ax.axhline(t0_global, color="k", ls=":", lw=1, label=rf"global $t_0={t0_global}$")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"integer $t_0$")
    ax.set_title("Discrete CAD unit-embedding $t_0$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "discrete_cad_t0.pdf")
    fig.savefig(FIG / "discrete_cad_t0.png", dpi=240)
    plt.close(fig)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
