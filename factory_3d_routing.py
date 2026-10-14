#!/usr/bin/env python3
"""PRIORITY E — Explicit Z^3 integer coordinates for Steane7 / RM15 factory routing.

Embeds:
  - Steane [[7,1,3]] and punctured RM15 factory blocks as local gadgets in Z^3
  - Injection port, courier path along a corridor, docking to a data slab via local gauging
Writes coordinate tables + a schematic figure.

SCOPE: SCHEMATIC ONLY. Explicit integer embedding of finite factory gadgets + corridor
routing for a figure; coordinates of different gadgets overlap by design, so this is
NOT a physical qubit allocation (see schedule_verifier.py for the checked layout);
NOT a full multilevel FTQC layout of an arbitrary W,T circuit.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SCOPE = (
    "explicit Z^3 integer coords for Steane7/RM15 factory gadgets, "
    "injection port, corridor courier, slab dock; NOT full asymptotic FTQC layout"
)


def steane_coords(origin=(0, 0, 0)):
    """Place 7 Steane data qubits on a compact 2x2x2 cube minus one corner."""
    ox, oy, oz = origin
    # nonzero F2^3 vectors as coords
    vecs = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    return [
        dict(
            block="steane7",
            role="data",
            index=i + 1,
            x=ox + v[0],
            y=oy + v[1],
            z=oz + v[2],
            label=f"S{i+1}",
        )
        for i, v in enumerate(vecs)
    ]


def rm15_coords(origin=(4, 0, 0)):
    """Place 15 RM qubits as nonzero F2^4 projected to a 2x2x4 prism."""
    ox, oy, oz = origin
    pts = []
    idx = 0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                for d in (0, 1):
                    if (a, b, c, d) == (0, 0, 0, 0):
                        continue
                    idx += 1
                    pts.append(
                        dict(
                            block="rm15",
                            role="data",
                            index=idx,
                            x=ox + a + 2 * d,
                            y=oy + b,
                            z=oz + c,
                            label=f"R{idx}",
                        )
                    )
    return pts


def corridor_path(start, end, axis="x"):
    """Axis-aligned courier path of integer sites from start to end."""
    sx, sy, sz = start
    ex, ey, ez = end
    path = []
    x, y, z = sx, sy, sz
    # Manhattan route: x then y then z
    while x != ex:
        x += 1 if ex > x else -1
        path.append((x, y, z))
    while y != ey:
        y += 1 if ey > y else -1
        path.append((x, y, z))
    while z != ez:
        z += 1 if ez > z else -1
        path.append((x, y, z))
    return path


def build_layout():
    rows = []
    # Factory blocks
    steane = steane_coords((0, 0, 0))
    rm = rm15_coords((4, 0, 0))
    rows.extend(steane)
    rows.extend(rm)

    # Injection port (bare resource birth) just west of Steane
    inj = dict(block="injection", role="port", index=0, x=-2, y=0, z=0, label="INJ")
    rows.append(inj)

    # Data slab dock: a constant-width slab patch at x=12, y in [0,2], z in [0,4]
    dock_pts = []
    for y in range(0, 3):
        for z in range(0, 5):
            dock_pts.append(
                dict(
                    block="data_slab",
                    role="dock",
                    index=len(dock_pts) + 1,
                    x=12,
                    y=y,
                    z=z,
                    label=f"D{len(dock_pts)+1}",
                )
            )
    rows.extend(dock_pts)

    # Courier: injection -> Steane center -> corridor -> slab dock
    steane_center = (1, 1, 1)
    dock_entry = (12, 1, 2)
    path1 = corridor_path((-2, 0, 0), steane_center)
    path2 = corridor_path(steane_center, (4, 1, 1))  # toward RM
    path3 = corridor_path((4, 1, 1), dock_entry)
    full = [(-2, 0, 0)] + path1 + path2 + path3
    # dedupe consecutive
    clean = [full[0]]
    for p in full[1:]:
        if p != clean[-1]:
            clean.append(p)
    for i, (x, y, z) in enumerate(clean):
        rows.append(
            dict(
                block="courier",
                role="path",
                index=i,
                x=x,
                y=y,
                z=z,
                label=f"C{i}",
            )
        )

    # Local gauging ancilla collar around dock (O(1) thickening)
    for dy, dz in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]:
        rows.append(
            dict(
                block="gauging",
                role="ancilla",
                index=len(rows),
                x=12,
                y=1 + dy,
                z=2 + dz,
                label="G",
            )
        )

    return rows, clean


def main():
    rows, path = build_layout()
    out = ROOT / "factory_3d_routing_coords.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "role", "index", "x", "y", "z", "label"])
        w.writeheader()
        w.writerows(rows)

    # Summary table
    summary = ROOT / "factory_3d_routing_summary.txt"
    counts = {}
    for r in rows:
        counts[r["block"]] = counts.get(r["block"], 0) + 1
    lines = [
        "Factory 3D routing — integer Z^3 embedding",
        f"SCOPE: {SCOPE}",
        f"courier_path_length={len(path)}",
        f"courier_start={path[0]} courier_end={path[-1]}",
        "block_counts=" + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
        "Steane origin=(0,0,0); RM15 origin=(4,0,0); injection=(-2,0,0); slab_dock x=12",
        "Dock uses local gauging ancilla collar (O(1) thickening of slab).",
        "STATUS: YES — explicit integer coordinates produced.",
    ]
    summary.write_text("\n".join(lines) + "\n")

    # 3D figure
    fig = plt.figure(figsize=(5.2, 4.0))
    ax = fig.add_subplot(111, projection="3d")
    colors = {
        "steane7": "C0",
        "rm15": "C1",
        "injection": "C3",
        "data_slab": "C2",
        "courier": "0.4",
        "gauging": "C4",
    }
    for block, color in colors.items():
        pts = [r for r in rows if r["block"] == block]
        if not pts:
            continue
        xs = [r["x"] for r in pts]
        ys = [r["y"] for r in pts]
        zs = [r["z"] for r in pts]
        marker = "o" if block != "courier" else "."
        ax.scatter(xs, ys, zs, c=color, label=block, s=28 if block != "courier" else 12, marker=marker)
    # path line
    ax.plot([p[0] for p in path], [p[1] for p in path], [p[2] for p in path], "k--", lw=1.0, alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Factory routing in $\\mathbb{Z}^3$")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "factory_3d_routing.pdf")
    fig.savefig(FIG / "factory_3d_routing.png", dpi=240)
    plt.close(fig)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
