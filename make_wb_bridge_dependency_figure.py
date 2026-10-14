#!/usr/bin/env python3
"""One-page WB microscopic bridge dependency map (exact-cell hierarchy)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

DERIVED = "#dcefdc"
AXIOM = "#f5d6d6"
AUDIT = "#f8e7b8"
EDGE = "#3f4752"
ARROW = "#59616b"
TEXT = "#171b21"

def box(ax, x, y, w, h, text, fc, fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.016,rounding_size=0.04",
        facecolor=fc, edgecolor=EDGE, linewidth=0.95))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.08, color=TEXT,
            wrap=False)

def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="->", lw=1.05, color=ARROW))

def main():
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    ax.text(7, 8.85, "WB microscopic bridge → slab theorem (exact-cell hierarchy)",
            ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT)

    box(ax, 0.3, 7.55, 3.3, 0.85, "Published WB\ncondensation generators\n+ Sec. VI B artwork", AXIOM, 6.0)
    box(ax, 5.2, 7.55, 3.6, 0.85, "WB-LR RETIRED\nexact finite-cell patches\n(no catalog-touch exemption)", DERIVED, 5.8, True)
    box(ax, 10.0, 7.55, 3.5, 0.85, "Finite cell catalog\n⇒ Δ_cell = O(1)\n(lem:wb-cell-locality)", DERIVED, 5.9)

    arrow(ax, (3.6, 7.95), (5.2, 7.95))
    arrow(ax, (8.8, 7.95), (10.0, 7.95))

    box(ax, 0.3, 5.85, 4.2, 1.05, "GF(2) derivation\nH_end=[1 1], H_mid=[1 1 1]\nker + 110 pass-through\n101/011 attachments", DERIVED, 6.0, True)
    box(ax, 5.0, 5.85, 4.2, 1.05, "VI B items 3–8 catalog\nr_inc^(∞)=1, r_inc^(G)=2\nmax wt 6; CSS; reflections\nexact patch classification", DERIVED, 6.0, True)
    box(ax, 9.7, 5.85, 3.9, 1.05, "PROOF-CRITICAL\nexact GF(2) word cert\nA_int p=0, B p=w\n16/16 PASS (R=2)", DERIVED, 5.7, True)

    arrow(ax, (2.0, 7.55), (2.4, 6.90))
    arrow(ax, (7.0, 7.55), (7.1, 6.90))
    arrow(ax, (11.7, 7.55), (11.6, 6.90))

    box(ax, 0.3, 4.15, 4.2, 1.0, "Prefix parity\neven words 2..8 + stress\nrefinement row-space IDs", AUDIT, 6.0)
    box(ax, 5.0, 4.15, 4.2, 1.0, "STRESS ONLY (demoted)\nR=2/R=3 exterior bulk\nbullet/plus SM cross-check", AUDIT, 6.0)
    box(ax, 9.7, 4.15, 3.9, 1.0, "Combinatorial incidence\n(wb_catalog_utils; no pymatching)\nwb_bridge_claim_status.csv", AUDIT, 5.9)

    arrow(ax, (2.4, 5.85), (2.4, 5.15))
    arrow(ax, (7.1, 5.85), (7.1, 5.15))
    arrow(ax, (11.6, 5.85), (11.6, 5.15))

    box(ax, 3.2, 2.35, 7.6, 1.15,
        "Theorem thm:slab-resolved\n"
        "inputs ONLY: (i) Δ_cell; (ii) H_end/H_mid incl. 110;\n"
        "(iii) Sec. VI B first/middle/last exact-cell certificates + prefix",
        DERIVED, 6.2, True)
    arrow(ax, (2.4, 4.15), (5.5, 3.50))
    arrow(ax, (7.1, 4.15), (7.0, 3.50))
    arrow(ax, (11.6, 4.15), (8.5, 3.50))
    arrow(ax, (11.75, 7.55), (10.2, 3.50))

    box(ax, 0.4, 0.55, 2.4, 0.55, "DERIVED / PROVED HERE", DERIVED, 5.8, True)
    box(ax, 3.1, 0.55, 2.8, 0.55, "PUBLISHED SOURCE (cited)", AXIOM, 5.8, True)
    box(ax, 6.2, 0.55, 2.6, 0.55, "MACHINE AUDIT LEDGER", AUDIT, 5.8, True)
    ax.text(7, 0.18,
            "No residual un-audited LR assumption beyond transcribed WB cells.  Bullet/plus = SM only.",
            ha="center", va="center", fontsize=6.2, color="#424950")

    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "wb_bridge_dependency.pdf", bbox_inches="tight")
    fig.savefig(OUT / "wb_bridge_dependency.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'wb_bridge_dependency.pdf'}")

if __name__ == "__main__":
    main()
