#!/usr/bin/env python3
"""Figure and table of the two-block scaling campaign (Supplemental Sec. two-block).

Reads numerical/scaling_campaign/mc_batches.csv (one row per batch, with its
seed), decoder_check.csv, and the CP-SAT distance files fault_distance.csv,
code_distance.csv and reported_bit_spacelike.csv, and writes
  figures/two_block_scaling.pdf/.png   the figure
  two_block_scaling_table.tex          the table body, generated from the data
  numerical/scaling_campaign/key_numbers.json   every number quoted in the text
so that no number in the paper is typed by hand.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "numerical" / "scaling_campaign"
sys.path.insert(0, str(ROOT / "scaling_campaign" / "campaign"))  # analyze, decoder_check
import analyze as A  # noqa: E402

PT = 1 / 72.27
TW = 510 * PT
INK, MUTED, BLUE, ORANGE = "#1f2328", "#8b9098", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.0, "lines.markersize": 3.6, "legend.frameon": False,
    "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def decoder_check():
    """Paired check on identical shots (decoder_check.py): failures of each decoder."""
    from decoder_check import mcnemar_p
    path = DATA / "decoder_check.csv"
    if not path.exists():
        return None
    agg, seen = {}, set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["point"], r["chunk"]) in seen:
                continue
            seen.add((r["point"], r["chunk"]))
            a = agg.setdefault(r["point"], dict(shots=0, full=0, sector=0, only_full=0, only_sector=0,
                                                sec_full=0.0, sec_sector=0.0))
            for k_csv, k in (("shots", "shots"), ("fail_full", "full"), ("fail_sector", "sector"),
                             ("only_full", "only_full"), ("only_sector", "only_sector")):
                a[k] += int(r[k_csv])
            a["sec_full"] += float(r["sec_full"]); a["sec_sector"] += float(r["sec_sector"])
    for a in agg.values():
        a["mcnemar_p"] = mcnemar_p(a["only_full"], a["only_sector"])
        a["speedup"] = a["sec_full"] / a["sec_sector"]
    return agg


def read_rows(name):
    """CP-SAT distance files (run_distances.py, reported_bit_spacelike_check.py), verbatim."""
    path = DATA / name
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    A.RES = DATA
    rows = [a for a in A.load() if a["decoder"] == "sector"]
    k2 = [a for a in rows if a["k"] == 2]

    def get(iface, basis, s, p, w=1, m=1, k=2):
        for a in rows:
            if (a["k"] == k and a["interface"] == iface and a["basis"] == basis and a["s"] == s
                    and a["p"] == p and (iface != "bridge" or a["width"] == w) and a["mult"] == m):
                return a
        return None

    styles = [("bridge", "thin bridge, $w=1$", ORANGE, "s"), ("thick", "spanning grid", BLUE, "o"),
              ("idle", "idle (no gauging)", MUTED, "D")]
    fig, axs = plt.subplots(1, 3, figsize=(TW, 0.36 * TW))
    for ax, p, tag in ((axs[0], 0.001, "(a)"), (axs[1], 0.002, "(b)")):
        for iface, lab, col, mk in styles:
            xs, ys, lo, hi, xz, uz = [], [], [], [], [], []
            for s in (2, 3, 4, 5):
                a = get(iface, "Z", s, p)
                if a is None:
                    continue
                kf, n = A.obs_count(a, "Z1aZ1b")
                if kf > 0:
                    r = kf / n
                    l, h = A.wilson(kf, n)
                    xs.append(s); ys.append(r); lo.append(r - l); hi.append(h - r)
                else:
                    xz.append(s); uz.append(A.wilson(0, n)[1])
            ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, capsize=2, label=lab, lw=0.9)
            if xz:
                ax.scatter(xz, uz, color=col, marker="v", facecolors="none", s=14, lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks([2, 3, 4, 5])
        ax.set_xticklabels([f"{s}\n({2*s+1},{2*s})" for s in (2, 3, 4, 5)])
        ax.set_xlabel(r"spacing $s$ \;($d_X,d_Z$)")
        ax.set_ylim(3e-6, 0.5)
        ax.set_title(f"{tag} $p={p:g}$", loc="left")
    axs[0].set_ylabel(r"failure of $\overline Z{}_1^{(a)}\overline Z{}_1^{(b)}$")
    axs[0].legend(loc="center right", fontsize=6.5)
    ax = axs[2]
    lam = {}
    for basis, col, mk, lab in (("X", BLUE, "o", "$X$ experiment"), ("Z", ORANGE, "s", "$Z$ experiment")):
        xs, ys, lo, hi = [], [], [], []
        for p in sorted({a["p"] for a in k2}):
            pts = []
            for s in (2, 3, 4, 5):
                a = get("thick", basis, s, p)
                if a is not None:
                    pts.append((s, a["errors"], a["shots"]))
            f = A.fit_lambda(pts)
            if f:
                xs.append(p); ys.append(f[0]); lo.append(f[0] - f[1]); hi.append(f[2] - f[0])
                lam[f"{basis}_{p:g}"] = dict(value=f[0], lo=f[1], hi=f[2], sizes=f[3])
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, capsize=2, label=lab, lw=0.9)
    ax.axhline(1, color=MUTED, lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter, NullLocator
    ticks = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.006]
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t * 1e3:g}" for t in ticks])
    ax.set_xlabel(r"physical error rate $p$ \;($10^{-3}$)")
    ax.set_ylabel(r"$\Lambda=$ failure$(s)/$failure$(s+1)$")
    ax.set_title("(c) spanning grid, all observables", loc="left")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(ROOT / "figures" / f"two_block_scaling.{ext}", **kw)
    plt.close(fig)

    # ---- table body (measured pair, Z sector) ------------------------------
    def cell(a):
        if a is None:
            return "--"
        kf, n = A.obs_count(a, "Z1aZ1b")
        return f"{kf}/{n}"
    lines = []
    for p in (0.0005, 0.001, 0.002, 0.003):
        first = True
        for iface, lab in (("bridge", "thin bridge, $w=1$"), ("thick", "spanning grid"),
                           ("idle", "idle")):
            ptxt = f"${p * 1e3:g}\\times10^{{-3}}$" if first else ""
            first = False
            lines.append(f"{ptxt} & {lab} & " + " & ".join(cell(get(iface, 'Z', s, p)) for s in (2, 3, 4, 5))
                         + r" \\")
        lines.append(r"\addlinespace[2pt]")
    for w in (2, 3):
        lines.append(f"$2\\times10^{{-3}}$ & thin bridge, $w={w}$ & -- & "
                     + " & ".join(cell(get('bridge', 'Z', s, 0.002, w=w)) for s in (3, 4)) + r" & -- \\")
    (ROOT / "two_block_scaling_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- every number quoted in the text -------------------------------------
    def rate(a, nm="Z1aZ1b"):
        kf, n = A.obs_count(a, nm)
        l, h = A.wilson(kf, n)
        return dict(k=kf, n=n, rate=kf / n, lo=l, hi=h)
    key = dict(
        total_shots=sum(a["shots"] for a in rows),
        core_hours=sum(a["seconds"] for a in rows) / 3600,
        tasks_with_data=len(rows),
        thin_p1e3={s: rate(get("bridge", "Z", s, 0.001)) for s in (2, 3, 4, 5)},
        thick_p1e3={s: rate(get("thick", "Z", s, 0.001)) for s in (2, 3, 4, 5)},
        idle_p1e3={s: rate(get("idle", "Z", s, 0.001)) for s in (2, 3, 4, 5)},
        thin_p5e4={s: rate(get("bridge", "Z", s, 0.0005)) for s in (2, 3, 4, 5)},
        thick_p5e4={s: rate(get("thick", "Z", s, 0.0005)) for s in (2, 3, 4, 5)},
        m_thick_p1e3={s: rate(get("thick", "X", s, 0.001), "m") for s in (2, 3, 4, 5)},
        widths_p2e3={f"w{w}_s{s}": rate(get("bridge", "Z", s, 0.002, w=w)) for w in (1, 2, 3) for s in (3, 4)},
        k4_thin_p1e3={s: rate(get("bridge", "Z", s, 0.001, k=4)) for s in (2, 3, 4)},
        k4_thick_p1e3={s: rate(get("thick", "Z", s, 0.001, k=4)) for s in (2, 3, 4)},
        k4_spect_X_thick_p1e3={s: (lambda a: dict(n=a["shots"], spect=sum(v for nm, v in zip(a["names"], a["obs"]) if len(nm) > 1 and nm[1] in "234")))(get("thick", "X", s, 0.001, k=4)) for s in (2, 3, 4)},
        mult2_m_p1e3={s: rate(get("thick", "X", s, 0.001, m=2), "m") for s in (2, 3, 4)},
        mult2_Z_p1e3={s: rate(get("thick", "Z", s, 0.001, m=2)) for s in (2, 3, 4)},
        lam=lam,
        decoder_check=decoder_check(),
        fault_distance=read_rows("fault_distance.csv"),
        steane_code_distance=read_rows("code_distance.csv"),
        reported_bit_spacelike=read_rows("reported_bit_spacelike.csv"),
    )
    (DATA / "key_numbers.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(json.dumps(key, indent=1))


if __name__ == "__main__":
    main()
