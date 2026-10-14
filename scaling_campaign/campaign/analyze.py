#!/usr/bin/env python3
"""Tables, fits and figures from results/mc_batches.csv and the other CSVs.

    python campaign/analyze.py            (or --res results_smoke)

Writes results/summary.csv, results/SUMMARY.md and results/fig_*.pdf/png.
Works on partial data: run it at any time, also while run_mc.py is running.

Sizes are labelled by the spacing s; the blocks then have distances
d_X = 2s + 1 (measured pair, Z sector) and d_Z = 2s (reported bit, X logicals).
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
Z95 = 1.959964


def wilson(k, n, z=Z95):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def load():
    agg, seen = {}, set()
    with open(RES / "mc_batches.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                key = (r["task_id"], int(r["batch"]))
                if key in seen:
                    continue
                shots, errors, sec = int(r["shots"]), int(r["errors"]), float(r["seconds"])
                obs = [int(v) for v in r["obs_errors"].split()]
            except (TypeError, ValueError, KeyError, AttributeError):
                continue
            seen.add(key)
            a = agg.get(r["task_id"])
            if a is None:
                names = r["obs_names"].split()
                a = agg[r["task_id"]] = dict(
                    task_id=r["task_id"], k=int(r["k"]), s=int(r["s"]), interface=r["interface"],
                    width=int(r["width"]), mult=int(r["mult"]), basis=r["basis"], p=float(r["p"]),
                    rounds=int(r["rounds"]), decoder=r["decoder"], shots=0, errors=0, seconds=0.0,
                    names=names, obs=[0] * len(names), batches=0)
            a["shots"] += shots
            a["errors"] += errors
            a["seconds"] += sec
            a["batches"] += 1
            for i, v in enumerate(obs):
                a["obs"][i] += v
    return list(agg.values())


def label(a):
    if a["interface"] == "bridge":
        return f"thin bridge w={a['width']}"
    if a["interface"] == "thick":
        return "thick grid" + (f" x{a['mult']}" if a["mult"] > 1 else "")
    return "idle memory"


def obs_count(a, nm):
    if nm == "any":
        return a["errors"], a["shots"]
    if nm not in a["names"]:
        return None
    return a["obs"][a["names"].index(nm)], a["shots"]


def write_summary_csv(rows):
    keys = ["task_id", "k", "s", "d_X", "d_Z", "interface", "width", "mult", "basis", "p", "rounds",
            "decoder", "shots", "errors", "rate_any", "wilson_lo", "wilson_hi", "sec_per_shot"]
    obs_cols = []
    for a in rows:
        for nm in a["names"]:
            if f"rate_{nm}" not in obs_cols:
                obs_cols += [f"nfail_{nm}", f"rate_{nm}", f"lo_{nm}", f"hi_{nm}"]
    with open(RES / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + obs_cols)
        w.writeheader()
        for a in sorted(rows, key=lambda a: (a["decoder"], a["k"], a["interface"], a["width"],
                                             a["mult"], a["basis"], a["s"], a["p"])):
            lo, hi = wilson(a["errors"], a["shots"])
            row = {kk: a[kk] for kk in keys if kk in a}
            row.update(d_X=2 * a["s"] + 1, d_Z=2 * a["s"], rate_any=a["errors"] / a["shots"],
                       wilson_lo=lo, wilson_hi=hi, sec_per_shot=a["seconds"] / a["shots"])
            for nm, v in zip(a["names"], a["obs"]):
                l2, h2 = wilson(v, a["shots"])
                row.update({f"nfail_{nm}": v, f"rate_{nm}": v / a["shots"], f"lo_{nm}": l2, f"hi_{nm}": h2})
            w.writerow(row)


def fit_lambda(points):
    """Weighted fit of ln(rate) = a - b s over points (s, k, n) with k >= 1.

    Weight k (the inverse variance of ln(k/n) for a binomial count).  Returns
    (Lambda, lo, hi, used) with Lambda = exp(b), the factor gained per unit of
    spacing (d_X and d_Z both grow by 2), and a 95% interval, or None.
    """
    pts = [(s, math.log(k / n), k) for s, k, n in points if k >= 1]
    if len(pts) < 2:
        return None
    W = sum(w for _, _, w in pts)
    mx = sum(x * w for x, _, w in pts) / W
    my = sum(y * w for _, y, w in pts) / W
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
    if sxx == 0:
        return None
    b = -sum(w * (x - mx) * (y - my) for x, y, w in pts) / sxx
    se = math.sqrt(1 / sxx)
    return math.exp(b), math.exp(b - Z95 * se), math.exp(b + Z95 * se), len(pts)


def crossing(ca, cb):
    """p where two rate-vs-p curves {p: rate} cross (first sign change, log-log)."""
    ps = sorted(set(ca) & set(cb))
    prev = None
    for p in ps:
        if ca[p] <= 0 or cb[p] <= 0:
            continue
        diff = math.log(cb[p]) - math.log(ca[p])
        if prev is not None and (prev[1] < 0) != (diff < 0):
            p0, d0 = prev
            t = d0 / (d0 - diff)
            return math.exp(math.log(p0) + t * (math.log(p) - math.log(p0)))
        prev = (p, diff)
    return None


def crossing_bootstrap(cnt_a, cnt_b, reps=1000, seed=1):
    """Point estimate and parametric-bootstrap 95% interval of the crossing.

    cnt_*: {p: (k, n)}.  Each replicate redraws every count from a normal
    approximation to Binomial(n, k/n); zero counts are replaced by 0.5.
    """
    rng = random.Random(seed)

    def rate(k, n):
        return max(k, 0.5) / n

    est = crossing({p: rate(*v) for p, v in cnt_a.items()}, {p: rate(*v) for p, v in cnt_b.items()})
    if est is None:
        return None
    vals = []
    for _ in range(reps):
        ra, rb = {}, {}
        for src, dst in ((cnt_a, ra), (cnt_b, rb)):
            for p, (k, n) in src.items():
                q = k / n
                kk = max(0.0, rng.gauss(k, math.sqrt(max(n * q * (1 - q), 0.25))))
                dst[p] = rate(kk, n)
        c = crossing(ra, rb)
        if c is not None:
            vals.append(c)
    if len(vals) < reps * 0.5:
        return est, None, None, len(vals) / reps
    vals.sort()
    return est, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1], len(vals) / reps


def figures(rows, md):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = [a for a in rows if a["decoder"] == "sector" and a["k"] == 2]
    ss = sorted({a["s"] for a in base})
    xt = [f"{s}\n({2*s+1},{2*s})" for s in ss]
    xlabel = "spacing s   (d_X, d_Z)"
    styles = {"thick grid": ("tab:blue", "o"), "thin bridge w=1": ("tab:red", "s"),
              "thin bridge w=2": ("tab:orange", "v"), "thin bridge w=3": ("tab:brown", "^"),
              "idle memory": ("tab:gray", "D"), "thick grid x2": ("tab:green", "P")}

    def errbar(ax, xs, ks, ns, lab, col=None, mk=None):
        c0, m0 = styles.get(lab, ("k", "x"))
        col, mk = col or c0, mk or m0
        xo, ys, lo, hi, xz, ub = [], [], [], [], [], []
        for x, k, n in zip(xs, ks, ns):
            if k > 0:
                r = k / n
                a, b = wilson(k, n)
                xo.append(x); ys.append(r); lo.append(r - a); hi.append(b - r)
            elif n > 0:
                xz.append(x); ub.append(wilson(0, n)[1])
        if xo:
            ax.errorbar(xo, ys, yerr=[lo, hi], color=col, marker=mk, capsize=3, label=lab)
        if xz:
            ax.scatter(xz, ub, color=col, marker="v", facecolors="none",
                       label=(lab + " (no failure: 95% upper bound)") if not xo else None)

    ps = sorted({a["p"] for a in base})
    show = [p for p in (0.001, 0.002, 0.003) if p in ps] or ps[:3]
    if show:
        fig, axs = plt.subplots(1, len(show), figsize=(4.3 * len(show), 3.6), squeeze=False)
        for ax, p in zip(axs[0], show):
            for lab in ("thick grid", "thin bridge w=1", "thin bridge w=2", "thin bridge w=3", "idle memory"):
                pts = sorted((a["s"], *obs_count(a, "Z1aZ1b")) for a in base
                             if a["basis"] == "Z" and a["p"] == p and label(a) == lab)
                if pts:
                    errbar(ax, [s for s, _, _ in pts], [k for _, k, _ in pts], [n for _, _, n in pts], lab)
            ax.set_yscale("log"); ax.set_xticks(ss); ax.set_xticklabels(xt); ax.set_xlabel(xlabel)
            ax.set_title(f"p = {p:g}"); ax.grid(alpha=0.3)
        axs[0][0].set_ylabel(r"failure of $\bar Z_1^{(a)}\bar Z_1^{(b)}$")
        axs[0][-1].legend(fontsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(RES / f"fig_A_thin_vs_thick.{ext}", dpi=160)
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
    cmap = plt.get_cmap("viridis")
    lines = []
    for ax, basis in zip(axs, ("X", "Z")):
        counts = {}
        for i, s in enumerate(ss):
            pts = sorted((a["p"], a["errors"], a["shots"]) for a in base
                         if a["basis"] == basis and a["s"] == s and label(a) == "thick grid")
            if not pts:
                continue
            counts[s] = {p: (k, n) for p, k, n in pts}
            errbar(ax, [p for p, _, _ in pts], [k for _, k, _ in pts], [n for _, _, n in pts],
                   f"s = {s}", col=cmap(i / max(1, len(ss) - 1)), mk="o")
        if len(counts) >= 2:
            s0, s1 = min(counts), max(counts)
            cb = crossing_bootstrap(counts[s0], counts[s1])
            if cb:
                est, lo, hi, frac = cb
                ax.axvline(est, color="k", ls="--", lw=0.8)
                rng = f"95% interval [{lo:.2e}, {hi:.2e}]" if lo else "interval undetermined"
                lines.append(f"{basis} experiment: s = {s0} and s = {s1} cross at p = {est:.2e} "
                             f"({rng}; crossing found in {frac:.0%} of bootstrap replicates)")
            else:
                lines.append(f"{basis} experiment: no crossing of s = {s0} and s = {s1} in the sampled "
                             "range (larger blocks are better at every sampled p, or worse at every p)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_xlabel("physical error rate p")
        ax.set_title(f"thick grid, {basis} experiment (any observable)")
    axs[0].set_ylabel("failure of the joint measurement")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RES / f"fig_B_thick_scaling.{ext}", dpi=160)
    plt.close(fig)
    if lines:
        md.append("\n## Crossings of the thick-grid curves (smallest vs largest s)\n")
        md += [f"- {c}" for c in lines]

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for i, p in enumerate(ps):
        pts = sorted((a["s"], *obs_count(a, "m")) for a in base
                     if a["basis"] == "X" and a["p"] == p and label(a) == "thick grid")
        if pts:
            errbar(ax, [s for s, _, _ in pts], [k for _, k, _ in pts], [n for _, _, n in pts],
                   f"p = {p:g}", col=cmap(i / max(1, len(ps) - 1)), mk="o")
    ax.set_yscale("log"); ax.set_xticks(ss); ax.set_xticklabels(xt); ax.set_xlabel(xlabel)
    ax.set_ylabel("failure of the reported bit m"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RES / f"fig_C_reported_bit.{ext}", dpi=160)
    plt.close(fig)


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(RES), help="results folder (default: results)")
    RES = Path(ap.parse_args().res)
    RES.mkdir(exist_ok=True)
    md = ["# Campaign summary (generated by analyze.py)\n"]
    rows = load() if (RES / "mc_batches.csv").exists() else []
    if rows:
        write_summary_csv(rows)
    md += [f"Tasks with data: {len(rows)}.  Total shots: {sum(a['shots'] for a in rows):,}.  "
           f"Decoder CPU time: {sum(a['seconds'] for a in rows) / 3600:.1f} core-hours.\n",
           "Rates are per complete joint measurement (R = 2s+1 rounds before, during and after "
           "gauging).  Brackets are Wilson 95% intervals.  Spacing s gives block distances "
           "d_X = 2s+1 and d_Z = 2s.\n"]
    base = [a for a in rows if a["decoder"] == "sector"]
    ss = sorted({a["s"] for a in base})

    if base:
        md.append("\n## Measured pair, Z sector (the observable the cut obstruction attacks)\n")
        md.append("| k | p | interface | " + " | ".join(f"s={s} ({2*s+1},{2*s})" for s in ss) + " |")
        md.append("|---|---|---|" + "---|" * len(ss))
        for k, p, lab in sorted({(a["k"], a["p"], label(a)) for a in base if a["basis"] == "Z"}):
            cells = []
            for s in ss:
                m = [a for a in base if a["basis"] == "Z" and a["k"] == k and a["p"] == p
                     and label(a) == lab and a["s"] == s]
                if m:
                    e, n = obs_count(m[0], "Z1aZ1b")
                    lo, hi = wilson(e, n)
                    cells.append(f"{e}/{n} [{lo:.1e}, {hi:.1e}]")
                else:
                    cells.append("")
            md.append(f"| {k} | {p:g} | {lab} | " + " | ".join(cells) + " |")

        md.append("\n## Suppression per unit of spacing, thick grid (weighted fit, 95% interval)\n")
        md.append("Lambda = rate(s) / rate(s+1).  Lambda > 1 means larger blocks fail less.  "
                  "Points with no failure are not in the fit; see the tables.\n")
        for k in sorted({a["k"] for a in base}):
            for basis, nm in (("X", "any"), ("Z", "any"), ("X", "m"), ("Z", "Z1aZ1b")):
                for p in sorted({a["p"] for a in base}):
                    pts = []
                    for a in base:
                        if a["k"] == k and a["basis"] == basis and a["p"] == p and label(a) == "thick grid":
                            r = obs_count(a, nm)
                            if r:
                                pts.append((a["s"], r[0], r[1]))
                    f = fit_lambda(pts)
                    if f:
                        lam, lo, hi, used = f
                        md.append(f"- k={k}, {basis} experiment, {nm}, p={p:g}: Lambda = {lam:.2f} "
                                  f"[{lo:.2f}, {hi:.2f}] from {used} sizes")

    dc = RES / "decoder_check.csv"
    if dc.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("dcheck", HERE / "decoder_check.py")
        dmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dmod)
        agg = {}
        with open(dc, newline="") as fh:
            for r in csv.DictReader(fh):
                a = agg.setdefault(r["point"], [0, 0, 0, 0, 0, 0.0, 0.0])
                for i, kk in enumerate(("shots", "fail_full", "fail_sector", "only_full", "only_sector")):
                    a[i] += int(r[kk])
                a[5] += float(r["sec_full"]); a[6] += float(r["sec_sector"])
        md.append("\n## Decoder check on identical shots (s = 2): paper's full-model vs per-sector BP+LSD\n")
        md.append("| point | shots | full fails | sector fails | only full wrong | only sector wrong "
                  "| McNemar p | time full/sector |")
        md.append("|---|---|---|---|---|---|---|---|")
        for pt, (n, ff, fs, of, os_, tf, ts) in sorted(agg.items()):
            md.append(f"| {pt} | {n} | {ff} | {fs} | {of} | {os_} | {dmod.mcnemar_p(of, os_):.3g} | "
                      f"{tf / max(ts, 1e-9):.0f}x |")

    for name, title in (("fault_distance.csv", "Circuit fault distances (CP-SAT)"),
                        ("code_distance.csv", "Code distances for the Steane input, d_in = 3 (CP-SAT)")):
        p = RES / name
        if p.exists():
            with open(p, newline="") as fh:
                rr = list(csv.DictReader(fh))
            if rr:
                md.append(f"\n## {title}\n")
                md.append("| " + " | ".join(rr[0]) + " |")
                md.append("|" + "---|" * len(rr[0]))
                for r in rr:
                    md.append("| " + " | ".join(str(v) for v in r.values()) + " |")
    if rows:
        try:
            figures(rows, md)
        except Exception as e:           # figures are optional; the tables are the record
            md.append(f"\n(figures skipped: {e})")
    (RES / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
