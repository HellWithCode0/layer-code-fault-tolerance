#!/usr/bin/env python3
"""Regenerate every figure included in manuscript.pdf and supplement.pdf.

Each figure is drawn at the exact width at which it is printed (REVTeX
reprint: \\textwidth = 510 pt, \\columnwidth = 246 pt, times the factor in
the corresponding \\includegraphics), so the text needs no rescaling and sits
at 7-8.5 pt next to the 10 pt body text.  LaTeX typesets all figure text in
Latin Modern, the font of the documents, so the script needs a TeX
installation with the lmodern package (the same one that builds the PDFs).

Numerical panels read the archived CSV files and redraw them without any new
simulation.  Schematics illustrate the constructions stated in the text and
carry no data.  Colours: code sizes are ordered, so they use one blue ramp
(light = small, dark = large) with a distinct marker per size; unordered
series use a fixed categorical order.  Both palettes were checked for
colour-vision separation.

Usage:  python make_publication_figures.py
Writes: figures/<name>.pdf (vector, used by LaTeX), figures/<name>.png
        (600 dpi preview), and revision_data_checks.json.
"""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)
from matplotlib.ticker import MultipleLocator, NullFormatter, LogLocator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
PT = 1 / 72.27                 # TeX point in inches
TW = 510 * PT                  # \textwidth
CW = 246 * PT                  # \columnwidth

# ---------------------------------------------------------------- style ----
INK = '#1f2328'        # primary text and outlines
INK2 = '#52514e'       # secondary text
MUTED = '#8b9098'      # tertiary marks
FRAME = '#7d858e'      # wireframe edges
GRID = '#e6e8eb'
BLUE = '#2a78d6'       # pure-X objects
ORANGE = '#eb6834'     # pure-Z objects
AQUA = '#1baf7a'       # ancillas and encoded resources
YELLOW = '#eda100'
MAGENTA = '#e87ba4'
GREEN = '#008300'
VIOLET = '#4a3aa7'     # X/Z intersections
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET]
RAMP = {2: ['#5598e7', '#104281'],
        3: ['#6da7ec', '#2a78d6', '#104281'],
        4: ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'],
        5: ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']}
MARKERS = ['o', 's', '^', 'D', 'p']     # open 'v' is reserved for zero counts
BAND = '#a3a9b0'

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{lmodern}'
                           r'\usepackage{amsmath,amssymb}',
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'text.color': INK, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'axes.linewidth': 0.6, 'axes.labelpad': 2.5, 'axes.titlepad': 4,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
    'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.45, 'ytick.minor.width': 0.45,
    'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
    'lines.linewidth': 1.0, 'lines.markersize': 3.6,
    'legend.frameon': False, 'legend.handlelength': 1.7,
    'legend.handletextpad': 0.5, 'legend.labelspacing': 0.3,
    'legend.columnspacing': 1.1, 'legend.borderaxespad': 0.35,
    'legend.borderpad': 0.2,
    'pdf.fonttype': 42, 'pdf.compression': 9,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})


def tint(color, white):
    """Mix a colour with white; white=0 keeps it, white=1 gives white."""
    return tuple(v + (1 - v) * white for v in to_rgb(color))


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf',
                metadata={'Creator': 'make_publication_figures.py',
                          'CreationDate': None})
    fig.savefig(OUT / f'{name}.png', dpi=600, metadata={'Software': None})
    plt.close(fig)


def canvas(fig, x, y, w, h):
    """Schematic axes whose data units are inches, placed at (x, y) in."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return ax


def plot_axes(fig, x, y, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def title(ax, letter, text, x=0.0, y=None, size=8.5, baseline=None):
    """Panel heading: bold letter, regular title, top-left aligned.  With
    baseline=y the text sits on y, to line up with a plot title."""
    head = (r'\textbf{(%s)}\enspace %s' % (letter, text)) if letter else text
    if baseline is not None:
        ax.text(x, baseline, head, ha='left', va='baseline', fontsize=size,
                color=INK)
        return
    if y is None:
        y = ax.get_ylim()[1]
    ax.text(x, y, head, ha='left', va='top', fontsize=size, color=INK)


TITLE_PAD = 5 / 72     # plot_title pad in inches


def plot_title(ax, letter, text):
    head = (r'\textbf{(%s)}\enspace %s' % (letter, text)) if letter else text
    ax.set_title(head, loc='left', fontsize=8.5, pad=5)


def txt(ax, x, y, s, size=7.5, color=INK, ha='center', va='center', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def rbox(ax, x, y, w, h, ec, fc=None, lw=0.8, r=0.045, ls='-', z=2):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f'round,pad=0,rounding_size={r}',
                           fc=tint(ec, 0.9) if fc is None else fc, ec=ec,
                           lw=lw, ls=ls, zorder=z)
    ax.add_patch(patch)
    return patch


def arrow(ax, p, q, color=INK2, lw=0.8, head=6.5, style='-|>', z=3, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=head,
                                 lw=lw, color=color, zorder=z, shrinkA=0,
                                 shrinkB=0, linestyle=ls))


def leader(ax, p, q, color=MUTED, lw=0.5):
    ax.plot(*zip(p, q), color=color, lw=lw, zorder=6, solid_capstyle='round')


class Oblique:
    """Cabinet-style oblique projection: depth y goes up and to the right."""

    def __init__(self, ox, oy, kx=0.42, ky=0.30):
        self.ox, self.oy, self.kx, self.ky = ox, oy, kx, ky

    def __call__(self, x, y, z):
        return (self.ox + x + self.kx * y, self.oy + z + self.ky * y)

    def poly(self, *pts):
        return [self(*p) for p in pts]


def wire_box(ax, P, x0, x1, y0, y1, z0, z1, color=FRAME, lw=0.7, z=5):
    """Wireframe box; the three edges at the hidden corner are dashed."""
    V = {(i, j, k): P((x0, x1)[i], (y0, y1)[j], (z0, z1)[k])
         for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    hidden = (0, 1, 0)
    for a in V:
        for b in V:
            if a < b and sum(abs(a[t] - b[t]) for t in range(3)) == 1:
                hid = hidden in (a, b)
                ax.plot(*zip(V[a], V[b]), color=color,
                        lw=0.55 * lw / 0.7 if hid else lw,
                        ls=(0, (2.2, 1.8)) if hid else '-',
                        zorder=z - 3 if hid else z, solid_capstyle='round',
                        dash_capstyle='butt')


def face(ax, P, pts, color, alpha=0.20, lw=0.9, z=3, ls='-', ec=None):
    ax.add_patch(Polygon(P.poly(*pts), closed=True,
                         fc=(*to_rgb(color), alpha),
                         ec=color if ec is None else ec, lw=lw, ls=ls,
                         zorder=z, joinstyle='round'))


def slab_block(ax, P, s, zx=0.5, xz=0.55, frame_lw=0.7, extra_x=(),
               tube=None):
    """Block with a horizontal X slab at height zx*s and a vertical Z slab
    at x = xz*s.  extra_x lists heights of dotted alternative X slabs."""
    wire_box(ax, P, 0, s, 0, s, 0, s, lw=frame_lw)
    for zf in extra_x:
        face(ax, P, [(0, 0, zf * s), (s, 0, zf * s), (s, s, zf * s),
                     (0, s, zf * s)], BLUE, alpha=0.0, lw=0.6, z=2,
             ls=(0, (1.2, 1.4)))
    zs, xs = zx * s, xz * s
    face(ax, P, [(0, 0, zs), (s, 0, zs), (s, s, zs), (0, s, zs)], BLUE,
         alpha=0.22, z=3)
    face(ax, P, [(xs, 0, 0), (xs, s, 0), (xs, s, s), (xs, 0, s)], ORANGE,
         alpha=0.20, z=4)
    if tube:
        t = tube
        face(ax, P, [(xs - t, 0, zs + t), (xs + t, 0, zs + t),
                     (xs + t, s, zs + t), (xs - t, s, zs + t)], VIOLET,
             alpha=0.55, lw=0.6, z=6)
        face(ax, P, [(xs - t, 0, zs - t), (xs + t, 0, zs - t),
                     (xs + t, 0, zs + t), (xs - t, 0, zs + t)], VIOLET,
             alpha=0.75, lw=0.6, z=6)
    else:
        ax.plot(*zip(P(xs, 0, zs), P(xs, s, zs)), color=VIOLET, lw=2.2,
                zorder=7, solid_capstyle='round')
    return zs, xs


def swatch_line(ax, x, y, color, lw=1.6, length=0.16, ls='-'):
    ax.plot([x, x + length], [y, y], color=color, lw=lw, ls=ls,
            solid_capstyle='butt', zorder=6)


# ------------------------------------------------------------ data I/O ----
def readcsv(path):
    with (ROOT / path).open(newline='') as fh:
        return list(csv.DictReader(fh))


def series(ax, x, y, lo, hi, color, marker, zorder=3, ms=3.6, lw=1.0):
    """Points with pointwise Wilson bars.  A zero-failure point is not drawn
    at a pseudocount; an open downward triangle marks its Wilson upper end."""
    x, y, lo, hi = map(np.asarray, (x, y, lo, hi))
    pos = y > 0
    ax.errorbar(x, np.where(pos, y, np.nan),
                yerr=[np.maximum(0, y - lo), np.maximum(0, hi - y)],
                color=color, marker=marker, ms=ms, mec='white', mew=0.45,
                lw=lw, elinewidth=0.6, capsize=1.5, capthick=0.6,
                zorder=zorder)
    if (~pos).any():
        ax.scatter(x[~pos], hi[~pos], marker='v', s=15, facecolors='white',
                   edgecolors=color, linewidths=0.8, zorder=zorder + 1)


def handle(color, marker, ls='-', lw=1.0):
    return Line2D([], [], color=color, marker=marker, ms=3.6, mec='white',
                  mew=0.45, lw=lw, ls=ls)


def zero_handle(color=INK2):
    return Line2D([], [], color=color, marker='v', ms=3.6, mfc='white',
                  mec=color, mew=0.8, lw=0)


def rows_curve(rr, xkey='p', scale=1.0):
    rr = sorted([r for r in rr if float(r[xkey]) > 0], key=lambda r: float(r[xkey]))
    x = np.array([float(r[xkey]) * scale for r in rr])
    y = np.array([float(r['logical_error_rate']) for r in rr])
    lo = np.array([float(r['wilson_lo']) for r in rr])
    hi = np.array([float(r['wilson_hi']) for r in rr])
    return x, y, lo, hi


def log_y(ax, lo, hi):
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10),
                                          numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', which='major', color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)


# ======================================================== main figures ====
def fig_architecture():
    W, H = TW, 2.62
    fig = plt.figure(figsize=(W, H))

    # (a) two blocks joined through a grid ancilla at the X-slab height
    ax = canvas(fig, 0.0, 0.0, 2.78, H)
    title(ax, 'a', 'Local logical access')
    s, g = 0.74, 0.46
    P = Oblique(0.42, 1.10)
    zs = 0.5 * s
    ha = 0.26
    # grid ancilla between the blocks
    x0, x1 = s, s + g
    face(ax, P, [(x0, 0, zs - ha / 2), (x1, 0, zs - ha / 2),
                 (x1, 0, zs + ha / 2), (x0, 0, zs + ha / 2)], AQUA,
         alpha=0.16, lw=0.0, z=2)
    face(ax, P, [(x0, 0, zs + ha / 2), (x1, 0, zs + ha / 2),
                 (x1, s, zs + ha / 2), (x0, s, zs + ha / 2)], AQUA,
         alpha=0.16, lw=0.0, z=2)
    xs_g = np.linspace(x0, x1, 5)
    ys_g = np.linspace(0, s, 5)
    zs_g = np.linspace(zs - ha / 2, zs + ha / 2, 3)
    for xv in xs_g:
        ax.plot(*zip(P(xv, 0, zs_g[0]), P(xv, 0, zs_g[-1])), color=AQUA,
                lw=0.5, zorder=3)
        ax.plot(*zip(P(xv, 0, zs_g[-1]), P(xv, s, zs_g[-1])), color=AQUA,
                lw=0.5, zorder=3)
    for zv in zs_g:
        ax.plot(*zip(P(x0, 0, zv), P(x1, 0, zv)), color=AQUA, lw=0.5,
                zorder=3)
    for yv in ys_g:
        ax.plot(*zip(P(x0, yv, zs_g[-1]), P(x1, yv, zs_g[-1])), color=AQUA,
                lw=0.5, zorder=3)
    for side in (0.0, 1.0):
        blk = Oblique(*P(side * (s + g), 0, 0), P.kx, P.ky)
        slab_block(ax, blk, s)
    # callouts
    pX = P(0, 0, zs)
    txt(ax, pX[0] - 0.06, pX[1], r'$X$ slab', ha='right')
    pZ = P(0.55 * s, s, s)
    txt(ax, pZ[0], pZ[1] + 0.10, r'$Z$ slab', va='bottom')
    for side, name in ((0, 'block'), (1, 'block')):
        c = P(side * (s + g) + s / 2, 0, 0)
        txt(ax, c[0], c[1] - 0.09, name, size=7, color=INK2, va='top')
    bottom = P(s + g / 2, 0, zs - ha / 2)
    rbox(ax, 0.40, 0.10, 2.05, 0.46, AQUA)
    txt(ax, 1.425, 0.43, 'Grid ancilla', size=8)
    txt(ax, 1.425, 0.22, r'Pauli parity $\longrightarrow$ frame update',
        size=7, color=INK2)
    arrow(ax, (bottom[0], bottom[1] - 0.04), (bottom[0], 0.58), color=AQUA)

    # (b) resource supply pipeline
    ax = canvas(fig, 2.96, 1.28, 4.09, 1.34)
    title(ax, 'b', 'Encoded resource supply')
    cw, gap, y0, ch = 0.85, 0.23, 0.33, 0.70
    xs = [i * (cw + gap) for i in range(4)]
    names = ['Inject', 'Transfer', 'Distill', 'Consume']
    subs = ['bare port', r'$d_0<d_1<\cdots$', 'fixed codes', 'data port']
    edges = [ORANGE, AQUA, AQUA, BLUE]
    for x, n, sub, ec in zip(xs, names, subs, edges):
        rbox(ax, x, y0, cw, ch, ec)
        txt(ax, x + cw / 2, y0 + ch - 0.13, n, size=8)
        txt(ax, x + cw / 2, y0 + 0.12, sub, size=7, color=INK2)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + cw + 0.03, y0 + ch / 2), (b - 0.03, y0 + ch / 2),
              color=INK, lw=0.8)
    cy = y0 + ch / 2 - 0.005

    def sq(x, y, side, color, a=0.35):
        ax.add_patch(Rectangle((x - side / 2, y - side / 2), side, side,
                               fc=(*to_rgb(color), a), ec=color, lw=0.6,
                               zorder=4))
    c0 = xs[0] + cw / 2
    ax.add_patch(Circle((c0 - 0.17, cy), 0.035, fc=ORANGE, ec=ORANGE,
                        zorder=4))
    arrow(ax, (c0 - 0.12, cy), (c0 + 0.03, cy), lw=0.6, head=5)
    sq(c0 + 0.12, cy, 0.11, AQUA)
    c1 = xs[1] + cw / 2
    for dx, side in ((-0.24, 0.07), (0.0, 0.11), (0.25, 0.155)):
        sq(c1 + dx, cy, side, AQUA)
    arrow(ax, (c1 - 0.195, cy), (c1 - 0.07, cy), lw=0.5, head=4.5)
    arrow(ax, (c1 + 0.065, cy), (c1 + 0.16, cy), lw=0.5, head=4.5)
    c2 = xs[2] + cw / 2
    for dy in (-0.075, 0.0, 0.075):
        sq(c2 - 0.2, cy + dy, 0.055, AQUA)
        ax.plot([c2 - 0.165, c2 + 0.07], [cy + dy, cy], color=INK2, lw=0.45,
                zorder=3)
    sq(c2 + 0.14, cy, 0.12, AQUA, a=0.55)
    c3 = xs[3] + cw / 2
    sq(c3 - 0.22, cy, 0.09, AQUA)
    arrow(ax, (c3 - 0.16, cy), (c3 - 0.02, cy), lw=0.6, head=5)
    sq(c3 + 0.13, cy, 0.19, BLUE, a=0.22)
    # bracket over the encoded stages
    yb = 0.21
    ax.plot([xs[1], xs[1], xs[3] + cw, xs[3] + cw], [yb + 0.05, yb, yb,
            yb + 0.05], color=MUTED, lw=0.6)
    txt(ax, (xs[1] + xs[3] + cw) / 2, yb - 0.02,
        'encoded blocks after raw injection', size=7, color=INK2, va='top')

    # (c) timeline of protected measurements
    ax = canvas(fig, 2.96, 0.0, 4.09, 1.10)
    title(ax, 'c', 'Protected measurement schedule')
    yb, hb = 0.40, 0.34
    segs = [(0.00, 1.12, 'gauging', BLUE), (1.12, 1.98, 'idle padding', MUTED),
            (1.98, 3.10, 'gauging', BLUE), (3.10, 3.56, '', MUTED)]
    for a, b, lab, col in segs:
        fade = 0.55 if lab == '' else 1.0
        ax.add_patch(Rectangle((a, yb), b - a, hb,
                               fc=tint(col, 0.86 if col == BLUE else 0.80),
                               ec='none', zorder=2, alpha=fade))
        for xr in np.arange(a + 0.07, b - 0.02, 0.10):
            ax.plot([xr, xr], [yb + 0.04, yb + hb - 0.04],
                    color=tint(col, 0.45), lw=0.5, zorder=3, alpha=fade)
        ax.add_patch(Rectangle((a, yb), b - a, hb, fc='none',
                               ec=col, lw=0.8, zorder=4, alpha=fade))
        if lab:
            txt(ax, (a + b) / 2, yb + hb / 2, lab, size=7.5,
                bbox=dict(boxstyle='round,pad=0.18,rounding_size=0.08',
                          fc='white', ec='none', alpha=0.92), zorder=5)
    txt(ax, 3.78, yb + hb / 2, r'$\cdots$', size=10)
    arrow(ax, (0.0, 0.20), (4.02, 0.20), color=INK, lw=0.8)
    txt(ax, 4.02, 0.12, 'time', size=7, color=INK2, ha='right', va='top')
    save(fig, 'revision_architecture')


def fig_slab_prefix():
    W, H = TW, 2.30
    fig = plt.figure(figsize=(W, H))

    # (a) transverse slab ports
    ax = canvas(fig, 0.0, 0.0, 2.55, H)
    title(ax, 'a', 'Transverse slab ports')
    s = 1.02
    P = Oblique(0.52, 0.62)
    zs, xs = slab_block(ax, P, s, extra_x=(0.22, 0.78))
    p = P(s, 0.55 * s, zs)
    q = (P(s, s, s)[0] + 0.10, p[1] - 0.10)
    leader(ax, p, (q[0] - 0.03, q[1]))
    txt(ax, q[0], q[1], r'$X$ slab', ha='left')
    p = P(xs, s, s)
    txt(ax, p[0] + 0.05, p[1] + 0.07, r'$Z$ slab', ha='left', va='bottom')
    pa, pb = P(0, 0, 0.22 * s), P(0, 0, 0.78 * s)
    arrow(ax, (pa[0] - 0.12, pa[1]), (pb[0] - 0.12, pb[1]), color=BLUE,
          lw=0.6, head=5, style='<|-|>')
    txt(ax, pa[0] - 0.22, (pa[1] + pb[1]) / 2, 'other gaps', size=6.5,
        color=INK2, rotation=90)
    txt(ax, 0.12, 0.36, r'$|X\ \mathrm{slab}|,\ |Z\ \mathrm{slab}|=O(N^2)$',
        ha='left')
    swatch_line(ax, 0.12, 0.15, VIOLET, lw=2.2, length=0.14)
    txt(ax, 0.32, 0.15, r'$|X\cap Z|=O(N)$', ha='left')

    # (b) prefix-parity repair of one Z check
    ax = canvas(fig, 2.72, 0.0, 4.33, H)
    title(ax, 'b', 'Prefix parity repairs one check')
    yb, ytop = 1.12, 1.72
    X = [0.62, 1.60, 2.58, 3.56]
    xv = [1, 0, 1, 0]
    pv = [0, 1, 1, 0, 0]
    words = ['11', '110', '101', '00']
    ax.plot([0.10, 4.08], [yb, yb], color=MUTED, lw=0.8, ls=(0, (1.2, 1.5)),
            zorder=2)
    ax.plot([X[0], X[2]], [yb, yb], color=GREEN, lw=3.0, zorder=3,
            solid_capstyle='butt')
    for i, (x, v) in enumerate(zip(X, xv)):
        if v:
            ax.plot([x, x], [yb, ytop], color=BLUE, lw=1.8, zorder=3,
                    solid_capstyle='butt')
        else:
            ax.plot([x, x], [yb, ytop], color=MUTED, lw=0.8,
                    ls=(0, (1.2, 1.5)), zorder=3)
        ax.add_patch(Circle((x, yb), 0.035, fc=INK, ec='white', lw=0.4,
                            zorder=5))
        txt(ax, x, ytop + 0.12, rf'$x_{i + 1}={v}$', size=8)
        txt(ax, x, yb - 0.20, words[i], size=8,
            color=GREEN if i == 1 else INK)
    mids = [(0.10 + X[0]) / 2 - 0.02] + [(a + b) / 2 for a, b in zip(X, X[1:])] \
        + [(X[-1] + 4.08) / 2 + 0.02]
    for i, (m, v) in enumerate(zip(mids, pv)):
        txt(ax, m, yb + 0.12, rf'$p_{i}={v}$', size=7)
    rbox(ax, X[1] - 0.24, yb - 0.34, 0.48, 0.64, GREEN, fc='none', lw=0.8,
         ls=(0, (3, 2)), r=0.03, z=4)
    txt(ax, X[1], yb - 0.46, 'pass-through', size=7, color=INK2)
    txt(ax, 1.72, 0.26, r'$p_{i-1}+p_i+x_i=0\pmod 2$', size=8.5)
    swatch_line(ax, 3.20, 0.36, GREEN, lw=3.0, length=0.18)
    txt(ax, 3.44, 0.36, r'backbone, $p_i=1$', size=7, color=INK2, ha='left')
    swatch_line(ax, 3.20, 0.16, BLUE, lw=1.8, length=0.18)
    txt(ax, 3.44, 0.16, r'data string, $x_i=1$', size=7, color=INK2,
        ha='left')
    save(fig, 'revision_slab_prefix')


def fig_resources():
    W, H = TW, 2.28
    fig = plt.figure(figsize=(W, H))
    ay, ah = 0.42, 1.62
    ax = canvas(fig, 0.0, 0.0, 3.30, H)
    title(ax, 'a', 'From fault distance to space',
          baseline=ay + ah + TITLE_PAD)
    bx, bw = 0.06, 2.05
    boxes = [(1.63, 0.30, r'$d=\Theta(f),\qquad N=\Theta(\sqrt f)$', BLUE, 8),
             (0.93, 0.30, r'$K=\Theta(\sqrt f),\qquad n=\Theta(f^{3/2})$',
              AQUA, 8),
             (0.14, 0.36, r'$Q=O\bigl(Wf+(B_f+1)f^{3/2}\bigr)$', VIOLET, 8.5)]
    for y, h, s, ec, size in boxes:
        rbox(ax, bx, y, bw, h, ec)
        txt(ax, bx + bw / 2, y + h / 2, s, size=size)
    xa = bx + bw / 2
    arrow(ax, (xa, 1.61), (xa, 1.25), color=INK, lw=0.8)
    arrow(ax, (xa, 0.91), (xa, 0.52), color=INK, lw=0.8)
    txt(ax, xa + 0.12, 1.50, r'layer code: $d=\Theta(N^2)$,', size=7,
        color=INK2, ha='left')
    txt(ax, xa + 0.12, 1.36, r'$K=\Theta(N)$, $n=\Theta(N^3)$', size=7,
        color=INK2, ha='left')
    txt(ax, xa + 0.12, 0.785, r'pack $K$ sites per cell:', size=7,
        color=INK2, ha='left')
    txt(ax, xa + 0.12, 0.645, r'$(W/K)\,n+(B_f+1)\,n$', size=7, color=INK2,
        ha='left')

    ax = plot_axes(fig, 3.98, ay, 2.98, ah)
    plot_title(ax, 'b', 'Bounded-aspect logical box')
    x = np.logspace(-2, 2, 400)
    ax.axvspan(1e-2, 1, color=tint(BLUE, 0.93), lw=0, zorder=0)
    ax.axvline(1, color=MUTED, lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)
    ax.loglog(x, x, color=AQUA, ls=(0, (4, 2)), lw=1.1, zorder=3)
    ax.axhline(1, color=BLUE, ls=(0, (1.2, 1.4)), lw=1.2, zorder=3)
    ax.loglog(x, 1 + x, color=INK, lw=1.4, zorder=4)
    ax.set(xlim=(1e-2, 1e2), ylim=(8e-3, 1.8e2),
           xlabel=r'Logical occupancy $x=W/\sqrt f$',
           ylabel=r'Space envelope$\,/\,f^{3/2}$')
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    txt(ax, 12, 30, r'envelope $1+x$', size=7.5, ha='right')
    txt(ax, 0.20, 0.07, r'extensive term $x$', size=7, color=INK2, ha='left')
    txt(ax, 0.013, 0.72, r'one-cell term $1$', size=7, color=INK2, ha='left',
        va='top')
    txt(ax, 0.1, 40, 'one-cell term\ndominates', size=6.5, color=INK2,
        linespacing=1.1)
    txt(ax, 10, 0.03, 'extensive term\ndominates', size=6.5, color=INK2,
        linespacing=1.1)
    save(fig, 'revision_resources')


def continuous_rows():
    data = ROOT / 'numerical/continuous_transition/threshold_estimate/all_points.csv'
    rows = list(csv.DictReader(data.open()))
    for r in rows:
        for k in ['p', 'logical_error_rate', 'wilson_lo', 'wilson_hi']:
            r[k] = float(r[k])
        for k in ['size', 'gauge_scale', 'shots', 'failures']:
            r[k] = int(float(r[k]))
        assert abs(r['logical_error_rate'] - r['failures'] / r['shots']) < 1e-12
    assert len(rows) == 255 and sum(r['shots'] for r in rows) == 4534000
    assert sum(r['failures'] for r in rows if r['p'] == 0) == 0
    return data, rows


def crossings():
    """Archived Jeffreys crossing medians and bands, in units of 1e-3."""
    fit = json.loads((ROOT / 'numerical/continuous_transition/threshold_estimate'
                      '/estimate/threshold_fit.json').read_text())
    out = {}
    for g in fit['by_G']:
        out[g['gauge_scale']] = tuple(1e3 * g[k] for k in
                                      ('jeffreys_median', 'jeffreys_95_lo',
                                       'jeffreys_95_hi'))
    # the values printed in Table S15 and the text
    table = {1: (4.471, 4.426, 4.522), 2: (4.431, 4.392, 4.466),
             3: (4.328, 4.265, 4.393)}
    for G, vals in table.items():
        assert all(abs(a - b) < 6e-4 for a, b in zip(out[G], vals)), G
    return out


def continuous_panel(ax, rows, G, cross):
    med, lo, hi = cross[G]
    ax.axvspan(lo, hi, color=tint(BAND, 0.55), lw=0, zorder=1)
    ax.axvline(med, color=INK2, ls=(0, (3, 2)), lw=0.7, zorder=2)
    for i, L in enumerate([3, 5, 7, 9, 11]):
        rr = [r for r in rows if r['gauge_scale'] == G and r['size'] == L
              and r['p'] > 0]
        rr.sort(key=lambda r: r['p'])
        series(ax, [1e3 * r['p'] for r in rr],
               [r['logical_error_rate'] for r in rr],
               [r['wilson_lo'] for r in rr], [r['wilson_hi'] for r in rr],
               RAMP[5][i], MARKERS[i], zorder=3 + i)
    log_y(ax, 1e-5, 1)
    ax.set_xlim(1.2, 6.2)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))


def size_handles(n, prefix='L'):
    sizes = {5: [3, 5, 7, 9, 11], 4: [3, 4, 5, 6], 3: [3, 5, 7], 2: [3, 5]}[n]
    return ([handle(RAMP[n][i], MARKERS[i]) for i in range(n)],
            [rf'${prefix}={L}$' for L in sizes])


def glabel(G):
    return '$G=' + ('d' if G == 1 else f'{G}d') + '$'


def fig_numerics(rows, cross):
    W, H = TW, 2.55
    fig = plt.figure(figsize=(W, H))
    left, right, gap, bottom, top = 0.52, 0.06, 0.12, 0.40, 0.50
    pw = (W - left - right - 2 * gap) / 3
    ph = H - bottom - top
    axs = []
    for k, G in enumerate([1, 2, 3]):
        ax = plot_axes(fig, left + k * (pw + gap), bottom, pw, ph)
        continuous_panel(ax, rows, G, cross)
        plot_title(ax, chr(97 + k), glabel(G))
        ax.set_xlabel(r'Noise strength $p\ (10^{-3})$')
        if k:
            ax.tick_params(labelleft=False)
        axs.append(ax)
    axs[0].set_ylabel('Logical failure probability')
    h, lab = size_handles(5)
    band = Rectangle((0, 0), 1, 1, fc=tint(BAND, 0.55), ec='none')
    med = Line2D([], [], color=INK2, ls=(0, (3, 2)), lw=0.7)
    fig.legend(h + [(band, med)], lab + [r'$L=3$ vs $11$ crossing, 95\% band'],
               loc='upper center', ncol=6, bbox_to_anchor=(0.53, 1.0),
               handlelength=1.6, columnspacing=1.3)
    save(fig, 'revision_numerics')


# ===================================================== supplement ========
def fig_symplectic_overlap():
    W, H = 0.95 * TW, 2.20
    fig = plt.figure(figsize=(W, H))
    pw = W / 3

    ax = canvas(fig, 0.0, 0.0, pw, H)
    title(ax, 'a', 'Overlap-corrected cleaning')
    ax.add_patch(Rectangle((0.18, 0.86), 1.78, 0.46,
                           fc=(*to_rgb(BLUE), 0.20), ec=BLUE, lw=0.8, zorder=2))
    ax.add_patch(Rectangle((0.84, 0.46), 0.46, 1.30,
                           fc=(*to_rgb(ORANGE), 0.20), ec=ORANGE, lw=0.8,
                           zorder=3))
    ax.add_patch(Rectangle((0.84, 0.86), 0.46, 0.46,
                           fc=(*to_rgb(VIOLET), 0.30), ec=VIOLET, lw=1.0,
                           zorder=4))
    txt(ax, 1.07, 1.09, r'$A\cap B$', size=7.5)
    txt(ax, 0.20, 1.40, r'$A$: $g_X(A)$', ha='left', va='bottom')
    txt(ax, 1.38, 1.62, r'$B$: $g_Z(B)$', ha='left')
    leader(ax, (1.14, 0.90), (1.52, 0.62))
    txt(ax, 1.55, 0.60, r'$r(A,B)\le|A\cap B|$', size=7, color=INK2,
        ha='left', va='top')
    txt(ax, pw / 2 - 0.05, 0.17, r'$g_X(A)+g_Z(B)\le K+|A\cap B|$', size=8.5)

    ax = canvas(fig, pw, 0.0, pw, H)
    title(ax, 'b', 'Layer-code transverse slabs')
    s = 0.95
    P = Oblique(0.56, 0.66)
    zs, xs = slab_block(ax, P, s, tube=0.042)
    p = P(0, 0, zs)
    txt(ax, p[0] - 0.06, p[1], r'$X$ slab', ha='right')
    p = P(xs, 0, 0)
    txt(ax, p[0], p[1] - 0.05, r'$Z$ slab', va='top')
    txt(ax, 0.36, 0.33, r'$|A|,\ |B|=\Theta(d)$', ha='left')
    swatch_line(ax, 0.36, 0.13, VIOLET, lw=3.2, length=0.14)
    txt(ax, 0.56, 0.13, r'$|A\cap B|=\Theta(K)$', ha='left')

    ax = canvas(fig, 2 * pw, 0.0, pw, H)
    title(ax, 'c', 'Structural consequence')
    bw = 1.90
    bx = (pw - bw) / 2
    rbox(ax, bx, 1.44, bw, 0.28, INK2, fc='#f5f6f7')
    txt(ax, pw / 2, 1.58, r'slab supports: $\Theta(d)$ each')
    rbox(ax, bx, 1.06, bw, 0.28, INK2, fc='#f5f6f7')
    txt(ax, pw / 2, 1.20, r'intersection: $\Theta(K)$')
    arrow(ax, (pw / 2, 1.04), (pw / 2, 0.84), color=INK, lw=0.8)
    rbox(ax, bx, 0.50, bw, 0.32, VIOLET)
    txt(ax, pw / 2, 0.66, r'$\chi_{XZ}(Q_N)=\Theta(K)=\Theta(n/d)$', size=8)
    txt(ax, pw / 2, 0.27, 'same slabs: fault-tolerant ports', size=7,
        color=INK2)
    txt(ax, pw / 2, 0.12, r'$+$ structural no-go witness', size=7,
        color=INK2)
    save(fig, 'symplectic_overlap_layer_ports')


def fig_slab_disjointness():
    W, H = 0.94 * TW, 2.10
    fig = plt.figure(figsize=(W, H))
    pw = W / 3

    ax = canvas(fig, 0.0, 0.0, pw, H)
    title(ax, 'a', 'Complete slab families')
    gx, gy, gw, gh = 0.30, 0.42, 1.55, 1.30
    for i in range(4):
        ax.add_patch(Rectangle((gx, gy + 0.10 + i * 0.30), gw, 0.13,
                               fc=(*to_rgb(BLUE), 0.22), ec=BLUE, lw=0.6,
                               zorder=2))
        ax.add_patch(Rectangle((gx + 0.14 + i * 0.37, gy), 0.13, gh,
                               fc=(*to_rgb(ORANGE), 0.22), ec=ORANGE, lw=0.6,
                               zorder=3))
    txt(ax, gx + gw + 0.06, gy + 0.165, r'$X$', size=7, ha='left')
    txt(ax, gx + 0.205, gy + gh + 0.05, r'$Z$', size=7, va='bottom')
    txt(ax, gx + gw / 2, 0.20, r'$r_X,\ r_Z=\Theta(N)$ disjoint slabs', size=7.5)

    ax = canvas(fig, pw, 0.0, pw, H)
    title(ax, 'b', 'Mixed logical classes')
    gx, gy, gw, gh = 0.12, 0.66, 0.95, 0.95
    bt = 0.115
    hy = [gy + 0.13, gy + 0.42, gy + 0.71]
    vx = [gx + 0.13, gx + 0.42, gx + 0.71]
    for i in range(3):
        on = i == 1
        ax.add_patch(Rectangle((gx, hy[i]), gw, bt,
                               fc=(*to_rgb(AQUA if on else BLUE),
                                   0.45 if on else 0.14),
                               ec=AQUA if on else BLUE, lw=0.9 if on else 0.5,
                               zorder=4 if on else 2))
        ax.add_patch(Rectangle((vx[i], gy), bt, gh,
                               fc=(*to_rgb(AQUA if on else ORANGE),
                                   0.45 if on else 0.14),
                               ec=AQUA if on else ORANGE,
                               lw=0.9 if on else 0.5, zorder=4 if on else 3))
    qx, qy = vx[2] + bt / 2, hy[0] + bt / 2
    ax.add_patch(Circle((qx, qy), 0.034, fc=INK, ec='white', lw=0.4,
                        zorder=6))
    tx = gx + gw + 0.08
    swatch_line(ax, tx, gy + 0.80, AQUA, lw=3.2, length=0.14)
    txt(ax, tx + 0.19, gy + 0.80, r'$P_2=X_2(u)\,Z_2(v)$', size=7, ha='left')
    leader(ax, (qx + 0.04, qy - 0.02), (tx - 0.01, gy + 0.20))
    txt(ax, tx + 0.02, gy + 0.25, 'qubit in', size=7, color=INK2, ha='left')
    txt(ax, tx + 0.02, gy + 0.11, r'$P_1,P_3$ only', size=7, color=INK2,
        ha='left')
    txt(ax, pw / 2, 0.40, r'each qubit lies in at most two $P_i$', size=7,
        color=INK2)
    txt(ax, pw / 2, 0.17, r'$\Delta\geq\tfrac12\min\{r_X,r_Z\}$', size=8.5)

    ax = canvas(fig, 2 * pw, 0.0, pw, H)
    title(ax, 'c', 'Static unitary restriction')
    txt(ax, pw / 2, 1.58, r'$\Delta=\Theta(N)$', size=8.5)
    txt(ax, pw / 2, 1.30, r'$d_\uparrow,\ d_\downarrow=\Theta(N^2)$', size=8.5)
    arrow(ax, (pw / 2, 1.12), (pw / 2, 0.84), color=INK, lw=0.8)
    rbox(ax, 0.20, 0.26, pw - 0.40, 0.56, VIOLET)
    txt(ax, pw / 2, 0.64, 'Logical Clifford', size=8)
    txt(ax, pw / 2, 0.42, 'fixed depth and gate arity', size=7, color=INK2)
    save(fig, 'slab_disjointness_clifford')


def fig_raw_injection():
    W, H = CW, 1.72
    fig = plt.figure(figsize=(W, H))
    ax = canvas(fig, 0.0, 0.0, W, H)
    txt(ax, W / 2, H - 0.02, r'\textbf{Raw injection: only the traveling '
        r'resource is cheap}', size=8, va='top')
    yc = 0.86
    qx = 0.31
    ax.add_patch(Circle((qx, yc), 0.12, fc=tint(ORANGE, 0.75), ec=ORANGE,
                        lw=0.9, zorder=3))
    txt(ax, qx, yc, r'$q$', size=8.5)
    txt(ax, qx, yc - 0.20, 'bare qubit', size=6.5, color=INK2, va='top')
    txt(ax, qx, yc - 0.32, '(weight 1)', size=6.5, color=INK2, va='top')
    arrow(ax, (qx + 0.14, yc), (0.89, yc), color=INK, lw=0.8)
    txt(ax, 0.67, yc + 0.07, r'gauge', size=6.5, color=INK2, va='bottom')
    txt(ax, 0.67, yc - 0.07, r'$Z_qZ_{B_0}$', size=7, va='top')
    bx, by, bw, bh = 0.92, 0.26, 1.56, 1.18
    rbox(ax, bx, by, bw, bh, AQUA, fc=tint(AQUA, 0.93))
    txt(ax, bx + bw / 2, by + bh - 0.13, r'block $B_0$', size=8)
    rows = [('spectator', r'weight $\Omega(d)$', BLUE),
            ('spectator', r'weight $\Omega(d)$', BLUE),
            ('traveling resource', r'weight $1$', ORANGE)]
    for k, (a, b, col) in enumerate(rows):
        y = by + bh - 0.42 - k * 0.25
        rbox(ax, bx + 0.08, y - 0.095, bw - 0.16, 0.19, col,
             fc=tint(col, 0.88), lw=0.6, r=0.03, z=3)
        txt(ax, bx + 0.15, y, a, size=6.5, ha='left', zorder=4)
        txt(ax, bx + bw - 0.15, y, b, size=6.5, ha='right', zorder=4)
    arrow(ax, (bx + bw + 0.03, yc), (bx + bw + 0.20, yc), color=INK, lw=0.8)
    x = bx + bw + 0.26
    for side in (0.12, 0.17, 0.23):
        ax.add_patch(Rectangle((x, yc - side / 2), side, side,
                               fc=tint(AQUA, 0.7), ec=AQUA, lw=0.6, zorder=3))
        x += side + 0.035
    txt(ax, bx + bw + 0.52, yc - 0.23, 'encoded', size=6.5, color=INK2,
        va='top')
    txt(ax, bx + bw + 0.52, yc - 0.35, 'courier', size=6.5, color=INK2,
        va='top')
    txt(ax, W / 2, 0.10, r'spectators keep $\Omega(d)$ in the quotient',
        size=6.5, color=INK2)
    save(fig, 'raw_injection_intuition')


def fig_gate_compilation():
    W, H = CW, 2.02
    fig = plt.figure(figsize=(W, H))
    ax = canvas(fig, 0.0, 0.0, W, H)
    txt(ax, W / 2, H - 0.02,
        r'\textbf{Clifford$\boldsymbol{+T}$ from pure-$\boldsymbol{X/Z}$ '
        r'measurements}', size=8, va='top')
    cols = [0.34, 1.62, 2.94]
    top, rh = 1.76, 0.25
    rows = [(r'$S$', r'phase teleportation $+\,|Y\rangle$',
             r'$Z^{m\oplus n}X^m$'),
            (r'$T$', r'phase teleportation $+\,|A\rangle$', r'$S^mZ^nX^m$'),
            (r'$H$', r'$S\to S_X\to S$\ \ ($3\,|Y\rangle$)', 'Pauli frame'),
            ('CNOT', r'$Z_cZ_a,\ X_aX_t,\ Z_a$', r'$Z_c^{\,v}X_t^{\,u\oplus w}$'),
            ('CZ', r'$H_t\cdot\mathrm{CNOT}\cdot H_t$', r'$\leq 6\,|Y\rangle$')]
    ax.plot([0.05, W - 0.05], [top, top], color=INK, lw=0.8)
    for x, h in zip(cols, ['Gate', 'Measurements / resources', 'Correction']):
        txt(ax, x, top - 0.13, r'\textbf{%s}' % h, size=7.5)
    ax.plot([0.05, W - 0.05], [top - 0.26, top - 0.26], color=INK, lw=0.5)
    y0 = top - 0.26
    for i, (g, m, c) in enumerate(rows):
        yc = y0 - (i + 0.5) * rh
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.05, yc - rh / 2), W - 0.10, rh,
                                   fc='#f3f5f7', ec='none', zorder=0))
        txt(ax, cols[0], yc, g, size=7.5)
        txt(ax, cols[1], yc, m, size=7.5)
        txt(ax, cols[2], yc, c, size=7.5)
    yb = y0 - len(rows) * rh
    ax.plot([0.05, W - 0.05], [yb, yb], color=INK, lw=0.8)
    txt(ax, W / 2, 0.09,
        r'\textit{Only pure-$X$ or pure-$Z$ logical measurements are required.}',
        size=7, color=INK2)
    save(fig, 'gate_compilation')


def fig_fragmentation():
    W, H = 0.92 * TW, 2.50
    fig = plt.figure(figsize=(W, H))
    ax = canvas(fig, 0.0, 0.0, W, H)
    txt(ax, W / 2, H - 0.02,
        r'\textbf{At fixed logical volume $\boldsymbol{W}$, fragmentation '
        r'$\boldsymbol{B_f(A)}$ controls the boundary-resource penalty}',
        size=8.5, va='top')
    n, c = 8, 0.155             # 8 x 8 coarse cells of side b
    gw = n * c
    pw = W / 3
    heads = [('Filled cube', r'$B_f(A)=O(W^{2/3}/b^{2}+1)$'),
             (r'Thin slab, thickness $h<b$', r'$B_f(A)=\Theta(W/(hb^{2}))$'),
             ('Isolated sites', r'$B_f(A)=W$')]
    feet = ['few partial coarse cells', r'same $W$, every cell partial',
            r'same $W$, one partial cell per site']
    full_c, part_c = tint(GREEN, 0.72), tint(ORANGE, 0.62)
    gy = 0.55
    regions = [
        [(1.0, 6.55, 1.0, 6.45)],
        [(0.25, 7.75, 3.36, 3.64)],
        [(x + 0.35, x + 0.65, y + 0.35, y + 0.65) for x, y in
         [(0, 6), (2, 7), (5, 6), (7, 4), (1, 3), (4, 4), (2, 1), (6, 1),
          (4, 0), (6, 7)]],
    ]
    for k in range(3):
        gx = k * pw + (pw - gw) / 2
        txt(ax, k * pw + pw / 2, 2.13, heads[k][0], size=8)
        txt(ax, k * pw + pw / 2, 1.96, heads[k][1], size=7.5)
        for i in range(n):
            for j in range(n):
                cover = 0.0
                for (a, b, cc, d) in regions[k]:
                    ox = max(0, min(b, i + 1) - max(a, i))
                    oy = max(0, min(d, j + 1) - max(cc, j))
                    cover += ox * oy
                fc = 'white'
                if cover > 1 - 1e-9:
                    fc = full_c
                elif cover > 1e-9:
                    fc = part_c
                ax.add_patch(Rectangle((gx + i * c, gy + j * c), c, c, fc=fc,
                                       ec='#c4c9cf', lw=0.4, zorder=2))
        for (a, b, cc, d) in regions[k]:
            ax.add_patch(Rectangle((gx + a * c, gy + cc * c), (b - a) * c,
                                   (d - cc) * c, fc='none', ec=INK, lw=0.8,
                                   zorder=4))
        txt(ax, k * pw + pw / 2, gy - 0.12, feet[k], size=7, color=INK2,
            va='top')
    gx = (pw - gw) / 2
    ax.annotate('', xy=(gx + gw + 0.05, gy + c), xytext=(gx + gw + 0.05, gy),
                arrowprops=dict(arrowstyle='<->', lw=0.5, color=INK2,
                                shrinkA=0, shrinkB=0, mutation_scale=4))
    txt(ax, gx + gw + 0.09, gy + c / 2, r'$b$', size=7, ha='left')
    lx, ly = W / 2 - 1.95, 0.12
    for fc, ec, lab, dx in [(full_c, '#c4c9cf', 'filled coarse cell', 0.0),
                            (part_c, '#c4c9cf',
                             r'partially filled cell (counted in $B_f(A)$)',
                             1.18),
                            ('none', INK, r'region $A$', 3.40)]:
        ax.add_patch(Rectangle((lx + dx, ly - 0.05), 0.10, 0.10, fc=fc,
                               ec=ec, lw=0.6 if fc != 'none' else 0.8,
                               zorder=3))
        txt(ax, lx + dx + 0.15, ly, lab, size=7, color=INK2, ha='left')
    save(fig, 'fragmentation_Bf')


def fig_ft_dependency():
    W, H = TW, 3.42
    fig = plt.figure(figsize=(W, H))
    ax = canvas(fig, 0.0, 0.0, W, H)
    PROVED, CITED, CHECK = GREEN, BLUE, YELLOW
    txt(ax, W / 2, H - 0.02, r'\textbf{Dependency chain for the universal '
        r'fault-tolerance theorem}', size=9, va='top')
    bw, bh = 1.50, 0.50
    xs = [0.12 + i * (bw + 0.27) for i in range(4)]
    yt, yb = 2.16, 1.18
    top = ['Explicit realization\n($k=K$, $d=\\Theta(N^2)$)',
           'Single-layer\nslab theorem',
           'Local 3D gauging\nand static distance',
           'Pure-$X/Z$ compilation\nand defective resources']
    bottom = ['Local detector catalog\nand $(s,\\lambda,\\rho)$ gluing',
              'Spacetime cleaning\nand rectangle distance',
              'Padded composition\nand encoded courier',
              'Universal resource bound\nand positive threshold']

    def node(x, y, w, h, s, col, size=7.5, num=None):
        rbox(ax, x, y, w, h, col, fc=tint(col, 0.88), lw=0.8, r=0.05)
        txt(ax, x + w / 2, y + h / 2, s, size=size, linespacing=1.15)
        if num is not None:
            ax.add_patch(Circle((x + 0.005, y + h - 0.005), 0.085, fc='white',
                                ec=col, lw=0.8, zorder=5))
            txt(ax, x + 0.005, y + h - 0.005, str(num), size=6.5, zorder=6)

    # proof chain: left to right on top, then right to left underneath
    for i, (x, s) in enumerate(zip(xs, top)):
        node(x, yt, bw, bh, s, PROVED, num=i + 1)
    for i, (x, s) in enumerate(zip(reversed(xs), bottom)):
        node(x, yb, bw, bh, s, PROVED, num=i + 5)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + bw + 0.03, yt + bh / 2), (b - 0.03, yt + bh / 2),
              color=INK, lw=0.8)
        arrow(ax, (b - 0.03, yb + bh / 2), (a + bw + 0.03, yb + bh / 2),
              color=INK, lw=0.8)
    arrow(ax, (xs[3] + bw / 2, yt - 0.03), (xs[3] + bw / 2, yb + bh + 0.03),
          color=INK, lw=0.8)

    # inputs: (left, bottom, width, height, text, colour, target box x)
    yc_top, yc_bot, ch = 2.94, 0.62, 0.27
    inputs = [
        (xs[0], yc_top, 1.95, ch, 'Williamson--Baspin condensation data',
         CITED, xs[0]),
        (xs[2] - 0.05, yc_top, 1.62, ch, 'Grid-gauging / expansion input',
         CITED, xs[2]),
        (xs[0] - 0.03, yc_bot, 2.12, ch,
         r'Balanced good CSS family $+$ noise model', CITED, xs[0]),
        (xs[2] + bw / 2 - 0.89, yc_bot, 1.78, ch,
         'Williamson--Yoder cleaning theorem', CITED, xs[2]),
        (xs[3] - 0.02, yc_bot - 0.10, 1.58, 0.40,
         'Complete-code certificate /\ncircuit checks (scripts $+$ CSV)',
         CHECK, xs[3]),
    ]
    for x, y, w, h, s, col, tgt in inputs:
        node(x, y, w, h, s, col, size=7)
        tx = tgt + bw / 2
        if y > yt:
            arrow(ax, (tx, y - 0.02), (tx, yt + bh + 0.03), color=col, lw=0.8)
        else:
            arrow(ax, (tx, y + h + 0.02), (tx, yb - 0.03), color=col, lw=0.8)

    items = [(PROVED, 'proved here'), (CITED, 'cited input'),
             (CHECK, 'supporting check')]
    widths = [0.60, 0.55, 0.85]
    lx = W / 2 - (sum(widths) + 0.35 * 2) / 2
    for (col, lab), w in zip(items, widths):
        rbox(ax, lx, 0.30 - 0.055, 0.20, 0.11, col, fc=tint(col, 0.88), lw=0.8,
             r=0.025)
        txt(ax, lx + 0.26, 0.30, lab, size=7, ha='left', color=INK2)
        lx += w + 0.35
    txt(ax, W / 2, 0.09, 'Finite numerical crossings test selected models; '
        'they are not inputs to the theorem.', size=7, color=INK2)
    save(fig, 'ft_proof_dependency')


def fig_sanity_checks():
    specs = [('phenomenological_mw_demo', 'distance', 'Toric memory control',
              r'Noise probability $p$', (5e-5, 1)),
             ('gauging_rectangle_mwpm', 'd', 'Gauging-strip control',
              r'Data-noise probability $p$', (4e-3, 1))]
    for fn, key, head, xlab, ylim in specs:
        rr = readcsv(fn + '.csv')
        sizes = sorted({int(r[key]) for r in rr})
        W, H = CW, 2.42
        fig = plt.figure(figsize=(W, H))
        ax = plot_axes(fig, 0.56, 0.40, W - 0.66, H - 0.66)
        cols = RAMP[len(sizes)]
        for i, d in enumerate(sizes):
            series(ax, *rows_curve([r for r in rr if int(r[key]) == d]),
                   cols[i], MARKERS[i], zorder=3 + i)
        log_y(ax, *ylim)
        ax.set_xlim(0, 0.105)
        ax.xaxis.set_major_locator(MultipleLocator(0.02))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        ax.set(xlabel=xlab, ylabel='Logical failure probability')
        plot_title(ax, None, head)
        h, lab = size_handles(len(sizes), 'd')
        ax.legend(h, lab, loc='lower right')
        save(fig, fn)


def fig_relative_expansion():
    W, H = CW, 2.30
    fig = plt.figure(figsize=(W, H))
    ax = plot_axes(fig, 0.52, 0.40, W - 0.62, H - 0.52)
    r = np.linspace(0.05, 1, 300)
    for i, t in enumerate((1, 2, 4)):
        ax.plot(r, r / (2 * t), color=RAMP[3][2 - i], lw=1.3,
                ls=['-', (0, (4, 1.6)), (0, (1.2, 1.2))][i])
        txt(ax, 0.95, 0.95 / (2 * t) + 0.014, rf'$t={t}$', size=7.5,
            ha='right', va='bottom')
    ax.set(xlim=(0, 1), ylim=(0, 0.55),
           xlabel=r'Aspect ratio $a_{\min}/a_{\max}$',
           ylabel=r'Lower bound on $\beta$')
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    txt(ax, 0.05, 0.50, r'$\beta\geq\dfrac{a_{\min}}{2t\,a_{\max}}$', size=8.5,
        ha='left', va='top')
    ax.grid(color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    save(fig, 'relative_expansion_bound')


def fig_padded_composition():
    W, H = CW, 2.30
    fig = plt.figure(figsize=(W, H))
    ax = plot_axes(fig, 0.52, 0.40, W - 0.62, H - 0.52)
    rect = 12
    x = np.linspace(1, 20, 4001)
    pad = 1 + np.floor((x - 1) + 1e-12)        # 1 + floor((R_pad - tau)/tau)
    ax.axhline(rect, color=BLUE, ls=(0, (1.2, 1.4)), lw=1.2, zorder=2)
    ax.plot(x, pad, color=AQUA, ls=(0, (4, 1.8)), lw=1.1, zorder=3)
    ax.plot(x, np.minimum(rect, pad), color=INK, lw=1.4, zorder=4)
    ax.set(xlim=(1, 20), ylim=(0, 14.5),
           xlabel=r'Padding ratio $R_{\mathrm{pad}}/\tau$',
           ylabel=r'Lower bound on $f_{\mathrm{ph}}$')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    h = [Line2D([], [], color=INK, lw=1.4),
         Line2D([], [], color=AQUA, ls=(0, (4, 1.8)), lw=1.1),
         Line2D([], [], color=BLUE, ls=(0, (1.2, 1.4)), lw=1.2)]
    ax.legend(h, [r'composed bound $f_{\mathrm{ph}}$',
                  r'padding term $1+\lfloor(R_{\mathrm{pad}}-\tau)/\tau\rfloor$',
                  r'$\delta_g=\delta_{\mathrm{mem}}=12$'],
              loc='lower right', fontsize=6.5, handlelength=2.0)
    save(fig, 'padded_composition_bound')


def fig_building_blocks(stage):
    W, H = TW, 2.45
    fig = plt.figure(figsize=(W, H))
    pw = 2.95
    memory = readcsv('wb_bposd_threshold.csv')
    ax = plot_axes(fig, 0.56, 0.40, pw, H - 0.68)
    for i, L in enumerate([3, 4, 5, 6]):
        rr = [r for r in memory if int(r['L']) == L
              and r['noise_model'] == 'code_capacity_independent_XZ']
        series(ax, *rows_curve(rr), RAMP[4][i], MARKERS[i], zorder=3 + i)
    log_y(ax, 5e-6, 1)
    ax.set_xlim(-0.002, 0.125)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.set(xlabel=r'Data-noise probability $p$',
           ylabel='Logical failure probability')
    plot_title(ax, 'a', 'WB-incidence memory')
    h, lab = size_handles(4)
    ax.legend(h + [zero_handle()], lab + ['zero failures'], loc='lower right',
              ncol=2)

    ax = plot_axes(fig, 0.56 + pw + 0.55, 0.40, pw, H - 0.68)
    for i, L in enumerate([3, 5, 7, 9, 11]):
        rr = [r for r in stage if r['experiment'] == 'primitive'
              and int(r['size']) == L]
        series(ax, *rows_curve(rr, scale=1000), RAMP[5][i], MARKERS[i],
               zorder=3 + i)
    log_y(ax, 1e-5, 1)
    ax.set_xlim(1.0, 6.8)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set(xlabel=r'Noise strength $p\ (10^{-3})$',
           ylabel='Logical failure probability')
    plot_title(ax, 'b', 'Circuit-level gauging')
    h, lab = size_handles(5, 'd')
    ax.legend(h, lab, loc='lower right', ncol=2)
    save(fig, 'fig10_building_blocks')


def fig_stage_composed(stage):
    W, H = 0.94 * TW, 2.45
    fig = plt.figure(figsize=(W, H))
    left, gap, right = 0.56, 0.14, 0.06
    pw = (W - left - gap - right) / 2
    for k, R in enumerate([1, 2]):
        ax = plot_axes(fig, left + k * (pw + gap), 0.40, pw, H - 0.68)
        for i, L in enumerate([3, 5, 7, 9, 11]):
            rr = [r for r in stage if r['experiment'] == 'hp_cas'
                  and int(r['rounds_scale']) == R and int(r['size']) == L]
            series(ax, *rows_curve(rr, scale=1000), RAMP[5][i], MARKERS[i],
                   zorder=3 + i)
        log_y(ax, 5e-6, 2)
        ax.set_xlim(1.0, 7.0)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_xlabel(r'Noise strength $p\ (10^{-3})$')
        plot_title(ax, 'ab'[k], '$R=d$' if R == 1 else '$R=2d$')
        if k == 0:
            ax.set_ylabel('Logical failure probability')
        else:
            ax.tick_params(labelleft=False)
            h, lab = size_handles(5)
            ax.legend(h, lab, loc='lower right', ncol=2)
    save(fig, 'supp_heavy_stage_composed')


def fig_threshold_panels(rows, cross):
    W, H = 0.325 * TW, 2.05
    for G in [1, 2, 3]:
        fig = plt.figure(figsize=(W, H))
        ax = plot_axes(fig, 0.46, 0.38, W - 0.53, H - 0.62)
        continuous_panel(ax, rows, G, cross)
        ax.set_xlabel(r'Noise strength $p\ (10^{-3})$')
        ax.set_ylabel('Logical failure probability', labelpad=1.5)
        plot_title(ax, None, glabel(G))
        h, lab = size_handles(5)
        leg = ax.legend(h, lab, loc='lower right', ncol=2, fontsize=6.5,
                        handlelength=1.4, columnspacing=0.8, frameon=True,
                        fancybox=False, borderpad=0.3)
        leg.get_frame().set(facecolor='white', edgecolor='none', alpha=1.0)
        save(fig, f'threshold_G{G}d')


def fig_controls():
    W, H = 0.48 * TW, 2.62
    rect = (0.56, 0.90, W - 0.64, H - 1.16)

    rr = readcsv('wb_layer_circuit_dem.csv')
    fig = plt.figure(figsize=(W, H))
    ax = plot_axes(fig, *rect)
    for i, L in enumerate([3, 4, 5, 6]):
        series(ax, *rows_curve([r for r in rr if int(r['L']) == L], scale=1000),
               RAMP[4][i], MARKERS[i], zorder=3 + i)
    log_y(ax, 1e-2, 1)
    ax.set_xlim(0, 21)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set(xlabel=r'Noise strength $p\ (10^{-3})$',
           ylabel='Logical failure probability')
    plot_title(ax, None, 'Truncated-WB circuit control')
    h, lab = size_handles(4)
    fig.legend(h, lab, loc='upper left', ncol=2,
               bbox_to_anchor=(rect[0] / W - 0.01, 0.56 / H))
    save(fig, 'final_circuit_level_arch_si1000_wb')

    rr = readcsv('magic_factory_rm15_si1000.csv')
    fig = plt.figure(figsize=(W, H))
    ax = plot_axes(fig, *rect)
    spec = [('Steane7_code_capacity_LUT', 'Steane: capacity', BLUE,
             (0, (3.5, 1.5)), 'o'),
            ('Steane7_project_Sdag_LUT', 'Steane: circuit', BLUE, '-', 's'),
            ('RM15_allones_code_capacity_LUT', 'RM15: capacity', ORANGE,
             (0, (3.5, 1.5)), '^'),
            ('RM15_project_Sdag_LUT', 'RM15: Clifford probe', ORANGE, '-',
             'D'),
            ('MultiLevel_Steane_then_RM15_capacity', 'Steane / RM15 stack',
             AQUA, '-', 'p'),
            ('Factory_plus_CAS_consume_d3_proxy', 'Factory $+$ consume proxy',
             VIOLET, '-', 'h')]
    handles = []
    for k, (key, lab, col, ls, mk) in enumerate(spec):
        x, y, lo, hi = rows_curve([r for r in rr if r['factory'] == key],
                                  scale=1000)
        pos = y > 0
        ax.errorbar(x, np.where(pos, y, np.nan),
                    yerr=[np.maximum(0, y - lo), np.maximum(0, hi - y)],
                    color=col, marker=mk, ms=3.6, mec='white', mew=0.45,
                    lw=1.0, ls=ls, elinewidth=0.6, capsize=1.5, capthick=0.6,
                    zorder=3 + k)
        if (~pos).any():
            ax.scatter(x[~pos], hi[~pos], marker='v', s=15,
                       facecolors='white', edgecolors=col, linewidths=0.8,
                       zorder=10)
        handles.append(handle(col, mk, ls=ls))
    log_y(ax, 1e-5, 2)
    ax.set_xlim(0, 10.6)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set(xlabel=r'Noise strength $p\ (10^{-3})$',
           ylabel='Logical failure probability')
    plot_title(ax, None, 'Finite factory controls')
    fig.legend(handles + [zero_handle()], [s[1] for s in spec] +
               ['zero failures'], loc='upper left', ncol=2,
               bbox_to_anchor=(rect[0] / W - 0.01, 0.56 / H), fontsize=6.5,
               handlelength=2.2, columnspacing=0.9)
    save(fig, 'final_magic_factory_rm15_si1000')


def main():
    OUT.mkdir(exist_ok=True)
    data, rows = continuous_rows()
    cross = crossings()
    stage = readcsv('numerical/stage_composed/all_points.csv')
    fig_architecture()
    fig_slab_prefix()
    fig_resources()
    fig_numerics(rows, cross)
    fig_symplectic_overlap()
    fig_slab_disjointness()
    fig_raw_injection()
    fig_gate_compilation()
    fig_fragmentation()
    fig_ft_dependency()
    fig_sanity_checks()
    fig_relative_expansion()
    fig_padded_composition()
    fig_building_blocks(stage)
    fig_stage_composed(stage)
    fig_threshold_panels(rows, cross)
    fig_controls()
    report = {'points': len(rows), 'shots': sum(r['shots'] for r in rows),
              'noiseless_points': sum(r['p'] == 0 for r in rows),
              'noiseless_failures': sum(r['failures'] for r in rows
                                        if r['p'] == 0),
              'all_rates_match_counts': True,
              'numerical_source': str(data.relative_to(ROOT)),
              'crossings': 'Archived rounded Jeffreys estimates; no new Monte '
                           'Carlo runs.'}
    (ROOT / 'revision_data_checks.json').write_text(json.dumps(report, indent=2),
                                                    encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
