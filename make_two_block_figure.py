#!/usr/bin/env python3
"""Main-text figure of the two-block measurement on explicit layer-code blocks.

Draws figures/two_block_main.pdf (and a 600 dpi PNG preview) in the style of
make_publication_figures.py.  (a) Monte Carlo failure of the measured pair
against the block size at p = 1e-3 (scaling campaign).  (b) Exact circuit
fault distance of the measured pair against the X distance of the blocks
(CP-SAT; colocated scaling campaign and routed campaign).  (c) Factor gained
per unit of spacing through the spanning grid.

Reads numerical/scaling_campaign/{mc_batches.csv, fault_distance.csv} and
numerical/routed_campaign/key_numbers.json; draws no number by hand.
The supplemental figure of the same campaign is drawn by make_scaling_figure.py.

Usage:  python make_two_block_figure.py
"""
import csv
import json
import sys
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator

import make_publication_figures as mp
from make_publication_figures import (AQUA, BLUE, INK2, MUTED, ORANGE, TW, VIOLET, handle, log_y, plot_axes,
                                      plot_title, save, series, tint, txt, zero_handle)

sys.path.insert(0, str(mp.ROOT / 'scaling_campaign' / 'campaign'))
import analyze as A  # noqa: E402

SC = mp.ROOT / 'numerical' / 'scaling_campaign'
RC = mp.ROOT / 'numerical' / 'routed_campaign'
BRIDGE = [VIOLET, tint(VIOLET, 0.35), tint(VIOLET, 0.6)]      # w = 1, 2, 3
GRID = AQUA
IDLE = MUTED
SPACINGS = (2, 3, 4, 5)


def campaign():
    A.RES = SC
    rows = [a for a in A.load() if a['decoder'] == 'sector' and a['k'] == 2]

    def get(iface, basis, s, p, w=1, m=1):
        for a in rows:
            if (a['interface'] == iface and a['basis'] == basis and a['s'] == s and a['p'] == p
                    and (iface != 'bridge' or a['width'] == w) and a['mult'] == m):
                return a
        return None
    return rows, get


def panel_failures(ax, get, p=0.001):
    out = {}
    for iface, col, mk in (('idle', IDLE, 'D'), ('thick', GRID, 'o'), ('bridge', BRIDGE[0], 's')):
        x, y, lo, hi = [], [], [], []
        for s in SPACINGS:
            a = get(iface, 'Z', s, p)
            if a is None:
                continue
            k, n = A.obs_count(a, 'Z1aZ1b')
            l, h = A.wilson(k, n)
            x.append(s)
            y.append(k / n)
            lo.append(l)
            hi.append(h)
        series(ax, x, y, lo, hi, col, mk, ms=4.2, lw=1.2, zorder=4 if iface != 'idle' else 3)
        out[iface] = (x, y)
    log_y(ax, 1e-5, 0.3)
    ax.set_xlim(1.6, 5.4)
    ax.set_xticks(SPACINGS)
    ax.set_xticklabels([f'{s}\n$({2 * s + 1},{2 * s})$' for s in SPACINGS], linespacing=1.15)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r'Spacing $s$ $\;(d_X,d_Z)$ of each block', labelpad=1)
    ax.set_ylabel(r'Failure of $\overline Z{}_1^{(a)}\overline Z{}_1^{(b)}$')
    plot_title(ax, 'a', 'Failures as the blocks grow')
    xb, yb = out['bridge']
    xg, yg = out['thick']
    txt(ax, 3.5, yb[1] * 2.6, r'one-row bridge: up $%.1f\times$' % (yb[-1] / yb[0]), size=6.6, color=VIOLET)
    txt(ax, 4.05, yg[2] * 5.5, r'spanning grid: down $%.0f\times$' % (yg[0] / yg[-1]), size=6.6,
        color='#0f7a54')
    txt(ax, 2.95, 4.6e-5, r'idle blocks', size=6.4, color=INK2)
    ax.legend([zero_handle(INK2)], [r'no failure (95\% bound)'], loc='lower left', fontsize=6.3, handlelength=1.2)


def fault_rows():
    with open(SC / 'fault_distance.csv', newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if r['observable'] == 'Z1aZ1b']
    routed = json.loads((RC / 'key_numbers.json').read_text())
    return rows, routed


def bounds(v):
    """'5' -> (5, 5); '4-9' -> (4, 9); '>=4' -> (4, None)."""
    v = str(v)
    if v.startswith('>='):
        return int(v[2:]), None
    if '-' in v:
        a, b = v.split('-')
        return int(a), int(b)
    return int(v), int(v)


def panel_distance(ax):
    rows, routed = fault_rows()
    xmax = 20
    ax.plot([0, xmax], [0, xmax], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    txt(ax, 6.9, 7.55, r'$=d_X$', size=6.6, color=INK2, ha='right')
    # colocated scaling campaign: exact (OPTIMAL) or proved bounds (FEASIBLE)
    for r in rows:
        dX = int(r['d_X'])
        if r['interface'].startswith('bridge'):
            w = int(r['interface'][-1])
            col, mk, dx = BRIDGE[w - 1], 's', -0.22
        else:
            col, mk, dx = GRID, 'o', -0.22
        lo, hi = int(r['lower_bound']), int(r['distance'])
        if r['status'] == 'OPTIMAL':
            ax.plot([dX + dx], [hi], marker=mk, color=col, ms=4.6, mec='white', mew=0.5, ls='none', zorder=5)
        else:                          # solver stopped: lower bound and the weight of a found pattern
            ax.plot([dX + dx] * 2, [lo, hi], color=col, lw=2.4, alpha=0.45, solid_capstyle='butt', zorder=3)
    # routed campaign: the measured pair through the one-row bridge, every input and round count
    P = routed['params']
    for name, (inp, rnd) in {'iceberg_422': ('iceberg_422', 'R=5'), 'band_m2': ('band_m2', 'R=7'),
                             'steane_713': ('steane_713', 'R=1')}.items():
        dX = int(P[name]['dX_theorem_lb'])
        d = bounds(routed['fault_distances'][f'{inp}|bridge|Z1aZ1b|{rnd}'])
        assert d == (1, 1), (name, d)
        ax.plot([dX + 0.22], [1], marker='s', color=BRIDGE[0], mfc='white', mew=0.9, ms=4.4, ls='none', zorder=5)
        if name != 'iceberg_422':      # block distance known only from below
            ax.annotate('', xy=(dX + 1.5, 1), xytext=(dX + 0.55, 1),
                        arrowprops=dict(arrowstyle='-|>', color=BRIDGE[0], lw=0.6, mutation_scale=5))
    g = bounds(routed['fault_distances']['iceberg_422|thick|Z1aZ1b|R=5'])
    ax.plot([int(P['iceberg_422']['dX_theorem_lb']) + 0.22], [g[0]], marker='o', color=GRID, mfc='white', mew=0.9,
            ms=4.4, ls='none', zorder=5)
    for w in (1, 2, 3):
        txt(ax, 9.2, w, r'bridge, $w=%d$' % w, size=6.3, color=BRIDGE[w - 1] if w == 1 else tint(VIOLET, 0.2),
            ha='left')
    txt(ax, 5.0, 5.75, r'grid', size=6.4, color='#0f7a54')
    txt(ax, 18.0, 1.55, r'Steane', size=6.0, color=INK2)
    ax.set_xlim(3.5, xmax)
    ax.set_ylim(0, 8)
    ax.set_xticks([5, 7, 10, 15, 18])
    ax.set_yticks(range(0, 9))
    ax.set_xlabel(r'$X$ distance $d_X$ of the blocks', labelpad=1)
    ax.set_ylabel(r'Exact fault distance', labelpad=2)
    ax.grid(axis='y', color=mp.GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)
    plot_title(ax, 'b', 'Exact fault distance')
    hs = [Line2D([], [], color=INK2, marker='s', ms=4.2, mec='white', mew=0.5, ls='none'),
          Line2D([], [], color=INK2, marker='s', ms=4.0, mfc='white', mew=0.9, ls='none'),
          Line2D([], [], color=INK2, lw=2.4, alpha=0.45)]
    ax.legend(hs, ['colocated', 'routed', 'proved bounds'], loc='upper right', fontsize=6.3, handlelength=1.2,
              bbox_to_anchor=(1.0, 0.86))


def panel_lambda(ax, rows, get):
    for basis, col, mk, lab in (('X', BLUE, 'o', r'$X$ experiment'), ('Z', ORANGE, 's', r'$Z$ experiment')):
        x, y, lo, hi = [], [], [], []
        for p in sorted({a['p'] for a in rows}):
            pts = []
            for s in SPACINGS:
                a = get('thick', basis, s, p)
                if a is not None:
                    pts.append((s, a['errors'], a['shots']))
            f = A.fit_lambda(pts)
            if f:
                x.append(p * 1e3)
                y.append(f[0])
                lo.append(f[1])
                hi.append(f[2])
        series(ax, x, y, lo, hi, col, mk, ms=4.2, lw=1.2)
    ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    txt(ax, 0.47, 0.86, r'no gain', size=6.2, color=INK2, ha='left')
    ax.set_xscale('log')
    log_y(ax, 0.6, 100)
    ticks = [0.5, 1, 2, 3, 4, 6]
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.set_xlim(0.42, 7.2)
    ax.set_xlabel(r'Noise strength $p$ $(10^{-3})$', labelpad=1)
    ax.set_ylabel(r'Gain $\Lambda$ per unit of $s$', labelpad=1)
    plot_title(ax, 'c', 'Grid: gain per size step')
    ax.legend([handle(BLUE, 'o'), handle(ORANGE, 's')], [r'$X$ experiment', r'$Z$ experiment'],
              loc='upper right', fontsize=6.3, handlelength=1.5)


def main():
    rows, get = campaign()
    W, H = TW, 2.42
    fig = plt.figure(figsize=(W, H))
    left, right, gap, bottom, top = 0.50, 0.05, 0.50, 0.50, 0.26
    widths = np.array([1.12, 1.0, 0.95])
    pw = (W - left - right - 2 * gap) * widths / widths.sum()
    ph = H - bottom - top
    xs = [left, left + pw[0] + gap, left + pw[0] + pw[1] + 2 * gap]
    panel_failures(plot_axes(fig, xs[0], bottom, pw[0], ph), get)
    panel_distance(plot_axes(fig, xs[1], bottom, pw[1], ph))
    panel_lambda(plot_axes(fig, xs[2], bottom, pw[2], ph), rows, get)
    save(fig, 'two_block_main')


if __name__ == '__main__':
    main()
