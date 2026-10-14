#!/usr/bin/env python3
"""Draw the schematic that explains why the measurement keeps the distance.

Writes figures/fault_distance_intuition.pdf (and a 600 dpi PNG preview),
the main-text figure of Sec. III B.  It is a schematic: it carries no data.
The style, fonts and colours are those of make_publication_figures.py,
whose helpers it imports; that import only defines functions and settings.

Usage:  python make_intuition_figure.py
"""
import numpy as np
from matplotlib.patches import Rectangle

import make_publication_figures as mp
from make_publication_figures import (AQUA, BLUE, INK, INK2, MUTED, VIOLET,
                                      TW, canvas, title, txt, tint, arrow,
                                      leader, save)
import matplotlib.pyplot as plt

H = 1.98
BX = (0.12, 1.30)          # left edges of blocks a and b
S = 0.84                   # block side
Y0 = 0.62                  # bottom of the blocks
YP = Y0 + 0.50 * S         # height of the X slab port


def blocks(ax):
    for x0, name in zip(BX, 'ab'):
        ax.add_patch(Rectangle((x0, Y0), S, S, fc='none', ec=mp.FRAME,
                               lw=0.7, zorder=4))
        txt(ax, x0 + S / 2, Y0 - 0.12, rf'block ${name}$', size=7,
            color=INK2)


def ports(ax):
    for x0 in BX:
        ax.plot([x0 + 0.05, x0 + S - 0.05], [YP, YP], color=BLUE, lw=1.9,
                zorder=7, solid_capstyle='butt')


def mesh(ax, x0, x1, y0, y1, step=0.07, lw=0.35, alpha=1.0):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           fc=tint(AQUA, 0.86), ec=AQUA, lw=0.6, zorder=2))
    for x in np.arange(x0 + step, x1 - 1e-9, step):
        ax.plot([x, x], [y0, y1], color=tint(AQUA, 0.45), lw=lw, zorder=3)
    for y in np.arange(y0 + step, y1 - 1e-9, step):
        ax.plot([x0, x1], [y, y], color=tint(AQUA, 0.45), lw=lw, zorder=3)


def wall(ax, x, y0, y1):
    ax.plot([x, x], [y0, y1], color=VIOLET, lw=1.5, ls=(0, (3.0, 1.6)),
            zorder=8, dash_capstyle='butt')


def fault(ax, x, y, size=0.026, color=INK):
    ax.plot([x - size, x + size], [y - size, y + size], color=color, lw=0.9,
            zorder=8, solid_capstyle='round')
    ax.plot([x - size, x + size], [y + size, y - size], color=color, lw=0.9,
            zorder=8, solid_capstyle='round')


def fig_fault_distance_intuition():
    fig = plt.figure(figsize=(TW, H))
    xm = BX[0] + S + (BX[1] - BX[0] - S) / 2      # middle of the gap
    right = BX[1] + S

    # (a) a thin interface has a short wall
    ax = canvas(fig, 0.0, 0.0, 2.32, H)
    title(ax, 'a', 'Thin interface: a short wall')
    blocks(ax)
    h = 0.07
    mesh(ax, BX[0], right, YP - h, YP + h, step=0.07)
    ports(ax)
    wall(ax, xm, YP - h - 0.10, YP + h + 0.10)
    txt(ax, xm, Y0 + S + 0.12, r"wall: few edges", size=7, color=VIOLET)
    p = (BX[0] + 0.30, YP)
    leader(ax, p, (BX[0] + 0.30, Y0 + S + 0.16))
    txt(ax, BX[0] + 0.30, Y0 + S + 0.22, r'port $\overline X_a$', size=7,
        color=BLUE, va='bottom')
    txt(ax, 1.16, 0.30, r'$X(\text{wall})\simeq\overline X_a$ after the '
        r'measurement,', size=7)
    txt(ax, 1.16, 0.12, r'so $\widetilde d_X\le$ (edges in the wall)',
        size=7)

    # (b) a grid that fills both blocks has only large walls
    ax = canvas(fig, 2.40, 0.0, 2.32, H)
    title(ax, 'b', 'Spanning grid: every wall is large')
    blocks(ax)
    mesh(ax, BX[0], right, Y0, Y0 + S, step=0.07)
    ports(ax)
    wall(ax, xm, Y0 - 0.04, Y0 + S + 0.04)
    txt(ax, xm, Y0 + S + 0.14, r'wall: a whole cross-section', size=7,
        color=VIOLET)
    txt(ax, 1.16, 0.30, r'$\Theta(N^2)$ edges, the scale of $d$;', size=7)
    txt(ax, 1.16, 0.12, r'$m_0$ ancillas per edge give $\widetilde d_X\ge d_X$',
        size=7)

    # (c) time: repeated rounds and idle padding
    ax = canvas(fig, 4.80, 0.0, TW - 4.80, H)
    title(ax, 'c', 'Time: repetition and padding')
    t0, t1, t2, t3, t4 = 0.10, 0.50, 1.40, 1.80, 2.18
    yb, yt = 0.60, 1.50
    for a, b, kind in ((t0, t1, 'pad'), (t1, t2, 'gauge'), (t2, t3, 'pad'),
                       (t3, t4, 'gauge')):
        c = AQUA if kind == 'gauge' else MUTED
        ax.add_patch(Rectangle((a, yb), b - a, yt - yb,
                               fc=tint(c, 0.86 if kind == 'gauge' else 0.80),
                               ec='none', zorder=1))
    for a in (t1, t2, t3):
        ax.plot([a, a], [yb, yt], color=INK2, lw=0.5, zorder=2)
    # repeated rounds inside the first window
    for t in np.linspace(t1, t2, 8)[1:-1]:
        ax.plot([t, t], [yb, yt], color=tint(AQUA, 0.45), lw=0.35, zorder=2)
    # a spacelike logical: many faults at one time
    for y in np.linspace(1.00, 1.42, 5):
        fault(ax, 0.66, y, size=0.022)
    txt(ax, 0.75, 1.27, r'$\ge\widetilde d$', size=7, ha='left')
    txt(ax, 0.75, 1.15, 'at one time', size=6.5, ha='left', color=INK2)
    # a timelike logical: one outcome wrong in every round
    ts = np.linspace(t1, t2, 8)
    for a, b in zip(ts[:-1], ts[1:]):
        fault(ax, (a + b) / 2, 0.72, size=0.022)
    txt(ax, (t1 + t2) / 2 + 0.06, 0.84, r'$\ge R$ in a row', size=7)
    # a pattern joining two windows must cross the pad
    arrow(ax, (t2 - 0.04, 0.98), (t3 + 0.04, 0.98), color=VIOLET, lw=0.8,
          head=5, style='<|-|>')
    txt(ax, (t2 + t3) / 2, 1.09, r'$\gtrsim R_{\rm pad}$', size=7,
        color=VIOLET)
    # labels and time axis
    txt(ax, (t1 + t2) / 2, yt + 0.10, 'measurement', size=7, color=INK2)
    txt(ax, (t0 + t1) / 2, yt + 0.10, 'idle', size=7, color=INK2)
    txt(ax, (t2 + t3) / 2, yt + 0.10, 'idle', size=7, color=INK2)
    txt(ax, (t3 + t4) / 2 + 0.02, yt + 0.10, 'next', size=7, color=INK2)
    arrow(ax, (t0, 0.44), (t4 + 0.02, 0.44), color=INK2, lw=0.6, head=5)
    txt(ax, t4 + 0.02, 0.33, 'time', size=7, color=INK2, ha='right')
    txt(ax, (t1 + t2) / 2, 0.33, r'$R$ rounds', size=7, color=INK2)
    txt(ax, (t2 + t3) / 2, 0.33, r'$R_{\rm pad}$', size=7, color=INK2)
    txt(ax, 0.04, (yb + yt) / 2, 'space', size=7, color=INK2, rotation=90)
    txt(ax, (t0 + t4) / 2, 0.12, r'fault distance $\gtrsim\min\{\widetilde d,R,'
        r'R_{\rm pad}/\tau\}$', size=7)
    save(fig, 'fault_distance_intuition')


if __name__ == '__main__':
    mp.OUT.mkdir(exist_ok=True)
    fig_fault_distance_intuition()
