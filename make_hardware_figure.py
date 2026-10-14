#!/usr/bin/env python3
"""Draw the three hardware figures from hardware/hardware_summary.json.

figures/hardware_overview  main text.  Top row: what was run (schematics that
                           carry no data, and the placement of the
                           cut-obstruction circuit on IQM Emerald).  Bottom
                           row: what was measured, with IQM runs pooled.
figures/hardware_results   supplement: every run separately.
figures/hardware_chips     supplement: every fixed placement on the chips.

Every number is read from hardware_summary.json; placements are read from the
placement files and the used couplers from the submitted circuit files.  The
chip coordinates are derived from the coupling maps in the archived backend
description (IQM) and calibration snapshot (IBM).  Style, fonts and colours
are those of make_publication_figures.py.

Usage:  python hardware/hardware_summary.py && python make_hardware_figure.py
Writes: figures/hardware_{overview,results,chips}.pdf and 600 dpi PNG previews.
"""
import json
import re
from collections import defaultdict
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.ticker import NullLocator

import make_publication_figures as mp
from make_publication_figures import (AQUA, BLUE, INK, INK2, MAGENTA, MUTED, ORANGE, TW, VIOLET, arrow,
                                      canvas, handle, log_y, plot_axes, plot_title, rbox, save, series,
                                      tint, title, txt, zero_handle)

HW = mp.ROOT / 'hardware'
S = json.loads((HW / 'hardware_summary.json').read_text())
BLUE2, BLUE3 = mp.RAMP[3][0], mp.RAMP[3][2]
LIGHT = '#d5d9de'        # unused qubits
LIGHTER = '#e4e7ea'      # unused couplers
WALL = VIOLET            # the wall (cut) of the cut obstruction, as in the distance figure
LEAK = MAGENTA           # errors that survive a reset


def J(name):
    return json.loads((HW / name).read_text())


# ------------------------------------------------------------- numbers ----
def wilson(f, n, z=1.96):
    p = f / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def iqm_cut(w, control=False):
    """IQM cut-obstruction error at width w, pooled over every run with that width."""
    rows = [v for k, v in S['cut'].items() if k.startswith('IQM') and k.endswith(f'w{w}')]
    if control:
        rows = [r['control'] for r in rows if 'control' in r]
    return wilson(sum(r['failures'] for r in rows), sum(r['kept'] for r in rows)), sum(r['kept'] for r in rows)


def triple(row):
    return row['error'], row['wilson95'][0], row['wilson95'][1]


def fold(a, b):
    r = a / b
    return f'{r:.0f}' if r >= 10 else f'{r:.1f}'


def pct(x, sig=2):
    """Percentage with `sig` significant figures, as pct() in hardware_summary.py."""
    from math import floor, log10
    v = 100 * x
    return f'{v:.{max(0, sig - 1 - int(floor(log10(abs(v)))))}f}' if v else '0'


def persistence():
    """Run B: rate of a check ancilla reading 1 in two consecutive rounds, pooled over R = 2, 3."""
    r = S['rounds']
    out = {}
    for v in ('single', 'double'):
        ps = [r[f'IBM kingston B R{R}_{v}']['persistence'] for R in (2, 3)]
        ev, tr = sum(p['events'] for p in ps), sum(p['trials'] for p in ps)
        out[v] = wilson(ev, tr)
    out['independent'] = float(np.mean([r[f'IBM kingston B R{R}_{v}']['persistence']['independent']
                                        for R in (2, 3) for v in ('single', 'double')]))
    return out


# --------------------------------------------------------------- chips ----
def neighbours(pairs):
    nb = defaultdict(set)
    for a, b in pairs:
        nb[a].add(b)
        nb[b].add(a)
    return nb


def square_lattice_xy(pairs, start):
    """Integer coordinates of a square-lattice chip in which every coupler is a
    unit step (exact search; a few lattice neighbours may lack a coupler)."""
    nb = neighbours(pairs)
    order, seen, queue = [], {start}, [start]
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in sorted(nb[u]):
            if v not in seen:
                seen.add(v)
                queue.append(v)
    steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    pos, taken = {}, set()

    def fits(v, p):
        return p not in taken and all(abs(pos[q][0] - p[0]) + abs(pos[q][1] - p[1]) == 1
                                      for q in nb[v] if q in pos)

    def place(i):
        if i == len(order):
            return True
        v = order[i]
        if i == 0:
            cands = [(0, 0)]
        else:
            u = next(q for q in order[:i] if q in nb[v])
            cands = [(pos[u][0] + dx, pos[u][1] + dy) for dx, dy in (steps[:1] if i == 1 else steps)]
        for p in cands:
            if fits(v, p):
                pos[v] = p
                taken.add(p)
                if place(i + 1):
                    return True
                del pos[v]
                taken.discard(p)
        return False

    assert place(0)
    return pos


def emerald():
    b = J('oq_backend_iqm_emerald.json')['constraint_data']
    pairs = [tuple(p) for p in b['topology']['coupling_map']]
    nb = neighbours(pairs)
    start = max(range(b['n_qubits']),       # an interior qubit: four couplers, four squares
                key=lambda q: (len(nb[q]), sum(1 for u in nb[q] for v in nb[q] if u < v and len(nb[u] & nb[v]) > 1)))
    pos = square_lattice_xy(pairs, start)
    assert len(pos) == b['n_qubits'] == len(set(pos.values()))
    return pos, pairs


def heron(name='ibm3_calibration_ibm_kingston.json'):
    """Heavy-hex Heron r2 (156 qubits): rows of 16 joined by four connector qubits."""
    pairs = sorted({tuple(sorted(p)) for p in J(name)['coupling']})
    nb = neighbours(pairs)
    n = max(max(p) for p in pairs) + 1
    pos = {q: (q % 20, -2 * (q // 20)) for q in range(n) if q % 20 < 16}
    for q in range(n):
        if q % 20 >= 16:
            up = [u for u in nb[q] if u in pos and u // 20 == q // 20]
            pos[q] = (pos[up[0]][0], -2 * (q // 20) - 1)
    assert all(abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) == 1 for a, b in pairs)
    return pos, pairs


def qasm_pairs(name):
    text = (HW / name).read_text()
    return {tuple(sorted(map(int, m))) for m in re.findall(r'c[zx] \$(\d+), \$(\d+);', text)}


def cut_roles(place, w):
    roles = {}
    for k, q in place.items():
        if k[0] in 'ab' and len(k) == 2:
            roles[q] = 'data'
        elif k.startswith('x_'):
            roles[q] = 'meas'
        elif k.startswith('br'):
            if k == 'br0p' or int(k[2]) < w:
                roles[q] = 'bridge'
        else:
            roles[q] = 'edge'
    return roles


def rounds_roles(place):
    return {q: ('data' if k[0] in 'ab' and len(k) == 2 else 'edge' if k == 'e' else 'meas')
            for k, q in place.items()}


def draw_chip(ax, pos, pairs, roles, used, x0, y0, s, rot=False, size=1.0):
    """Chip at lower-left corner (x0, y0) with lattice spacing s (inches)."""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    mx, my = min(xs), min(ys)

    def at(q):
        x, y = pos[q]
        return x0 + s * (x - mx), y0 + s * (y - my)

    for a, b in pairs:
        u = tuple(sorted((a, b))) in used
        ax.plot(*zip(at(a), at(b)), color=INK2 if u else LIGHTER, lw=(0.85 if u else 0.5) * size,
                zorder=3 if u else 1, solid_capstyle='round')
    r = 0.3 * s
    for q in pos:
        x, y = at(q)
        role = roles.get(q)
        if role is None:
            ax.add_patch(Circle((x, y), 0.55 * r, fc=LIGHT, ec='none', zorder=2))
        elif role == 'data':
            ax.add_patch(Circle((x, y), r, fc=BLUE, ec='white', lw=0.4 * size, zorder=4))
        elif role == 'meas':
            ax.add_patch(Circle((x, y), 0.85 * r, fc='white', ec=INK2, lw=0.7 * size, zorder=4))
        else:
            h = 0.85 * r
            ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, fc=AQUA, lw=0.9 * size, zorder=4,
                                   ec=WALL if role == 'bridge' else 'white'))
    return (x0, y0, x0 + s * (max(xs) - mx), y0 + s * (max(ys) - my))


def chip_legend(ax, x, y, size=6.5, gap=0.62, bridge=True):
    items = [('data', 'data qubit'), ('edge', 'edge ancilla'), ('meas', 'measured ancilla')]
    if bridge:
        items.insert(2, ('bridge', 'bridge ancilla'))
    for i, (role, lab) in enumerate(items):
        cx = x + i * gap
        if role == 'data':
            ax.add_patch(Circle((cx, y), 0.035, fc=BLUE, ec='white', lw=0.4, zorder=4))
        elif role == 'meas':
            ax.add_patch(Circle((cx, y), 0.03, fc='white', ec=INK2, lw=0.7, zorder=4))
        else:
            ax.add_patch(Rectangle((cx - 0.03, y - 0.03), 0.06, 0.06, fc=AQUA, lw=0.9, zorder=4,
                                   ec=WALL if role == 'bridge' else 'white'))
        txt(ax, cx + 0.07, y, lab, size=size, color=INK2, ha='left')


# ---------------------------------------------------------- schematics ----
def vertex(ax, x, y, r=0.036):
    ax.add_patch(Circle((x, y), r, fc=BLUE, ec='white', lw=0.5, zorder=5))


def edge_qubit(ax, x, y, h=0.03, wall=False):
    ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, fc=AQUA, ec=WALL if wall else 'white',
                           lw=0.8 if wall else 0.4, zorder=5))


def link(ax, p, q, color=INK2, lw=0.8, ls='-'):
    ax.plot(*zip(p, q), color=color, lw=lw, ls=ls, zorder=3, solid_capstyle='round')


def two_blocks(ax, xa, xb, ys):
    for x, lab in ((xa, r'block $a$'), (xb, r'block $b$')):
        rbox(ax, x - 0.115, ys[2] - 0.13, 0.23, ys[0] - ys[2] + 0.26, ec=tint(BLUE, 0.45), fc=tint(BLUE, 0.9),
             lw=0.6, r=0.06, z=1)
        txt(ax, x, ys[2] - 0.22, lab, size=6.8, color=INK2)
        for i in range(2):
            link(ax, (x, ys[i]), (x, ys[i + 1]))
            edge_qubit(ax, x, (ys[i] + ys[i + 1]) / 2)
        for y in ys:
            vertex(ax, x, y)


def wall_line(ax, x, y0, y1, label=None, ly=None):
    ax.plot([x, x], [y0, y1], color=WALL, lw=1.3, ls=(0, (2.6, 1.6)), zorder=6, dash_capstyle='butt')
    if label:
        txt(ax, x, y1 + 0.08 if ly is None else ly, label, size=6.8, color=WALL)


def schematic_cut(ax, w, h, pos, pairs):
    title(ax, 'a', 'Cut obstruction', y=h)
    xa, xb = 0.28, 1.08
    ys = [1.30, 0.94, 0.58]
    xm = (xa + xb) / 2
    for i in range(2):              # cycle checks of the bridge squares
        ax.add_patch(Rectangle((xa, ys[i + 1]), xb - xa, ys[i] - ys[i + 1], fc=tint(AQUA, 0.84), ec='none',
                               zorder=0))
    two_blocks(ax, xa, xb, ys)
    for y in ys:
        link(ax, (xa, y), (xb, y))
        edge_qubit(ax, xm, y, wall=True)
    txt(ax, xm + 0.2, (ys[1] + ys[2]) / 2, r'cycle' '\n' r'check', size=6, color=tint(INK2, 0.1),
        linespacing=0.95)
    wall_line(ax, xm, ys[2] - 0.17, ys[0] + 0.17, r'wall: $w$ bridge edges')
    txt(ax, xm, 0.13, r'$X(\mathrm{wall})\simeq\overline X_a$, so $\widetilde d_X=w$', size=7)

    # the same circuit on the chip (width 3, runs 1 and 3)
    place = J('iqm_robust_placement.json')[0]
    xs = [p[0] for p in pos.values()]
    yy = [p[1] for p in pos.values()]
    s = min((w - 1.50) / (max(xs) - min(xs)), 1.0 / (max(yy) - min(yy)))
    cw, chh = s * (max(xs) - min(xs)), s * (max(yy) - min(yy))
    x0 = w - cw - 0.03
    y0 = 0.50
    draw_chip(ax, pos, pairs, cut_roles(place, 3), qasm_pairs('qasm3_iqm_w3.qasm'), x0, y0, s, size=0.8)
    txt(ax, x0 + cw / 2, y0 - 0.15, r'on IQM Emerald', size=6.8, color=INK2)
    txt(ax, x0 + cw / 2, y0 - 0.29, r'19 of 54 qubits, no SWAP', size=6.2, color=MUTED)
    arrow(ax, (xb + 0.17, ys[1]), (x0 - 0.04, ys[1]), color=MUTED, lw=0.6, head=5)


def schematic_parallel(ax, w, h):
    title(ax, 'b', 'Parallel ancillas', y=h)
    xa, xb = 0.17, w - 0.17
    ys = [1.30, 0.94, 0.58]
    xm = (xa + xb) / 2
    two_blocks(ax, xa, xb, ys)
    y0, bow = ys[0], 0.15
    t = np.linspace(0, 1, 60)
    curves = []
    for sgn in (1, -1):             # quadratic Bezier arcs between the two vertices
        x = (1 - t) ** 2 * xa + 2 * t * (1 - t) * xm + t ** 2 * xb
        y = y0 + 2 * t * (1 - t) * sgn * bow
        curves.append((x, y))
        ax.plot(x, y, color=INK2, lw=0.8, zorder=3, solid_capstyle='round')
        edge_qubit(ax, xm, y0 + sgn * bow / 2, wall=True)
    lens = np.vstack([np.column_stack(curves[0]), np.column_stack(curves[1])[::-1]])
    ax.add_patch(Polygon(lens, closed=True, fc=tint(AQUA, 0.78), ec='none', zorder=0))
    wall_line(ax, xm, y0 - 0.2, y0 + 0.2, r'$m_0=2$ on one edge', ly=y0 + 0.3)
    txt(ax, xm, ys[1] - 0.02, r'new check' '\n' r'$Z_eZ_{e^\prime}$', size=6.2, color=INK2, linespacing=1.0)
    txt(ax, xm, 0.13, r'$\widetilde d_X=2$ with the right gate order', size=6.6)


def schematic_rounds(ax, w, h):
    title(ax, 'c', 'Repeated rounds', y=h)
    x0, sw, gap = 0.66, 0.40, 0.15
    slots = [x0 + i * (sw + gap) for i in range(3)]
    xend = slots[-1] + sw
    lanes = [(1.12, r'one fault', [1, 0, 0], AQUA, r'caught'),
             (0.66, r'persistent' '\n' r'fault', [1, 1, 1], LEAK, r'missed')]
    ytop, ybot = 1.36, 0.42
    for i, sx in enumerate(slots):
        rbox(ax, sx, ybot, sw, ytop - ybot, ec=tint(INK2, 0.55), fc=tint(INK2, 0.94), lw=0.5, r=0.04, z=1)
        txt(ax, sx + sw / 2, ytop + 0.09, f'round {i + 1}', size=6.6, color=INK2)
    for i in range(2):
        gx = slots[i] + sw + gap / 2
        ax.plot([gx, gx], [ybot + 0.04, ytop - 0.04], color=MUTED, lw=0.5, ls=(0, (1, 1.2)), zorder=1)
    txt(ax, slots[0] + sw + gap / 2, ybot - 0.09, r'reset', size=6.2, color=MUTED)
    txt(ax, slots[1] + sw + gap / 2, ybot - 0.09, r'reset', size=6.2, color=MUTED)
    for y, lab, bits, col, verdict in lanes:
        txt(ax, 0.0, y, lab, size=6.8, ha='left', color=INK, linespacing=0.95)
        ax.plot([slots[0] - 0.05, xend + 0.05], [y, y], color=INK2, lw=0.5, zorder=2)
        if col == LEAK:
            ax.add_patch(Rectangle((slots[0] + 0.03, y - 0.075), xend - slots[0] - 0.06, 0.15,
                                   fc=tint(LEAK, 0.55), ec='none', zorder=1.5))
        for sx, b in zip(slots, bits):
            wrong = b == 1
            ax.add_patch(Circle((sx + sw / 2, y), 0.068, fc='white', ec=LEAK if wrong else INK2,
                                lw=0.9 if wrong else 0.5, zorder=3))
            txt(ax, sx + sw / 2, y - 0.004, f'${b}$', size=6.8, color=LEAK if wrong else INK, zorder=4)
        txt(ax, xend + 0.1, y, verdict, size=6.8, ha='left', color=tint(col, 0.0) if col != AQUA else '#0f7a54')
    txt(ax, w / 2, 0.13, r'star outcome in each round: a change reveals a fault', size=6.6)


# ------------------------------------------------------------ overview ----
def data_cut(ax):
    c = S['cut']
    ws = np.array([1, 2, 3])
    iq = [iqm_cut(w)[0] for w in ws]
    y, lo, hi = map(np.array, zip(*iq))
    ib = [triple(c[f'IBM fez w{w}']) for w in ws]
    yb, lob, hib = map(np.array, zip(*ib))
    ctrl_q = [iqm_cut(w, control=True)[0][0] for w in (2, 3)]
    ctrl_b = [c[f'IBM fez w{w}']['control']['error'] for w in (2, 3)]
    for yy, col, mk in ((ctrl_q, BLUE, 'o'), (ctrl_b, ORANGE, 'D')):
        ax.plot([2, 3], yy, ls=(0, (2, 1.5)), lw=0.7, color=tint(col, 0.35), marker=mk, ms=3.6, mfc='white',
                mec=col, mew=0.8, zorder=2)
    series(ax, ws, yb, lob, hib, ORANGE, 'D', ms=4.2, lw=1.2)
    series(ax, ws, y, lo, hi, BLUE, 'o', ms=4.4, lw=1.2)
    ax.set_xlim(0.6, 3.4)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel(r'Bridge width $w$ (edges in the wall)')
    ax.set_ylabel(r'Error after postselection')
    plot_title(ax, 'd', 'Wider bridge, fewer errors')
    kept3 = iqm_cut(3)[1]
    txt(ax, 3.0, hi[2] * 0.42, r'0 of %s' % f'{kept3:,}'.replace(',', '{,}'), size=6.2, color=BLUE)
    gm = lambda a, b: sqrt(a * b)
    txt(ax, 1.38, gm(y[0], y[1]) * 0.55, r'$%s\times$' % fold(y[0], y[1]), size=7.2, color=BLUE, ha='right')
    txt(ax, 1.6, gm(yb[0], yb[1]) * 1.9, r'$%s\times$' % fold(yb[0], yb[1]), size=7.2, color=ORANGE, ha='left')
    txt(ax, 2.6, gm(yb[1], yb[2]) * 1.9, r'$%s\times$' % fold(yb[1], yb[2]), size=7.2, color=ORANGE, ha='left')
    txt(ax, 2.5, 0.165, r'cycle checks ignored', size=6.3, color=INK2)
    hs = [handle(BLUE, 'o'), handle(ORANGE, 'D'), zero_handle(BLUE)]
    ax.legend(hs, [r'IQM Emerald (runs pooled)', r'IBM \texttt{ibm\_fez}', r'no failure (95\% bound)'],
              loc='lower left', fontsize=6.3, handlelength=1.5)


def data_parallel(ax):
    one = iqm_cut(1)[0]
    two = iqm_cut(2)[0]
    m = S['multiplicity']['IQM w1 m2']
    pts = [one, triple(m), two]
    for i, (col, (v, lo, hi)) in enumerate(zip([BLUE, AQUA, BLUE], pts)):
        series(ax, [i], [v], [lo], [hi], col, 'o', ms=4.4, lw=1.2)
    ax.plot([1], [m['control']['error']], ls='none', marker='o', ms=3.6, mfc='white', mec=AQUA, mew=0.8, zorder=2)
    txt(ax, 1.17, m['control']['error'], r'check' '\n' r'ignored', size=6, color=INK2, ha='left',
        linespacing=0.95)
    arrow(ax, (0.14, one[0] * 0.8), (0.86, m['error'] * 1.25), color=AQUA, lw=0.8, head=5)
    txt(ax, 0.42, sqrt(one[0] * m['error']) * 0.5, r'$%s\times$' % fold(one[0], m['error']), size=7.2,
        color='#0f7a54', ha='center')
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r'one' '\n' r'ancilla', r'two' '\n' r'ancillas', r'two' '\n' r'edges'], linespacing=1.0)
    ax.tick_params(labelleft=False)
    ax.xaxis.set_minor_locator(NullLocator())
    plot_title(ax, 'e', 'Doubled edge')


def data_rounds(ax):
    r = S['rounds']
    iq = [triple(r[f'IQM fresh R{R}']) for R in (1, 2)]
    ib = [triple(r[f'IBM kingston A R{R}']) for R in (1, 2, 3)]
    mq = [r[f'IQM fresh R{R}']['model'] for R in (1, 2)]
    mb = [r[f'IBM kingston A R{R}']['model'] for R in (1, 2, 3)]
    yb = np.array([v[0] for v in ib])
    ax.fill_between([1, 2, 3], mb, yb, color=tint(LEAK, 0.72), lw=0, zorder=0.5)
    ax.plot([1, 2, 3], mb, color=ORANGE, lw=0.8, ls=(0, (1.2, 1.3)), zorder=1)
    ax.plot([1, 2], mq, color=BLUE, lw=0.8, ls=(0, (1.2, 1.3)), zorder=1)
    for rows, col, mk, Rs in ((ib, ORANGE, 'D', [1, 2, 3]), (iq, BLUE, 'o', [1, 2])):
        y, lo, hi = map(np.array, zip(*rows))
        series(ax, Rs, y, lo, hi, col, mk, ms=4.4 if mk == 'o' else 4.2, lw=1.2)
    txt(ax, 2.62, sqrt(yb[2] * mb[2]) * 1.6, r'persistent' '\n' r'errors', size=6.4, color='#b0476f',
        linespacing=0.95)
    txt(ax, 2.2, mb[1] * (mb[2] / mb[1]) ** 0.2 * 0.38, r'model', size=6.2, color=ORANGE, ha='left')
    txt(ax, 1.55, sqrt(iq[0][0] * iq[1][0]) * 0.45, r'$%s\times$' % fold(iq[0][0], iq[1][0]), size=7.2,
        color=BLUE, ha='right')
    ax.set_xlim(0.6, 3.4)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel(r'Rounds $R$')
    ax.tick_params(labelleft=False)
    plot_title(ax, 'f', 'More rounds')
    hs = [handle(BLUE, 'o'), handle(ORANGE, 'D'),
          Line2D([], [], color=INK2, lw=0.8, ls=(0, (1.2, 1.3)))]
    ax.legend(hs, [r'IQM, fresh ancillas', r'IBM \texttt{ibm\_kingston}, reset', r'calibrated Pauli model'],
              loc='lower left', fontsize=6.3, handlelength=1.5)


def data_persistence(ax):
    P = persistence()
    vals = [P['independent'], P['single'][0], P['double'][0]]
    cols = [tint(INK2, 0.55), LEAK, tint(LEAK, 0.4)]
    top = 100 * P['single'][2]
    ax.bar([0, 1, 2], [100 * v for v in vals], width=0.62, color=cols, zorder=2)
    for i, (v, lo, hi) in enumerate([(P['independent'], None, None), P['single'], P['double']]):
        if lo is not None:
            ax.errorbar([i], [100 * v], yerr=[[100 * (v - lo)], [100 * (hi - v)]], color=INK, lw=0, elinewidth=0.6,
                        capsize=1.5, capthick=0.6, zorder=3)
        txt(ax, i, 100 * (hi if hi else v) + 0.04 * top, pct(v, 2 if lo is None else 3) + r'\%', size=6.2,
            va='bottom', color=INK)
    yb = 1.3 * top                  # bracket from the independent rate to the one-reset rate
    ax.plot([0, 0, 1, 1], [100 * vals[0] + 0.2 * top, yb, yb, top + 0.2 * top], color='#b0476f', lw=0.6)
    txt(ax, 0.5, yb + 0.03 * top, r'$%d\times$' % round(P['single'][0] / P['independent']), size=7.2,
        color='#b0476f', va='bottom')
    txt(ax, 2.6, 1.78 * top, r'ancilla reads 1' '\n' r'in two rounds', size=6.1, color=INK2, ha='right',
        va='top', linespacing=0.95)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 1.8 * top)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r'indep.' '\n' r'errors', r'one' '\n' r'reset', r'two' '\n' r'resets'], linespacing=1.0,
                       fontsize=6.4)
    ax.tick_params(left=False, right=False, top=False, labelleft=False, length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_minor_locator(NullLocator())
    plot_title(ax, 'g', 'Survives reset')


def overview():
    pos, pairs = emerald()
    W, H = TW, 4.32
    fig = plt.figure(figsize=(W, H))
    left, right, gap = 0.47, 0.03, 0.22
    avail = W - left - right - 2 * gap
    cw = np.array([2.18, 1.17, 2.71])
    cw = cw * avail / cw.sum()
    xs = [left, left + cw[0] + gap, left + cw[0] + cw[1] + 2 * gap]
    yb, hb = 0.43, 1.66
    ys = yb + hb + 0.36
    ht = H - ys - 0.02
    # top row: schematics (the first starts at the y-label margin)
    a = canvas(fig, 0.06, ys, xs[0] + cw[0] - 0.06, ht)
    schematic_cut(a, xs[0] + cw[0] - 0.06, ht - 0.02, pos, pairs)
    schematic_parallel(canvas(fig, xs[1], ys, cw[1], ht), cw[1], ht - 0.02)
    schematic_rounds(canvas(fig, xs[2], ys, cw[2], ht), cw[2], ht - 0.02)
    # bottom row: data
    ax = plot_axes(fig, xs[0], yb, cw[0], hb)
    log_y(ax, 1e-5, 0.3)
    data_cut(ax)
    ax = plot_axes(fig, xs[1], yb, cw[1], hb)
    log_y(ax, 1e-5, 0.3)
    data_parallel(ax)
    wp = 0.98
    ax = plot_axes(fig, xs[2], yb, cw[2] - wp - 0.12, hb)
    log_y(ax, 1e-5, 0.3)
    data_rounds(ax)
    data_persistence(plot_axes(fig, xs[2] + cw[2] - wp, yb, wp, hb))
    save(fig, 'hardware_overview')


# ------------------------------------------------------- supplementary ----
def hollow(ax, x, y, color, marker):
    ax.plot(x, y, ls=(0, (2, 1.5)), lw=0.7, color=tint(color, 0.35), marker=marker, ms=3.6, mfc='white',
            mec=color, mew=0.8, zorder=2)


def pts(keys, xs, table):
    rows = [table[k] for k in keys]
    y = np.array([r['error'] for r in rows])
    lo = np.array([r['wilson95'][0] for r in rows])
    hi = np.array([r['wilson95'][1] for r in rows])
    return np.asarray(xs, float), y, lo, hi


def results():
    W, H = TW, 2.45
    fig = plt.figure(figsize=(W, H))
    left, right, gap, bottom, top = 0.50, 0.06, 0.30, 0.42, 0.30
    widths = np.array([1.25, 0.85, 1.0])
    pw = (W - left - right - 2 * gap) * widths / widths.sum()
    ph = H - bottom - top
    xs = [left, left + pw[0] + gap, left + pw[0] + pw[1] + 2 * gap]
    axs = [plot_axes(fig, xs[i], bottom, pw[i], ph) for i in range(3)]
    for ax in axs:
        log_y(ax, 1e-5, 0.2)
    c = S['cut']

    ax = axs[0]
    sets = [('IQM run 1', [1, 2, 3], BLUE, 'o', -0.09), ('IQM run 2', [1, 2], BLUE2, 's', -0.03),
            ('IQM run 3', [2, 3], BLUE3, '^', 0.03), ('IBM fez', [1, 2, 3], ORANGE, 'D', 0.09)]
    for name, ws, col, mk, dx in sets:
        x, y, lo, hi = pts([f'{name} w{w}' for w in ws], [w + dx for w in ws], c)
        series(ax, x, y, lo, hi, col, mk)
        cw = [w for w in ws if 'control' in c[f'{name} w{w}']]
        if cw:
            hollow(ax, [w + dx for w in cw], [c[f'{name} w{w}']['control']['error'] for w in cw], col, mk)
    ax.set_xlim(0.6, 3.4)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel(r'Bridge width $w$')
    ax.set_ylabel('Error after postselection')
    plot_title(ax, 'a', 'Cut obstruction, every run')
    hs = [handle(BLUE, 'o'), handle(BLUE2, 's'), handle(BLUE3, '^'), handle(ORANGE, 'D'),
          Line2D([], [], color=tint(INK2, 0.35), ls=(0, (2, 1.5)), lw=0.7, marker='o', ms=3.6, mfc='white',
                 mec=INK2, mew=0.8), zero_handle()]
    ax.legend(hs, ['IQM run 1', 'IQM run 2', 'IQM run 3', r'IBM \texttt{ibm\_fez}', 'cycle checks ignored',
                   'no failure (95\\% bound)'], loc='lower left', fontsize=6.3, ncol=2, columnspacing=0.8,
              handlelength=1.5)

    ax = axs[1]
    one, two = iqm_cut(1)[0], iqm_cut(2)[0]
    m = S['multiplicity']['IQM w1 m2']
    for i, (col, (v, lo, hi)) in enumerate(zip([BLUE, AQUA, BLUE], [one, triple(m), two])):
        series(ax, [i], [v], [lo], [hi], col, 'o')
    hollow(ax, [1], [m['control']['error']], AQUA, 'o')
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r'$w=1$', r'$w=1$' '\n' r'$m_0=2$', r'$w=2$'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(labelleft=False)
    plot_title(ax, 'b', 'Parallel ancillas')

    ax = axs[2]
    r = S['rounds']
    rs = [('IQM fresh R{}', [1, 2], BLUE, 'o', -0.09, 'IQM, fresh ancillas'),
          ('IBM kingston A R{}', [1, 2, 3], ORANGE, 'D', -0.03, 'IBM run A, one reset'),
          ('IBM kingston B R{}_single', [2, 3], tint(ORANGE, 0.35), 's', 0.03, 'IBM run B, one reset'),
          ('IBM kingston B R{}_double', [2, 3], LEAK, '^', 0.09, 'IBM run B, two resets')]
    hs, labs = [], []
    for fmt, Rs, col, mk, dx, lab in rs:
        keys = [fmt.format(R) for R in Rs]
        if 'B' in fmt:      # run B shares one R = 1 circuit between the two variants
            keys = ['IBM kingston B R1'] + keys
            Rs = [1] + Rs
        x, y, lo, hi = pts(keys, [R + dx for R in Rs], r)
        series(ax, x, y, lo, hi, col, mk)
        hs.append(handle(col, mk))
        labs.append(lab)
    ax.set_xlim(0.6, 3.4)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel(r'Rounds $R$')
    ax.tick_params(labelleft=False)
    plot_title(ax, 'c', 'Repeated rounds, every run')
    ax.legend(hs, labs, loc='lower left', fontsize=6.3, handlelength=1.5)
    save(fig, 'hardware_results')


def chips():
    epos, epairs = emerald()
    hpos, hpairs = heron()
    W, H = TW, 3.95
    fig = plt.figure(figsize=(W, H))
    # top row: four IQM Emerald placements
    iqm = [('a', r'Cut, runs 1 and 3', cut_roles(J('iqm_robust_placement.json')[0], 3),
            qasm_pairs('qasm3_iqm_w3.qasm')),
           ('b', r'Cut, run 2', cut_roles(J('iqm_second_placement.json')[0], 2),
            qasm_pairs('qasm3_iqm2_w2.qasm')),
           ('c', r'Parallel ancillas', cut_roles(J('iqm_m2_placement.json')[0], 1),
            qasm_pairs('qasm3_iqm_m2.qasm')),
           ('d', r'Rounds, two copies',
            {**rounds_roles(J('iqm_rounds_two_copies.json')['A']), **rounds_roles(J('iqm_rounds_two_copies.json')['B'])},
            qasm_pairs('qasm3_iqm_rounds_R2.qasm'))]
    pw, top_h, x0 = (W - 0.1) / 4, 1.62, 0.05
    xs = [p[0] for p in epos.values()]
    ys = [p[1] for p in epos.values()]
    s = min((pw - 0.3) / (max(xs) - min(xs)), (top_h - 0.3) / (max(ys) - min(ys)))
    for i, (letter, head, roles, used) in enumerate(iqm):
        ax = canvas(fig, x0 + i * pw, H - top_h, pw, top_h)
        title(ax, letter, head, y=top_h - 0.02, size=8)
        cwid = s * (max(xs) - min(xs))
        assert all(r in roles for p in used for r in p), letter       # every gate acts on a placed qubit
        draw_chip(ax, epos, epairs, roles, used, (pw - cwid) / 2, 0.08, s)
    # bottom row: IBM ibm_kingston, repeated rounds
    ibm = [('e', r'\texttt{ibm\_kingston}, repeated rounds run A', J('ibm2_rounds_plan.json')['layout']),
           ('f', r'\texttt{ibm\_kingston}, repeated rounds run B', J('ibm3_reset_plan.json')['layout'])]
    text = (HW / 'ibm_rounds_hw.py').read_text()
    path_names = json.loads(re.search(r'^PATH = (\[.*?\])', text, re.M).group(1).replace("'", '"'))
    bot_h = H - top_h - 0.35
    hx = [p[0] for p in hpos.values()]
    hy = [p[1] for p in hpos.values()]
    hs_ = min((W / 2 - 0.4) / (max(hx) - min(hx)), (bot_h - 0.3) / (max(hy) - min(hy)))
    hpairset = set(hpairs)
    for i, (letter, head, layout) in enumerate(ibm):
        ax = canvas(fig, x0 + i * W / 2, 0.3, W / 2 - 0.1, bot_h)
        title(ax, letter, head, y=bot_h - 0.02, size=8)
        roles, used = {}, set()
        for path in layout.values():
            for name, q in zip(path_names, path):
                roles[q] = 'data' if name[0] in 'ab' and len(name) == 2 else 'edge' if name == 'e' else 'meas'
            for a, b in zip(path, path[1:]):
                assert tuple(sorted((a, b))) in hpairset, (a, b)      # the path is SWAP-free
                used.add(tuple(sorted((a, b))))
        cwid = hs_ * (max(hx) - min(hx))
        draw_chip(ax, hpos, hpairs, roles, used, (W / 2 - 0.1 - cwid) / 2, 0.05, hs_, size=0.9)
    leg = canvas(fig, 0, 0, W, 0.3)
    chip_legend(leg, 1.3, 0.13, gap=1.15)
    save(fig, 'hardware_chips')


if __name__ == '__main__':
    overview()
    results()
    chips()
