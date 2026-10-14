#!/usr/bin/env python3
"""Draw the junction and wall geometry of the explicit two-sheet realization.

The figure illustrates Sec. "The explicit two-sheet realization" of the
supplement (Table "Junction and wall rules", Lemma "Commuting checks and point
defects") on the smallest instance that has every rule in it: three input
qubits, one Z check Z_j = {q_1, q_2, q_3} and one X check X_i = {q_1, q_2}
(|X_i cap Z_j| = 2, so the input is CSS), spacing s = 3.

  (a) isometric view of the sheets D_1, D_2, D_3, C_j, A_i with the Z-junction
      lines (first, middle, last), the X-junction lines, the crossing line
      l_ij (the wall) and one single-layer X slab;
  (b) the middle Z-junction cell: D_q star + c_y(u), C_j face + d_z(u);
  (c) the wall cell on l_ij: A_i star + sigma_y c_z, C_j face + tau_y a_y,
      and the coefficient path sigma_l, tau_l, ..., sigma_h.

The drawing is schematic, but its coordinates and the wall coefficients are
not invented: the script builds this instance with layer_code_explicit.py,
checks that all stabilizers commute, reads the sheet coordinates from the
builder, and prints in panel (c) the (sigma, tau) values the builder's GF(2)
solve returns.  Style (Latin Modern via LaTeX, print width, colours) is
imported from make_publication_figures.py, so a TeX installation with lmodern
is required.

Usage:  python make_junction_figure.py
Writes: figures/junction_wall_geometry.pdf (vector) and
        figures/junction_wall_geometry.png (600 dpi preview).
Prints a final PASS/FAIL line for the instance checks.
"""
from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon, Rectangle

from make_publication_figures import (  # noqa: F401  (rcParams set on import)
    TW, INK, INK2, MUTED, FRAME, BLUE, ORANGE, AQUA, VIOLET, tint, canvas,
    title, txt, arrow, leader)
from layer_code_explicit import build_layer_code

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
NAME = 'junction_wall_geometry'

SHEET = '#9aa1a9'        # neutral grey for data sheets
SHEET_FILL = 0.10


# ------------------------------------------------------------ instance ----
def tiny_instance(s):
    """Build the three-qubit instance and return geometry in paper indexing
    (Y_q = s q for q = 1..n; the builder uses q = 0..n-1)."""
    HX = np.array([[1, 1, 0]], dtype=np.uint8)
    HZ = np.array([[1, 1, 1]], dtype=np.uint8)
    lc = build_layer_code(HX, HZ, spacing=s)
    g = lc.geometry
    hx = lc.HX.toarray().astype(np.int64)
    hz = lc.HZ.toarray().astype(np.int64)
    commute = not np.any((hx @ hz.T) % 2)
    rep = lc.wall_report[0]
    shift = s                                   # builder y + s = paper y
    Y = [y + shift for y in g['Yq']]
    X, Z = g['Xj'][0], g['Zi'][0]
    lo = max(Y[g['suppZ'][0][0]], Y[g['suppX'][0][0]])
    hi = min(Y[g['suppZ'][0][-1]], Y[g['suppX'][0][-1]])
    sig = [int((y - shift) in rep['wall_stars']) for y in range(lo, hi + 1)]
    tau = [int((y - shift) in rep['wall_plaqs']) for y in range(lo, hi)]
    return dict(Y=Y, X=X, Z=Z, Lx=g['Lx'], Lz=g['Lz'], lo=lo, hi=hi,
                sigma=sig, tau=tau, commute=commute, status=rep['status'],
                common=[int(q) + 1 for q in rep['common']], n=lc.n)


# ---------------------------------------------------------- projection ----
class Proj:
    """Fixed oblique projection, right handed: y to the right, z up, x toward
    the viewer (drawn down and to the left)."""

    def __init__(self, ox, oy, u, kx=0.50, ky=0.30):
        self.ox, self.oy, self.u, self.kx, self.ky = ox, oy, u, kx, ky

    def __call__(self, x, y, z):
        return (self.ox + self.u * (y - self.kx * x),
                self.oy + self.u * (z - self.ky * x))

    def poly(self, pts):
        return [self(*p) for p in pts]


def patch(ax, P, pts, color, alpha, lw=0.6, z=2, ec=None, ls='-'):
    ax.add_patch(Polygon(P.poly(pts), closed=True, fc=(*to_rgb(color), alpha),
                         ec=color if ec is None else ec, lw=lw, ls=ls,
                         zorder=z, joinstyle='round'))


def seg(ax, P, a, b, color, lw=1.0, z=5, ls='-', off=(0.0, 0.0)):
    pa, pb = P(*a), P(*b)
    ax.plot([pa[0] + off[0], pb[0] + off[0]], [pa[1] + off[1], pb[1] + off[1]],
            color=color, lw=lw, ls=ls, zorder=z, solid_capstyle='round',
            dash_capstyle='butt')


def mid(P, a, b, off=(0.0, 0.0)):
    pa, pb = P(*a), P(*b)
    return ((pa[0] + pb[0]) / 2 + off[0], (pa[1] + pb[1]) / 2 + off[1])


def qdot(ax, p, color, r=0.027, z=9, hollow=False):
    ax.add_patch(Circle(p, r, fc='white' if hollow else color, ec=color,
                        lw=0.8 if hollow else 0.4, zorder=z))


def added(ax, p, r=0.036, z=10):
    """An added qubit: violet disc with a white rim."""
    ax.add_patch(Circle(p, r, fc=VIOLET, ec='white', lw=0.6, zorder=z))


LABEL_BOX = dict(boxstyle='round,pad=0.10,rounding_size=0.04', fc='white',
                 ec='none', alpha=0.9)


# -------------------------------------------------------------- panel a ----
def panel_overview(fig, geo, x0, w, H):
    ax = canvas(fig, x0, 0.0, w, H)
    title(ax, 'a', 'Sheets, junction lines and the wall')
    Y, X, Z, Lx, Lz = geo['Y'], geo['X'], geo['Z'], geo['Lx'], geo['Lz']
    # kx*L_x must differ from s, or a sheet's back edge would sit on the
    # next sheet's front edge in projection
    P = Proj(-0.05, 0.80, 0.255, kx=0.40)
    z0 = Z + 1                                   # the drawn X slab [z0, z0+1]
    lo, hi = geo['lo'], geo['hi']

    # data sheets D_q (plane y = Y_q)
    for q, y in enumerate(Y, 1):
        patch(ax, P, [(0, y, 0), (Lx, y, 0), (Lx, y, Lz), (0, y, Lz)], SHEET,
              SHEET_FILL, lw=0.6, z=2, ec=FRAME)
        p = P(0, y, Lz)
        txt(ax, p[0] + 0.03, p[1] + 0.03, rf'$D_{q}$', size=7.5, ha='left',
            va='bottom')
    # Z-check sheet C_j (plane x = X_j, Y_1 <= y <= Y_3)
    patch(ax, P, [(X, Y[0], 0), (X, Y[-1], 0), (X, Y[-1], Lz),
                  (X, Y[0], Lz)], ORANGE, 0.12, lw=0.7, z=3)
    # X-check sheet A_i (plane z = Z_i, Y_1 <= y <= Y_2)
    patch(ax, P, [(0, Y[0], Z), (Lx, Y[0], Z), (Lx, Y[1], Z), (0, Y[1], Z)],
          BLUE, 0.16, lw=0.7, z=4)
    # single-layer X slab: the z-edges spanning [z0, z0+1] on every sheet
    band = dict(color=BLUE, alpha=0.18, lw=0.0, z=3.5)
    for y in Y:
        patch(ax, P, [(0, y, z0), (Lx, y, z0), (Lx, y, z0 + 1),
                      (0, y, z0 + 1)], **band)
    patch(ax, P, [(X, Y[0], z0), (X, Y[-1], z0), (X, Y[-1], z0 + 1),
                  (X, Y[0], z0 + 1)], **band)
    for zz in (z0, z0 + 1):
        for y in Y:
            seg(ax, P, (0, y, zz), (Lx, y, zz), BLUE, lw=0.4, z=3.6,
                ls=(0, (1.0, 1.2)))
        seg(ax, P, (X, Y[0], zz), (X, Y[-1], zz), BLUE, lw=0.4, z=3.6,
            ls=(0, (1.0, 1.2)))
    p = P(0, Y[-1], z0 + 0.5)
    txt(ax, p[0] + 0.05, p[1] + 0.06, r'$X$ slab', size=7, ha='left',
        color=BLUE)
    txt(ax, p[0] + 0.05, p[1] - 0.08, r'$[z_0,z_0{+}1]$', size=6.5,
        ha='left', color=INK2)

    # Z-junction lines (x = X_j, y = Y_q): first, middle, last
    for y in Y:
        seg(ax, P, (X, y, 0), (X, y, Lz), ORANGE, lw=1.7, z=7)
    for y, lab in zip(Y, ('first', 'middle', 'last')):
        p = P(X, y, Lz)
        txt(ax, p[0] - 0.03, p[1] + 0.04, lab, size=6.5, color=ORANGE,
            ha='right', va='bottom')
    # X-junction lines (y = Y_q, z = Z_i) for q in supp X_i
    for y in Y[:2]:
        seg(ax, P, (0, y, Z), (Lx, y, Z), BLUE, lw=1.7, z=7)
    # wall l_ij (x = X_j, z = Z_i, lo <= y <= hi) with switches at X_i cap Z_j
    seg(ax, P, (X, lo, Z), (X, hi, Z), VIOLET, lw=2.4, z=8)
    for y in (lo, hi):
        ax.add_patch(Circle(P(X, y, Z), 0.030, fc='white', ec=VIOLET, lw=0.9,
                            zorder=9))
    # sheet labels
    p = P(X, Y[-1], 0)
    txt(ax, p[0] + 0.03, p[1] - 0.03, r'$C_j$', size=7.5, ha='left',
        va='top', color=ORANGE)
    p = P(Lx, (Y[0] + Y[1]) / 2 + 0.3, Z)
    txt(ax, p[0], p[1] - 0.07, r'$A_i$', size=7.5, va='top', color=BLUE,
        bbox=LABEL_BOX, zorder=10)

    # legend row
    yl = 0.13
    items = [(ORANGE, 1.7, r'$Z$ junction'), (BLUE, 1.7, r'$X$ junction'),
             (VIOLET, 2.4, r'wall $\ell_{ij}$')]
    for xl, (col, lw, lab) in zip((0.02, 0.80, 1.58), items):
        ax.plot([xl, xl + 0.14], [yl, yl], color=col, lw=lw,
                solid_capstyle='butt')
        txt(ax, xl + 0.18, yl, lab, size=7, ha='left')
    ax.add_patch(Circle((2.30, yl), 0.030, fc='white', ec=VIOLET, lw=0.9))
    txt(ax, 2.36, yl, 'switch', size=7, ha='left')

    # axes triad, lower right (free of sheets)
    T = Proj(2.44, 0.56, 0.18, P.kx, P.ky)
    o = T(0, 0, 0)
    for v, lab, dx, dy in (((1.4, 0, 0), 'x', -0.04, -0.02),
                           ((0, 1, 0), 'y', 0.04, 0.0),
                           ((0, 0, 1), 'z', 0.0, 0.04)):
        e = T(*v)
        arrow(ax, o, e, color=INK2, lw=0.6, head=4.5)
        txt(ax, e[0] + dx, e[1] + dy, rf'${lab}$', size=7, color=INK2,
            ha='right' if dx < 0 else ('left' if dx > 0 else 'center'),
            va='bottom' if dy > 0 else ('top' if dy < 0 else 'center'))
    return ax


# -------------------------------------------------------------- panel b ----
def panel_zjunction(fig, x0, w, H):
    ax = canvas(fig, x0, 0.0, w, H)
    title(ax, 'b', r'Middle $Z$ junction at $u$')
    P = Proj(0.98, 1.62, 0.56)
    # screen offset of doubled edges: 2*dd must exceed the radius sum of an
    # added disc and a plain qubit dot (0.036 + 0.027), or the two dots merge
    dd = 0.040
    # D_q patch (y = Y_q) and C_j patch (x = X_j) around u
    patch(ax, P, [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)], SHEET,
          SHEET_FILL, lw=0.5, z=2, ec=FRAME)
    patch(ax, P, [(0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1)], ORANGE,
          0.09, lw=0.5, z=3)
    p = P(-1, 0, 1)
    txt(ax, p[0] + 0.03, p[1] + 0.02, r'$D_q$', size=7.5, ha='left',
        va='bottom')
    p = P(0, -1, 1)
    txt(ax, p[0] + 0.02, p[1] + 0.03, r'$C_j$', size=7.5, ha='left',
        va='bottom', color=ORANGE)
    # the junction z-line carries two qubits per edge: D (left), C (right)
    offD, offC = (-dd, 0.0), (dd, 0.0)
    seg(ax, P, (0, 0, -1), (0, 0, 1), FRAME, lw=0.6, z=4, off=offD)
    seg(ax, P, (0, 0, -1), (0, 0, 1), tint(ORANGE, 0.35), lw=0.6, z=4,
        off=offC)
    # C_j face on [Y_q, Y_q+1] x [z, z+1]
    ax.add_patch(Polygon(P.poly([(0, 0, 0), (0, 1, 0), (0, 1, 1),
                                 (0, 0, 1)]), closed=True,
                         fc=(*to_rgb(ORANGE), 0.16), ec='none', zorder=5))
    for a, b, off in (((0, 0, 0), (0, 1, 0), (0, 0)),
                      ((0, 1, 0), (0, 1, 1), (0, 0)),
                      ((0, 1, 1), (0, 0, 1), (0, 0)),
                      ((0, 0, 1), (0, 0, 0), offC)):
        seg(ax, P, a, b, ORANGE, lw=1.5, z=6, off=off)
    # D_q star at u: two x-edges and the two z-edges of the data line
    for a, b, off in (((0, 0, 0), (1, 0, 0), (0, 0)),
                      ((0, 0, 0), (-1, 0, 0), (0, 0)),
                      ((0, 0, 0), (0, 0, 1), offD),
                      ((0, 0, 0), (0, 0, -1), offD)):
        seg(ax, P, a, b, BLUE, lw=1.5, z=7, off=off)
    # qubits: star only (blue), face only (orange), added and shared (violet)
    for a, b in (((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (-1, 0, 0))):
        qdot(ax, mid(P, a, b), BLUE)
    qdot(ax, mid(P, (0, 0, 0), (0, 0, -1), offD), BLUE)
    for a, b in (((0, 1, 0), (0, 1, 1)), ((0, 1, 1), (0, 0, 1))):
        qdot(ax, mid(P, a, b), ORANGE)
    qdot(ax, mid(P, (0, 0, 0), (0, 0, 1), offC), ORANGE)
    pc = mid(P, (0, 0, 0), (0, 1, 0))
    pd = mid(P, (0, 0, 0), (0, 0, 1), offD)
    added(ax, pc)
    added(ax, pd)
    pu = P(0, 0, 0)
    ax.add_patch(Circle(pu, 0.020, fc=INK, ec='white', lw=0.4, zorder=11))
    txt(ax, pu[0] - 0.05, pu[1] + 0.04, r'$u$', size=7.5, ha='right',
        va='bottom')
    txt(ax, pc[0] + 0.02, pc[1] - 0.07, r'$c_y(u)$', size=7, color=VIOLET,
        va='top')
    txt(ax, pd[0] - 0.07, pd[1] + 0.06, r'$d_z(u)$', size=7, color=VIOLET,
        ha='right', bbox=LABEL_BOX, zorder=10)
    # prefix bits: C_j on either side of the line, D_q on its line
    p = P(0, -0.75, -0.80)
    txt(ax, p[0], p[1], r'$p_{t-1}$', size=6.5, color=INK2)
    p = P(0, 0.75, -0.80)
    txt(ax, p[0], p[1], r'$p_t$', size=6.5, color=INK2)
    p = P(0.75, 0, -0.30)
    txt(ax, p[0], p[1], r'$x_{q_t}$', size=6.5, color=INK2)

    # annotations
    ya = 0.60
    txt(ax, 0.10, ya, r'star $+\,c_y(u)$,\enspace face $+\,d_z(u)$', size=7,
        ha='left')
    txt(ax, 0.10, ya - 0.18, r'$p_{t-1}+p_t+x_{q_t}=0$', size=7.5,
        ha='left')
    txt(ax, 0.10, ya - 0.36, r'word $110$: backbone on, $x_{q_t}=0$',
        size=6.5, color=INK2, ha='left')
    return ax


# -------------------------------------------------------------- panel c ----
def panel_wall(fig, geo, x0, w, H):
    ax = canvas(fig, x0, 0.0, w, H)
    title(ax, 'c', r'Wall cell on $\ell_{ij}$')
    P = Proj(0.95, 1.72, 0.56)
    dd = 0.040                                   # see panel (b)
    # A_i patch (z = 0, horizontal) and C_j patch (x = 0, vertical)
    patch(ax, P, [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], BLUE,
          0.10, lw=0.5, z=2)
    patch(ax, P, [(0, -1, -0.55), (0, 1, -0.55), (0, 1, 1), (0, -1, 1)],
          ORANGE, 0.10, lw=0.5, z=3)
    p = P(1, -1, 0)
    txt(ax, p[0] - 0.03, p[1], r'$A_i$', size=7.5, ha='right',
        color=BLUE)
    p = P(0, 1, 1)
    txt(ax, p[0] + 0.03, p[1], r'$C_j$', size=7.5, ha='left', va='top',
        color=ORANGE)
    # wall line: doubled y-edges, A (below) and C (above)
    offA, offC = (0.0, -dd), (0.0, dd)
    # the wall line as a light band behind both of its qubits, so the offset
    # A and C edges stay visible on top of it
    pw0, pw1 = P(0, -1.25, 0), P(0, 1.25, 0)
    ax.plot([pw0[0], pw1[0]], [pw0[1], pw1[1]], color=tint(VIOLET, 0.62),
            lw=2 * dd * 72 + 1.2, zorder=4, solid_capstyle='butt')
    # C_j face on [y, y+1] x [Z_i, Z_i+1]
    ax.add_patch(Polygon(P.poly([(0, 0, 0), (0, 1, 0), (0, 1, 1),
                                 (0, 0, 1)]), closed=True,
                         fc=(*to_rgb(ORANGE), 0.16), ec='none', zorder=5))
    for a, b, off in (((0, 0, 0), (0, 1, 0), offC),
                      ((0, 1, 0), (0, 1, 1), (0, 0)),
                      ((0, 1, 1), (0, 0, 1), (0, 0)),
                      ((0, 0, 1), (0, 0, 0), (0, 0))):
        seg(ax, P, a, b, ORANGE, lw=1.5, z=6, off=off)
    # A_i star at w: x-edges and the two A y-edges along the wall
    for a, b, off in (((0, 0, 0), (1, 0, 0), (0, 0)),
                      ((0, 0, 0), (-1, 0, 0), (0, 0)),
                      ((0, 0, 0), (0, 1, 0), offA),
                      ((0, 0, 0), (0, -1, 0), offA)):
        seg(ax, P, a, b, BLUE, lw=1.5, z=7, off=off)
    for a, b in (((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (-1, 0, 0))):
        qdot(ax, mid(P, a, b), BLUE)
    qdot(ax, mid(P, (0, 0, 0), (0, -1, 0), offA), BLUE)
    for a, b in (((0, 1, 0), (0, 1, 1)), ((0, 1, 1), (0, 0, 1))):
        qdot(ax, mid(P, a, b), ORANGE)
    qdot(ax, mid(P, (0, 0, 0), (0, 1, 0), offC), ORANGE)
    ps = mid(P, (0, 0, 0), (0, 0, 1))
    pt = mid(P, (0, 0, 0), (0, 1, 0), offA)
    added(ax, ps)
    added(ax, pt)
    pw = P(0, 0, 0)
    ax.add_patch(Circle(pw, 0.020, fc=INK, ec='white', lw=0.4, zorder=11))
    txt(ax, ps[0] - 0.06, ps[1], r'$\sigma_y\,c_z$', size=7, color=VIOLET,
        ha='right')
    txt(ax, pt[0], pt[1] - 0.07, r'$\tau_y\,a_y$', size=7, color=VIOLET,
        va='top', bbox=LABEL_BOX, zorder=10)
    txt(ax, pw[0] - 0.05, pw[1] + 0.05, r'$w$', size=7.5, ha='right',
        va='bottom')
    pe = P(0, 1.25, 0)
    txt(ax, pe[0] + 0.06, pe[1], r'$\ell_{ij}$', size=7, color=VIOLET,
        ha='left')

    # coefficient path sigma_l, tau_l, ..., sigma_h from the builder's solve
    sig, tau = geo['sigma'], geo['tau']
    vals, kinds = [], []
    for k in range(len(tau)):
        vals += [sig[k], tau[k]]
        kinds += ['s', 't']
    vals.append(sig[-1])
    kinds.append('s')
    nb = len(vals)
    m = nb // 2
    names = []
    for k in range(nb):
        i = k // 2
        if kinds[k] == 's':
            names.append(r'\sigma_{\ell}' if i == 0 else
                         (r'\sigma_{h}' if k == nb - 1 else
                          rf'\sigma_{{\ell+{i}}}'))
        else:
            names.append(r'\tau_{\ell}' if i == 0 else
                         (r'\tau_{h-1}' if k == nb - 2 else
                          rf'\tau_{{\ell+{i}}}'))
    bw, gap = 0.19, 0.075
    total = nb * bw + (nb - 1) * gap
    bx0 = (w - total) / 2
    by, bh = 0.50, 0.19
    xs = [bx0 + k * (bw + gap) for k in range(nb)]
    for k, (x, v, kd) in enumerate(zip(xs, vals, kinds)):
        ec = BLUE if kd == 's' else ORANGE
        fixed = k in (0, nb - 1)
        ax.add_patch(Rectangle((x, by), bw, bh,
                               fc=tint(VIOLET, 0.72) if v else 'white',
                               ec=ec, lw=1.3 if fixed else 0.7, zorder=4))
        txt(ax, x + bw / 2, by + bh / 2, str(v), size=7, zorder=5)
        txt(ax, x + bw / 2, by - 0.05, rf'${names[k]}$', size=6.5,
            color=INK2, va='top')
        if k < nb - 1:
            ax.plot([x + bw, x + bw + gap], [by + bh / 2] * 2, color=INK2,
                    lw=0.6, zorder=3)
    # fixed ends: short anchors outside the path
    for x, sgn in ((xs[0], -1), (xs[-1] + bw, 1)):
        ax.plot([x, x + sgn * 0.075], [by + bh / 2] * 2, color=INK, lw=0.8,
                zorder=3)
        ax.plot([x + sgn * 0.075] * 2, [by + 0.03, by + bh - 0.03], color=INK,
                lw=1.1, zorder=3, solid_capstyle='butt')
    # switch positions: data qubits of X_i cap Z_j.  q_1 (first incidence of
    # Z_j) couples sigma_l and tau_l; q_2 (middle incidence) sits in the
    # equation that fixes sigma_h through the C_j face on [h, h+1].
    sw = [((xs[0] + bw + xs[1]) / 2, r'$q_1$'),
          (xs[-1] + bw + 0.035, r'$q_2$')]
    for x, lab in sw:
        ax.plot([x, x], [by - 0.015, by + bh + 0.045], color=VIOLET, lw=1.1,
                zorder=6, solid_capstyle='butt')
        txt(ax, x, by + bh + 0.06, lab, size=6.5, color=VIOLET, va='bottom')
    txt(ax, w / 2, by + bh + 0.26, r'fixed ends, a switch at each '
        r'$q\in X_i\cap Z_j$', size=6.5, color=INK2)
    txt(ax, w / 2, 0.10, r'solvable iff $|X_i\cap Z_j|$ is even', size=7.5)
    return ax, m


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--spacing', type=int, default=3,
                    help='lattice spacing s of the drawn instance (default 3)')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    geo = tiny_instance(args.spacing)

    W, H = TW, 2.90
    fig = plt.figure(figsize=(W, H))
    wa, wb = 2.75, 2.00
    panel_overview(fig, geo, 0.0, wa, H)
    panel_zjunction(fig, wa + 0.06, wb, H)
    panel_wall(fig, geo, wa + wb + 0.12, W - wa - wb - 0.12, H)
    fig.savefig(OUT / f'{NAME}.pdf',
                metadata={'Creator': 'make_junction_figure.py',
                          'CreationDate': None})
    fig.savefig(OUT / f'{NAME}.png', dpi=600, metadata={'Software': None})
    plt.close(fig)

    # instance checks: commuting checks, wall solved, switch pattern
    sig, tau = geo['sigma'], geo['tau']
    path = [v for pair in zip(sig, tau) for v in pair] + [sig[-1]]
    ok = (geo['commute'] and geo['status'] == 'solved'
          and geo['common'] == [1, 2] and sig[0] == 0
          and all(v == 1 for v in path[1:]))
    print(f"instance: n_in=3, Z_j={{q1,q2,q3}}, X_i={{q1,q2}}, s={args.spacing}, "
          f"n={geo['n']} physical qubits")
    print(f"sheets: Y_q={geo['Y']}, X_j={geo['X']}, Z_i={geo['Z']}, "
          f"L_x={geo['Lx']}, L_z={geo['Lz']}, wall y in [{geo['lo']},{geo['hi']}]")
    print(f"wall coefficients: sigma={sig}, tau={tau}; "
          f"common qubits={geo['common']}")
    print(f"wrote {OUT / (NAME + '.pdf')} and .png")
    print(('PASS' if ok else 'FAIL')
          + ': all checks commute, wall solved, coefficients switch at q1 '
          'and q2 (path 0,1,...,1)')


if __name__ == '__main__':
    main()
