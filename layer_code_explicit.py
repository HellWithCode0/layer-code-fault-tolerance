#!/usr/bin/env python3
"""Explicit microscopic layer-code realization with complete check matrices.

This module builds, for an arbitrary input CSS code (H_X, H_Z), a complete
three-dimensional CSS stabilizer code whose defect network has the
Williamson--Baspin layer-code content:

  * one xz data sheet D_q per input qubit (plane y = Y_q),
  * one yz Z-check sheet C_j per input Z check (plane x = X_j),
  * one xy X-check sheet A_i per input X check (plane z = Z_i).

Every sheet is an edge surface code (qubits on unit edges, X stars on
vertices, Z plaquettes on faces) and owns its own qubits, so sheets that meet
without an input incidence cross transparently.  Input incidences are realized
by explicit line junctions:

  Z junction, q in supp(Z_j), along the z-line (X_j, Y_q, *)
    middle: D_q star at the line vertex u gets C_j's edge u -> u+y,
            C_j plaquette on the +y side of the line gets D_q's line z-edge.
    first : C_j's boundary column *is* D_q's line (shared qubits); the merged
            star is D_q's star plus C_j's edge u -> u+y.
    last  : mirror image (C_j's edge u-y -> u).

  X junction, q in supp(X_i), along the x-line (*, Y_q, Z_i)
    (first, middle and last alike)
            A_i star at the line vertex u gets D_q's edge u -> u+z,
            D_q plaquette above the line gets A_i's line x-edge.

  Crossing lines C_j cap A_i, along the y-line (X_j, *, Z_i), carry "wall"
  terms: an A_i star on the y-line may get C_j's edge u -> u+z, and a C_j
  plaquette above the y-line may get A_i's y-edge.  Which ones do is fixed by
  solving the (linear, GF(2)) commutation conditions; the solution switches
  at every qubit shared by the two checks, i.e. at the point defects where a
  z-line and an x-line cross inside a data sheet.

Nothing is inferred from published artwork.  The script verifies every
property it relies on: all checks commute, the number of encoded qubits
equals that of the input code, and the prefix-parity slab operators commute
with every stabilizer, in every gap, with the input symplectic pairing.

The mode junction_mode="single_chart" instead builds the plain union of the
sheets with shared line edges at every incidence, which is what one obtains by
reading the six junction cells in a single unit-cubic chart.  It is used as a
negative control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import scipy.sparse as sp
from ldpc import mod2


def E(a, b):
    a = tuple(int(t) for t in a)
    b = tuple(int(t) for t in b)
    return (a, b) if a <= b else (b, a)


def add(v, d):
    return (v[0] + d[0], v[1] + d[1], v[2] + d[2])


X_, Y_, Z_ = (1, 0, 0), (0, 1, 0), (0, 0, 1)
NX_, NY_, NZ_ = (-1, 0, 0), (0, -1, 0), (0, 0, -1)


# --------------------------------------------------------------------------
# GF(2) helpers
# --------------------------------------------------------------------------

def gf2_rank(M) -> int:
    if M.shape[0] == 0 or M.shape[1] == 0:
        return 0
    return int(mod2.rank(sp.csr_matrix(M, dtype=np.uint8)))


def gf2_solve(A: np.ndarray, b: np.ndarray):
    """Particular solution of A x = b over F2 (free variables 0) or None."""
    A = (np.asarray(A, dtype=np.uint8) & 1).copy()
    b = (np.asarray(b, dtype=np.uint8) & 1).reshape(-1).copy()
    m, n = A.shape
    M = np.hstack([A, b.reshape(-1, 1)])
    row, piv = 0, []
    for c in range(n):
        nz = np.nonzero(M[row:, c])[0]
        if nz.size == 0:
            continue
        p = row + nz[0]
        if p != row:
            M[[row, p]] = M[[p, row]]
        others = np.nonzero(M[:, c])[0]
        others = others[others != row]
        M[others] ^= M[row]
        piv.append(c)
        row += 1
        if row == m:
            break
    if np.any(M[row:, n]):
        return None
    x = np.zeros(n, dtype=np.uint8)
    for r, c in enumerate(piv):
        x[c] = M[r, n]
    return x


def in_rowspace(H, V) -> np.ndarray:
    """For each row v of V, whether v lies in the row space of H."""
    H = sp.csr_matrix(H, dtype=np.uint8)
    r = gf2_rank(H)
    out = []
    for v in np.atleast_2d(V):
        out.append(gf2_rank(sp.vstack([H, sp.csr_matrix(v.reshape(1, -1))])) == r)
    return np.array(out, dtype=bool)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

@dataclass
class LayerCode:
    HX_in: np.ndarray
    HZ_in: np.ndarray
    spacing: int
    mode: str
    qubits: list = field(default_factory=list)          # qubit keys
    qindex: dict = field(default_factory=dict)          # key -> index
    xchecks: list = field(default_factory=list)         # dicts
    zchecks: list = field(default_factory=list)
    alias: dict = field(default_factory=dict)           # shared-qubit aliases
    geometry: dict = field(default_factory=dict)
    wall_report: list = field(default_factory=list)

    # ---- qubit bookkeeping ------------------------------------------------
    def key(self, sheet, edge):
        k = (sheet, edge)
        return self.alias.get(k, k)

    def q(self, sheet, edge, create=True):
        k = self.key(sheet, edge)
        if k not in self.qindex:
            if not create:
                return None
            self.qindex[k] = len(self.qubits)
            self.qubits.append(k)
        return self.qindex[k]

    def has(self, sheet, edge):
        return self.key(sheet, edge) in self.qindex

    # ---- matrices ----------------------------------------------------------
    def matrix(self, checks):
        rows, cols = [], []
        for r, c in enumerate(checks):
            for qi in c["support"]:
                rows.append(r)
                cols.append(qi)
        data = np.ones(len(rows), dtype=np.uint8)
        M = sp.csr_matrix((data, (rows, cols)), shape=(len(checks), len(self.qubits)),
                          dtype=np.uint8)
        M.data &= 1
        M.eliminate_zeros()
        return M

    @property
    def HX(self):
        return self.matrix(self.xchecks)

    @property
    def HZ(self):
        return self.matrix(self.zchecks)

    @property
    def n(self):
        return len(self.qubits)


def _supports(H):
    return [list(np.nonzero(row)[0]) for row in np.asarray(H)]


def build_layer_code(HX_in, HZ_in, spacing: int = 2, junction_mode: str = "two_sheet",
                     solve_walls: bool = True, require_css: bool = True) -> LayerCode:
    HX_in = np.asarray(HX_in, dtype=np.uint8) & 1
    HZ_in = np.asarray(HZ_in, dtype=np.uint8) & 1
    nin = HX_in.shape[1]
    assert HZ_in.shape[1] == nin
    if require_css:
        assert not np.any((HX_in.astype(int) @ HZ_in.T.astype(int)) % 2), "input not CSS"
    s = int(spacing)
    assert s >= 2
    suppX, suppZ = _supports(HX_in), _supports(HZ_in)
    for sup in suppX + suppZ:
        assert len(sup) >= 2, "input checks of weight < 2 are not supported"
    mX, mZ = len(suppX), len(suppZ)
    Yq = [s * q for q in range(nin)]
    Xj = [s * (j + 1) for j in range(mZ)]
    Zi = [s * (i + 1) for i in range(mX)]
    Lx, Lz = s * (mZ + 1), s * (mX + 1)
    two = junction_mode == "two_sheet"
    lc = LayerCode(HX_in, HZ_in, s, junction_mode)
    lc.geometry = dict(Yq=Yq, Xj=Xj, Zi=Zi, Lx=Lx, Lz=Lz, suppX=suppX, suppZ=suppZ)

    # ---------------- shared-qubit aliases ----------------------------------
    for j, sup in enumerate(suppZ):
        X = Xj[j]
        ends = (sup[0], sup[-1]) if two else tuple(sup)
        for q in ends:
            for z in range(Lz):
                e = E((X, Yq[q], z), (X, Yq[q], z + 1))
                lc.alias[(("C", j), e)] = (("D", q), e)
    if not two:
        for i, sup in enumerate(suppX):
            Z = Zi[i]
            for q in sup:
                for x in range(Lx):
                    e = E((x, Yq[q], Z), (x + 1, Yq[q], Z))
                    lc.alias[(("A", i), e)] = (("D", q), e)

    # ---------------- data sheets -------------------------------------------
    stars, plaqs = {}, {}   # label -> set of qubit indices
    for q in range(nin):
        sh, Y = ("D", q), Yq[q]
        for x in range(Lx):
            for z in range(1, Lz):
                lc.q(sh, E((x, Y, z), (x + 1, Y, z)))
        for x in range(Lx + 1):
            for z in range(Lz):
                lc.q(sh, E((x, Y, z), (x, Y, z + 1)))

    def star(sheet, v, dirs):
        sup = set()
        for d in dirs:
            e = E(v, add(v, d))
            if lc.has(sheet, e):
                sup ^= {lc.q(sheet, e)}
        return sup

    def face(sheet, v, d1, d2):
        corners = [v, add(v, d1), add(add(v, d1), d2), add(v, d2)]
        sup = set()
        for a, b in zip(corners, corners[1:] + corners[:1]):
            e = E(a, b)
            if lc.has(sheet, e):
                sup ^= {lc.q(sheet, e)}
        return sup

    for q in range(nin):
        sh, Y = ("D", q), Yq[q]
        for x in range(Lx + 1):
            for z in range(1, Lz):
                stars[("D", q, (x, Y, z))] = star(sh, (x, Y, z), (X_, NX_, Z_, NZ_))
        for x in range(Lx):
            for z in range(Lz):
                plaqs[("D", q, (x, Y, z))] = face(sh, (x, Y, z), X_, Z_)

    # ---------------- Z-check sheets ----------------------------------------
    for j, sup in enumerate(suppZ):
        sh, X = ("C", j), Xj[j]
        Yf, Yl = Yq[sup[0]], Yq[sup[-1]]
        for y in range(Yf, Yl):
            for z in range(1, Lz):
                lc.q(sh, E((X, y, z), (X, y + 1, z)))
        for y in range(Yf, Yl + 1):
            for z in range(Lz):
                lc.q(sh, E((X, y, z), (X, y, z + 1)))
        ends = {Yq[q]: q for q in ((sup[0], sup[-1]) if two else sup)}
        for y in range(Yf, Yl + 1):
            for z in range(1, Lz):
                v = (X, y, z)
                if y in ends:            # merged into the data star
                    qd = ends[y]
                    extra = star(sh, v, (Y_, NY_))
                    stars[("D", qd, v)] ^= extra
                    if not two:          # single chart: shared z-edges already in D star
                        pass
                else:
                    stars[("C", j, v)] = star(sh, v, (Y_, NY_, Z_, NZ_))
        for y in range(Yf, Yl):
            for z in range(Lz):
                plaqs[("C", j, (X, y, z))] = face(sh, (X, y, z), Y_, Z_)
        if two:
            for q in sup[1:-1]:          # middle junctions
                Y = Yq[q]
                for z in range(1, Lz):
                    v = (X, Y, z)
                    stars[("D", q, v)] ^= {lc.q(sh, E(v, add(v, Y_)))}
                for z in range(Lz):
                    plaqs[("C", j, (X, Y, z))] ^= {lc.q(("D", q), E((X, Y, z), (X, Y, z + 1)))}

    # ---------------- X-check sheets ----------------------------------------
    for i, sup in enumerate(suppX):
        sh, Z = ("A", i), Zi[i]
        Yf, Yl = Yq[sup[0]], Yq[sup[-1]]
        for y in range(Yf, Yl + 1):
            for x in range(Lx):
                lc.q(sh, E((x, y, Z), (x + 1, y, Z)))
        for y in range(Yf, Yl):
            for x in range(Lx + 1):
                lc.q(sh, E((x, y, Z), (x, y + 1, Z)))
        ends = {Yq[q]: q for q in sup} if not two else {}
        for y in range(Yf, Yl + 1):
            for x in range(Lx + 1):
                v = (x, y, Z)
                if y in ends:            # single chart: merged into data star
                    stars[("D", ends[y], v)] ^= star(sh, v, (Y_, NY_))
                else:
                    stars[("A", i, v)] = star(sh, v, (X_, NX_, Y_, NY_))
        for y in range(Yf, Yl):
            for x in range(Lx):
                plaqs[("A", i, (x, y, Z))] = face(sh, (x, y, Z), X_, Y_)
        if two:
            for q in sup:
                Y = Yq[q]
                for x in range(Lx + 1):
                    v = (x, Y, Z)
                    stars[("A", i, v)] ^= {lc.q(("D", q), E(v, add(v, Z_)))}
                for x in range(Lx):
                    plaqs[("D", q, (x, Y, Z))] ^= {lc.q(sh, E((x, Y, Z), (x + 1, Y, Z)))}

    # ---------------- walls on C_j cap A_i ------------------------------------
    if two and solve_walls:
        _solve_walls(lc, stars, plaqs)

    lc.xchecks = [dict(label=k, support=sorted(v)) for k, v in stars.items() if v]
    lc.zchecks = [dict(label=k, support=sorted(v)) for k, v in plaqs.items() if v]
    return lc


def _solve_walls(lc: LayerCode, stars, plaqs):
    g = lc.geometry
    Yq, Xj, Zi = g["Yq"], g["Xj"], g["Zi"]
    # qubit -> Z checks containing it, X checks containing it (for base overlaps)
    for j, supZ in enumerate(g["suppZ"]):
        for i, supX in enumerate(g["suppX"]):
            X, Z = Xj[j], Zi[i]
            lo = max(Yq[supZ[0]], Yq[supX[0]])
            hi = min(Yq[supZ[-1]], Yq[supX[-1]])
            if lo > hi:
                continue
            svars = [("A", i, (X, y, Z)) for y in range(lo, hi + 1)]
            tvars = [("C", j, (X, y, Z)) for y in range(lo, hi)]
            sq = [lc.q(("C", j), E((X, y, Z), (X, y, Z + 1))) for y in range(lo, hi + 1)]
            tq = [lc.q(("A", i), E((X, y, Z), (X, y + 1, Z))) for y in range(lo, hi)]
            svars_ok = [k in stars for k in svars]
            # Relevant Z checks: all containing an s-qubit, plus t-plaquettes.
            zkeys = set(tvars)
            xkeys = set(k for k, ok in zip(svars, svars_ok) if ok)
            sset, tset = set(sq), set(tq)
            for k, v in plaqs.items():
                if v & sset:
                    zkeys.add(k)
            for k, v in stars.items():
                if v & tset:
                    xkeys.add(k)
            zkeys = sorted(k for k in zkeys if k in plaqs)
            xkeys = sorted(xkeys)
            nv = len(svars) + len(tvars)
            rows, rhs = [], []
            for a in xkeys:
                for b in zkeys:
                    base = len(stars[a] & plaqs[b]) & 1
                    coeff = np.zeros(nv, dtype=np.uint8)
                    if a in svars:
                        ia = svars.index(a)
                        if svars_ok[ia] and sq[ia] in plaqs[b]:
                            coeff[ia] ^= 1
                    if b in tvars:
                        ib = tvars.index(b)
                        if tq[ib] in stars[a]:
                            coeff[len(svars) + ib] ^= 1
                    if base or coeff.any():
                        rows.append(coeff)
                        rhs.append(base)
            if not rows:
                continue
            sol = gf2_solve(np.array(rows), np.array(rhs))
            if sol is None:
                lc.wall_report.append(dict(j=j, i=i, status="INCONSISTENT"))
                continue
            for ia, a in enumerate(svars):
                if sol[ia] and svars_ok[ia]:
                    stars[a] ^= {sq[ia]}
            for ib, b in enumerate(tvars):
                if sol[len(svars) + ib]:
                    plaqs[b] ^= {tq[ib]}
            lc.wall_report.append(dict(
                j=j, i=i, status="solved", n_eq=len(rows), n_var=nv,
                wall_stars=[svars[k][2][1] for k in range(len(svars)) if sol[k]],
                wall_plaqs=[tvars[k][2][1] for k in range(len(tvars)) if sol[len(svars) + k]],
                common=[q for q in supZ if q in supX]))


# --------------------------------------------------------------------------
# Logical operators: prefix-parity slabs
# --------------------------------------------------------------------------

def prefix_bits(word):
    p, out = 0, []
    for b in word:
        p ^= int(b)
        out.append(p)
    return out


def x_slab(lc: LayerCode, x_in, z0: int, override=None):
    """Prefix-parity X representative of the input X logical x_in at height z0.

    Returns (vector, support-z-range).  `override` maps (j, segment) -> bit to
    build negative controls that violate the prefix recurrence.
    """
    g = lc.geometry
    v = np.zeros(lc.n, dtype=np.uint8)
    for q in range(len(x_in)):
        if x_in[q]:
            for x in range(g["Lx"] + 1):
                v[lc.q(("D", q), E((x, g["Yq"][q], z0), (x, g["Yq"][q], z0 + 1)), False)] ^= 1
    for j, sup in enumerate(g["suppZ"]):
        X = g["Xj"][j]
        word = [int(x_in[q]) for q in sup]
        p = prefix_bits(word)
        for t in range(len(sup) - 1):
            bit = p[t]
            if override is not None and (j, t) in override:
                bit = override[(j, t)]
            if not bit:
                continue
            y0, y1 = g["Yq"][sup[t]], g["Yq"][sup[t + 1]]
            # the C-owned z-edges on (y0, y1]; the edge at y1 is C's own line
            # edge for a middle junction and the shared data edge (not added)
            # for the last junction.  At y0 the edge is C's own line edge for a
            # middle junction and shared (not added) for the first junction.
            for y in range(y0, y1 + 1):
                key = lc.key(("C", j), E((X, y, z0), (X, y, z0 + 1)))
                if key[0][0] == "D":
                    continue
                if y == y0 and t == 0:
                    continue
                if y == y1 and t + 1 == len(sup) - 1:
                    continue
                # C's own line edge at a middle junction carries p_{t-1} on the
                # -y side, i.e. it belongs to the segment ending there.
                if y == y0 and 0 < t:
                    continue
                v[lc.qindex[key]] ^= 1
    return v


def z_slab(lc: LayerCode, z_in, x0: int, override=None):
    g = lc.geometry
    v = np.zeros(lc.n, dtype=np.uint8)
    for q in range(len(z_in)):
        if z_in[q]:
            for z in range(g["Lz"]):
                v[lc.q(("D", q), E((x0, g["Yq"][q], z), (x0, g["Yq"][q], z + 1)), False)] ^= 1
    for i, sup in enumerate(g["suppX"]):
        Z = g["Zi"][i]
        word = [int(z_in[q]) for q in sup]
        p = prefix_bits(word)
        for t in range(len(sup) - 1):
            bit = p[t]
            if override is not None and (i, t) in override:
                bit = override[(i, t)]
            if not bit:
                continue
            y0, y1 = g["Yq"][sup[t]], g["Yq"][sup[t + 1]]
            for y in range(y0, y1):
                v[lc.q(("A", i), E((x0, y, Z), (x0, y + 1, Z)), False)] ^= 1
    return v


def support_extent(lc: LayerCode, vec, axis: int):
    """Min/max coordinate along `axis` of all edges carrying the operator."""
    lo, hi = None, None
    for qi in np.nonzero(vec)[0]:
        (_, e) = lc.qubits[qi]
        for pt in e:
            c = pt[axis]
            lo = c if lo is None else min(lo, c)
            hi = c if hi is None else max(hi, c)
    return lo, hi
