#!/usr/bin/env python3
"""Thin versus thick measurement interfaces for a joint logical measurement.

Two blocks of the periodic three-dimensional toric code are used as a small
stand-in for codes whose X logical is a membrane of weight Theta(L^2) supported
in one plane (the situation of a layer-code slab representative).  The joint
logical X_a X_b is gauged with a cubic-grid ancilla of thickness H placed across
both port planes:

  * H = 1  : a single added measurement layer (a 2D sheet),
  * H = 2  : a thin slab,
  * H = L  : the thick three-dimensional grid box used in the paper.

For every instance the script builds the deformed CSS code explicitly, checks
H_X H_Z^T = 0, and verifies over GF(2) that the cut operator X(delta T), with
T the block-a half of the grid, is a nontrivial X logical of the deformed code
(Proposition "Cut obstruction" of the Supplemental Material).  Its weight L*H
is an upper bound on the deformed X distance, to be compared with the
undeformed membrane distance L^2.

Output: measurement_interface_check.csv
"""
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- GF(2) utils
class RowSpace:
    """Incremental GF(2) row space; rows are Python ints used as bitsets."""

    def __init__(self) -> None:
        self.pivots: dict[int, int] = {}

    def reduce(self, v: int) -> int:
        while v:
            top = v.bit_length() - 1
            p = self.pivots.get(top)
            if p is None:
                return v
            v ^= p
        return 0

    def add(self, v: int) -> bool:
        r = self.reduce(v)
        if r:
            self.pivots[r.bit_length() - 1] = r
            return True
        return False


def bits(idx) -> int:
    v = 0
    for i in idx:
        v ^= 1 << i
    return v


# ---------------------------------------------------------- 3D toric code
def toric3d(L: int, offset: int):
    """Edges (x,y,z,dir) of an L^3 periodic cubic lattice, dir in {0,1,2}."""
    def e(x, y, z, d):
        return offset + ((((x % L) * L + (y % L)) * L + (z % L)) * 3 + d)

    stars, plaqs = [], []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                stars.append([e(x, y, z, 0), e(x, y, z, 1), e(x, y, z, 2),
                              e(x - 1, y, z, 0), e(x, y - 1, z, 1), e(x, y, z - 1, 2)])
                plaqs.append([e(x, y, z, 0), e(x + 1, y, z, 1), e(x, y + 1, z, 0), e(x, y, z, 1)])  # xy
                plaqs.append([e(x, y, z, 0), e(x + 1, y, z, 2), e(x, y, z + 1, 0), e(x, y, z, 2)])  # xz
                plaqs.append([e(x, y, z, 1), e(x, y + 1, z, 2), e(x, y, z + 1, 1), e(x, y, z, 2)])  # yz
    # X membrane: all z-edges from height 0 to 1, indexed by (x, y).
    membrane = {(x, y): e(x, y, 0, 2) for x in range(L) for y in range(L)}
    return stars, plaqs, membrane, 3 * L ** 3


# ---------------------------------------------------------------- instance
def run(L: int, H: int, m: int = 1) -> dict:
    stars_a, plaqs_a, mem_a, n_a = toric3d(L, 0)
    stars_b, plaqs_b, mem_b, n_b = toric3d(L, n_a)
    n_data = n_a + n_b

    # Grid ancilla: vertices (X,Y,Z), X in [0,2L), Y in [0,L), Z in [0,H).
    NX, NY, NZ = 2 * L, L, H
    vid = {(X, Y, Z): i for i, (X, Y, Z) in enumerate(
        (X, Y, Z) for X in range(NX) for Y in range(NY) for Z in range(NZ))}
    edges, adj = [], {v: [] for v in vid}
    for (X, Y, Z) in vid:
        for dX, dY, dZ in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            w = (X + dX, Y + dY, Z + dZ)
            if w in vid:
                for copy in range(m):          # m parallel edges per grid edge
                    k = n_data + len(edges)
                    edges.append(((X, Y, Z), w, copy))
                    adj[(X, Y, Z)].append((w, k, copy))
                    adj[w].append(((X, Y, Z), k, copy))
    n_total = n_data + len(edges)
    zp = H // 2
    port = {}                       # data qubit -> grid vertex
    for (x, y), q in mem_a.items():
        port[q] = (x, y, zp)
    for (x, y), q in mem_b.items():
        port[q] = (L + x, y, zp)
    vport = {v: q for q, v in port.items()}

    def path(u, v):
        prev = {u: None}
        dq = deque([u])
        while dq:
            a = dq.popleft()
            if a == v:
                break
            for b, k, copy in adj[a]:
                if copy == 0 and b not in prev:
                    prev[b] = (a, k)
                    dq.append(b)
        out, a = [], v
        while prev[a] is not None:
            a, k = prev[a]
            out.append(k)
        return out

    # X-type generators: unchanged vertex stars and Gauss operators A_v.
    hx = [bits(s) for s in stars_a + stars_b]
    for v in vid:
        sup = [k for _, k, _c in adj[v]]
        if v in vport:
            sup.append(vport[v])
        hx.append(bits(sup))
    # Z-type generators: plaquettes deformed along pairing paths, unit squares.
    hz = []
    for pl in plaqs_a + plaqs_b:
        ports = [port[q] for q in pl if q in port]
        assert len(ports) % 2 == 0
        sup = list(pl)
        for i in range(0, len(ports), 2):
            sup += path(ports[i], ports[i + 1])
        hz.append(bits(sup))
    for (X, Y, Z) in vid:
        for (d1, d2) in (((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1))):
            c = [(X, Y, Z), tuple(a + b for a, b in zip((X, Y, Z), d1)),
                 tuple(a + b + c_ for a, b, c_ in zip((X, Y, Z), d1, d2)),
                 tuple(a + b for a, b in zip((X, Y, Z), d2))]
            if all(p in vid for p in c):
                ks = []
                for i in range(4):
                    u, w = c[i], c[(i + 1) % 4]
                    ks += [k for b, k, copy in adj[u] if b == w and copy == 0]
                hz.append(bits(ks))
    # Bundle cycles Z_e Z_e' generate the extra cycle space of the multigraph.
    for (u, w, copy), k in zip(edges, range(n_data, n_total)):
        if copy > 0:
            hz.append(bits([k - copy, k]))

    css_ok = all(bin(a & b).count("1") % 2 == 0 for a in hx for b in hz)

    # Cut operator X(delta T), T = block-a half of the grid.
    cut = [k for (u, w, _c), k in zip(edges, range(n_data, n_total)) if (u[0] < L) != (w[0] < L)]
    v_cut = bits(cut)
    commutes = all(bin(v_cut & z).count("1") % 2 == 0 for z in hz)
    rs = RowSpace()
    for r in hx:
        rs.add(r)
    nontrivial = rs.reduce(v_cut) != 0
    # Sanity: the measured product X_a X_b is a deformed stabilizer, while
    # X_a alone is a nontrivial deformed logical equivalent to the cut operator.
    prod_is_stab = rs.reduce(bits(list(mem_a.values()) + list(mem_b.values()))) == 0
    equiv = rs.reduce(v_cut ^ bits(mem_a.values())) == 0
    # Two-sided bracket for the deformed X distance.  Lower bound: Proposition
    # "Distance under gauging" with beta >= m*min{1, H*h(G0)} for one port
    # plane (thickening lemma, t = 1) and h(G0) >= 1/(2L) for the 2L x L
    # cross-section.  Upper bound: the cut logical, or X(D_a) itself.
    beta_lb = m * min(1.0, H / (2 * L))
    d_lb = min(1.0, beta_lb) * L * L
    d_ub = min(len(cut), L * L)
    name = {1: "sheet (H=1)", 2: "thin slab (H=2)"}.get(H, f"thick box (H=L={L})")
    if m > 1:
        name += f", edge multiplicity {m}"
    return dict(L=L, H=H, multiplicity=m, ancilla=name,
                data_qubits=n_data, grid_edge_qubits=len(edges),
                undeformed_dX=L * L, cut_logical_weight=len(cut),
                beta_lower_bound=round(beta_lb, 4),
                deformed_dX_lower=d_lb, deformed_dX_upper=d_ub,
                css_valid=css_ok, cut_commutes_with_Z=commutes,
                cut_is_nontrivial=nontrivial, cut_equiv_to_Xa=equiv,
                measured_product_is_stabilizer=prod_is_stab)


def main() -> None:
    rows = []
    for L in (3, 4, 5, 6):
        for H, m in [(1, 1), (2, 1), (L, 1), (L, 2)]:
            r = run(L, H, m)
            rows.append(r)
            print(r)
            assert r["css_valid"] and r["cut_commutes_with_Z"] and r["cut_is_nontrivial"]
            assert r["cut_equiv_to_Xa"] and r["measured_product_is_stabilizer"]
            assert r["cut_logical_weight"] == m * L * H
            assert r["deformed_dX_lower"] <= r["deformed_dX_upper"]
    out = ROOT / "measurement_interface_check.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.name}: {len(rows)} rows; all checks PASS")


if __name__ == "__main__":
    main()
