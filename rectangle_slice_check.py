#!/usr/bin/env python3
"""Exact check of the slice bound for gauging rectangles.

Supplemental Proposition "Slice bound for a gauging rectangle" states that,
in the zero-boundary relative problem of one gauging measurement of an
X-type logical, every detector-trivial phenomenological fault vector with
nonzero effect has at least min{d~_X, d_Z, R} faults, where d~_X is the X
distance of the deformed code, d_Z the Z distance of the undeformed code and
R the number of deformed rounds.  The proof splits the faults into the two
CSS sectors: X-type faults reach at least d~_X, and Z-type faults at least
min{d_Z, R}.  This script tests both statements exactly.

It builds the rectangle problem from the definitions of the phenomenological
model, independently of the proof:

  * rounds 1..T: `pre` undeformed rounds, R deformed rounds, `post`
    undeformed rounds;
  * faults: a Pauli on a data qubit in any gap between rounds (gap g lies
    just before round g, gap T+1 just before the right cut), a Pauli on an
    edge qubit while it exists (its preparation and readout faults are the
    first and last of these gaps), and a flipped record of any check in any
    round;
  * detectors: consecutive rounds of every repeated check, the entry rows
    (deformed Z check against its undeformed version and the |0> edges, unit
    cycles against the |0> edges), the exit rows (undeformed Z check against
    its deformed version and the edge readouts, unit cycles against the edge
    readouts), zero records in the first round (zero relative input), and
    zero syndrome and zero records at the right cut;
  * effects: the reported outcome, read as the product of the vertex-star
    records of the first deformed round; the residual Pauli on the data at
    the right cut; and the change of the frame update, which the protocol
    computes from the edge readouts along a spanning tree of the grid.

CP-SAT then finds the exact minimum number of faults with a nonzero effect,
separately for X-type and Z-type faults, and the exact static distances d~_X
and d_Z.  The instance is the pair of periodic three-dimensional toric codes
of measurement_interface_check.py (X logical: a membrane in one plane), with
a cubic-grid ancilla of thickness H and m parallel ancillas per edge.

Output: rectangle_slice_check.csv.  Every row must satisfy
X-sector = d~_X and Z-sector = min{d_Z, R}; the script asserts the lower
bounds of the proposition and reports whether they are attained.
"""
from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

from measurement_interface_check import toric3d

ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------- GF(2) algebra
def rref(M):
    M = M.copy() % 2
    rows, cols = M.shape
    piv, r = [], 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        M[[r, p]] = M[[p, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv.append(c)
        r += 1
    return M[:r], piv


def nullspace(M):
    R_, piv = rref(M)
    n = M.shape[1]
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = R_[i, f]
        basis.append(v)
    return np.array(basis, dtype=np.uint8).reshape(len(basis), n)


def rank(M):
    return len(rref(M)[1]) if len(M) else 0


def logical_basis(Hker, Hrow):
    """Basis of ker(Hker) modulo rowspace(Hrow): representatives."""
    K = nullspace(Hker)
    out, cur = [], Hrow.copy()
    base = rank(cur)
    for v in K:
        test = np.vstack([cur, v[None, :]])
        rk = rank(test)
        if rk > base:
            out.append(v)
            cur, base = test, rk
    return out


def mat(rows, n):
    M = np.zeros((len(rows), n), dtype=np.uint8)
    for i, r in enumerate(rows):
        for q in r:
            M[i, q] ^= 1
    return M


# ------------------------------------------------------------------ instance
def build(L, H, m, two_blocks=True):
    stars_a, plaqs_a, mem_a, n_a = toric3d(L, 0)
    if two_blocks:
        stars_b, plaqs_b, mem_b, n_b = toric3d(L, n_a)
    else:
        stars_b, plaqs_b, mem_b, n_b = [], [], {}, 0
    nd = n_a + n_b
    hx = stars_a + stars_b                       # X checks of C (data indices)
    hz = plaqs_a + plaqs_b                       # Z checks of C
    NX = 2 * L if two_blocks else L
    verts = [(X, Y, Z) for X in range(NX) for Y in range(L) for Z in range(H)]
    vid = {v: i for i, v in enumerate(verts)}
    edges, adj = [], {v: [] for v in verts}      # edge j -> (u, w, copy)
    for u in verts:
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            w = tuple(a + b for a, b in zip(u, d))
            if w in vid:
                for c in range(m):
                    j = len(edges)
                    edges.append((u, w, c))
                    adj[u].append((w, j, c))
                    adj[w].append((u, j, c))
    zp = H // 2
    port = {q: (x, y, zp) for (x, y), q in mem_a.items()}
    port.update({q: (L + x, y, zp) for (x, y), q in mem_b.items()})
    vport = {v: q for q, v in port.items()}
    D = sorted(port)

    def path(u, v):
        prev, dq = {u: None}, deque([u])
        while dq:
            a = dq.popleft()
            if a == v:
                break
            for b, j, c in adj[a]:
                if c == 0 and b not in prev:
                    prev[b] = (a, j)
                    dq.append(b)
        out, a = [], v
        while prev[a] is not None:
            a, j = prev[a]
            out.append(j)
        return out

    def pair_edges(qs):
        ps = [port[q] for q in qs if q in port]
        assert len(ps) % 2 == 0
        out = set()
        for i in range(0, len(ps), 2):
            out ^= set(path(ps[i], ps[i + 1]))
        return sorted(out)

    mu = [pair_edges(s) for s in hz]            # pairing paths of each Z check
    cycles = []
    for (X, Y, Z) in verts:
        for d1, d2 in (((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1))):
            c = [(X, Y, Z), tuple(a + b for a, b in zip((X, Y, Z), d1)),
                 tuple(a + b + e for a, b, e in zip((X, Y, Z), d1, d2)),
                 tuple(a + b for a, b in zip((X, Y, Z), d2))]
            if all(p in vid for p in c):
                ks = []
                for i in range(4):
                    u, w = c[i], c[(i + 1) % 4]
                    ks += [j for b, j, cc in adj[u] if b == w and cc == 0]
                cycles.append(ks)
    for j, (u, w, c) in enumerate(edges):
        if c > 0:
            cycles.append([j - c, j])
    stars = [([vport[v]] if v in vport else [], [j for _, j, _c in adj[v]]) for v in verts]
    # spanning tree from vertex 0 (copy-0 edges): tree path of every vertex
    root = verts[0]
    prev, dq = {root: None}, deque([root])
    while dq:
        a = dq.popleft()
        for b, j, c in adj[a]:
            if c == 0 and b not in prev:
                prev[b] = (a, j)
                dq.append(b)
    tree_path = {}
    for v in verts:
        out, a = [], v
        while prev[a] is not None:
            a, j = prev[a]
            out.append(j)
        tree_path[v] = out
    # logical bases of C
    HX, HZ = mat(hx, nd), mat(hz, nd)
    xlog = [set(np.nonzero(v)[0]) for v in logical_basis(HZ, HX)]   # X logicals
    zlog = [set(np.nonzero(v)[0]) for v in logical_basis(HX, HZ)]   # Z logicals
    Dset = set(D)
    # Z logicals commuting with P = X(D): annihilator of D, modulo stabilizers
    zl_arr = np.array([[1 if q in z else 0 for q in range(nd)] for z in zlog], dtype=np.uint8)
    par = np.array([len(z & Dset) % 2 for z in zlog], dtype=np.uint8)
    zcomm = []
    if par.any():
        i0 = int(np.nonzero(par)[0][0])
        for i in range(len(zlog)):
            v = zl_arr[i] ^ (zl_arr[i0] if par[i] else 0)
            if i != i0:
                zcomm.append(set(np.nonzero(v)[0]))
    else:
        zcomm = zlog
    assert all(len(z & Dset) % 2 == 0 for z in zcomm)
    return dict(nd=nd, ne=len(edges), hx=hx, hz=hz, mu=mu, cycles=cycles, stars=stars,
                verts=verts, port=port, vport=vport, D=D, tree_path=tree_path,
                xlog=xlog, zlog=zlog, zcomm=zcomm, pair_edges=pair_edges)


# ------------------------------------------------------------ CP-SAT helpers
class Model:
    def __init__(self):
        self.m = cp_model.CpModel()
        self.vars = []
        self.one = self.m.NewConstant(1)

    def var(self):
        v = self.m.NewBoolVar(f"v{len(self.vars)}")
        self.vars.append(v)
        return len(self.vars) - 1

    def xor_zero(self, form):
        if form:
            self.m.AddBoolXOr([self.vars[i] for i in form] + [self.one])

    def indicator(self, form):
        """Boolean equal to the parity of `form`."""
        y = self.m.NewBoolVar("y")
        self.m.AddBoolXOr([self.vars[i] for i in form] + [y, self.one])
        return y

    def solve(self, effects, limit=600.0):
        self.m.AddBoolOr(effects)
        self.m.Minimize(sum(self.vars))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = limit
        s.parameters.num_workers = 8
        st = s.Solve(self.m)
        return (s.StatusName(st), int(round(s.ObjectiveValue())),
                int(round(s.BestObjectiveBound())))


def static_distance(inst, deformed, sector):
    """Exact static distance: d~_X of the deformed code, or d_Z of C."""
    M = Model()
    nd, ne = inst["nd"], inst["ne"]
    if sector == "X":
        n = nd + (ne if deformed else 0)
        x = [M.var() for _ in range(n)]
        zch = [list(s) + ([nd + j for j in mu] if deformed else [])
               for s, mu in zip(inst["hz"], inst["mu"])]
        if deformed:
            zch += [[nd + j for j in c] for c in inst["cycles"]]
            funcs = [sorted(z) + [nd + j for j in inst["pair_edges"](sorted(z))]
                     for z in inst["zcomm"]]
        else:
            funcs = [sorted(z) for z in inst["zlog"]]
        for c in zch:
            M.xor_zero({x[q] for q in c})
        eff = [M.indicator({x[q] for q in f}) for f in funcs]
    else:
        x = [M.var() for _ in range(nd)]
        for c in inst["hx"]:
            M.xor_zero({x[q] for q in c})
        eff = [M.indicator({x[q] for q in sorted(f)}) for f in inst["xlog"]]
    return M.solve(eff)


# ------------------------------------------------------- the rectangle model
def rectangle(inst, R, pre, post, sector):
    """Exact minimum number of faults of one sector with a nonzero effect."""
    M = Model()
    nd, ne = inst["nd"], inst["ne"]
    T = pre + R + post
    t_in, t_out = pre + 1, pre + R                 # deformed rounds
    deformed = lambda t: t_in <= t <= t_out
    gaps = range(1, T + 2)
    dat = {(q, g): M.var() for q in range(nd) for g in gaps}
    if sector == "X":
        # X on an edge in gap t_in is a preparation fault, in gap t_out+1 a
        # readout fault; in between it is an ordinary edge fault.
        e_gaps = range(t_in, t_out + 2)
    else:
        # Z on a |0> edge at preparation and Z before the Z readout act
        # trivially; they are still allowed as (useless) faults.
        e_gaps = range(t_in, t_out + 2)
    edg = {(j, g): M.var() for j in range(ne) for g in e_gaps} if R > 0 else {}

    def cum_d(q, t):
        return {dat[q, g] for g in gaps if g <= t}

    def cum_e(j, t):
        return {edg[j, g] for g in e_gaps if g <= t}

    def syn(data, edges_, t):
        f = set()
        for q in data:
            f ^= cum_d(q, t)
        for j in edges_:
            f ^= cum_e(j, t)
        return f

    rec = {}
    if sector == "X":
        # Z-type checks: Z checks of C (undeformed), deformed Z checks and
        # unit cycles (deformed rounds).
        for t in range(1, T + 1):
            for i, s in enumerate(inst["hz"]):
                ed = inst["mu"][i] if deformed(t) else []
                rec["s", i, t] = syn(s, ed, t) ^ {M.var()}
            if deformed(t):
                for i, c in enumerate(inst["cycles"]):
                    rec["c", i, t] = syn([], c, t) ^ {M.var()}
        dz = {j: cum_e(j, t_out + 1) for j in range(ne)} if R > 0 else {}
        for i in range(len(inst["hz"])):
            M.xor_zero(rec["s", i, 1])                          # left cut
            for t in range(1, T):
                f = rec["s", i, t] ^ rec["s", i, t + 1]          # incl. entry
                if R > 0 and t == t_out:                         # exit row
                    for j in inst["mu"][i]:
                        f = f ^ dz[j]
                M.xor_zero(f)
            M.xor_zero(syn(inst["hz"][i], [], T + 1))            # right cut
            M.xor_zero(rec["s", i, T])
        if R > 0:
            for i, c in enumerate(inst["cycles"]):
                M.xor_zero(rec["c", i, t_in])                    # entry, |0> edges
                for t in range(t_in, t_out):
                    M.xor_zero(rec["c", i, t] ^ rec["c", i, t + 1])
                f = set(rec["c", i, t_out])
                for j in c:
                    f ^= dz[j]
                M.xor_zero(f)                                    # exit
        # effect: residual data error plus the change of the frame update,
        # paired with Z logicals of C that commute with the measured logical
        eff = []
        for z in inst["zcomm"]:
            f = set()
            for q in z:
                f ^= cum_d(q, T + 1)
            if R > 0:
                for q in z:
                    if q in inst["port"]:
                        for j in inst["tree_path"][inst["port"][q]]:
                            f ^= dz[j]
            eff.append(M.indicator(f))
        if R == 0:
            eff = [M.indicator(_pair(cum_d, z, T)) for z in inst["zlog"]]
    else:
        for t in range(1, T + 1):
            for i, a in enumerate(inst["hx"]):
                rec["a", i, t] = syn(a, [], t) ^ {M.var()}
            if deformed(t):
                for v, (dq_, ej) in enumerate(inst["stars"]):
                    rec["v", v, t] = syn(dq_, ej, t) ^ {M.var()}
        for i in range(len(inst["hx"])):
            M.xor_zero(rec["a", i, 1])
            for t in range(1, T):
                M.xor_zero(rec["a", i, t] ^ rec["a", i, t + 1])
            M.xor_zero(syn(inst["hx"][i], [], T + 1))
            M.xor_zero(rec["a", i, T])
        for v in range(len(inst["stars"]) if R > 0 else 0):
            for t in range(t_in, t_out):
                M.xor_zero(rec["v", v, t] ^ rec["v", v, t + 1])
        eff = [M.indicator(_pair(cum_d, x, T)) for x in inst["xlog"]]
        if R > 0:
            f = set()
            for v in range(len(inst["stars"])):
                f ^= rec["v", v, t_in]
            eff.append(M.indicator(f))                           # outcome flip
    return M.solve(eff)


def _pair(cum_d, logical, T):
    f = set()
    for q in logical:
        f ^= cum_d(q, T + 1)
    return f


# ---------------------------------------------------------------------- main
CONFIGS = [
    # (two blocks, H, m, R, pre, post)
    (True, 1, 1, 0, 1, 1),
    (True, 1, 1, 1, 1, 1), (True, 1, 1, 2, 1, 1), (True, 1, 1, 4, 1, 1),
    (True, 2, 1, 2, 1, 1), (True, 2, 1, 4, 1, 1),
    (True, 3, 1, 2, 1, 1), (True, 3, 1, 4, 1, 1), (True, 3, 1, 3, 2, 2),
    (True, 3, 2, 1, 1, 1), (True, 3, 2, 2, 1, 1), (True, 3, 2, 4, 1, 1),
    (False, 1, 1, 2, 1, 1), (False, 3, 1, 4, 1, 1),
]


def main():
    L = 3
    rows, cache = [], {}
    for two, H, m, R, pre, post in CONFIGS:
        key = (two, H, m)
        if key not in cache:
            inst = build(L, H, m, two)
            dX = static_distance(inst, False, "X")
            dZ = static_distance(inst, False, "Z")
            dXt = static_distance(inst, True, "X")
            cache[key] = (inst, dX, dZ, dXt)
        inst, dX, dZ, dXt = cache[key]
        t0 = time.time()
        sx = rectangle(inst, R, pre, post, "X")
        sz = rectangle(inst, R, pre, post, "Z")
        if R == 0:
            pred_x, pred_z = dX[1], dZ[1]
        else:
            pred_x, pred_z = dXt[1], min(dZ[1], R)
        row = dict(blocks=2 if two else 1, L=L, H=H, multiplicity=m, R=R, pre=pre, post=post,
                   static_dX=dX[1], static_dZ=dZ[1], static_deformed_dX=dXt[1] if R else "",
                   sector_X=sx[1], sector_X_status=sx[0],
                   sector_Z=sz[1], sector_Z_status=sz[0],
                   predicted_X=pred_x, predicted_Z=pred_z,
                   rectangle_distance=min(sx[1], sz[1]),
                   bound_min_dXt_dZ_R=min(pred_x, pred_z),
                   attained=(sx[1] == pred_x and sz[1] == pred_z),
                   seconds=round(time.time() - t0, 1))
        print(row, flush=True)
        for s in (dX, dZ, dXt, sx, sz):
            assert s[0] == "OPTIMAL", s
        assert sx[1] >= pred_x and sz[1] >= pred_z, row        # the proposition
        rows.append(row)
    out = ROOT / "rectangle_slice_check.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_att = sum(r["attained"] for r in rows)
    print(f"wrote {out.name}: {len(rows)} rows; lower bounds hold in all; "
          f"attained with equality in {n_att}")


if __name__ == "__main__":
    main()
