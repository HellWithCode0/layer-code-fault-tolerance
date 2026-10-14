#!/usr/bin/env python3
"""Two-block joint logical measurement on explicit layer-code blocks, any size.

This is the circuit of the paper's two_block_measurement.py (Supplemental
Sec. "Two-block measurement"), generalized in three ways:

  * the input code is any iceberg code [[k+2, k, 2]] (H_X = H_Z = all ones),
    so k = 2 gives the paper's [[4,2,2]] blocks and k = 4 gives [[6,4,2]];
  * the spacing s, the rounds per phase and the interface are free;
  * interface "idle" gives the same blocks, rounds and observables with no
    gauging at all (a memory baseline for the same duration).

For k = 2 and the paper's parameters the circuit is identical, character for
character, to the one built by two_block_measurement.py; smoke_test.py checks
this against a frozen hash.

Logical 1 of each block is measured jointly, P = X1(a) X1(b), by
Williamson--Yoder gauging on a planar grid attached to the single-layer slab
representatives; logicals 2..k of each block are spectators.  Every logical
qubit starts maximally entangled with a noiseless reference qubit, so the
observables certify the output Choi state:

  X experiment  m + x(Ra1) + x(Rb1)        reported measurement bit
                Xi(a) + x(Rai), Xi(b) + x(Rbi)   for i = 1..k
  Z experiment  Z1(a) Z1(b) + z(Ra1) z(Rb1)      measured pair, Z sector
                Zi(a) + z(Rai), Zi(b) + z(Rbi)   spectators, i = 2..k

Interfaces:
  bridge-w   each block's ports carry their own grid; the grids are joined by
             w parallel rows (w = 1 is the thin interface of the cut obstruction);
  thick      one grid spans both blocks;
  thick x m  the same grid with m parallel ancillas on every edge;
  idle       no gauging (memory baseline).

Noise (uniform circuit model, strength p): DEPOLARIZE2(p) after every CNOT,
DEPOLARIZE1(p) on every data/edge qubit once per round, X/Z flips of
probability p after every reset and before every measurement, including the
final data readout.  The initial encoding and the reference qubits are
noiseless.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp
import stim

from layer_code_explicit import build_layer_code, gf2_rank, x_slab, z_slab


def iceberg_logicals(k):
    """Logical pairs of the [[k+2, k, 2]] code: X_i = X_0 X_i, Z_i = Z_i Z_{k+1}."""
    n = k + 2
    xs, zs = [], []
    for i in range(1, k + 1):
        x = np.zeros(n, np.uint8); x[0] = 1; x[i] = 1
        z = np.zeros(n, np.uint8); z[i] = 1; z[n - 1] = 1
        xs.append(x); zs.append(z)
    return xs, zs


def rows_of(M):
    M = sp.csr_matrix(M)
    return [list(M.indices[M.indptr[r]:M.indptr[r + 1]]) for r in range(M.shape[0])]


class TwoBlock:
    def __init__(self, k=2, s=2, interface="thick", width=1, mult=1, z0=1, gap=2):
        n_in = k + 2
        H = np.ones((1, n_in), np.uint8)
        self.lc = lc = build_layer_code(H, H, spacing=s)
        n = self.n = lc.n
        g = lc.geometry
        self.k, self.s, self.interface, self.width, self.mult = k, s, interface, width, mult
        Lx, Lz = g["Lx"], g["Lz"]
        self.delta = Lx + gap
        xs_in, zs_in = iceberg_logicals(k)
        # heights of the X slabs: logical 1 (measured) at z0, spectators elsewhere
        zh = [z0] + [z for z in range(Lz) if z != z0][: k - 1]
        # positions of the Z planes: 0, Lx, 1, Lx-1, 2, ...
        xp = []
        lo, hi = 0, Lx
        while len(xp) < k:
            xp.append(lo)
            if len(xp) < k:
                xp.append(hi)
            lo, hi = lo + 1, hi - 1
        assert len(zh) == k, "block too thin for this many spectators"
        self.Xs = [x_slab(lc, xs_in[i], zh[i]) for i in range(k)]
        self.Zs = [z_slab(lc, zs_in[i], xp[i]) for i in range(k)]
        # paper names (k = 2): X1 at z0, X2 at z = 0, Z1 at x = 0, Z2 at x = Lx
        self.X1 = self.Xs[0]
        bx, bz = rows_of(lc.HX), rows_of(lc.HZ)
        self.orig_x = [c for c in bx] + [[q + n for q in c] for c in bx]
        self.orig_z = [c for c in bz] + [[q + n for q in c] for c in bz]
        ports = {}
        for blk in (0, 1):
            for qi in np.nonzero(self.X1)[0]:
                (sheet, e) = lc.qubits[qi]
                (x, y, z) = e[0]
                off = 1 if sheet[0] == "C" else 0
                ports[qi + blk * n] = (2 * (x + blk * self.delta) + off, 2 * y)
        self.ports = ports
        ys = [v[1] for v in ports.values()]
        xa = [v[0] for q, v in ports.items() if q < n]
        xb = [v[0] for q, v in ports.items() if q >= n]
        self.ylo, self.yhi = min(ys), max(ys)
        verts = set()
        if interface in ("thick", "idle"):
            for X in range(min(xa), max(xb) + 1):
                for Y in range(self.ylo, self.yhi + 1):
                    verts.add((X, Y))
        elif interface == "bridge":
            for (lo_, hi_) in ((min(xa), max(xa)), (min(xb), max(xb))):
                for X in range(lo_, hi_ + 1):
                    for Y in range(self.ylo, self.yhi + 1):
                        verts.add((X, Y))
            rows = list(range(self.ylo, self.yhi + 1))
            mid = len(rows) // 2
            chosen = sorted(rows, key=lambda r: abs(r - rows[mid]))[:width]
            for X in range(max(xa) + 1, min(xb)):
                for Y in chosen:
                    verts.add((X, Y))
        else:
            raise ValueError(interface)
        self.verts = sorted(verts)
        vid = {v: i for i, v in enumerate(self.verts)}
        self.vid = vid
        edges, adj = [], {v: [] for v in self.verts}
        for (X, Y) in self.verts:
            for d in ((1, 0), (0, 1)):
                w = (X + d[0], Y + d[1])
                if w in vid:
                    for c in range(mult):
                        kk = len(edges)
                        edges.append(((X, Y), w, c))
                        adj[(X, Y)].append((w, kk, c))
                        adj[w].append(((X, Y), kk, c))
        self.edges, self.adj = edges, adj
        self.nE = len(edges)
        self.E0 = 2 * n
        vport = {}
        for q, v in ports.items():
            assert v in vid and v not in vport, "ports must map to distinct grid vertices"
            vport[v] = q
        self.vport = vport
        self.gauss = []
        for v in self.verts:
            sup = [self.E0 + kk for _, kk, _ in adj[v]]
            if v in vport:
                sup.append(vport[v])
            self.gauss.append(sorted(sup))
        self.flux = []
        for (X, Y) in self.verts:
            sq = [(X, Y), (X + 1, Y), (X + 1, Y + 1), (X, Y + 1)]
            if all(p in vid for p in sq):
                ks = []
                for a, b in zip(sq, sq[1:] + sq[:1]):
                    ks += [self.E0 + kk for w, kk, c in adj[a] if w == b and c == 0]
                self.flux.append(sorted(ks))
        for kk, (u, w, c) in enumerate(edges):
            if c > 0:
                k0 = next(k2 for ww, k2, cc in adj[u] if ww == w and cc == 0)
                self.flux.append([self.E0 + k0, self.E0 + kk])
        self.def_z, self.gamma = [], []
        for c in self.orig_z:
            pp = [self.ports[q] for q in c if q in self.ports]
            assert len(pp) % 2 == 0
            path = []
            for i in range(0, len(pp), 2):
                path += self.path(pp[i], pp[i + 1])
            self.gamma.append(sorted(path))
            self.def_z.append(sorted(set(c) ^ set(path)))
        self.tree_parent = self.bfs_tree(self.verts[0])

    def path(self, u, v):
        prev = {u: None}
        dq = deque([u])
        while dq:
            a = dq.popleft()
            if a == v:
                break
            for b, kk, c in self.adj[a]:
                if c == 0 and b not in prev:
                    prev[b] = (a, kk)
                    dq.append(b)
        out, a = [], v
        while prev[a] is not None:
            a, kk = prev[a]
            out.append(self.E0 + kk)
        return out

    def bfs_tree(self, root):
        prev = {root: None}
        dq = deque([root])
        while dq:
            a = dq.popleft()
            for b, kk, c in self.adj[a]:
                if c == 0 and b not in prev:
                    prev[b] = (a, self.E0 + kk)
                    dq.append(b)
        assert len(prev) == len(self.verts), "interface graph must be connected"
        return prev

    def tree_path_edges(self, v):
        out, a = [], v
        while self.tree_parent[a] is not None:
            a, e = self.tree_parent[a]
            out.append(e)
        return out

    def byproduct_edges(self, data_support):
        cnt = {}
        for q in data_support:
            if q in self.ports:
                for e in self.tree_path_edges(self.ports[q]):
                    cnt[e] = cnt.get(e, 0) ^ 1
        return sorted(e for e, b in cnt.items() if b)

    def sup(self, vec, blk):
        return [int(q) + blk * self.n for q in np.nonzero(vec)[0]]

    def deformed_matrices(self):
        N = self.E0 + self.nE

        def mat(rows):
            r, c = [], []
            for i, row in enumerate(rows):
                r += [i] * len(row)
                c += row
            M = sp.csr_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(len(rows), N))
            M.data &= 1
            return M
        return mat(self.orig_x + self.gauss), mat(self.def_z + self.flux)

    def cut_size(self):
        xa = max(v[0] for q, v in self.ports.items() if q < self.n)
        xb = min(v[0] for q, v in self.ports.items() if q >= self.n)
        mid = (xa + xb) / 2
        return sum(1 for (u, w, c) in self.edges if (u[0] < mid) != (w[0] < mid))

    def check_deformed_code(self):
        """Deformed checks commute and encode 2k - 1 logical qubits."""
        HX, HZ = self.deformed_matrices()
        anti = int(((HX.astype(np.int32) @ HZ.T.astype(np.int32)).toarray() % 2).sum())
        N = HX.shape[1]
        k_def = N - gf2_rank(HX) - gf2_rank(HZ)
        return anti == 0 and k_def == 2 * self.k - 1, dict(anticommuting=anti, k_deformed=k_def)


class Builder:
    def __init__(self, p):
        self.c = stim.Circuit()
        self.p = p
        self.nmeas = 0

    def meas(self, gate, qubits, flip=True):
        if flip and self.p > 0:
            self.c.append("Z_ERROR" if gate == "MX" else "X_ERROR", qubits, self.p)
        self.c.append(gate, qubits)
        idx = list(range(self.nmeas, self.nmeas + len(qubits)))
        self.nmeas += len(qubits)
        return idx

    def reset(self, gate, qubits, noisy=True):
        self.c.append(gate, qubits)
        if noisy and self.p > 0:
            self.c.append("Z_ERROR" if gate == "RX" else "X_ERROR", qubits, self.p)

    def rec(self, idxs):
        return [stim.target_rec(i - self.nmeas) for i in idxs]

    def detector(self, idxs, coords=()):
        cnt = {}
        for i in idxs:
            cnt[i] = cnt.get(i, 0) ^ 1
        idxs = sorted(i for i, b in cnt.items() if b)
        self.c.append("DETECTOR", self.rec(idxs), list(coords))

    def observable(self, idxs, k):
        cnt = {}
        for i in idxs:
            cnt[i] = cnt.get(i, 0) ^ 1
        idxs = sorted(i for i, b in cnt.items() if b)
        self.c.append("OBSERVABLE_INCLUDE", self.rec(idxs), k)

    def mpp(self, pauli_lists):
        out = []
        for plist in pauli_lists:
            t = []
            for P, q in plist:
                t += [stim.target_pauli(q, P), stim.target_combiner()]
            self.c.append("MPP", t[:-1])
            out.append(self.nmeas)
            self.nmeas += 1
        return out

    def extract(self, checks, basis, anc):
        p = self.p
        if not checks:
            return []
        a = [anc[i] for i in range(len(checks))]
        self.reset("RX" if basis == "X" else "R", a)
        wmax = max(len(c) for c in checks)
        for kk in range(wmax):
            pairs = []
            for i, c in enumerate(checks):
                if kk < len(c):
                    pairs += [a[i], c[kk]] if basis == "X" else [c[kk], a[i]]
            if pairs:
                self.c.append("CX", pairs)
                if p > 0:
                    self.c.append("DEPOLARIZE2", pairs, p)
        return self.meas("MX" if basis == "X" else "M", a)


def observable_names(k, basis, idle=False):
    if basis == "X":
        names = [] if idle else ["m"]
        names += ["X1a", "X1b"]
        for i in range(2, k + 1):
            names += [f"X{i}a", f"X{i}b"]
    else:
        names = ["Z1aZ1b"]
        for i in range(2, k + 1):
            names += [f"Z{i}a", f"Z{i}b"]
    return names


def build_circuit(tb: TwoBlock, p: float, basis: str, pre: int, rg: int, post: int, idle=False):
    """Returns (circuit, observable names).  idle=True: no gauging, same duration."""
    n, E0, nE, k = tb.n, tb.E0, tb.nE, tb.k
    N = E0 + nE
    labels = [f"a{i}" for i in range(1, k + 1)] + [f"b{i}" for i in range(1, k + 1)]
    refs = {lab: N + i for i, lab in enumerate(labels)}
    anc0 = N + len(labels)
    n_ox, n_oz = len(tb.orig_x), len(tb.orig_z)
    n_g, n_f = len(tb.gauss), len(tb.flux)
    anc_ox = list(range(anc0, anc0 + n_ox))
    anc_oz = list(range(anc0 + n_ox, anc0 + n_ox + n_oz))
    anc_g = list(range(anc0 + n_ox + n_oz, anc0 + n_ox + n_oz + n_g))
    anc_f = list(range(anc0 + n_ox + n_oz + n_g, anc0 + n_ox + n_oz + n_g + n_f))
    B = Builder(p)
    data = list(range(2 * n))
    edgeq = list(range(E0, E0 + nE))

    B.reset("R", data + list(refs.values()), noisy=False)
    lastx = B.mpp([[("X", q) for q in c] for c in tb.orig_x])
    lastz = [[] for _ in tb.orig_z]
    L = {}
    for i in range(k):
        L[f"a{i+1}"] = (tb.sup(tb.Xs[i], 0), tb.sup(tb.Zs[i], 0))
        L[f"b{i+1}"] = (tb.sup(tb.Xs[i], 1), tb.sup(tb.Zs[i], 1))
    L = {lab: L[lab] for lab in labels}
    sig = {}
    for lab, (xs, zs) in L.items():
        sig[lab] = B.mpp([[("X", q) for q in xs] + [("X", refs[lab])]])[0]

    def idle_noise(qs):
        if p > 0:
            B.c.append("DEPOLARIZE1", qs, p)

    def original_round(t):
        nonlocal lastx, lastz
        idle_noise(data)
        mx = B.extract(tb.orig_x, "X", anc_ox)
        mz = B.extract(tb.orig_z, "Z", anc_oz)
        for i, r in enumerate(mx):
            B.detector([r] + list(np.atleast_1d(lastx[i])), (i, t, 0))
        for i, r in enumerate(mz):
            B.detector([r] + list(lastz[i]), (i, t, 1))
        lastx = [[r] for r in mx]
        lastz = [[r] for r in mz]

    if idle:
        for t in range(pre + rg + post):
            original_round(t)
        T = pre + rg + post
        m_first, erec = None, {}
    else:
        for t in range(pre):
            original_round(t)
        B.reset("R", edgeq)
        lastg, lastf = None, None
        m_first = None
        for t in range(rg):
            idle_noise(data + edgeq)
            mx = B.extract(tb.orig_x, "X", anc_ox)
            mg = B.extract(tb.gauss, "X", anc_g)
            mz = B.extract(tb.def_z, "Z", anc_oz)
            mf = B.extract(tb.flux, "Z", anc_f)
            for i, r in enumerate(mx):
                B.detector([r] + list(np.atleast_1d(lastx[i])), (i, pre + t, 0))
            for i, r in enumerate(mz):
                B.detector([r] + list(lastz[i]), (i, pre + t, 1))
            if t == 0:
                m_first = mg
                for i, r in enumerate(mf):
                    B.detector([r], (i, pre + t, 3))
            else:
                for i, r in enumerate(mg):
                    B.detector([r, lastg[i]], (i, pre + t, 2))
                for i, r in enumerate(mf):
                    B.detector([r, lastf[i]], (i, pre + t, 3))
            lastx = [[r] for r in mx]
            lastz = [[r] for r in mz]
            lastg, lastf = mg, mf
        me = B.meas("M", edgeq)
        erec = {E0 + kk: me[kk] for kk in range(nE)}
        for i, c in enumerate(tb.flux):
            B.detector([lastf[i]] + [erec[q] for q in c], (i, pre + rg, 3))
        lastz = [list(lastz[i]) + [erec[q] for q in tb.gamma[i]] for i in range(len(tb.orig_z))]
        for t in range(post):
            original_round(pre + rg + 1 + t)
        T = pre + rg + 1 + post

    def byp(zs):
        return [] if idle else [erec[e] for e in tb.byproduct_edges(zs)]

    if basis == "X":
        md = B.meas("MX", data)
        mr = B.meas("MX", list(refs.values()), flip=False)
        for i, c in enumerate(tb.orig_x):
            B.detector([md[q] for q in c] + list(np.atleast_1d(lastx[i])), (i, T, 0))
        rref = dict(zip(refs, mr))
        j = 0
        if not idle:
            B.observable(list(m_first) + [sig["a1"], sig["b1"], rref["a1"], rref["b1"]], 0)
            j = 1
        order = ["a1", "b1"] + [lab for i in range(2, k + 1) for lab in (f"a{i}", f"b{i}")]
        for lab in order:
            B.observable([md[q] for q in L[lab][0]] + [sig[lab], rref[lab]], j)
            j += 1
    else:
        md = B.meas("M", data)
        mr = B.meas("M", list(refs.values()), flip=False)
        for i, c in enumerate(tb.orig_z):
            B.detector([md[q] for q in c] + list(lastz[i]), (i, T, 1))
        rref = dict(zip(refs, mr))
        zz = L["a1"][1] + L["b1"][1]
        B.observable([md[q] for q in zz] + [rref["a1"], rref["b1"]] + byp(zz), 0)
        j = 1
        for i in range(2, k + 1):
            for lab in (f"a{i}", f"b{i}"):
                zs = L[lab][1]
                B.observable([md[q] for q in zs] + [rref[lab]] + byp(zs), j)
                j += 1
    return B.c, observable_names(k, basis, idle)


def block_distances(k, s):
    """Code distances of one block, (d_X, d_Z) = (2s+1, 2s).

    Exact for s <= 4 and k in {2, 4} by the paper's CP-SAT runs
    (explicit_layer_code_distance.csv); for larger s these are the values of
    the distance theorem, (L_x + 1) ceil(2/eta) and L_z ceil(2/eta) with
    L = 2s, which the paper's search meets on every computed instance.
    """
    return 2 * s + 1, 2 * s


def noiseless_check(circuit, shots=64):
    c0 = circuit.without_noise()
    s = c0.compile_detector_sampler(seed=1)
    d, o = s.sample(shots, separate_observables=True)
    return (not d.any()) and (not o.any())
