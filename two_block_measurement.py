#!/usr/bin/env python3
"""Complete two-block logical measurement on explicit layer-code blocks (review point 3).

Two blocks of the explicit layer code (layer_code_explicit.py) with the
[[4,2,2]] input each carry K = 2 logical qubits.  Logical 1 of each block is
measured jointly, P = X1(a) X1(b), by Williamson--Yoder gauging on an ancilla
graph G attached to the slab representatives; logical 2 of each block is a
spectator.  Every logical qubit starts maximally entangled with a noiseless
reference qubit, so the protocol is tested on an unknown, entangled input: the
eight observables below are a complete generating set of the output Choi
state, split between an X-basis and a Z-basis final readout.

  X experiment  m + x(Ra1) + x(Rb1)      reported measurement bit
                X1(a) + x(Ra1)           measured pair, X sector
                X1(b) + x(Rb1)
                X2(a) + x(Ra2)           spectators, X sector
                X2(b) + x(Rb2)
  Z experiment  Z1(a) Z1(b) + z(Ra1) z(Rb1)   measured pair, Z sector
                Z2(a) + z(Ra2)           spectators, Z sector
                Z2(b) + z(Rb2)

(initial signs of the noiseless encoding are folded into each observable).

Interfaces compared:
  bridge-w   each block's ports carry their own planar grid; the two grids
             are joined by w parallel grid rows (w = 1 is a thin bridge);
  thick      one planar grid spans both blocks;
  thick x m  the same grid with m parallel ancillas on every edge.

For each interface the script (i) validates the deformed code, (ii) checks
the noiseless circuit (every detector and observable deterministic, zero
syndrome), (iii) runs stim's undetectable-logical-error search on the full
circuit-level detector error model to find the shortest undetectable fault
pattern for each observable, and optionally (iv) samples logical failure
rates with BP+LSD (localized-statistics decoding).

Noise (uniform circuit model, strength p): DEPOLARIZE2(p) after every CNOT,
DEPOLARIZE1(p) on every data/edge qubit once per round, X/Z flips of
probability p after every reset and before every measurement, including the
final data readout.  The initial encoding and the reference qubits are
noiseless (they define the test input and the ideal comparison).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from layer_code_explicit import build_layer_code, gf2_rank, x_slab, z_slab  # noqa: E402

X1_IN, X2_IN = np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
Z1_IN, Z2_IN = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])


def rows_of(M):
    M = sp.csr_matrix(M)
    return [list(M.indices[M.indptr[r]:M.indptr[r + 1]]) for r in range(M.shape[0])]


# --------------------------------------------------------------------------
# Deformed code
# --------------------------------------------------------------------------

class TwoBlock:
    def __init__(self, s=2, interface="thick", width=1, mult=1, z0=1, gap=2):
        H = np.ones((1, 4), np.uint8)
        self.lc = lc = build_layer_code(H, H, spacing=s)
        n = self.n = lc.n
        g = lc.geometry
        self.s, self.interface, self.width, self.mult = s, interface, width, mult
        Lx = g["Lx"]
        self.delta = Lx + gap
        # logical supports within a block
        self.X1 = x_slab(lc, X1_IN, z0)
        self.X2 = x_slab(lc, X2_IN, 0)
        self.Z1 = z_slab(lc, Z1_IN, 0)
        self.Z2 = z_slab(lc, Z2_IN, Lx)
        # block checks
        bx, bz = rows_of(lc.HX), rows_of(lc.HZ)
        self.orig_x = [c for c in bx] + [[q + n for q in c] for c in bx]
        self.orig_z = [c for c in bz] + [[q + n for q in c] for c in bz]
        # ports: qubit -> grid vertex (planar, doubled resolution)
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
        # grid vertices
        verts = set()
        if interface == "thick":
            for X in range(min(xa), max(xb) + 1):
                for Y in range(self.ylo, self.yhi + 1):
                    verts.add((X, Y))
        else:   # bridge
            for (lo, hi) in ((min(xa), max(xa)), (min(xb), max(xb))):
                for X in range(lo, hi + 1):
                    for Y in range(self.ylo, self.yhi + 1):
                        verts.add((X, Y))
            rows = list(range(self.ylo, self.yhi + 1))
            mid = len(rows) // 2
            chosen = sorted(rows, key=lambda r: abs(r - rows[mid]))[:width]
            for X in range(max(xa) + 1, min(xb)):
                for Y in chosen:
                    verts.add((X, Y))
        self.verts = sorted(verts)
        vid = {v: i for i, v in enumerate(self.verts)}
        self.vid = vid
        edges, adj = [], {v: [] for v in self.verts}
        for (X, Y) in self.verts:
            for d in ((1, 0), (0, 1)):
                w = (X + d[0], Y + d[1])
                if w in vid:
                    for c in range(mult):
                        k = len(edges)
                        edges.append(((X, Y), w, c))
                        adj[(X, Y)].append((w, k, c))
                        adj[w].append(((X, Y), k, c))
        self.edges, self.adj = edges, adj
        self.nE = len(edges)
        self.E0 = 2 * n            # first edge-qubit index
        vport = {}
        for q, v in ports.items():
            assert v in vid and v not in vport, "ports must map to distinct grid vertices"
            vport[v] = q
        self.vport = vport
        # Gauss laws
        self.gauss = []
        for v in self.verts:
            sup = [self.E0 + k for _, k, _ in adj[v]]
            if v in vport:
                sup.append(vport[v])
            self.gauss.append(sorted(sup))
        # flux: unit squares (copy 0) and parallel-copy pairs
        self.flux = []
        for (X, Y) in self.verts:
            sq = [(X, Y), (X + 1, Y), (X + 1, Y + 1), (X, Y + 1)]
            if all(p in vid for p in sq):
                ks = []
                for a, b in zip(sq, sq[1:] + sq[:1]):
                    ks += [self.E0 + k for w, k, c in adj[a] if w == b and c == 0]
                self.flux.append(sorted(ks))
        for k, (u, w, c) in enumerate(edges):
            if c > 0:
                k0 = next(kk for ww, kk, cc in adj[u] if ww == w and cc == 0)
                self.flux.append([self.E0 + k0, self.E0 + k])
        # deformed Z checks
        self.def_z, self.gamma = [], []
        for c in self.orig_z:
            pp = [self.ports[q] for q in c if q in self.ports]
            assert len(pp) % 2 == 0
            path = []
            for i in range(0, len(pp), 2):
                path += self.path(pp[i], pp[i + 1])
            self.gamma.append(sorted(path))
            self.def_z.append(sorted(set(c) ^ set(path)))
        # spanning tree for the byproduct frame
        self.tree_parent = self.bfs_tree(self.verts[0])

    def path(self, u, v):
        prev = {u: None}
        dq = deque([u])
        while dq:
            a = dq.popleft()
            if a == v:
                break
            for b, k, c in self.adj[a]:
                if c == 0 and b not in prev:
                    prev[b] = (a, k)
                    dq.append(b)
        out, a = [], v
        while prev[a] is not None:
            a, k = prev[a]
            out.append(self.E0 + k)
        return out

    def bfs_tree(self, root):
        prev = {root: None}
        dq = deque([root])
        while dq:
            a = dq.popleft()
            for b, k, c in self.adj[a]:
                if c == 0 and b not in prev:
                    prev[b] = (a, self.E0 + k)
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
        """Edge records whose parity gives the X-byproduct's effect on a Z operator."""
        cnt = {}
        for q in data_support:
            if q in self.ports:
                for e in self.tree_path_edges(self.ports[q]):
                    cnt[e] = cnt.get(e, 0) ^ 1
        return sorted(e for e, b in cnt.items() if b)

    # full logical supports (global indices)
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
        HX = mat(self.orig_x + self.gauss)
        HZ = mat(self.def_z + self.flux)
        return HX, HZ

    def cut_size(self):
        """Edges between the two blocks' port regions (min cut in the bridge)."""
        xa = max(v[0] for q, v in self.ports.items() if q < self.n)
        xb = min(v[0] for q, v in self.ports.items() if q >= self.n)
        mid = (xa + xb) / 2
        return sum(1 for (u, w, c) in self.edges if (u[0] < mid) != (w[0] < mid))


# --------------------------------------------------------------------------
# Circuit
# --------------------------------------------------------------------------

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
        idxs = [i for i in idxs]
        # cancel duplicates
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
        """Noiseless multi-qubit Pauli product measurements."""
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
        """One ancilla-based measurement of each check; returns record indices."""
        p = self.p
        if not checks:
            return []
        a = [anc[i] for i in range(len(checks))]
        self.reset("RX" if basis == "X" else "R", a)
        wmax = max(len(c) for c in checks)
        for k in range(wmax):
            pairs = []
            for i, c in enumerate(checks):
                if k < len(c):
                    pairs += [a[i], c[k]] if basis == "X" else [c[k], a[i]]
            if pairs:
                self.c.append("CX", pairs)
                if p > 0:
                    self.c.append("DEPOLARIZE2", pairs, p)
        return self.meas("MX" if basis == "X" else "M", a)


def build_circuit(tb: TwoBlock, p: float, basis: str, pre: int, rg: int, post: int):
    n, E0, nE = tb.n, tb.E0, tb.nE
    N = E0 + nE
    refs = {k: N + i for i, k in enumerate(("a1", "a2", "b1", "b2"))}
    anc0 = N + 4
    n_ox, n_oz = len(tb.orig_x), len(tb.orig_z)
    n_g, n_f = len(tb.gauss), len(tb.flux)
    anc_ox = list(range(anc0, anc0 + n_ox))
    anc_oz = list(range(anc0 + n_ox, anc0 + n_ox + n_oz))
    anc_g = list(range(anc0 + n_ox + n_oz, anc0 + n_ox + n_oz + n_g))
    anc_f = list(range(anc0 + n_ox + n_oz + n_g, anc0 + n_ox + n_oz + n_g + n_f))
    B = Builder(p)
    data = list(range(2 * n))
    edgeq = list(range(E0, E0 + nE))

    # ---- noiseless Choi-state encoding -----------------------------------
    B.reset("R", data + list(refs.values()), noisy=False)
    lastx = B.mpp([[("X", q) for q in c] for c in tb.orig_x])
    lastz = [[] for _ in tb.orig_z]            # deterministic +1
    L = {"a1": (tb.sup(tb.X1, 0), tb.sup(tb.Z1, 0)), "a2": (tb.sup(tb.X2, 0), tb.sup(tb.Z2, 0)),
         "b1": (tb.sup(tb.X1, 1), tb.sup(tb.Z1, 1)), "b2": (tb.sup(tb.X2, 1), tb.sup(tb.Z2, 1))}
    sig = {}
    for k, (xs, zs) in L.items():
        sig[k] = B.mpp([[("X", q) for q in xs] + [("X", refs[k])]])[0]

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

    for t in range(pre):
        original_round(t)

    # ---- gauging ------------------------------------------------------------
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
    # ungauge: read edges in Z
    me = B.meas("M", edgeq)
    erec = {E0 + k: me[k] for k in range(nE)}
    for i, c in enumerate(tb.flux):
        B.detector([lastf[i]] + [erec[q] for q in c], (i, pre + rg, 3))
    lastz = [list(lastz[i]) + [erec[q] for q in tb.gamma[i]] for i in range(len(tb.orig_z))]

    for t in range(post):
        original_round(pre + rg + 1 + t)

    # ---- final readout --------------------------------------------------------
    T = pre + rg + 1 + post
    if basis == "X":
        md = B.meas("MX", data)
        mr = B.meas("MX", list(refs.values()), flip=False)
        for i, c in enumerate(tb.orig_x):
            B.detector([md[q] for q in c] + list(np.atleast_1d(lastx[i])), (i, T, 0))
        rref = dict(zip(refs, mr))
        B.observable(list(m_first) + [sig["a1"], sig["b1"], rref["a1"], rref["b1"]], 0)
        for j, k in enumerate(("a1", "b1", "a2", "b2")):
            B.observable([md[q] for q in L[k][0]] + [sig[k], rref[k]], j + 1)
        names = ["m", "X1a", "X1b", "X2a", "X2b"]
    else:
        md = B.meas("M", data)
        mr = B.meas("M", list(refs.values()), flip=False)
        for i, c in enumerate(tb.orig_z):
            B.detector([md[q] for q in c] + list(lastz[i]), (i, T, 1))
        rref = dict(zip(refs, mr))
        zz = L["a1"][1] + L["b1"][1]
        B.observable([md[q] for q in zz] + [rref["a1"], rref["b1"]]
                     + [erec[e] for e in tb.byproduct_edges(zz)], 0)
        for j, k in enumerate(("a2", "b2")):
            zs = L[k][1]
            B.observable([md[q] for q in zs] + [rref[k]]
                         + [erec[e] for e in tb.byproduct_edges(zs)], j + 1)
        names = ["Z1aZ1b", "Z2a", "Z2b"]
    return B.c, names


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def logical_basis(HX, HZ):
    """Return (Xlogicals, Zlogicals) as dense arrays for a CSS code (small n)."""
    from ldpc import mod2
    def ker(H):
        K = mod2.nullspace(sp.csr_matrix(H))
        return np.asarray(K.todense() if sp.issparse(K) else K, np.uint8)
    def complement(V, H):
        # rows of V independent modulo rowspace(H)
        base = sp.csr_matrix(H)
        r = gf2_rank(base)
        out = []
        cur = base
        for v in V:
            trial = sp.vstack([cur, sp.csr_matrix(v.reshape(1, -1))])
            rr = gf2_rank(trial)
            if rr > r:
                out.append(v)
                cur, r = trial, rr
        return np.array(out, np.uint8)
    return complement(ker(HZ), HX), complement(ker(HX), HZ)


def code_distances(tb: TwoBlock):
    HX, HZ = tb.deformed_matrices()
    anti = int(((HX.astype(np.int32) @ HZ.T.astype(np.int32)).toarray() % 2).sum())
    N = HX.shape[1]
    k_def = N - gf2_rank(HX) - gf2_rank(HZ)
    res = dict(deformed_qubits=N, deformed_anticommuting=anti, deformed_k=k_def)
    # code-capacity distances with stim (X errors -> Z checks, and vice versa)
    XL, ZL = logical_basis(HX, HZ)
    for sector, Hdet, Lobs, P, R, Err in (("dX", HZ, ZL, "Z", "R", "X_ERROR"),
                                          ("dZ", HX, XL, "X", "RX", "Z_ERROR")):
        c = stim.Circuit()
        c.append(R, range(N))
        c.append(Err, range(N), 0.01)
        for row in rows_of(Hdet):
            t = []
            for q in row:
                t += [stim.target_pauli(q, P), stim.target_combiner()]
            c.append("MPP", t[:-1])
            c.append("DETECTOR", [stim.target_rec(-1)])
        # logical observables: only those deterministic on the product state
        for j, v in enumerate(Lobs):
            t = []
            for q in np.nonzero(v)[0]:
                t += [stim.target_pauli(int(q), P), stim.target_combiner()]
            c.append("MPP", t[:-1])
            c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], j)
        from min_weight_logical import min_weight_logical
        dem = c.detector_error_model(decompose_errors=False)
        ds = [min_weight_logical(dem, j, 120.0) for j in range(len(Lobs))]
        vals = [d["distance"] for d in ds if d["distance"] is not None]
        res[sector] = min(vals) if vals else None
        res[sector + "_exact"] = all(d["status"] == "OPTIMAL" for d in ds)
    return res


def sample_failures(circuit, names, shots, seed=0):
    from ldpc import BpLsdDecoder
    dem = circuit.detector_error_model(decompose_errors=False)
    # DEM -> matrices
    H_rows, L_rows, pri = [], [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        obs = [t.val for t in inst.targets_copy() if t.is_logical_observable_id()]
        H_rows.append(dets); L_rows.append(obs); pri.append(inst.args_copy()[0])
    nd, no, ne = dem.num_detectors, dem.num_observables, len(pri)
    def mat(rows, m):
        r, c = [], []
        for j, row in enumerate(rows):
            for i in row:
                r.append(i); c.append(j)
        M = sp.csr_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(m, ne))
        M.data &= 1
        return M
    Hm, Lm = mat(H_rows, nd), mat(L_rows, no)
    dec = BpLsdDecoder(Hm, error_channel=list(pri), max_iter=30, bp_method="minimum_sum",
                       ms_scaling_factor=0.9, schedule="serial", lsd_method="lsd_cs", lsd_order=4)
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    fails = np.zeros(no, int)
    anyfail = 0
    for i in range(shots):
        corr = dec.decode(dets[i].astype(np.uint8))
        pred = (Lm @ corr) % 2
        bad = pred.astype(bool) ^ obs[i]
        fails += bad
        anyfail += int(bad.any())
    return dict(zip(names, (fails / shots).tolist())), anyfail / shots


def noiseless_check(circuit, shots=64):
    c0 = circuit.without_noise()
    s = c0.compile_detector_sampler(seed=1)
    d, o = s.sample(shots, separate_observables=True)
    return (not d.any()) and (not o.any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=int, default=2)
    ap.add_argument("--configs", default="bridge:1:1,bridge:2:1,bridge:3:1,thick:0:1,thick:0:2",
                    help="interface:width:multiplicity, comma separated")
    ap.add_argument("--rounds", type=int, default=0, help="rounds per phase (default 2*spacing)")
    ap.add_argument("--distance", action="store_true", help="exact fault distances (CP-SAT)")
    ap.add_argument("--time-limit", type=float, default=240.0)
    ap.add_argument("--p", type=float, default=0.002)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260926,
                    help="base seed; each (interface, basis) pair adds a CRC32 offset, recorded in the CSV")
    ap.add_argument("--skip-code-distance", action="store_true",
                    help="skip the code-level deformed distances (e.g. for a sampling-only run)")
    ap.add_argument("--out", default=str(ROOT / "two_block_measurement.csv"))
    args = ap.parse_args()
    from min_weight_logical import circuit_distances
    s = args.spacing
    rows = []
    for spec in args.configs.split(","):
        interface, w, m = spec.split(":")
        w, m = int(w), int(m)
        tb = TwoBlock(s=s, interface=interface, width=w, mult=m)
        lab = f"bridge w={w}" if interface == "bridge" else f"thick x{m}"
        t0 = time.time()
        r = args.rounds or (2 * s)
        info = dict(spacing=s, interface=lab, block_qubits=tb.n, edge_qubits=tb.nE,
                    ports=len(tb.ports), interface_cut=tb.cut_size(), rounds_per_phase=r)
        if not args.skip_code_distance:
            info.update(code_distances(tb))
        for basis in ("X", "Z"):
            circ, names = build_circuit(tb, args.p, basis, pre=r, rg=r, post=r)
            info[f"{basis}_noiseless_deterministic"] = noiseless_check(circ)
            info[f"{basis}_detectors"] = circ.num_detectors
            if args.distance:
                dist = circuit_distances(circ, names, time_limit=args.time_limit, sector_coord=2,
                                         sector_values={0, 2} if basis == "X" else {1, 3},
                                         only={"m", "X1a", "X2a", "Z1aZ1b", "Z2a"})
                for nm, d in dist.items():
                    info[f"fd_{nm}"] = d["distance"]
                    info[f"fd_{nm}_lower"] = d["lower_bound"]
                    info[f"fd_{nm}_status"] = d["status"]
            if args.shots:
                # Recorded integer seed: stable across processes (Python's hash() of a
                # string is salted per process and must not be used for seeding).
                seed = (args.seed + zlib.crc32(f"{lab}|{basis}".encode())) % 2**31
                fr, anyf = sample_failures(circ, names, args.shots, seed=seed)
                info[f"seed_{basis}"] = seed
                for nm, f in fr.items():
                    info[f"fail_{nm}"] = round(f, 5)
                    info[f"nfail_{nm}"] = int(round(f * args.shots))
                info[f"fail_any_{basis}"] = round(anyf, 5)
                info[f"nfail_any_{basis}"] = int(round(anyf * args.shots))
                info["p"] = args.p
                info["shots"] = args.shots
        info["seconds"] = round(time.time() - t0, 1)
        rows.append(info)
        print(info, flush=True)
        keys = []
        for rr in rows:
            for k in rr:
                if k not in keys:
                    keys.append(k)
        with open(args.out, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(rows)


if __name__ == "__main__":
    main()
