#!/usr/bin/env python3
"""Direct-versus-composed check of the boundary interface (boundary composition).

A referee found that the worked gluing example of the previous version let a
measurement-record error change the quantum boundary syndrome: two flipped
records around a cut, with no data error, produce the physical detector
history (1, 0, 1), but the old boundary equations accepted the assignment by
setting s_mid = 1.  The repair carries, across every cut, the classical record
information needed by detector rows that straddle the cut, separately from
the quantum boundary class:

    s_t = s_{t-1} + f_t        (physical syndrome, quantum)
    r_t = s_t + mu_t           (reported record, classical)
    d_t = r_{t-1} + r_t        (detector)

so a cut carries (s, lambda, rho): the syndrome and logical frame of the
inherited Pauli class in the frame of the realized branch, and the records rho
of the last round before the cut.

This script tests that interface on the paper's own objects.  Two explicit
[[4,2,2]]-input layer-code blocks (s = 2, n = 224 each, from
two_block_measurement.TwoBlock with the spanning grid) run the schedule

    idle(2) | gauge | idle(2) | gauge | idle(1)

where each gauge primitive is: one undeformed entry round, edge preparation
in |0>, two rounds of the deformed code measuring X1(a) X1(b), edge readout in
Z, one undeformed exit round.  Checks are measured as multi-qubit Pauli
products (phenomenological model); every data or edge Pauli before a round,
every record flip and every edge-readout flip is a fault mechanism.  The
detectors are the local catalog rows T0, T3a/b and T4a/b of the supplement,
including the rows that straddle each cut.  The protected outcomes are the two
reported measurement bits; the gauging byproduct is carried in the frame.

  direct    one Stim frame simulation of the whole physical circuit (no
            reduction at cuts; byproducts are applied to the frame only at the
            end);
  composed  each primitive simulated in isolation, with its input given only
            by (s, lambda, rho) at the cut and re-expanded as a canonical
            representative Pauli; outputs chained by linear algebra.  The
            logical representatives change from cut to cut (slabs at
            different heights and columns), so the syndrome terms C s_in of
            the frame propagation are exercised.

Checks (written to boundary_composition_check.csv):
  1. Every detector of the annotated noiseless circuit is deterministic
     (Stim builds its detector error model).
  2. For EVERY single fault mechanism of the schedule, and for every unit
     inherited input (s0, lambda0, rho0) at the first cut, the direct and
     composed maps agree on all detectors, both reported bits, and the final
     (s, lambda, rho).  By linearity this is the equality of the two maps,
     including inherited syndromes and inherited record errors at every cut.
  3. Negative control: gluing with (s, lambda) only, i.e. assuming the last
     record before each cut equals the syndrome, disagrees exactly on record
     flips in the last round before an internal cut.
  4. The referee's pattern (the records of one check flipped in the rounds on
     both sides of a cut, clean records around them) gives the detector
     history (1, 0, 1) in the direct circuit and in the composed map.
"""
from __future__ import annotations

import csv
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from layer_code_explicit import x_slab, z_slab  # noqa: E402
from two_block_measurement import TwoBlock, X1_IN, X2_IN, Z1_IN, Z2_IN  # noqa: E402

CHUNK = 4096


# ---------------------------------------------------------------------------
# GF(2) helpers
# ---------------------------------------------------------------------------

def mm2(A, B):
    """Matrix product over GF(2) via exact float32 BLAS (entries 0/1)."""
    return (np.asarray(A, np.float32) @ np.asarray(B, np.float32)).astype(np.int64) % 2


def rref(M):
    """Row-reduce a copy of M over GF(2); return (R, pivots, T) with R = T M."""
    A = (np.asarray(M, np.uint8) & 1).copy()
    m, ncol = A.shape
    T = np.eye(m, dtype=np.uint8)
    piv, row = [], 0
    for c in range(ncol):
        if row == m:
            break
        nz = np.nonzero(A[row:, c])[0]
        if nz.size == 0:
            continue
        p = row + nz[0]
        if p != row:
            A[[row, p]] = A[[p, row]]
            T[[row, p]] = T[[p, row]]
        others = np.nonzero(A[:, c])[0]
        others = others[others != row]
        A[others] ^= A[row]
        T[others] ^= T[row]
        piv.append(c)
        row += 1
    return A, piv, T


def independent_rows(M):
    """Indices of a maximal independent subset of the rows of M (greedy)."""
    keep, basis, piv = [], [], []
    for i, v in enumerate(np.asarray(M, np.uint8)):
        w = v.copy()
        for b, c in zip(basis, piv):
            if w[c]:
                w ^= b
        nz = np.nonzero(w)[0]
        if nz.size:
            basis.append(w)
            piv.append(nz[0])
            keep.append(i)
    return keep


def right_inverse(A):
    """A (r x m) of full row rank; return B (m x r) with A B = I over GF(2)."""
    r, m = A.shape
    _, piv, T = rref(A)
    assert len(piv) == r, "rows must be independent"
    B = np.zeros((m, r), np.uint8)
    for i, c in enumerate(piv):
        B[c] = T[i]
    assert np.array_equal(mm2(A, B), np.eye(r, dtype=np.int64))
    return B


# ---------------------------------------------------------------------------
# The two-block instance, checks and representatives
# ---------------------------------------------------------------------------

TB = TwoBlock(s=2, interface="thick", width=1, mult=1)
NB = TB.n
N = TB.E0 + TB.nE
DATA = list(range(2 * NB))
EDGES = list(range(TB.E0, N))
LX, LZ = TB.lc.geometry["Lx"], TB.lc.geometry["Lz"]

# undeformed check set C0 (on data) and deformed set; a check is (key, index, type, support)
C0 = [("x", i, "X", tuple(c)) for i, c in enumerate(TB.orig_x)] + \
     [("z", j, "Z", tuple(c)) for j, c in enumerate(TB.orig_z)]
DEF = [("x", i, "X", tuple(c)) for i, c in enumerate(TB.orig_x)] + \
      [("g", v, "X", tuple(c)) for v, c in enumerate(TB.gauss)] + \
      [(("zt" if TB.gamma[j] else "z"), j, "Z", tuple(c)) for j, c in enumerate(TB.def_z)] + \
      [("b", p, "Z", tuple(c)) for p, c in enumerate(TB.flux)]
C0_KEYS = [(k, i) for k, i, _, _ in C0]
C0_INDEX = {key: a for a, key in enumerate(C0_KEYS)}


def symp(ptype, sup):
    v = np.zeros(2 * N, np.uint8)
    off = 0 if ptype == "X" else N
    for q in sup:
        v[off + q] ^= 1
    return v


def comm_matrix(rows):
    """Rows R (k x 2N); return M with (M @ P) mod 2 = symplectic <R_i, P>."""
    R = np.asarray(rows, np.uint8)
    return np.hstack([R[:, N:], R[:, :N]])


C0_SYMP = np.array([symp(t, s) for _, _, t, s in C0])
C0_COMM = comm_matrix(C0_SYMP)
GENS = C0_SYMP[independent_rows(C0_SYMP)]      # independent generating set at a cut
NG, NL, NC = len(GENS), 8, len(C0)
NIN = NG + NL + NC                              # dimension of (s, lambda, rho)


@lru_cache(maxsize=None)
def reps_at_cut(k):
    """Frame partners at cut k: Zbar(1a,2a,1b,2b) then Xbar(1a,2a,1b,2b).
    The slabs move from cut to cut, so consecutive cuts use different
    representatives of the same classes (they differ by stabilizers)."""
    zc = [k % (LX + 1), (k + 2) % (LX + 1)]
    xh = [k % LZ, (k + 1) % LZ]
    out = []
    for blk in (0, 1):
        for zin, x0 in ((Z1_IN, zc[0]), (Z2_IN, zc[1])):
            out.append(symp("Z", [int(q) + blk * NB for q in np.nonzero(z_slab(TB.lc, zin, x0))[0]]))
    for blk in (0, 1):
        for xin, z0 in ((X1_IN, xh[0]), (X2_IN, xh[1])):
            out.append(symp("X", [int(q) + blk * NB for q in np.nonzero(x_slab(TB.lc, xin, z0))[0]]))
    return np.array(out)


@lru_cache(maxsize=None)
def rep_inverse(k):
    """(2N x (NG+NL)): column u is a canonical representative Pauli whose
    (s, lambda) coordinates at cut k are the u-th unit vector."""
    return right_inverse(comm_matrix(np.vstack([GENS, reps_at_cut(k)])))


# gauging byproduct: X on port q with exponent the sum of edge readouts along
# the tree path from the root to q's grid vertex
BYP = np.zeros((2 * N, TB.nE), np.uint8)        # edge-readout flips -> X deviation
for q in TB.ports:
    for e in TB.tree_path_edges(TB.ports[q]):
        BYP[q, e - TB.E0] ^= 1


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def mpp_targets(rows_or_checks):
    tg = []
    for item in rows_or_checks:
        if isinstance(item, tuple):
            _, _, t, sup = item
            ps = [(q, t) for q in sup]
        else:
            ps = [(q, "Y" if (item[q] and item[N + q]) else ("X" if item[q] else "Z"))
                  for q in range(N) if item[q] or item[N + q]]
        for a, (q, t) in enumerate(ps):   # products are listed one after another
            if a:
                tg.append(stim.target_combiner())
            tg.append({"X": stim.target_x, "Z": stim.target_z, "Y": stim.target_y}[t](q))
    c = stim.Circuit()
    c.append("MPP", tg)
    return c


RC0, RDEF = mpp_targets(C0), mpp_targets(DEF)


class Primitive:
    """A catalog primitive: its rounds, record layout, detectors and outcomes."""

    def __init__(self, kind, R):
        self.kind = kind
        if kind == "idle":
            self.rounds = [("C0", DATA)] * R
        else:   # undeformed entry, R deformed rounds, undeformed exit
            self.rounds = [("C0", DATA)] + [("DEF", DATA + EDGES)] * R + [("C0", DATA)]
        last = len(self.rounds) - 1
        self.rec, self.edge_rec, m = {}, {}, 0
        for r, (lab, _) in enumerate(self.rounds):
            if kind == "gauge" and r == last:          # edge readout precedes the exit round
                for e in range(TB.nE):
                    self.edge_rec[e] = m
                    m += 1
            for key, i, _, _ in (C0 if lab == "C0" else DEF):
                self.rec[(r, (key, i))] = m
                m += 1
        self.nrec = m
        self.last_c0 = [self.rec[(last, key)] for key in C0_KEYS]
        # detector rows: (record indices inside the primitive, rho_in indices)
        self.dets = []
        for r, (lab, _) in enumerate(self.rounds):
            for key, i, _, _ in (C0 if lab == "C0" else DEF):
                me = self.rec[(r, (key, i))]
                if r == 0:                                       # T0 across the input cut
                    self.dets.append(([me], [C0_INDEX[(key, i)]]))
                elif (r - 1, (key, i)) in self.rec:              # T0, persistent check
                    self.dets.append(([self.rec[(r - 1, (key, i))], me], []))
                elif key == "zt":                                # T3a (edges prepared in |0>)
                    self.dets.append(([self.rec[(r - 1, ("z", i))], me], []))
                elif key == "b":                                 # T3b
                    self.dets.append(([me], []))
                elif key == "z" and (r - 1, ("zt", i)) in self.rec:   # T4a
                    ids = [self.rec[(r - 1, ("zt", i))], me] + \
                          [self.edge_rec[e - TB.E0] for e in TB.gamma[i]]
                    self.dets.append((ids, []))
                # key "g" in the first deformed round: open temporal endpoint, no row
            if kind == "gauge" and r == last:                    # T4b
                for p, cyc in enumerate(TB.flux):
                    ids = [self.rec[(r - 1, ("b", p))]] + [self.edge_rec[e - TB.E0] for e in cyc]
                    self.dets.append((ids, []))
        self.outcomes = [[self.rec[(1, ("g", v))] for v in range(len(TB.gauss))]] if kind == "gauge" else []
        self.mechs = []
        for r, (_, qs) in enumerate(self.rounds):
            for q in qs:
                self.mechs.append(("pauli", r, q, "X"))
                self.mechs.append(("pauli", r, q, "Z"))
        for (r, key), idx in self.rec.items():
            self.mechs.append(("meas", r, key, idx))
        for e, idx in self.edge_rec.items():
            self.mechs.append(("edge", e, idx))
        self.Dr = np.zeros((len(self.dets), self.nrec), np.uint8)
        self.Di = np.zeros((len(self.dets), NC), np.uint8)
        for a, (ids, rin) in enumerate(self.dets):
            for i in ids:
                self.Dr[a, i] ^= 1
            for i in rin:
                self.Di[a, i] ^= 1

    @property
    def nd(self):
        return len(self.dets)

    def find_row(self, ids, rin=()):
        for a, (ii, rr) in enumerate(self.dets):
            if sorted(ii) == sorted(ids) and sorted(rr) == sorted(rin):
                return a
        raise KeyError((ids, rin))


SCHEDULE = [Primitive("idle", 2), Primitive("gauge", 2), Primitive("idle", 2),
            Primitive("gauge", 2), Primitive("idle", 1)]
OFFS = np.cumsum([0] + [p.nd for p in SCHEDULE])


# ---------------------------------------------------------------------------
# Frame simulation of a sequence of primitives
# ---------------------------------------------------------------------------

def run(prims, P0, rho0, assign, frame_at_each_cut):
    """Simulate primitives in sequence for a batch.

    P0      (B x 2N) inherited deviation, injected at the start
    rho0    (B x NC) inherited records of the round before the first cut
    assign  per instance: a set of (primitive index, mechanism)
    frame_at_each_cut: True for an isolated primitive (the byproduct enters the
            frame at the output cut, so the output is in the realized frame);
            False for the direct physical circuit (byproducts are applied to
            the frame only at the very end).
    Returns detector and outcome flips per primitive, the last-round records
    (frame convention if frame_at_each_cut), the simulator, and the byproduct
    deviation still to be applied to the frame.
    """
    B = P0.shape[0]
    sim = stim.FlipSimulator(batch_size=B, disable_stabilizer_randomization=True, num_qubits=N)
    for pauli, off in (("X", 0), ("Z", N)):
        mask = P0[:, off:off + N].T.astype(bool)
        if mask.any():
            sim.broadcast_pauli_errors(pauli=pauli, mask=mask)
    pending = np.zeros((B, 2 * N), np.uint8)
    out_det, out_obs = [], []
    prev = rho0.astype(np.uint8)
    m0 = 0
    for g, prim in enumerate(prims):
        post = np.zeros((prim.nrec, B), np.uint8)
        inj = {}
        for b, mset in enumerate(assign):
            for (gg, mech) in mset:
                if gg != g:
                    continue
                if mech[0] == "meas":
                    post[mech[3], b] ^= 1
                elif mech[0] == "edge":
                    post[mech[2], b] ^= 1
                else:
                    inj.setdefault((mech[1], mech[3]), []).append((mech[2], b))
        last_r = len(prim.rounds) - 1
        for r, (lab, _) in enumerate(prim.rounds):
            if prim.kind == "gauge" and r == 1:
                sim.do(stim.Circuit("R " + " ".join(map(str, EDGES))))
            if prim.kind == "gauge" and r == last_r:
                sim.do(stim.Circuit("M " + " ".join(map(str, EDGES))))
            for pauli in ("X", "Z"):
                if (r, pauli) in inj:
                    mask = np.zeros((N, B), bool)
                    for q, b in inj[(r, pauli)]:
                        mask[q, b] ^= True
                    sim.broadcast_pauli_errors(pauli=pauli, mask=mask)
            sim.do(RC0 if lab == "C0" else RDEF)
        flips = sim.get_measurement_flips(bit_packed=False).astype(np.uint8)
        recs = flips[m0:m0 + prim.nrec] ^ post
        m0 += prim.nrec
        det = (mm2(prim.Dr, recs) + mm2(prim.Di, prev.T)) % 2
        obs = np.array([recs[ids].sum(0) % 2 for ids in prim.outcomes], np.uint8).reshape(-1, B)
        out_det.append(det.T.astype(np.uint8))
        out_obs.append(obs.T)
        last = recs[prim.last_c0].T.copy()
        if prim.kind == "gauge":
            dz = recs[[prim.edge_rec[e] for e in range(TB.nE)]].T
            dX = mm2(dz, BYP.T).astype(np.uint8)            # B x 2N, X part only
            if frame_at_each_cut:
                mask = dX[:, :N].T.astype(bool)
                if mask.any():
                    sim.broadcast_pauli_errors(pauli="X", mask=mask)
                # the exit round was measured before the byproduct entered the
                # frame: convert its records to the frame convention
                last ^= mm2(dX, C0_COMM.T).astype(np.uint8)
            else:
                pending ^= dX
        prev = last
    return out_det, out_obs, prev, sim, pending


def final_coordinates(sim, pending, k_cut):
    """Noiseless readout of (s, lambda) at cut k_cut, in the frame convention."""
    mask = pending[:, :N].T.astype(bool)
    if mask.any():
        sim.broadcast_pauli_errors(pauli="X", mask=mask)
    before = sim.num_measurements
    sim.do(mpp_targets(list(np.vstack([GENS, reps_at_cut(k_cut)]))))
    fl = sim.get_measurement_flips(bit_packed=False).astype(np.uint8)[before:]
    return fl[:NG].T, fl[NG:].T


def unit_input(u, k):
    P = np.zeros(2 * N, np.uint8)
    rho = np.zeros(NC, np.uint8)
    if u < NG + NL:
        P = rep_inverse(k)[:, u].copy()
    else:
        rho[u - NG - NL] = 1
    return P, rho


# ---------------------------------------------------------------------------
# Composed map
# ---------------------------------------------------------------------------

def isolated_response(g, prim):
    """Outputs (detectors, outcomes, s_out, lambda_out, rho_out) of primitive g
    for the NIN unit inputs at cut g (matrix In) and for each internal
    mechanism with zero input (matrix Int)."""
    tests = [(*unit_input(u, g), set()) for u in range(NIN)]
    tests += [(np.zeros(2 * N, np.uint8), np.zeros(NC, np.uint8), {(0, m)}) for m in prim.mechs]
    outs = []
    for c0 in range(0, len(tests), CHUNK):
        ch = tests[c0:c0 + CHUNK]
        det, obs, last, sim, pend = run([prim], np.array([t[0] for t in ch]),
                                        np.array([t[1] for t in ch]), [t[2] for t in ch], True)
        s, lam = final_coordinates(sim, pend, g + 1)
        outs.append(np.hstack([det[0], obs[0], s, lam, last]))
    O = np.vstack(outs)
    return O[:NIN], O[NIN:]


def chain(responses, g0, first, rho_rule):
    """Chain the responses of primitives g0, g0+1, ... starting from the
    outputs `first` (M x out_g0) of primitive g0.  rho_rule "full" glues with
    (s, lambda, rho); "no_rho" glues with (s, lambda) only, replacing rho by the
    records a fault-free last round would report for the representative."""
    M = first.shape[0]
    dets = [np.zeros((M, SCHEDULE[g].nd), np.uint8) for g in range(g0)]
    obss = [np.zeros((M, len(SCHEDULE[g].outcomes)), np.uint8) for g in range(g0)]
    o = first
    for g in range(g0, len(SCHEDULE)):
        if g > g0:
            b = bvec
            if rho_rule == "no_rho":
                P = mm2(b[:, :NG + NL], rep_inverse(g).T)
                b = np.hstack([b[:, :NG + NL], mm2(P, C0_COMM.T)])
            o = mm2(b, responses[g][0]).astype(np.uint8)
        nd, no = SCHEDULE[g].nd, len(SCHEDULE[g].outcomes)
        dets.append(o[:, :nd])
        obss.append(o[:, nd:nd + no])
        bvec = o[:, nd + no:]
    return np.hstack(dets + obss + [bvec]).astype(np.uint8)


def direct(tests):
    rows = []
    for c0 in range(0, len(tests), CHUNK):
        ch = tests[c0:c0 + CHUNK]
        det, obs, last, sim, pend = run(SCHEDULE, np.array([t[0] for t in ch]),
                                        np.array([t[1] for t in ch]), [t[2] for t in ch], False)
        s, lam = final_coordinates(sim, pend, len(SCHEDULE))
        last = last ^ mm2(pend, C0_COMM.T).astype(np.uint8)   # physical -> frame convention
        rows.append(np.hstack(det + [np.hstack(obs), s, lam, last]).astype(np.uint8))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def determinism_check():
    """Annotated noiseless circuit (a leading round stands in for the inherited
    records); Stim refuses to build a detector error model if any detector is
    not deterministic."""
    c = stim.Circuit()
    c += RC0
    prev = list(range(NC))
    base = NC
    for prim in SCHEDULE:
        for r, (lab, _) in enumerate(prim.rounds):
            if prim.kind == "gauge" and r == 1:
                c.append("R", EDGES)
            if prim.kind == "gauge" and r == len(prim.rounds) - 1:
                c.append("M", EDGES)
            c.append("DEPOLARIZE1", DATA, 1e-3)
            c += RC0 if lab == "C0" else RDEF
        total = base + prim.nrec
        for ids, rin in prim.dets:
            absids = [base + i for i in ids] + [prev[i] for i in rin]
            c.append("DETECTOR", [stim.target_rec(a - total) for a in absids])
        prev = [base + i for i in prim.last_c0]
        base = total
    try:
        dem = c.detector_error_model()
        return True, f"{dem.num_detectors} detectors, all deterministic"
    except ValueError as exc:
        return False, str(exc)[:300]


def referee_pattern(responses):
    """Record flips of the persistent check X_0 in the last round of the first
    idle primitive and in the entry round of the first gauge primitive."""
    key = ("x", 0)
    p0, p1 = SCHEDULE[0], SCHEDULE[1]
    ra, rb = len(p0.rounds) - 1, 0
    ma = ("meas", ra, key, p0.rec[(ra, key)])
    mb = ("meas", rb, key, p1.rec[(rb, key)])
    rows = [OFFS[0] + p0.find_row([p0.rec[(ra - 1, key)], p0.rec[(ra, key)]]),   # before the cut
            OFFS[1] + p1.find_row([p1.rec[(0, key)]], [C0_INDEX[key]]),          # straddling row
            OFFS[1] + p1.find_row([p1.rec[(0, key)], p1.rec[(1, key)]])]         # after the cut
    z = np.zeros((1, 2 * N), np.uint8), np.zeros((1, NC), np.uint8)
    d = direct([(z[0][0], z[1][0], {(0, ma), (1, mb)})])[0]
    ka = p0.mechs.index(ma)
    kb = p1.mechs.index(mb)
    comp = chain(responses, 0, responses[0][1][ka:ka + 1], "full")[0] ^ \
        chain(responses, 1, responses[1][1][kb:kb + 1], "full")[0]
    return [int(d[i]) for i in rows], [int(comp[i]) for i in rows], bool(np.array_equal(d, comp))


def main():
    print(f"instance: two [[4,2,2]] blocks, {2 * NB} data + {TB.nE} edge qubits, "
          f"{NC} undeformed checks ({NG} independent), {len(DEF)} deformed checks")
    ok_det, info = determinism_check()
    print("determinism:", ok_det, info, flush=True)
    responses = []
    for g, prim in enumerate(SCHEDULE):
        responses.append(isolated_response(g, prim))
        print(f"  isolated primitive {g} ({prim.kind}): {len(prim.mechs)} mechanisms, "
              f"{prim.nd} detectors", flush=True)
    groups = [("inherited", 0, responses[0][0],
               [(*unit_input(u, 0), set()) for u in range(NIN)])]
    for g, prim in enumerate(SCHEDULE):
        groups.append((f"primitive {g}", g, responses[g][1],
                       [(np.zeros(2 * N, np.uint8), np.zeros(NC, np.uint8), {(g, m)})
                        for m in prim.mechs]))
    n_tests = mism = mism_norho = 0
    norho_ok = True
    n_last = 0
    for name, g0, first, tests in groups:
        D = direct(tests)
        C = chain(responses, g0, first, "full")
        Cn = chain(responses, g0, first, "no_rho")
        bad = (C != D).any(1)
        badn = (Cn != D).any(1)
        n_tests += len(tests)
        mism += int(bad.sum())
        mism_norho += int(badn.sum())
        if name != "inherited":
            prim = SCHEDULE[g0]
            expect = np.array([m[0] == "meas" and m[1] == len(prim.rounds) - 1
                               and g0 < len(SCHEDULE) - 1 for m in prim.mechs])
            n_last += int(expect.sum())
            norho_ok &= bool(np.array_equal(badn, expect))
        else:
            norho_ok &= not badn.any()
        print(f"  {name}: {len(tests)} test inputs, composed mismatches {int(bad.sum())}, "
              f"(s,lambda)-only mismatches {int(badn.sum())}", flush=True)
    pat_d, pat_c, pat_eq = referee_pattern(responses)
    rows = [
        dict(test="every catalog detector deterministic (Stim DEM of the annotated circuit)",
             value=info, expected="builds", passed=ok_det),
        dict(test="direct vs composed with (s, lambda, rho): test inputs that disagree",
             value=f"{mism} of {n_tests} ({n_tests - NIN} single mechanisms and "
                   f"{NIN} unit inherited inputs)", expected="0", passed=mism == 0),
        dict(test="negative control, glue with (s, lambda) only: test inputs that disagree",
             value=f"{mism_norho} of {n_tests}", expected=f"{n_last} (last-round record flips "
             f"before the four internal cuts)", passed=norho_ok and mism_norho == n_last),
        dict(test="referee pattern, detector rows (before, straddling, after the first cut)",
             value=f"direct {pat_d}, composed {pat_c}, full outputs equal {pat_eq}",
             expected="[1, 0, 1]", passed=pat_d == [1, 0, 1] and pat_c == [1, 0, 1] and pat_eq),
    ]
    for r in rows:
        print(r)
    with open(ROOT / "boundary_composition_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = all(r["passed"] for r in rows)
    print("BOUNDARY COMPOSITION CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
