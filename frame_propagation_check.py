#!/usr/bin/env python3
"""Inherited-syndrome terms in the logical-frame propagation (review point 2).

At a cut with stabilizer generators S_1..S_r and chosen logical representatives
{Xbar_a, Zbar_a}, a Pauli deviation P has coordinates

    s_i(P)      = <P, S_i>                (syndrome)
    lambda(P)   = (<P, Zbar_a>, <P, Xbar_a>)_a   (frame)

where <.,.> is the symplectic commutator.  The frame depends on the chosen
representatives whenever s(P) != 0.  Across a primitive whose two cuts use
representatives that differ by stabilizers, the fault-free propagation is

    lambda_out = L lambda_in + C s_in,                                    (1)

with C built from the stabilizer coefficients of Rbar_out Rbar_in^{-1}.  The
script checks:

  1. the reviewer's example (stabilizer X1X2, representatives X1 and X2,
     error Z1): C = [1];
  2. the explicit layer code (layer_code_explicit.py), idle primitive whose
     input cut uses the X slabs at height 0 and whose output cut uses the
     slabs at height 1: C is computed, is nonzero, and (1) holds for every
     inherited Pauli in a random sample; dropping the C term fails for
     inherited Paulis with nonzero syndrome;
  3. the relative effect  Delta = lambda_out + L lambda_in + C s_in  depends
     only on internal faults (random inherited classes, random internal faults);
  4. telescoping: on a repeated-measurement idle chain, every detector-trivial
     fault pattern whose support lies strictly between two cuts has zero
     syndrome at both cuts (exhaustive enumeration), so the C s_in term never
     contributes to a contained component.

Output: frame_propagation_check.csv
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def comm(p, q):
    """Symplectic commutator of binary Paulis p=(x|z), q=(x|z)."""
    n = len(p) // 2
    return int((p[:n] @ q[n:] + p[n:] @ q[:n]) % 2)


def gf2_solve(A, b):
    A = (np.asarray(A, np.uint8) & 1).copy()
    b = (np.asarray(b, np.uint8) & 1).reshape(-1).copy()
    m, n = A.shape
    M = np.hstack([A, b.reshape(-1, 1)])
    row, piv = 0, []
    for c in range(n):
        nz = np.nonzero(M[row:, c])[0]
        if nz.size == 0:
            continue
        p = row + nz[0]
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
    x = np.zeros(n, np.uint8)
    for r, c in enumerate(piv):
        x[c] = M[r, n]
    return x


def coords(P, stabs, logs):
    s = np.array([comm(P, S) for S in stabs], np.uint8)
    lam = np.array([comm(P, L) for L in logs], np.uint8)
    return s, lam


def frame_maps(stabs, logs_in, logs_out):
    """L and C for an idle primitive whose cuts use different representatives.

    Each output representative equals the input one times a stabilizer
    product: logs_out[a] = logs_in[a] * prod_i S_i^{c_{a i}}.  Then
    <P, logs_out[a]> = <P, logs_in[a]> + sum_i c_{a i} s_i(P).
    """
    S = np.array(stabs, np.uint8)
    Cm = []
    for Li, Lo in zip(logs_in, logs_out):
        diff = (np.asarray(Li, np.uint8) ^ np.asarray(Lo, np.uint8))
        c = gf2_solve(S.T, diff)
        assert c is not None, "representatives must differ by a stabilizer"
        Cm.append(c)
    return np.eye(len(logs_in), dtype=np.uint8), np.array(Cm, np.uint8)


def reviewer_example():
    # two qubits, Paulis as (x1 x2 | z1 z2)
    S = [np.array([1, 1, 0, 0], np.uint8)]            # X1 X2
    logs_in = [np.array([1, 0, 0, 0], np.uint8)]      # frame partner Xbar = X1
    logs_out = [np.array([0, 1, 0, 0], np.uint8)]     # Xbar = X2
    Lm, Cm = frame_maps(S, logs_in, logs_out)
    Z1 = np.array([0, 0, 1, 0], np.uint8)
    s_in, lam_in = coords(Z1, S, logs_in)
    _, lam_out = coords(Z1, S, logs_out)
    return dict(test="reviewer example X1X2", C=Cm.tolist(), s_in=s_in.tolist(),
                lam_in=lam_in.tolist(), lam_out=lam_out.tolist(),
                law_with_C=bool(np.array_equal(lam_out, (Lm @ lam_in + Cm @ s_in) % 2)),
                law_without_C=bool(np.array_equal(lam_out, (Lm @ lam_in) % 2)))


def layer_code_example(samples=2000, seed=7):
    from layer_code_explicit import build_layer_code, x_slab, z_slab
    H = np.ones((1, 4), np.uint8)
    lc = build_layer_code(H, H, spacing=2)
    n = lc.n
    HX, HZ = lc.HX.toarray().astype(np.uint8), lc.HZ.toarray().astype(np.uint8)
    # stabilizers as symplectic vectors
    stabs = [np.concatenate([r, np.zeros(n, np.uint8)]) for r in HX] + \
            [np.concatenate([np.zeros(n, np.uint8), r]) for r in HZ]
    # independent generating set
    from layer_code_explicit import gf2_rank
    keep, cur = [], np.zeros((0, 2 * n), np.uint8)
    for v in stabs:
        t = np.vstack([cur, v])
        if gf2_rank(sp.csr_matrix(t)) > cur.shape[0]:
            cur = t
            keep.append(v)
    stabs = keep
    Xin = [np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])]
    Zin = [np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])]

    def xs(x, z0):
        return np.concatenate([x_slab(lc, x, z0), np.zeros(n, np.uint8)])

    def zs(z, x0):
        return np.concatenate([np.zeros(n, np.uint8), z_slab(lc, z, x0)])
    # frame partners: <P, Zbar> gives the X-frame bit, <P, Xbar> the Z-frame bit
    logs_in = [zs(z, 0) for z in Zin] + [xs(x, 0) for x in Xin]
    logs_out = [zs(z, lc.geometry["Lx"]) for z in Zin] + [xs(x, 1) for x in Xin]
    Lm, Cm = frame_maps(stabs, logs_in, logs_out)
    rng = np.random.default_rng(seed)
    ok_with = ok_without = n_s = 0
    rel_ok = 0
    for _ in range(samples):
        P = (rng.random(2 * n) < 0.01).astype(np.uint8)       # inherited deviation
        s_in, lam_in = coords(P, stabs, logs_in)
        _, lam_out = coords(P, stabs, logs_out)
        pred = (Lm @ lam_in + Cm @ s_in) % 2
        ok_with += np.array_equal(pred, lam_out)
        if s_in.any():
            n_s += 1
            ok_without += np.array_equal((Lm @ lam_in) % 2, lam_out)
        # relative effect of an internal fault e (a Pauli applied inside the
        # primitive): Delta = lambda_out(P e) + L lambda_in(P) + C s_in(P)
        e = (rng.random(2 * n) < 0.005).astype(np.uint8)
        _, lam_out_pe = coords(P ^ e, stabs, logs_out)
        delta = (lam_out_pe + Lm @ lam_in + Cm @ s_in) % 2
        _, lam_e = coords(e, stabs, logs_out)                 # effect of e alone
        rel_ok += np.array_equal(delta, lam_e)
    return dict(test="explicit layer code, idle with slab change", n=n,
                stabilizer_generators=len(stabs), C_nonzero_entries=int(Cm.sum()),
                samples=samples, law_with_C=f"{ok_with}/{samples}",
                law_without_C_on_nonzero_syndrome=f"{ok_without}/{n_s}",
                relative_effect_independent_of_inherited_class=f"{rel_ok}/{samples}")


def telescoping_check(T=4, checks=2):
    """Idle chain: persistent checks measured T times between two cuts.

    Variables: data flips f[t][c] before round t (t = 1..T, one bit per check
    pattern, i.e. the syndrome change it causes), measurement flips mu[t][c].
    Detectors: round 1 vs cut-in syndrome, round t vs round t-1, and the
    cut-out syndrome vs round T (the continuation after the cut).  A pattern
    is 'contained' if s_in = 0.  Claim: detector-trivial and contained implies
    s_out = 0.
    """
    total = trivial = contained_trivial = contained_trivial_nonzero_out = 0
    nvar = 2 * T * checks
    for bits in itertools.product((0, 1), repeat=nvar):
        f = np.array(bits[:T * checks]).reshape(T, checks)
        mu = np.array(bits[T * checks:]).reshape(T, checks)
        s = np.zeros(checks, int)          # s_in = 0 (contained)
        rec_prev = s.copy()
        ok = True
        for t in range(T):
            s = (s + f[t]) % 2
            rec = (s + mu[t]) % 2
            if np.any((rec + rec_prev) % 2):
                ok = False
                break
            rec_prev = rec
        # continuation: fault-free rounds after the cut report s exactly
        if ok and np.any((s + rec_prev) % 2):
            ok = False
        total += 1
        if ok:
            contained_trivial += 1
            contained_trivial_nonzero_out += int(np.any(s))
    return dict(test="telescoping idle chain", rounds=T, checks=checks, patterns=total,
                contained_detector_trivial=contained_trivial,
                with_nonzero_syndrome_at_output_cut=contained_trivial_nonzero_out)


def main():
    rows = [reviewer_example(), layer_code_example(), telescoping_check()]
    for r in rows:
        print(r)
    ok = rows[0]["law_with_C"] and not rows[0]["law_without_C"]
    a, b = rows[1]["law_with_C"].split("/")
    ok &= a == b
    a, b = rows[1]["relative_effect_independent_of_inherited_class"].split("/")
    ok &= a == b
    ok &= rows[2]["with_nonzero_syndrome_at_output_cut"] == 0
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(ROOT / "frame_propagation_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("FRAME PROPAGATION CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
