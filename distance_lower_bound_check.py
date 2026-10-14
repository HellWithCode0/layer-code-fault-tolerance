#!/usr/bin/env python3
"""Checks of the distance theorem for the explicit realization.

Theorem (distance, strengthened in this revision).  For the explicit
realization of an input CSS code,
    d_X >= (L_x + 1) * ceil(d_X^in / eta_X),   d_Z >= L_z * ceil(d_Z^in / eta_Z),
    eta_X = max(1, floor(w_X / 2)),            eta_Z = max(1, floor(w_Z / 2)),
where w_X, w_Z are the largest X- and Z-check weights of the input.  (The
previous version had d_X >= (L_x + 1) d_X^in / w_X.  A backbone piece
contributes a PREFIX of a check, which may be traded for the complementary
suffix modulo that check, and the number of pieces per plane is an integer.)

Key lemma (plane restriction).  For an X-type logical O of class x and a
column x0, let a_q be the parity of O on the data z-edges of D_q in column
x0 and b_{i,t} its parity on the A_i y-edges of column x0 in backbone
segment t.  Then
    v(x0)_q = a_q + sum_{i : q in X_i} sum_{t >= pos_i(q)} b_{i,t}
lies in the coset x + rowspace(H_X) of the input code; in particular it is
a nontrivial input X logical, so |v(x0)| >= d_X^in.  Each occupied backbone
piece b_{i,t} contributes the prefix of check X_i of length t; adding the row
X_i replaces it by the complementary suffix and keeps v(x0) in the same coset.
Choosing the shorter side for every occupied piece gives a vector v'(x0) in
x + rowspace(H_X) with |v'(x0)| <= eta_X * (#occupied pieces), so column x0
contains at least ceil(d_X^in / eta_X) qubits of O.  The dual lemma uses the
layers of z-edges at every height.

This script (1) checks the facts about the slabs that the proof uses: the
support of every basis slab is exactly the union of the pieces above, it lies
in its own plane x = x0 (layer of z-edges [z0, z0+1]), and its weight obeys
the upper bound of the theorem; (2) verifies the lemma on random
representatives O = slab operator times random stabilizers, for every column
(height), on the certificate's input family; and (3) compares the resulting
bound with the exact distances of explicit_layer_code_distance.csv.
Outputs: distance_lower_bound_check.csv, distance_bound_vs_exact.csv
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from layer_code_explicit import E, build_layer_code, gf2_rank, x_slab, z_slab  # noqa: E402
from slab_gluing_certificate import input_codes, nullspace  # noqa: E402


def input_distance(Hc, Hs):
    """min |v| over v in ker(Hc) outside rowspace(Hs) (brute force, small n)."""
    n = Hc.shape[1]
    rs = gf2_rank(Hs) if Hs.shape[0] else 0
    for w in range(1, n + 1):
        for supp in itertools.combinations(range(n), w):
            v = np.zeros(n, np.uint8)
            v[list(supp)] = 1
            if Hc.shape[0] and np.any((Hc.astype(int) @ v) % 2):
                continue
            if Hs.shape[0] == 0 or gf2_rank(np.vstack([Hs, v])) > rs:
                return w
    return None


def in_coset(v, x, H):
    d = (v ^ x).astype(np.uint8)
    if not d.any():
        return True
    return gf2_rank(np.vstack([H, d])) == gf2_rank(H)


def plane_vector_X(lc, O, x0):
    """v(x0) for an X operator O and column x0; also the count of occupied pieces."""
    g = lc.geometry
    nin = lc.HX_in.shape[1]
    v = np.zeros(nin, np.uint8)
    pieces = 0
    for q in range(nin):
        par = 0
        for z in range(g["Lz"]):
            par ^= int(O[lc.q(("D", q), E((x0, g["Yq"][q], z), (x0, g["Yq"][q], z + 1)), False)])
        v[q] ^= par
        pieces += par
    vs = v.copy()                     # shortest-side representative v'(x0)
    for i, sup in enumerate(g["suppX"]):
        Z = g["Zi"][i]
        b = []
        for t in range(len(sup) - 1):
            par = 0
            for y in range(g["Yq"][sup[t]], g["Yq"][sup[t + 1]]):
                par ^= int(O[lc.q(("A", i), E((x0, y, Z), (x0, y + 1, Z)), False)])
            b.append(par)
            pieces += par
        for h, q in enumerate(sup):
            v[q] ^= sum(b[h:]) % 2
        _shortest_side(vs, sup, b)
    return v, pieces, vs


def _shortest_side(vs, sup, b):
    """Add, for every occupied segment t of a check with ordered support sup,
    the shorter of its prefix sup[:t+1] and the complementary suffix sup[t+1:].
    The two differ by the check row itself, so the coset is unchanged."""
    k = len(sup)
    for t, bit in enumerate(b):
        if bit:
            part = sup[:t + 1] if t + 1 <= k - t - 1 else sup[t + 1:]
            for q in part:
                vs[q] ^= 1


def plane_vector_Z(lc, O, z0):
    g = lc.geometry
    nin = lc.HX_in.shape[1]
    v = np.zeros(nin, np.uint8)
    pieces = 0
    for q in range(nin):
        par = 0
        for xx in range(g["Lx"] + 1):
            par ^= int(O[lc.q(("D", q), E((xx, g["Yq"][q], z0), (xx, g["Yq"][q], z0 + 1)), False)])
        v[q] ^= par
        pieces += par
    zero = np.zeros(nin, np.uint8)
    vs = v.copy()
    for j, sup in enumerate(g["suppZ"]):
        b = []
        for t in range(len(sup) - 1):
            ov = {(jj, tt): 0 for jj, s2 in enumerate(g["suppZ"]) for tt in range(len(s2) - 1)}
            ov[(j, t)] = 1
            seg = x_slab(lc, zero, z0, override=ov)
            par = int((seg.astype(int) @ O.astype(int)) % 2)
            b.append(par)
            pieces += par
        for h, q in enumerate(sup):
            v[q] ^= sum(b[h:]) % 2
        _shortest_side(vs, sup, b)
    return v, pieces, vs


def x_pieces(lc, x, z0):
    """Union of the height-z0 pieces of the X slab: data rows and C_j segments."""
    g = lc.geometry
    nin = lc.HX_in.shape[1]
    U = np.zeros(lc.n, bool)
    for q in range(nin):
        if x[q]:
            for xx in range(g["Lx"] + 1):
                U[lc.q(("D", q), E((xx, g["Yq"][q], z0), (xx, g["Yq"][q], z0 + 1)), False)] = True
    zero = np.zeros(nin, np.uint8)
    for j, sup in enumerate(g["suppZ"]):
        for t in range(len(sup) - 1):
            if sum(int(x[q]) for q in sup[:t + 1]) % 2:
                ov = {(jj, tt): 0 for jj, s2 in enumerate(g["suppZ"]) for tt in range(len(s2) - 1)}
                ov[(j, t)] = 1
                U |= x_slab(lc, zero, z0, override=ov).astype(bool)
    return U


def z_pieces(lc, z, x0):
    """Union of the column-x0 pieces of the Z slab: data columns and A_i segments."""
    g = lc.geometry
    U = np.zeros(lc.n, bool)
    for q in range(lc.HX_in.shape[1]):
        if z[q]:
            for zz in range(g["Lz"]):
                U[lc.q(("D", q), E((x0, g["Yq"][q], zz), (x0, g["Yq"][q], zz + 1)), False)] = True
    for i, sup in enumerate(g["suppX"]):
        for t in range(len(sup) - 1):
            if sum(int(z[q]) for q in sup[:t + 1]) % 2:
                for y in range(g["Yq"][sup[t]], g["Yq"][sup[t + 1]]):
                    U[lc.q(("A", i), E((x0, y, g["Zi"][i]), (x0, y + 1, g["Zi"][i])), False)] = True
    return U


def slab_structure(lc, HX, HZ, s):
    """(checked, passed) over basis slabs: support = pieces, own plane/layer, weight bound."""
    g = lc.geometry
    nin = HX.shape[1]
    ok = tot = 0
    for x in nullspace(HZ):
        for z0 in range(g["Lz"]):
            S = x_slab(lc, x, z0).astype(bool)
            layer = all(a[0] == b[0] and a[1] == b[1] and {a[2], b[2]} == {z0, z0 + 1}
                        for _, (a, b) in (lc.qubits[k] for k in np.flatnonzero(S)))
            weight = S.sum() <= (g["Lx"] + 1) * x.sum() + s * HZ.shape[0] * nin
            tot += 1
            ok += int(np.array_equal(S, x_pieces(lc, x, z0)) and layer and weight)
    for z in nullspace(HX):
        for x0 in range(g["Lx"] + 1):
            S = z_slab(lc, z, x0).astype(bool)
            plane = all(a[0] == x0 and b[0] == x0 for _, (a, b) in (lc.qubits[k] for k in np.flatnonzero(S)))
            weight = S.sum() <= g["Lz"] * z.sum() + s * HX.shape[0] * nin
            tot += 1
            ok += int(np.array_equal(S, z_pieces(lc, z, x0)) and plane and weight)
    return tot, ok


def main(samples=6, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for name, HX, HZ in input_codes():
        if HX.shape[1] > 15:
            continue
        lc = build_layer_code(HX, HZ, spacing=2)
        g = lc.geometry
        HXf, HZf = lc.HX.toarray().astype(np.uint8), lc.HZ.toarray().astype(np.uint8)
        dXin, dZin = input_distance(HZ, HX), input_distance(HX, HZ)
        wX, wZ = int(HX.sum(1).max()), int(HZ.sum(1).max())
        etaX, etaZ = max(1, wX // 2), max(1, wZ // 2)
        n_slab, ok_slab = slab_structure(lc, HX, HZ, 2)
        ok_x = ok_z = 0
        tot_x = tot_z = 0
        for x in nullspace(HZ):
            if gf2_rank(np.vstack([HX, x])) == gf2_rank(HX):
                continue                      # trivial class
            for _ in range(samples):
                O = x_slab(lc, x, int(rng.integers(g["Lz"])))
                mask = rng.random(HXf.shape[0]) < 0.3
                O = (O + HXf[mask].sum(0)) % 2
                for x0 in range(g["Lx"] + 1):
                    v, pieces, vs = plane_vector_X(lc, O.astype(np.uint8), x0)
                    tot_x += 1
                    ok_x += int(in_coset(v, x, HX) and pieces * wX >= v.sum() >= dXin
                                and in_coset(vs, x, HX) and pieces * etaX >= vs.sum() >= dXin
                                and pieces >= -(-dXin // etaX))
        for z in nullspace(HX):
            if gf2_rank(np.vstack([HZ, z])) == gf2_rank(HZ):
                continue
            for _ in range(samples):
                O = z_slab(lc, z, int(rng.integers(g["Lx"] + 1)))
                mask = rng.random(HZf.shape[0]) < 0.3
                O = (O + HZf[mask].sum(0)) % 2
                for z0 in range(g["Lz"]):
                    v, pieces, vs = plane_vector_Z(lc, O.astype(np.uint8), z0)
                    tot_z += 1
                    ok_z += int(in_coset(v, z, HZ) and pieces * wZ >= v.sum() >= dZin
                                and in_coset(vs, z, HZ) and pieces * etaZ >= vs.sum() >= dZin
                                and pieces >= -(-dZin // etaZ))
        bX = (g["Lx"] + 1) * dXin / wX
        bZ = g["Lz"] * dZin / wZ
        rows.append(dict(code=name, spacing=2, n_phys=lc.n, dX_in=dXin, dZ_in=dZin, wX=wX, wZ=wZ,
                         Lx=g["Lx"], Lz=g["Lz"], old_bound_dX=int(np.ceil(bX)), old_bound_dZ=int(np.ceil(bZ)),
                         bound_dX=(g["Lx"] + 1) * -(-dXin // etaX), bound_dZ=g["Lz"] * -(-dZin // etaZ),
                         slab_structure=f"{ok_slab}/{n_slab}",
                         plane_lemma_X=f"{ok_x}/{tot_x}", plane_lemma_Z=f"{ok_z}/{tot_z}"))
        print(rows[-1], flush=True)
    # compare with exact distances where available
    exact = {}
    p = ROOT / "explicit_layer_code_distance.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            exact[(r["input"], int(r["spacing"]))] = r
    cmp_rows = []
    for (inp, s), r in exact.items():
        n_in = 4 if inp == "[[4,2,2]]" else 6
        H = np.ones((1, n_in), np.uint8)
        lc = build_layer_code(H, H, spacing=s)
        g = lc.geometry
        eta = max(1, n_in // 2)                      # a single check of weight n_in
        oX = int(np.ceil((g["Lx"] + 1) * 2 / n_in))  # previous bound (L_x+1) d_in / w
        oZ = int(np.ceil(g["Lz"] * 2 / n_in))
        bX = (g["Lx"] + 1) * -(-2 // eta)            # strengthened bound
        bZ = g["Lz"] * -(-2 // eta)
        # consistent: the proved bound never exceeds the explicit logical found by
        # the search (columns dX, dZ); closed: it meets that logical, so the
        # distance is determined exactly even where CP-SAT stopped short.
        cmp_rows.append(dict(code=inp, spacing=s, old_bound_dX=oX, bound_dX=bX, dX_lower=r["dX_lower"],
                             dX=r["dX"], old_bound_dZ=oZ, bound_dZ=bZ, dZ_lower=r["dZ_lower"], dZ=r["dZ"],
                             consistent=bX <= int(r["dX"]) and bZ <= int(r["dZ"]),
                             closed=bX == int(r["dX"]) and bZ == int(r["dZ"])))
        print(cmp_rows[-1])
    with open(ROOT / "distance_lower_bound_check.csv", "w", newline="") as fh:
        keys = list(rows[0]) if rows else []
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "distance_bound_vs_exact.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cmp_rows[0]))
        w.writeheader()
        w.writerows(cmp_rows)
    def full(v):
        a, b = v.split("/")
        return a == b
    good = all(full(r["slab_structure"]) and full(r["plane_lemma_X"]) and full(r["plane_lemma_Z"])
               for r in rows)
    good &= all(r["consistent"] for r in cmp_rows)
    print("DISTANCE LEMMA CHECK:", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
