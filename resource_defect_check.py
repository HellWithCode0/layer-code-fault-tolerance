#!/usr/bin/env python3
"""Numerical certificate for the defective-resource lemma of the magic-state factory.

Background.  The previous version assumed that every bad magic-state candidate
is diagonal in the basis {|A>, Z|A>} (|A> = T|+>) and enforced this with the
twirl T_A = e^{-i pi/4} S X, implemented with a |Y> resource by phase
teleportation.  A referee showed that if the raw candidate is X|A> and the
twirl's own |Y> resource is bad, the nontrivial twirl branch applies Z T_A and
the channel (rho + Z T_A rho T_A^dag Z)/2 leaves rho = X|A><A|X unchanged, so
coherence survives.  The repaired argument drops the twirl and uses a
bad-subsystem model: a bad resource may be in an ARBITRARY state, entangled
with an environment.  Key lemma: phase teleportation with an arbitrary
resource applies, in every measurement branch, an operator in
span{I, Z} (x) B(H_E) on the data coordinate relative to the ideal rotation.
Hence a code-based always-output factory whose code corrects any single Z
error outputs exactly the ideal resource, unentangled from everything,
whenever at most one coordinate is defective (principal candidate or its
conditional correction resource bad).

Gadget (outcome bit 0 = eigenvalue +1): R_theta = diag(1, e^{i theta}); data
d holds |psi>, auxiliary a holds the resource (ideally R_theta|+>).  Measure
Z_d Z_a (bit m), then X_d (bit n), discard d, apply
C_mn = R_{2 theta}^m Z_a^n X_a^m (rightmost first).  theta = pi/4 uses |A>
and needs R_{pi/2} = S, itself realised by a second phase teleportation with
a |Y> = S|+> resource; theta = pi/2 uses |Y> and needs R_pi = Z (free).

Checks (one CSV row each: test, parameters, value, expected, pass):
  1. the referee counterexample (faulty twirl keeps coherence 1/2, ideal twirl
     removes it);
  2. the lemma: every branch Kraus map of the gadget with a random resource
     entangled with an environment is diagonal on the data wire relative to
     R_theta (also with the S correction teleported through an arbitrary
     entangled |Y> slot), the Kraus maps are complete, and ideal resources
     give R_theta exactly;
  3. factory exactness: state-vector simulation of the punctured Reed-Muller
     [[15,1,3]] T factory and the Steane [[7,1,3]] S factory, gadget applied
     coordinate by coordinate, one arbitrary entangled defect, Born-sampled
     branches plus full branch enumeration, X-check syndrome projection and
     predetermined recovery; the code register must equal the ideal output
     (fidelity >= 1 - 1e-9) in every branch.  Negative controls: two
     defective coordinates give a logical error; two bad slots in the same
     coordinate are still corrected;
  4. always-output and postselected malignant Z-pattern counts by weight
     (used by resource_estimate.py).

Output: resource_defect_check.csv (next to this script).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "resource_defect_check.csv")

TOL = 1e-10          # numerical zero for amplitudes
FID_TOL = 1e-9       # required fidelity margin
PROB_CUT = 1e-12     # branches below this probability are not examined

ROWS: list[dict] = []


def record(test, params, value, expected, ok):
    ROWS.append({"test": test, "parameters": params, "value": value,
                 "expected": expected, "pass": bool(ok)})
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {test} ({params}): {value}  expected {expected}")


# ---------------------------------------------------------------------------
# single-qubit primitives
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
S = np.diag([1, 1j])
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)


def R(theta):
    return np.diag([1, np.exp(1j * theta)])


def ket_R(theta):
    """Ideal resource R_theta|+>."""
    return R(theta) @ PLUS


def rand_state(rng, dim):
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return v / np.linalg.norm(v)


def phase_free_equal(a, b, tol=TOL):
    """a == c b for some unit complex c (both arrays, same shape)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    ov = np.vdot(b, a)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if nb < tol:
        return na < tol
    return abs(abs(ov) - na * nb) < tol * max(1.0, na * nb) and abs(na - nb) < tol


# ---------------------------------------------------------------------------
# phase-teleportation gadget as a family of Kraus maps
# ---------------------------------------------------------------------------
def gadget_kraus(theta, res, res_env_dim, s_slot=None, s_env_dim=1):
    """Branch Kraus maps of the phase-teleportation gadget.

    res: resource state of a (x) E, flattened with a as the leading index.
    s_slot: if given (theta = pi/4 only), the R_{2 theta} = S correction is
      itself teleported with this state of (Y qubit) (x) E_Y as resource.
    Returns dict label -> K with K[a_out, e, b_in] (map H_d -> H_a (x) H_E).
    The gadget is simulated directly on d (x) a (x) E for inputs |0>, |1>.
    """
    r = np.asarray(res).reshape(2, res_env_dim)
    out = {}
    for m in (0, 1):
        for n in (0, 1):
            K = np.zeros((2, res_env_dim, 2), dtype=complex)
            for b in (0, 1):
                psi = np.zeros((2, 2, res_env_dim), dtype=complex)  # d, a, e
                psi[b] = r
                # project onto Z_d Z_a parity m
                for d in (0, 1):
                    for a in (0, 1):
                        if (d ^ a) != m:
                            psi[d, a] = 0
                # measure X_d with outcome n, discard d
                xa = np.array([1, (-1) ** n], dtype=complex) / np.sqrt(2)
                st = np.tensordot(xa.conj(), psi, axes=([0], [0]))  # a, e
                if m:
                    st = X @ st
                if n:
                    st = Z @ st
                K[:, :, b] = st
            if m and s_slot is None:
                K = np.einsum("ij,jeb->ieb", R(2 * theta), K)
                out[(m, n)] = K
            elif m:
                # teleport R_{2 theta} (must be S here) with the Y slot
                assert abs(2 * theta - np.pi / 2) < 1e-12
                K2 = gadget_kraus(np.pi / 2, s_slot, s_env_dim)
                for (m2, n2), k2 in K2.items():
                    # compose: env ordering (eA, eY)
                    Kc = np.einsum("pfa,aeb->pefb", k2, K)
                    out[(m, n, m2, n2)] = Kc.reshape(2, res_env_dim * s_env_dim, 2)
            else:
                out[(m, n)] = K
    return out


def kraus_diag_leak(K, theta_total_phase):
    """Max leakage of (R^{-1} (x) I) K out of span{|b>} (x) H_E for b = 0, 1."""
    Ri = np.linalg.inv(theta_total_phase)
    M = np.einsum("ij,jeb->ieb", Ri, K)
    return max(np.linalg.norm(M[1 - b, :, b]) for b in (0, 1))


# ---------------------------------------------------------------------------
# codes
# ---------------------------------------------------------------------------
def gf2_nullspace(A):
    A = A.copy() % 2
    rows, cols = A.shape
    piv = []
    r = 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i, c]), None)
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for i, c in enumerate(piv):
            if A[i, f]:
                v[c] = 1
        basis.append(v)
    return np.array(basis, dtype=np.uint8)


class Code:
    """Punctured Reed-Muller type code: coordinates = nonzero v in F_2^k."""

    def __init__(self, k, name):
        self.name = name
        self.k = k
        self.n = 2 ** k - 1
        self.v = [np.array([(i >> (k - 1 - t)) & 1 for t in range(k)], dtype=np.uint8)
                  for i in range(1, self.n + 1)]
        self.H = np.array(self.v, dtype=np.uint8).T          # k x n, X checks
        ker = gf2_nullspace(self.H)                          # Hamming code
        # even-weight subcode of ker(H): Z stabilizers
        odd = [row for row in ker if row.sum() % 2]
        even = [row for row in ker if row.sum() % 2 == 0]
        if odd:
            o0 = odd[0]
            even += [(row ^ o0) for row in odd[1:]]
        self.Zst = np.array(even, dtype=np.uint8)
        # syndrome value -> recovery coordinate
        self.syn_to_j = {}
        for j, vj in enumerate(self.v):
            self.syn_to_j[int("".join(map(str, vj)), 2)] = j
        # all basis indices, bit j of index = coordinate j (qubit axis j)
        idx = np.arange(2 ** self.n)
        self.bits = ((idx[:, None] >> (self.n - 1 - np.arange(self.n))[None, :]) & 1).astype(np.uint8)

    def plus_L(self):
        ok = np.all((self.bits @ self.Zst.T) % 2 == 0, axis=1)
        psi = ok.astype(complex)
        return psi / np.linalg.norm(psi), ok

    def logical_basis(self):
        # |0_L>: rowspace of H (simplex); |1_L>: its complement
        simplex = set()
        for coeffs in itertools.product((0, 1), repeat=self.k):
            w = np.array(coeffs, dtype=np.uint8) @ self.H % 2
            simplex.add(int("".join(map(str, w)), 2))
        z0 = np.zeros(2 ** self.n, dtype=complex)
        z1 = np.zeros(2 ** self.n, dtype=complex)
        full = 2 ** self.n - 1
        for s in simplex:
            z0[s] = 1
            z1[s ^ full] = 1
        return z0 / np.linalg.norm(z0), z1 / np.linalg.norm(z1)


# ---------------------------------------------------------------------------
# register operations: state shape (2,)*n + (D,)
# ---------------------------------------------------------------------------
def apply_kraus(psi, K, j, n):
    dnew = K.shape[1]
    new = np.tensordot(K, psi, axes=([2], [j]))          # out, e, q..., D
    new = np.moveaxis(new, 1, -1)                          # out, q..., D, e
    new = np.moveaxis(new, 0, j)                           # q..., D, e
    D = psi.shape[-1]
    return new.reshape((2,) * n + (D * dnew,))


def hadamard_all(psi, n):
    H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    for j in range(n):
        psi = np.moveaxis(np.tensordot(H2, psi, axes=([1], [j])), 0, j)
    return psi


def decode_and_score(code, psi, target_h, syn_of_idx, n):
    """Project on each X-check syndrome, recover, return list of (syn, prob, fid)."""
    D = psi.shape[-1]
    ph = hadamard_all(psi, n).reshape(2 ** n, D)
    res = []
    for s in range(2 ** code.k):
        mask = syn_of_idx == s
        blk = np.where(mask[:, None], ph, 0)
        p = float(np.vdot(blk, blk).real)
        if p < PROB_CUT:
            res.append((s, p, None))
            continue
        if s:
            j = code.syn_to_j[s]
            # Z_j in the Hadamard frame is a bit flip on coordinate j
            blk = blk.reshape((2,) * n + (D,))
            blk = np.flip(blk, axis=j).reshape(2 ** n, D)
        amp = target_h.conj() @ blk                          # vector over env
        fid = float(np.vdot(amp, amp).real) / p
        res.append((s, p, fid))
    return res


class Factory:
    def __init__(self, code, theta, teleport_s):
        self.code = code
        self.theta = theta
        self.teleport_s = teleport_s
        n = code.n
        self.n = n
        plus, _ = code.plus_L()
        self.plus = plus.reshape((2,) * n + (1,))
        # syndrome of each Hadamard-frame basis index: H y
        sy = code.bits @ code.H.T % 2
        self.syn_of_idx = sy @ (2 ** np.arange(code.k - 1, -1, -1))
        self.ideal_kraus = self.kraus_for(None, 1, None, 1)

    def kraus_for(self, res, dres, yslot, dy):
        th = self.theta
        if res is None:
            res, dres = ket_R(th), 1
        if self.teleport_s:
            if yslot is None:
                yslot, dy = ket_R(np.pi / 2), 1
            return gadget_kraus(th, res, dres, yslot, dy)
        return gadget_kraus(th, res, dres)

    def run(self, kraus_per_coord, rng=None, forced=None):
        """Apply coordinate gadgets; sample branches (rng) or use forced labels."""
        psi = self.plus.copy()
        labels = []
        for j in range(self.n):
            ks = kraus_per_coord[j]
            keys = sorted(ks.keys())
            if forced is not None and forced[j] is not None:
                lab = forced[j]
                psi = apply_kraus(psi, ks[lab], j, self.n)
                labels.append(lab)
                continue
            cands = [apply_kraus(psi, ks[kk], j, self.n) for kk in keys]
            probs = np.array([np.vdot(c, c).real for c in cands])
            probs = probs / probs.sum()
            i = rng.choice(len(keys), p=probs)
            psi = cands[i] / np.linalg.norm(cands[i])
            labels.append(keys[i])
        nrm = np.linalg.norm(psi)
        return psi / nrm, labels, nrm ** 2

    def ideal_target(self, rng):
        psi, _, _ = self.run([self.ideal_kraus] * self.n, rng=rng)
        D = psi.shape[-1]
        assert D == 1
        return psi.reshape(-1)


# ---------------------------------------------------------------------------
# check 1: referee counterexample
# ---------------------------------------------------------------------------
def check_referee():
    print("[1] referee counterexample for the old twirl")
    A = ket_R(np.pi / 4)
    ZA = Z @ A
    TA = np.exp(-1j * np.pi / 4) * S @ X
    e1 = np.linalg.norm(TA @ A - A)
    e2 = np.linalg.norm(TA @ ZA + ZA)
    record("twirl_eigen", "T_A|A>=|A>", f"{e1:.2e}", "0", e1 < TOL)
    record("twirl_eigen", "T_A Z|A>=-Z|A>", f"{e2:.2e}", "0", e2 < TOL)
    rho = X @ np.outer(A, A.conj()) @ X
    F = Z @ TA
    rho_f = 0.5 * (rho + F @ rho @ F.conj().T)
    e3 = np.linalg.norm(rho_f - rho)
    record("faulty_twirl_fixed_point", "rho=X|A><A|X", f"{e3:.2e}", "0", e3 < TOL)
    coh = abs(np.vdot(A, rho_f @ ZA))
    record("faulty_twirl_coherence", "|<A|rho Z|A>|", f"{coh:.12f}", "0.5",
           abs(coh - 0.5) < TOL)
    rho_i = 0.5 * (rho + TA @ rho @ TA.conj().T)
    coh_i = abs(np.vdot(A, rho_i @ ZA))
    record("ideal_twirl_coherence", "|<A|rho Z|A>|", f"{coh_i:.2e}", "0", coh_i < TOL)


# ---------------------------------------------------------------------------
# check 2: defective phase-teleportation lemma
# ---------------------------------------------------------------------------
def check_lemma(rng, trials):
    print("[2] defective phase teleportation lemma")
    for name, th in (("pi/4", np.pi / 4), ("pi/2", np.pi / 2)):
        # sanity with ideal resources
        ks = gadget_kraus(th, ket_R(th), 1)
        worst = 0.0
        for lab, K in ks.items():
            M = K[:, 0, :]
            c = M[0, 0] / R(th)[0, 0]
            worst = max(worst, np.linalg.norm(M - c * R(th)), abs(abs(c) - 0.5))
        record("ideal_gadget_exact", f"theta={name}, all 4 branches K=R/2 up to phase",
               f"{worst:.2e}", "0", worst < TOL)
        for dE in (1, 2, 4):
            leak = 0.0
            comp = 0.0
            for _ in range(trials):
                r = rand_state(rng, 2 * dE)
                ks = gadget_kraus(th, r, dE)
                tot = np.zeros((2, 2), dtype=complex)
                for K in ks.values():
                    leak = max(leak, kraus_diag_leak(K, R(th)))
                    Kf = K.reshape(2 * dE, 2)
                    tot += Kf.conj().T @ Kf
                comp = max(comp, np.linalg.norm(tot - I2))
            record("defective_gadget_diagonal",
                   f"theta={name}, dimE={dE}, trials={trials}, 4 branches",
                   f"max leak {leak:.2e}", "0", leak < TOL)
            record("defective_gadget_complete",
                   f"theta={name}, dimE={dE}, sum K^dag K = I",
                   f"{comp:.2e}", "0", comp < TOL)
    # theta = pi/4 with the S correction teleported through an arbitrary Y slot
    th = np.pi / 4
    ks = gadget_kraus(th, ket_R(th), 1, ket_R(np.pi / 2), 1)
    worst = 0.0
    for K in ks.values():
        M = K[:, 0, :]
        c = M[0, 0]
        worst = max(worst, np.linalg.norm(M - c * R(th)))
    record("ideal_gadget_exact", "theta=pi/4 with teleported S, 10 branches",
           f"{worst:.2e}", "0", worst < TOL)
    for dA, dY in ((1, 2), (2, 2), (4, 2), (2, 4)):
        leak = 0.0
        comp = 0.0
        for _ in range(trials):
            r = rand_state(rng, 2 * dA)
            y = rand_state(rng, 2 * dY)
            ks = gadget_kraus(th, r, dA, y, dY)
            tot = np.zeros((2, 2), dtype=complex)
            for K in ks.values():
                leak = max(leak, kraus_diag_leak(K, R(th)))
                Kf = K.reshape(-1, 2)
                tot += Kf.conj().T @ Kf
            comp = max(comp, np.linalg.norm(tot - I2))
        record("defective_gadget_diagonal",
               f"theta=pi/4, teleported S, dimE_A={dA}, dimE_Y={dY}, trials={trials}, 10 branches",
               f"max leak {leak:.2e}", "0", leak < TOL)
        record("defective_gadget_complete",
               f"theta=pi/4, teleported S, dimE_A={dA}, dimE_Y={dY}",
               f"{comp:.2e}", "0", comp < TOL)
    # ideal A resource, arbitrary Y slot only
    leak = 0.0
    for _ in range(trials):
        y = rand_state(rng, 4)
        ks = gadget_kraus(th, ket_R(th), 1, y, 2)
        for K in ks.values():
            leak = max(leak, kraus_diag_leak(K, R(th)))
    record("defective_gadget_diagonal",
           f"theta=pi/4, ideal |A>, arbitrary Y slot dimE_Y=2, trials={trials}",
           f"max leak {leak:.2e}", "0", leak < TOL)


# ---------------------------------------------------------------------------
# check 3: factory exactness
# ---------------------------------------------------------------------------
def orientation(fac, target):
    z0, z1 = fac.code.logical_basis()
    a0, a1 = np.vdot(z0, target), np.vdot(z1, target)
    rel = a1 / a0
    tp = np.exp(1j * fac.theta)
    if abs(abs(a0) - 1 / np.sqrt(2)) > 1e-9 or abs(abs(a1) - 1 / np.sqrt(2)) > 1e-9:
        return "neither", rel
    if abs(rel - tp) < 1e-9:
        return "R_theta|+>_L", rel
    if abs(rel - np.conj(tp)) < 1e-9:
        return "R_theta^dag|+>_L", rel
    return "other", rel


def factory_case(fac, rng, trials, dA, y_mode, dY, tag):
    """Random single defect per trial, sampled branches."""
    n = fac.n
    target = fac.ideal_target(rng)
    target_h = hadamard_all(target.reshape((2,) * n + (1,)), n).reshape(-1)
    worst = 1.0
    nbr = 0
    for _ in range(trials):
        j0 = int(rng.integers(n))
        r = rand_state(rng, 2 * dA)
        y = rand_state(rng, 2 * dY) if y_mode else None
        kd = fac.kraus_for(r, dA, y, dY if y_mode else 1)
        kr = [fac.ideal_kraus] * n
        kr = list(kr)
        kr[j0] = kd
        psi, _, _ = fac.run(kr, rng=rng)
        for s, p, f in decode_and_score(fac.code, psi, target_h, fac.syn_of_idx, n):
            if f is not None:
                worst = min(worst, f)
                nbr += 1
    record("factory_single_defect_sampled",
           f"{tag}, trials={trials}, syndrome branches examined={nbr}",
           f"min fidelity {worst:.15f}", f">= 1-{FID_TOL:g}", nbr > 0 and worst >= 1 - FID_TOL)
    return worst


def factory_enumerate_defect(fac, rng, dA, y_mode, dY, tag):
    """All branches of the defective coordinate enumerated, others sampled."""
    n = fac.n
    target = fac.ideal_target(rng)
    target_h = hadamard_all(target.reshape((2,) * n + (1,)), n).reshape(-1)
    j0 = int(rng.integers(n))
    r = rand_state(rng, 2 * dA)
    y = rand_state(rng, 2 * dY) if y_mode else None
    kd = fac.kraus_for(r, dA, y, dY if y_mode else 1)
    kr = [fac.ideal_kraus] * n
    kr = list(kr)
    kr[j0] = kd
    # sample the ideal coordinates once, then enumerate the defect's branches
    _, labels, _ = fac.run([fac.ideal_kraus] * n, rng=rng)
    # completeness at the defect: summed weight must equal the weight of the
    # fixed ideal-coordinate outcomes, prod_{j != j0} ||K_j||_F^2 / 2
    ref = 1.0
    for j in range(n):
        if j != j0:
            Kj = fac.ideal_kraus[labels[j]]
            ref *= float(np.vdot(Kj, Kj).real) / 2
    worst = 1.0
    ptot = 0.0
    nbr = 0
    for lab in sorted(kd.keys()):
        forced = list(labels)
        forced[j0] = lab
        psi, _, w = fac.run(kr, forced=forced)
        # weight relative to the fixed ideal-coordinate outcomes
        ptot += w
        for s, p, f in decode_and_score(fac.code, psi, target_h, fac.syn_of_idx, n):
            if f is not None and w / ref > PROB_CUT:
                worst = min(worst, f)
                nbr += 1
    psum = ptot / ref
    record("factory_single_defect_enumerated",
           f"{tag}, defect coord={j0 + 1}, {len(kd)} defect branches x all syndromes, examined={nbr}",
           f"min fidelity {worst:.15f}; summed defect-branch prob {psum:.12f}",
           f">= 1-{FID_TOL:g}; 1", nbr > 0 and worst >= 1 - FID_TOL and abs(psum - 1) < 1e-9)
    return worst


def factory_full_enumeration(fac, rng, dA, tag):
    """Every branch of every coordinate (small codes only)."""
    n = fac.n
    target = fac.ideal_target(rng)
    target_h = hadamard_all(target.reshape((2,) * n + (1,)), n).reshape(-1)
    j0 = int(rng.integers(n))
    r = rand_state(rng, 2 * dA)
    kd = fac.kraus_for(r, dA, None, 1)
    kr = [fac.ideal_kraus] * n
    kr = list(kr)
    kr[j0] = kd
    worst = [1.0]
    ptot = [0.0]
    nleaf = [0]

    def dfs(j, psi):
        if j == n:
            w = float(np.vdot(psi, psi).real)
            ptot[0] += w
            nleaf[0] += 1
            if w < PROB_CUT:
                return
            for s, p, f in decode_and_score(fac.code, psi / np.sqrt(w), target_h,
                                            fac.syn_of_idx, n):
                if f is not None:
                    worst[0] = min(worst[0], f)
            return
        for lab in sorted(kr[j].keys()):
            dfs(j + 1, apply_kraus(psi, kr[j][lab], j, n))

    dfs(0, fac.plus.copy())
    record("factory_single_defect_full_enumeration",
           f"{tag}, defect coord={j0 + 1}, leaves={nleaf[0]}",
           f"min fidelity {worst[0]:.15f}; total prob {ptot[0]:.12f}",
           f">= 1-{FID_TOL:g}; 1", worst[0] >= 1 - FID_TOL and abs(ptot[0] - 1) < 1e-9)


def factory_two_defects(fac, rng, tag, pairs):
    n = fac.n
    target = fac.ideal_target(rng)
    target_h = hadamard_all(target.reshape((2,) * n + (1,)), n).reshape(-1)
    zbar_t = target * (1 - 2 * (fac.code.bits.sum(axis=1).astype(np.int64) % 2))
    zbar_h = hadamard_all(zbar_t.reshape((2,) * n + (1,)), n).reshape(-1)
    bad = Z @ ket_R(fac.theta)
    kbad = fac.kraus_for(bad, 1, None, 1)
    max_f = 0.0
    min_fz = 1.0
    for (i, j) in pairs:
        kr = list([fac.ideal_kraus] * n)
        kr[i] = kbad
        kr[j] = kbad
        psi, _, _ = fac.run(kr, rng=rng)
        for s, p, f in decode_and_score(fac.code, psi, target_h, fac.syn_of_idx, n):
            if f is not None:
                max_f = max(max_f, f)
        for s, p, f in decode_and_score(fac.code, psi, zbar_h, fac.syn_of_idx, n):
            if f is not None:
                min_fz = min(min_fz, f)
    record("negative_control_two_defects",
           f"{tag}, Z R_theta|+> at 2 coords, pairs={len(pairs)}",
           f"max fidelity with ideal {max_f:.2e}; min fidelity with Zbar*ideal {min_fz:.12f}",
           "0; 1", max_f < 1e-9 and min_fz > 1 - FID_TOL)


def check_factories(rng, trials, full_steane):
    print("[3] factory exactness with one arbitrary defect")
    rm = Code(4, "RM15")
    st = Code(3, "Steane7")
    for code in (rm, st):
        plus, _ = code.plus_L()
        n = code.n
        errs = []
        # X checks (rows of H) and Xbar permute basis states; Z stabs are phases
        idx = np.arange(2 ** n)
        for h in list(code.H) + [np.ones(n, dtype=np.uint8)]:
            mask = int("".join(map(str, h)), 2)
            errs.append(np.linalg.norm(plus[idx ^ mask] - plus))
        for zs in code.Zst:
            ph = (-1.0) ** ((code.bits @ zs) % 2)
            errs.append(np.linalg.norm(ph * plus - plus))
        record("encoded_plus_stabilized",
               f"{code.name}: {code.H.shape[0]} X checks, {len(code.Zst)} Z stabs, Xbar",
               f"max residual {max(errs):.2e}", "0", max(errs) < TOL)
        record("code_parameters", f"{code.name}: n, #Zstab generators",
               f"n={n}, Zst={len(code.Zst)}, Xchk={code.H.shape[0]}",
               f"n={n}, k=1", n - len(code.Zst) - code.H.shape[0] == 1)

    fac_rm = Factory(rm, np.pi / 4, teleport_s=True)
    fac_rm_plain = Factory(rm, np.pi / 4, teleport_s=False)
    fac_st = Factory(st, np.pi / 2, teleport_s=False)

    for fac, lab in ((fac_rm, "RM15 T factory (teleported S)"),
                     (fac_rm_plain, "RM15 T factory (direct S)"),
                     (fac_st, "Steane S factory")):
        tgt = fac.ideal_target(rng)
        tgt2 = fac.ideal_target(rng)
        same = phase_free_equal(tgt, tgt2, 1e-9)
        ori, rel = orientation(fac, tgt)
        gate = "T" if fac.theta == np.pi / 4 else "S"
        record("ideal_output_orientation", lab,
               f"{ori.replace('R_theta', gate)} (rel phase {np.angle(rel) / np.pi:+.4f} pi); branch independent={same}",
               "encoded magic state (T or T^dag, S or S^dag)",
               ori in ("R_theta|+>_L", "R_theta^dag|+>_L") and same)

    t0 = time.time()
    # RM15: defective |A> only, 1- or 2-qubit environment
    for dA in (2, 4):
        factory_case(fac_rm, rng, trials, dA, False, 1, f"RM15, arbitrary |A> slot dimE={dA}")
    # RM15: defective |A> and defective S-correction Y slot in the SAME coordinate
    for dA, dY in ((2, 2), (4, 2)):
        factory_case(fac_rm, rng, trials, dA, True, dY,
                     f"RM15, arbitrary |A> slot dimE={dA} and same-coordinate Y slot dimE={dY}")
    factory_case(fac_rm_plain, rng, trials, 4, False, 1, "RM15 direct S, arbitrary |A> slot dimE=4")
    for dA in (2, 4):
        factory_case(fac_st, rng, trials, dA, False, 1, f"Steane, arbitrary |Y> slot dimE={dA}")
    print(f"    sampled cases done in {time.time() - t0:.1f}s")

    factory_enumerate_defect(fac_rm, rng, 4, True, 2,
                             "RM15, arbitrary |A> dimE=4 + same-coordinate Y slot dimE=2")
    factory_enumerate_defect(fac_rm_plain, rng, 4, False, 1, "RM15 direct S, arbitrary |A> dimE=4")
    if full_steane:
        factory_full_enumeration(fac_st, rng, 4, "Steane, arbitrary |Y> dimE=4, all 4^7 branches")

    # negative controls
    pairs_rm = list(itertools.combinations(range(rm.n), 2))
    factory_two_defects(fac_rm_plain, rng, "RM15 direct S", pairs_rm)
    pairs_st = list(itertools.combinations(range(st.n), 2))
    factory_two_defects(fac_st, rng, "Steane", pairs_st)
    print(f"    factory checks done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# check 4: malignant counts
# ---------------------------------------------------------------------------
def malignant_counts(code):
    n = code.n
    bits = code.bits.astype(np.int64)
    syn = (bits @ code.H.T.astype(np.int64)) % 2
    sval = syn @ (2 ** np.arange(code.k - 1, -1, -1))
    w = bits.sum(axis=1)
    # recovery Z_j flips bit j; residual parity = |e| + [s != 0] mod 2 (flip changes weight by +-1)
    resid_par = (w + (sval != 0)) % 2
    # residual lies in ker(H); stabilizer iff even weight; logical iff odd
    logical = resid_par == 1
    ao = np.array([int(np.sum(logical & (w == k))) for k in range(n + 1)])
    ps = np.array([int(np.sum(logical & (sval == 0) & (w == k))) for k in range(n + 1)])
    return ao, ps


def check_counts():
    print("[4] malignant Z-pattern counts")
    out = {}
    for k, name, exp_ao, exp_ps in ((4, "RM15", {2: 105, 3: 35}, {3: 35}),
                                    (3, "Steane7", {2: 21}, {3: 7})):
        code = Code(k, name)
        ao, ps = malignant_counts(code)
        s_ao = "[" + ",".join(map(str, ao)) + "]"
        s_ps = "[" + ",".join(map(str, ps)) + "]"
        ok_ao = all(ao[w] == c for w, c in exp_ao.items()) and ao[0] == 0 and ao[1] == 0
        ok_ps = all(ps[w] == c for w, c in exp_ps.items()) and ps[:3].sum() == 0
        record("malignant_always_output", f"{name}, min-weight recovery, counts by weight 0..{code.n}",
               s_ao, ", ".join(f"w{w}={c}" for w, c in exp_ao.items()) + ", w0=w1=0", ok_ao)
        record("malignant_postselected", f"{name}, accept syndrome 0, counts by weight 0..{code.n}",
               s_ps, ", ".join(f"w{w}={c}" for w, c in exp_ps.items()) + ", w<3 = 0", ok_ps)
        out[name] = (ao, ps)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--lemma-trials", type=int, default=200)
    ap.add_argument("--factory-trials", type=int, default=30)
    ap.add_argument("--no-full-steane", action="store_true",
                    help="skip the exhaustive 4^7 branch enumeration of the Steane factory")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    check_referee()
    check_lemma(rng, args.lemma_trials)
    check_factories(rng, args.factory_trials, not args.no_full_steane)
    check_counts()
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["test", "parameters", "value", "expected", "pass"])
        wr.writeheader()
        wr.writerows(ROWS)
    npass = sum(r["pass"] for r in ROWS)
    print(f"\n{npass}/{len(ROWS)} checks passed in {time.time() - t0:.1f}s; wrote {OUT_CSV}")
    ok = npass == len(ROWS)
    print(f"RESOURCE DEFECT CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
