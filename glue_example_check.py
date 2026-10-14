#!/usr/bin/env python3
"""GF(2) check of the worked gluing example (two adjacent idle primitives).

The previous version of the example wrote the T0 row of each primitive as
f + mu + s_in + s_out = 0 with s the physical syndrome at the cut.  That lets a
record error change the quantum boundary class: with f1 = f2 = 0,
mu1 = mu2 = 1 and s_in = s_out = 0 the old equations are satisfied by
s_mid = 1, although no data error occurred, and summing the two rows loses
the two nonzero endpoints of the physical detector history (1, 0, 1).  Part A
reproduces that finding.

The corrected example separates the physical syndrome, the reported record
and the logical frame.  For one persistent check C and one round per
primitive,

    s_out = s_in + f,        r = s_out + mu,        delta = rho_in + r,
    rho_out = r,             lambda_out = lambda_in + f,

where rho_in is the record of the last round before the input cut (classical
boundary data) and delta the primitive's T0 detector.  Part B checks, on all
2^7 assignments of (f1, mu1, f2, mu2, s_in, rho_in, lambda_in), that
eliminating the mid-cut variables (s_mid, rho_mid, lambda_mid) reproduces the
detector/effect map of the two-round circuit written directly.  Part C shows
that the referee's assignment now gives the detector history (1, 0, 1) when a
clean third round follows.  Part D shows that gluing with (s, lambda) only,
i.e. assuming rho_mid = s_mid, fails exactly when mu1 = 1.

This toy illustrates the boundary interface; boundary_composition_check.py
tests it on explicit layer-code blocks through idle -> gauging -> idle ->
gauging.  Neither replaces the general proof.
"""
from __future__ import annotations

import itertools


def primitive(s_in, rho_in, lam_in, f, mu):
    """One idle round on one check: returns (delta, s_out, rho_out, lam_out)."""
    s_out = s_in ^ f
    r = s_out ^ mu
    return rho_in ^ r, s_out, r, lam_in ^ f


def direct_two_rounds(s_in, rho_in, lam_in, f1, mu1, f2, mu2):
    """The same two rounds written as one circuit."""
    s1 = s_in ^ f1
    r1 = s1 ^ mu1
    s2 = s1 ^ f2
    r2 = s2 ^ mu2
    return (rho_in ^ r1, r1 ^ r2), s2, r2, lam_in ^ f1 ^ f2


def main() -> int:
    ok = True

    # A. the old equations accept a record error as a syndrome change
    f1 = f2 = 0
    mu1 = mu2 = 1
    s_in = s_out = 0
    old_accepts = [s_mid for s_mid in (0, 1)
                   if (f1 ^ mu1 ^ s_in ^ s_mid) == 0 and (f2 ^ mu2 ^ s_mid ^ s_out) == 0]
    s, r_prev, hist = 0, 0, []
    for f, mu in ((f1, mu1), (f2, mu2), (0, 0)):   # clean round afterwards
        s ^= f
        r = s ^ mu
        hist.append(r_prev ^ r)
        r_prev = r
    print(f"A. old equations accept s_mid = {old_accepts}; physical syndrome stays 0; "
          f"physical detector history {hist}")
    ok &= old_accepts == [1] and hist == [1, 0, 1]

    # B. corrected fiber product equals the direct two-round circuit
    n = 0
    for f1, mu1, f2, mu2, s_in, rho_in, lam_in in itertools.product((0, 1), repeat=7):
        d1, s_mid, rho_mid, lam_mid = primitive(s_in, rho_in, lam_in, f1, mu1)
        d2, s_o, rho_o, lam_o = primitive(s_mid, rho_mid, lam_mid, f2, mu2)
        assert ((d1, d2), s_o, rho_o, lam_o) == direct_two_rounds(s_in, rho_in, lam_in, f1, mu1, f2, mu2)
        n += 1
    print(f"B. glued (s, rho, lambda) system equals the direct two-round circuit on all {n} assignments")
    ok &= n == 2 ** 7

    # C. referee's assignment in the corrected variables, then a clean round
    d1, s_mid, rho_mid, lam_mid = primitive(0, 0, 0, 0, 1)
    d2, s_o, rho_o, lam_o = primitive(s_mid, rho_mid, lam_mid, 0, 1)
    d3, *_ = primitive(s_o, rho_o, lam_o, 0, 0)
    print(f"C. corrected history {[d1, d2, d3]}, s_mid = {s_mid}, s_out = {s_o}")
    ok &= [d1, d2, d3] == [1, 0, 1] and s_mid == 0 and s_o == 0

    # D. gluing with (s, lambda) only fails exactly when mu1 = 1
    fails = []
    for f1, mu1, f2, mu2, s_in, rho_in, lam_in in itertools.product((0, 1), repeat=7):
        d1, s_mid, _, lam_mid = primitive(s_in, rho_in, lam_in, f1, mu1)
        d2, s_o, rho_o, lam_o = primitive(s_mid, s_mid, lam_mid, f2, mu2)   # rho_mid := s_mid
        if ((d1, d2), s_o, rho_o, lam_o) != direct_two_rounds(s_in, rho_in, lam_in, f1, mu1, f2, mu2):
            fails.append(mu1)
    print(f"D. (s, lambda)-only gluing disagrees on {len(fails)} of 128 assignments, "
          f"all with mu1 = 1: {set(fails) == {1}}")
    ok &= len(fails) == 64 and set(fails) == {1}

    print("glue_example_check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
