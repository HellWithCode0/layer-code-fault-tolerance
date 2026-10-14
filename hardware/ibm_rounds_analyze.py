"""Analyze the IBM repeated-rounds job (ibm_rounds_submit.py).

Analysis plan, fixed before submission (2026-10-13):
  Primary tests, pooled over the two copies, postselected on all detectors:
    reported-bit error at R = 2 below R = 1, and at R = 3 below R = 2
    (one-sided Fisher exact tests, each at level 0.05).
  Also reported: each copy alone with Wilson intervals, and the control that
  ignores the round-to-round star comparisons (it should return to the R = 1
  level).
  Bit order: column i is clbit i (BitArray.to_bool_array(order="little")).
  It cannot be checked from these data: reversing the bit string maps copy A
  round r onto copy B round R+1-r, and the detector set onto itself.  The same
  convention was confirmed on the first IBM run, where the reversed reading
  put every detector at 50%.
Input: ibm2_rounds_bits.npz with one packed bit array per round count.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ibm_rounds_hw as rh
from cut_obstruction_hw import wilson


def fisher_greater(k1, n1, k2, n2):
    """One-sided: group 1 fails more often.  Log-space sum for large counts."""
    K, N = k1 + k2, n1 + n2
    lg = lambda n, k: math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    base = lg(N, K)
    terms = [lg(n1, k) + lg(n2, K - k) - base for k in range(k1, min(K, n1) + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def detector_total(bits, R):
    tot = 0.0
    for k in (0, 1):
        for d in rh.detectors(R, k):
            tot += float((bits[:, d].sum(axis=1) % 2).mean())
    return tot


def analyze(bits_by_R):
    out = {"per_R": {}}
    for R, bits in sorted(bits_by_R.items()):
        res = dict(detector_total=detector_total(bits, R), copies=[rh.analyze_bits(bits, R, k) for k in (0, 1)])
        if R >= 2:
            res["control_no_star_comparisons"] = [rh.analyze_bits(bits, R, k, include_star_comparisons=False)
                                                  for k in (0, 1)]
        f = sum(c["postselected_failures"] for c in res["copies"])
        a = sum(c["accepted"] for c in res["copies"])
        res["pooled"] = dict(failures=f, accepted=a, error=f / max(a, 1), wilson95=wilson(f, a))
        out["per_R"][R] = res
    tests = {}
    for lo, hi in ((1, 2), (2, 3)):
        if lo in out["per_R"] and hi in out["per_R"]:
            p1, p2 = out["per_R"][lo]["pooled"], out["per_R"][hi]["pooled"]
            tests[f"R{hi}_below_R{lo}"] = fisher_greater(p1["failures"], p1["accepted"], p2["failures"], p2["accepted"])
    out["primary_tests"] = tests
    return out


def report(out):
    for R, res in out["per_R"].items():
        print(f"R={R} (sum of detector rates over both copies {res['detector_total']:.2f})")
        for k, c in enumerate(res["copies"]):
            print(f"   copy {'AB'[k]}: {c['postselected_failures']} wrong of {c['accepted']} kept "
                  f"({100 * c['accept_rate']:.1f}% kept) = {100 * c['postselected_error']:.4f}% "
                  f"[{100 * c['postselected_error_wilson95'][0]:.4f}%, {100 * c['postselected_error_wilson95'][1]:.4f}%]")
        if "control_no_star_comparisons" in res:
            cs = res["control_no_star_comparisons"]
            print("   control without star comparisons: " + ", ".join(
                f"copy {'AB'[k]} {100 * c['postselected_error']:.2f}%" for k, c in enumerate(cs)))
        p = res["pooled"]
        print(f"   pooled: {p['failures']}/{p['accepted']} = {100 * p['error']:.4f}% "
              f"[{100 * p['wilson95'][0]:.4f}%, {100 * p['wilson95'][1]:.4f}%]")
    for name, p in out["primary_tests"].items():
        print(f"primary test {name}: one-sided Fisher p = {p:.3g}{' (below 1e-300)' if p == 0 else ''} -> {'CLEAR (p < 0.05)' if p < 0.05 else 'not clear at 0.05'}")


def load(path=HERE / "ibm2_rounds_bits.npz"):
    z = np.load(path)
    return {int(k[1:]): np.unpackbits(z[k], axis=1, count=int(z[f"n{k[1:]}"]), bitorder="little")
            for k in z.files if k.startswith("R")}


def main():
    out = analyze(load())
    report(out)
    (HERE / "ibm2_rounds_results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
