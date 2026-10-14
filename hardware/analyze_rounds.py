"""Analyze the repeated-rounds run on IQM Emerald (export_rounds.py circuits).

Analysis plan, fixed before submission (2026-10-13):
  Primary test: reported-bit error after postselection on all detectors,
  pooled over the two copies; one-sided Fisher exact test that R = 2 fails less
  often than R = 1, at level 0.05.
  Control: R = 2 data postselected on the block checks only (ignoring the
  round-to-round star comparisons) should return to the R = 1 level, because
  then nothing catches a single wrong star outcome.
  The output bit order is found from the data (lowest total detector rate
  among candidate orders) and reported.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rounds_hw as rh
from cut_obstruction_hw import wilson

COPIES = json.loads((HERE / "iqm_rounds_two_copies.json").read_text())
N = len(rh.QUBITS)
L = N * len(COPIES)


def candidates(length):
    """position in the string -> (copy index, qubit name)."""
    names = [(k, q) for k in range(len(COPIES)) for q in rh.QUBITS]          # clbit order
    phys = {(k, q): p[q] for k, p in enumerate(COPIES.values()) for q in rh.QUBITS}
    by_phys = sorted(names, key=lambda kq: phys[kq])
    if length != L:
        return {}
    return {"clbits right to left (Qiskit)": {kq: L - 1 - i for i, kq in enumerate(names)},
            "clbits left to right": {kq: i for i, kq in enumerate(names)},
            "sorted physical, left to right": {kq: i for i, kq in enumerate(by_phys)},
            "sorted physical, right to left": {kq: L - 1 - i for i, kq in enumerate(by_phys)}}


def analyze(raw, R, dets=None, order=None):
    counts = {s.replace(" ", ""): c for s, c in raw.items()}
    length = len(next(iter(counts)))
    n = sum(counts.values())
    cands = candidates(length)
    if not cands:
        raise ValueError(f"unexpected bit-string length {length} (expected {L})")
    score = {name: sum(c for s, c in counts.items() for k in range(len(COPIES)) for d in rh.detectors(R)
                       if sum(int(s[pos[(k, q)]]) for q in d) % 2) / n
             for name, pos in cands.items()}
    best = order if order is not None else min(score, key=score.get)
    pos = cands[best]
    per_copy = []
    for k in range(len(COPIES)):
        q_counts = {}
        for s, c in counts.items():
            key = "".join(s[pos[(k, q)]] for q in reversed(rh.QUBITS))
            q_counts[key] = q_counts.get(key, 0) + c
        per_copy.append(rh.analyze_counts(q_counts, R, dets))
    return dict(bit_order=best, order_scores=score, copies=per_copy)


def fisher_greater(k1, n1, k2, n2):
    K, N_ = k1 + k2, n1 + n2
    return sum(math.comb(n1, k) * math.comb(n2, K - k) for k in range(k1, min(K, n1) + 1)) / math.comb(N_, K)


def pooled(res):
    f = sum(c["postselected_failures"] for c in res["copies"])
    a = sum(c["accepted"] for c in res["copies"])
    return f, a


def main():
    # Both jobs use the same layout, so the bit order is found once from R = 2
    # (which has the round-to-round star detectors) and applied to R = 1.
    files = {R: HERE / f"oq_rounds_R{R}_output.json" for R in (2, 1)}
    out, order = {}, None
    for R, f in files.items():
        if not f.exists():
            print(f"R={R}: no output yet")
            continue
        raw = json.loads(f.read_text())
        res = analyze(raw, R, order=order if R == 1 else None)
        if R == 2:
            order = res["bit_order"]
        out[R] = res
        for k, c in enumerate(res["copies"]):
            print(f"R={R} copy {'AB'[k]}: {c['postselected_failures']} wrong of {c['accepted']} kept "
                  f"({100 * c['accept_rate']:.1f}% kept) = {100 * c['postselected_error']:.2f}% "
                  f"{[round(100 * x, 2) for x in c['postselected_error_wilson95']]}")
        print(f"   bit order: {res['bit_order']}; scores " + ", ".join(f"{k} {v:.2f}" for k, v in res["order_scores"].items()))
        if R == 2:
            ctrl = analyze(raw, 2, dets=rh.detectors(2)[:4], order=order)
            out["R2_control"] = ctrl
            fc, ac = pooled(ctrl)
            print(f"   control (block checks only, star comparisons ignored): {fc}/{ac} = {100 * fc / ac:.2f}%")
    if 1 in out and 2 in out:
        f1, a1 = pooled(out[1])
        f2, a2 = pooled(out[2])
        p = fisher_greater(f1, a1, f2, a2)
        lo1, hi1 = wilson(f1, a1)
        lo2, hi2 = wilson(f2, a2)
        out["primary"] = dict(R1=[f1, a1, [lo1, hi1]], R2=[f2, a2, [lo2, hi2]], p_one_sided=p)
        print(f"pooled R=1: {f1}/{a1} = {100 * f1 / a1:.2f}% [{100 * lo1:.2f}%, {100 * hi1:.2f}%]")
        print(f"pooled R=2: {f2}/{a2} = {100 * f2 / a2:.3f}% [{100 * lo2:.3f}%, {100 * hi2:.3f}%]")
        print(f"primary test, R=2 below R=1: one-sided Fisher p = {p:.3g} -> "
              f"{'CLEAR (p < 0.05)' if p < 0.05 else 'not clear at 0.05'}")
    (HERE / "rounds_results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
