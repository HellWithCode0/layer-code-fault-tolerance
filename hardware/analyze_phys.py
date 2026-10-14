"""Analyze the SWAP-free Rigetti run (export_physical.py circuits).

Each output holds two copies (A, B) of the experiment.  Before decoding, the
bit order is checked from the data: the intended detectors must be the
parities with the lowest violation rates among all two- to four-bit parities
of each copy.  If the reversed order fits better, it is used and reported.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw

N = len(hw.QUBITS)


def matrix(counts, reverse):
    rows, cnt = [], []
    for s, c in counts.items():
        s = s.replace(" ", "")
        rows.append([int(s[i] if reverse else s[-1 - i]) for i in range(2 * N)])
        cnt.append(c)
    return np.array(rows, dtype=np.uint8), np.array(cnt)


def copy_counts(B, C, k):
    out = {}
    for row, c in zip(B[:, k * N:(k + 1) * N], C):
        key = "".join(str(b) for b in row[::-1])          # qiskit order for hw.analyze_counts
        out[key] = out.get(key, 0) + int(c)
    return out


def det_rate(B, C, k, d):
    cols = [k * N + hw.IDX[q] for q in d]
    return float(((B[:, cols].sum(1) % 2) * C).sum() / C.sum())


def order_check(B, C, w):
    """Mean violation of the intended detectors, and their rank among all small parities of copy A."""
    intended = [tuple(sorted(hw.IDX[q] for q in d)) for d in hw.detectors(w)]
    rates = {}
    for r in (2, 3, 4):
        for t in itertools.combinations(range(N), r):
            v = float(((B[:, list(t)].sum(1) % 2) * C).sum() / C.sum())
            rates[t] = min(v, 1 - v)
    ranked = sorted(rates, key=rates.get)
    return float(np.mean([rates[t] for t in intended])), max(ranked.index(t) for t in intended)


def main():
    out = {}
    for w in (1, 2, 3):
        f = HERE / f"oq_phys_output_w{w}.json"
        if not f.exists():
            print(f"w={w}: no output yet")
            continue
        counts = json.loads(f.read_text())
        fits = {rev: order_check(*matrix(counts, rev), w) for rev in (False, True)}
        rev = min(fits, key=lambda r: fits[r][0])
        B, C = matrix(counts, rev)
        res = dict(bit_order="reversed" if rev else "qiskit", order_fit=fits[rev])
        for k, name in enumerate(("A", "B")):
            a = hw.analyze_counts(w, copy_counts(B, C, k))
            a["detector_violation_rates"] = [round(det_rate(B, C, k, d), 3) for d in hw.detectors(w)]
            res[name] = a
            print(f"w={w} copy {name}: accept {a['accept_rate']:.3f}, postselected error "
                  f"{a['postselected_error']:.4f} {[round(x, 4) for x in a['postselected_error_wilson95']]}, "
                  f"decoded error {a['decoded_error']:.4f}, detectors {a['detector_violation_rates']}")
        print(f"   bit order {res['bit_order']}; intended detectors rank within the top {fits[rev][1] + 1} parities")
        out[f"w{w}"] = res
    (HERE / "oq_phys_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
