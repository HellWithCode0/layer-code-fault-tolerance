"""Analyze the IQM Emerald edge-multiplicity run (export_multiplicity.py).

Finds the output bit order from the data (as analyze_iqm.py), then reports the
postselected error with all checks and, as the built-in control, without the
two-edge check Z(br0) Z(br0p).  Without that check a single bridge fault is
invisible again, so the control should sit at the width-one level.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multiplicity_hw as mh

PLACE = json.loads((HERE / "iqm_m2_placement.json").read_text())[0]
Q = mh.QUBITS


def candidates(length):
    by_phys = sorted(Q, key=lambda q: PLACE[q])
    if length == len(Q):
        return {"clbits right to left (Qiskit)": {q: length - 1 - mh.IDX[q] for q in Q},
                "clbits left to right": {q: mh.IDX[q] for q in Q},
                "sorted physical, left to right": {q: by_phys.index(q) for q in Q},
                "sorted physical, right to left": {q: length - 1 - by_phys.index(q) for q in Q}}
    out = {}
    for base in (0, 1):
        idx = {q: PLACE[q] - base for q in Q}
        if max(idx.values()) < length:
            out[f"physical {base}-based, left to right"] = idx
            out[f"physical {base}-based, right to left"] = {q: length - 1 - i for q, i in idx.items()}
    return out


def analyze(raw):
    counts = {s.replace(" ", ""): c for s, c in raw.items()}
    length = len(next(iter(counts)))
    n = sum(counts.values())
    score = {}
    for name, pos in candidates(length).items():
        score[name] = (pos, [sum(c for s, c in counts.items() if sum(int(s[pos[q]]) for q in d) % 2) / n
                             for d in mh.detectors()])
    best = min(score, key=lambda k: sum(score[k][1]))
    pos = score[best][0]
    q_counts = {}
    for s, c in counts.items():
        key = "".join(s[pos[q]] for q in reversed(Q))
        q_counts[key] = q_counts.get(key, 0) + c
    full = mh.analyze_counts(q_counts)
    ctrl = mh.analyze_counts(q_counts, dets=mh.detectors()[:4])
    return dict(bit_order=best, order_scores={k: round(sum(v[1]), 3) for k, v in score.items()},
                detector_violation_rates=[round(x, 3) for x in score[best][1]],
                all_checks=full, without_two_edge_check=ctrl)


def main():
    f = HERE / "oq_iqm_m2_output.json"
    if not f.exists():
        print("no output yet")
        return
    r = analyze(json.loads(f.read_text()))
    (HERE / "oq_iqm_m2_results.json").write_text(json.dumps(r, indent=1))
    a, c = r["all_checks"], r["without_two_edge_check"]
    print(f"Emerald width 1, two parallel bridge qubits: shots {a['shots']}")
    print(f"   all checks:              {a['postselected_failures']} wrong of {a['accepted']} kept = "
          f"{100 * a['postselected_error']:.2f}% {[round(100 * x, 2) for x in a['postselected_error_wilson95']]}")
    print(f"   without two-edge check:  {c['postselected_failures']} wrong of {c['accepted']} kept = "
          f"{100 * c['postselected_error']:.2f}% {[round(100 * x, 2) for x in c['postselected_error_wilson95']]}")
    print(f"   detectors {r['detector_violation_rates']}; bit order {r['bit_order']}; scores {r['order_scores']}")


if __name__ == "__main__":
    main()
