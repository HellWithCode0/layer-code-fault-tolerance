"""Analyze the IQM Emerald run (export_iqm.py circuits, one copy).

The output bit order is not documented, so it is found from the data.  Every
candidate order maps string positions to qubit names; the order whose intended
detectors have the lowest total violation rate is used, and all candidates are
reported.  Candidates for 19-bit strings: classical bits right to left (Qiskit)
or left to right, and physical qubits sorted ascending, read either way.  For
longer strings: one position per physical qubit (0-based or 1-based), read
either way.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = __import__("os").environ.get("IQM_RUN", "")   # "" first qubit set, "2" second set
PLACEMENT_FILE = {"": "iqm_robust_placement.json", "2": "iqm_second_placement.json",
                  "3": "iqm_robust_placement.json"}[RUN]   # run 3 repeats run 1's qubits
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw

PLACE = json.loads((HERE / PLACEMENT_FILE).read_text())[0]


def candidates(length):
    by_phys = sorted(hw.QUBITS, key=lambda q: PLACE[q])
    if length == len(hw.QUBITS):
        return {
            "clbits right to left (Qiskit)": {q: length - 1 - hw.IDX[q] for q in hw.QUBITS},
            "clbits left to right": {q: hw.IDX[q] for q in hw.QUBITS},
            "sorted physical, left to right": {q: by_phys.index(q) for q in hw.QUBITS},
            "sorted physical, right to left": {q: length - 1 - by_phys.index(q) for q in hw.QUBITS},
        }
    out = {}
    for base in (0, 1):
        idx = {q: PLACE[q] - base for q in hw.QUBITS}
        if max(idx.values()) < length:
            out[f"physical {base}-based, left to right"] = idx
            out[f"physical {base}-based, right to left"] = {q: length - 1 - i for q, i in idx.items()}
    return out


def rates(counts, w, pos):
    n = sum(counts.values())
    return [round(sum(c for s, c in counts.items() if sum(int(s[pos[q]]) for q in d) % 2) / n, 3)
            for d in hw.detectors(w)]


def analyze(w, raw):
    if not (isinstance(raw, dict) and all(isinstance(v, int) for v in raw.values())):
        print(f"w={w}: unexpected output format {type(raw).__name__}: {str(raw)[:300]}")
        return None
    counts = {s.replace(" ", ""): c for s, c in raw.items()}
    length = len(next(iter(counts)))
    cands = {name: (pos, rates(counts, w, pos)) for name, pos in candidates(length).items()}
    best = min(cands, key=lambda k: sum(cands[k][1]))
    pos = cands[best][0]
    # rewrite in Qiskit order so analyze_counts reads it as designed
    q_counts = {}
    for s, c in counts.items():
        key = "".join(s[pos[q]] for q in reversed(hw.QUBITS))
        q_counts[key] = q_counts.get(key, 0) + c
    a = hw.analyze_counts(w, q_counts)
    a["bit_order"] = best
    a["detector_violation_rates"] = cands[best][1]
    a["order_candidates"] = {k: v[1] for k, v in cands.items()}
    return a


def main():
    out = {}
    for w in (1, 2, 3):
        f = HERE / f"oq_iqm{RUN}_output_w{w}.json"
        if not f.exists():
            print(f"w={w}: no output yet")
            continue
        a = analyze(w, json.loads(f.read_text()))
        if a is None:
            continue
        out[f"w{w}"] = a
        print(f"Emerald w={w}: shots {a['shots']}, accept {a['accept_rate']:.3f}, postselected error "
              f"{a['postselected_error']:.4f} {[round(x, 4) for x in a['postselected_error_wilson95']]}, "
              f"decoded error {a['decoded_error']:.4f}, detectors {a['detector_violation_rates']}")
        print("   bit order: " + a["bit_order"] + "; all candidates: "
              + "; ".join(f"{k} {sum(v):.2f}" for k, v in a["order_candidates"].items()))
    (HERE / f"oq_iqm{RUN}_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
