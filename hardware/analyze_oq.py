"""Analyze the Open Quantum (Rigetti) outputs with the same detectors, observable and decoder.

w = 0 used the full 19-qubit register; w = 1, 2, 3 used the compact registers of
compact_maps.json.  Bitstrings are Qiskit-ordered (clbit 0 rightmost); this was
checked on the w = 2 output, where the reversed order gives 50% violation on
every detector.  Also reports, per detector, the violation rate: 0.5 means no
information survives.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw


def analyze(w, counts, names):
    idx = {q: i for i, q in enumerate(names)}
    old = hw.IDX
    hw.IDX = idx                         # analyze_counts reads bits through hw.IDX
    try:
        a = hw.analyze_counts(w, counts)
    finally:
        hw.IDX = old
    n = sum(counts.values())
    viol = []
    for d in hw.detectors(w):
        v = sum(c for s, c in counts.items() if sum(int(s[-1 - idx[q]]) for q in d) % 2)
        viol.append(round(v / n, 3))
    a["detector_violation_rates"] = viol
    a["observable_raw_error"] = round(sum(c for s, c in counts.items()
                                          if sum(int(s[-1 - idx[q]]) for q in hw.observable(w)) % 2) / n, 4)
    return a


def main():
    maps = json.loads((HERE / "compact_maps.json").read_text())
    out = {}
    for w in (0, 1, 2, 3):
        f = HERE / f"oq_output_w{w}.json"
        if not f.exists():
            print(f"w={w}: no output yet")
            continue
        names = hw.QUBITS if w == 0 else maps[str(w)]
        a = analyze(w, json.loads(f.read_text()), names)
        out[f"w{w}"] = a
        print(f"Rigetti w={w}: shots {a['shots']}, accept {a['accept_rate']:.3f}, postselected error "
              f"{a['postselected_error']:.4f} {[round(x, 4) for x in a['postselected_error_wilson95']]}, "
              f"decoded error {a['decoded_error']:.4f}, raw observable error {a['observable_raw_error']}, "
              f"detector violations {a['detector_violation_rates']}")
    (HERE / "oq_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
