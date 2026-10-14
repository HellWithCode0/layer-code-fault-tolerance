"""IQM Emerald circuits with fixed physical qubits and no SWAPs (one copy).

The placement comes from embed_iqm.py (0-based Open Quantum numbering) and is
stored in iqm_robust_placement.json.  Classical bits c[0..18] follow the order
of cut_obstruction_hw.QUBITS.  The QASM is parsed back into Stim and checked
noiselessly; predictions use IQM-like noise (CZ 0.5% to 1%, readout 2% / 4%).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = __import__("os").environ.get("IQM_RUN", "")   # "" first qubit set, "2" second set
PLACEMENT_FILE = {"": "iqm_robust_placement.json", "2": "iqm_second_placement.json",
                  "3": "iqm_robust_placement.json"}[RUN]   # run 3 repeats run 1's qubits
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw
import export_physical as ep
from embed_iqm import chip, placements

N = len(hw.QUBITS)


def pick_two():
    g = chip("emerald")
    ps = placements("emerald")
    dist = {}
    best = None
    for i, A in enumerate(ps):
        sa = set(A.values())
        for B in ps[i + 1:]:
            sb = set(B.values())
            if sa & sb or any(g.has_edge(a, b) for a in sa for b in sb):
                continue
            if best is None:
                best = (A, B)
                return best
    return best


def qasm(w, copies):
    E = {tuple(sorted(e)) for e in json.loads((HERE / "oq_backend_iqm_emerald.json").read_text())
         ["constraint_data"]["topology"]["coupling_map"]}
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{N * len(copies)}] c;"]
    for place in copies.values():
        es = hw.edges_for(w)
        for v in hw.DATA:
            a = place[f"x_{v}"]
            lines.append(f"h ${a};")
            for t in [v] + [e for e in es if v in hw.EDGES[e]]:
                p = place[t]
                assert tuple(sorted((a, p))) in E, (v, t, a, p)
                lines += [f"h ${p};", f"cz ${a}, ${p};", f"h ${p};"]
            lines.append(f"h ${a};")
    allq = [place[q] for place in copies.values() for q in hw.QUBITS]
    assert len(set(allq)) == len(allq)
    lines.append("barrier " + ", ".join(f"${q}" for q in allq) + ";")
    for k, place in enumerate(copies.values()):
        for i, q in enumerate(hw.QUBITS):
            lines.append(f"c[{k * N + i}] = measure ${place[q]};")
    return "\n".join(lines) + "\n"


def main():
    # No two disjoint placements exist on Emerald, so one copy.  The placement is
    # chosen so that every CZ pair is also a coupler if the device reads the
    # qubit numbers as 1-based (each number one lower): SWAP-free either way.
    copies = {"A": json.loads((HERE / PLACEMENT_FILE).read_text())[0]}
    ep.COPIES = copies                       # rates() reads the copies from export_physical
    rep = {"copies": copies, "prediction": {}}
    for w in (1, 2, 3):
        text = qasm(w, copies)
        (HERE / f"qasm3_iqm{RUN}_w{w}.qasm").write_text(text)
        ideal = ep.rates(w, text, 0, 0, 0, 0, shots=20000)
        assert all(v["accept"] == 1.0 and v["postselected_error"] == 0.0 for v in ideal.values()), ideal
        print(f"w={w}: {len(copies)} copies, {text.count('cz ')} CZ gates; noiseless: detectors silent, observable correct")
        for p2 in (0.005, 0.01):
            pr = ep.rates(w, text, 1e-3, p2, 0.02, 0.04)
            rep["prediction"][f"w={w} p2={p2}"] = pr
            v = pr["A"]
            print(f"   CZ error {p2}: accept {v['accept']:.3f}, postselected error {v['postselected_error']:.4f}")
    print("copies:", json.dumps(copies))
    (HERE / f"iqm{RUN}_plan.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
