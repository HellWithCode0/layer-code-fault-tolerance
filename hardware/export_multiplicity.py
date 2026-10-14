"""Place the edge-multiplicity circuit (multiplicity_hw.py) on IQM Emerald and check it.

Placement: a SWAP-free embedding that stays SWAP-free if the device reads the
qubit numbers as 1-based (each number one lower), preferring qubits of the
first Emerald run.  Checks before submission, as for the width runs:
1. Qiskit parses the OpenQASM 3 file.
2. Its gates equal the reference circuit multiplicity_hw.build() (Clifford).
3. A noiseless Aer run passes every detector with the correct observable.
4. Every CZ is on a coupler under both numberings; measured qubit k is QUBITS[k].
5. The exact circuit fault distance is 2 (CP-SAT on the Stim DEM).
Also prints the expected rates under IQM-like noise.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import rustworkx as rx
import stim
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.quantum_info import Clifford
from qiskit_aer import AerSimulator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scaling_campaign" / "campaign"))
import multiplicity_hw as mh
import export_physical as ep
from embed_iqm import chip
from min_weight_logical import min_weight_logical

E = {tuple(sorted(e)) for e in json.loads((HERE / "oq_backend_iqm_emerald.json").read_text())
     ["constraint_data"]["topology"]["coupling_map"]}
PAIRS = [(f"x_{v}", t) for v in mh.DATA for t in mh.targets(v)]


def placement():
    g = rx.PyGraph()
    idx = {q: g.add_node(q) for q in mh.QUBITS}
    for a, t in PAIRS:
        g.add_edge(idx[a], idx[t], None)
    run1 = set(json.loads((HERE / "iqm_robust_placement.json").read_text())[0].values())
    best = None
    for m in rx.vf2_mapping(chip("emerald"), g, subgraph=True, induced=False, call_limit=None):
        p = {mh.QUBITS[hq]: gq for gq, hq in m.items()}
        if min(p.values()) < 1:
            continue
        if not all(tuple(sorted((p[a] - 1, p[t] - 1))) in E for a, t in PAIRS):
            continue
        key = len(run1 & set(p.values()))
        if best is None or key > best[0]:
            best = (key, p)
    return best


def qasm(place):
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{len(mh.QUBITS)}] c;"]
    for v in mh.DATA:
        a = place[f"x_{v}"]
        lines.append(f"h ${a};")
        for t in mh.targets(v):
            p = place[t]
            assert tuple(sorted((a, p))) in E
            lines += [f"h ${p};", f"cz ${a}, ${p};", f"h ${p};"]
        lines.append(f"h ${a};")
    lines.append("barrier " + ", ".join(f"${place[q]}" for q in mh.QUBITS) + ";")
    for i, q in enumerate(mh.QUBITS):
        lines.append(f"c[{i}] = measure ${place[q]};")
    return "\n".join(lines) + "\n"


def unitary_part(qc):
    out = QuantumCircuit(qc.num_qubits)
    for ins in qc.data:
        if ins.operation.name not in ("measure", "barrier"):
            out.append(ins.operation, [out.qubits[qc.find_bit(q).index] for q in ins.qubits])
    return out


def stim_with_detectors(text, p1, p2, pm):
    c, clbit = ep.to_stim(text, p1, p2)
    phys = sorted({int(x) for x in re.findall(r"\$(\d+)", text)})
    noisy = stim.Circuit()
    for inst in c:
        if inst.name == "M" and pm:
            noisy.append("X_ERROR", inst.targets_copy(), pm)
        noisy.append(inst)
        if inst.name == "R" and pm:
            noisy.append("X_ERROR", phys, pm)
    m = len(clbit)
    rec = lambda q: stim.target_rec(clbit[mh.IDX[q]] - m)
    for d in mh.detectors():
        noisy.append("DETECTOR", [rec(q) for q in d])
    noisy.append("OBSERVABLE_INCLUDE", [rec(q) for q in mh.observable()], 0)
    return noisy, clbit


def main():
    overlap, place = placement()
    (HERE / "iqm_m2_placement.json").write_text(json.dumps([place]))
    text = qasm(place)
    (HERE / "qasm3_iqm_m2.qasm").write_text(text)
    print(f"placement (shares {overlap} qubits with run 1): {place}")
    qc = qasm3.loads(text)
    ref = QuantumCircuit(qc.num_qubits)
    b = mh.build()
    for ins in b.data:
        if ins.operation.name in ("measure", "barrier"):
            continue
        ref.append(ins.operation, [ref.qubits[place[mh.QUBITS[b.find_bit(q).index]]] for q in ins.qubits])
    same = Clifford(unitary_part(qc)) == Clifford(ref)
    sim = AerSimulator(method="stabilizer")
    a = mh.analyze_counts(sim.run(transpile(qc, sim), shots=4000).result().get_counts())
    pairs = [tuple(int(x) for x in ln[4:-1].replace("$", "").split(", ")) for ln in text.splitlines() if ln.startswith("cz ")]
    ok0 = all(tuple(sorted(p)) in E for p in pairs)
    ok1 = all(tuple(sorted((p[0] - 1, p[1] - 1))) in E for p in pairs)
    meas = [ln for ln in text.splitlines() if "measure" in ln]
    order_ok = all(m == f"c[{i}] = measure ${place[q]};" for i, (m, q) in enumerate(zip(meas, mh.QUBITS)))
    noisy, _ = stim_with_detectors(text, 1e-3, 1e-3, 1e-3)
    dist = min_weight_logical(noisy.detector_error_model(decompose_errors=False), 0, time_limit=60)
    print(f"parses with Qiskit {qc.num_qubits}q | equals reference: {same} | noiseless accept "
          f"{a['accept_rate']:.3f} errors {a['postselected_failures']} | {len(pairs)} CZ on couplers "
          f"0-based {ok0} 1-based {ok1} | measure order {order_ok} | fault distance {dist['distance']} ({dist['status']})")
    assert same and a["accept_rate"] == 1.0 and a["postselected_failures"] == 0 and ok0 and ok1 and order_ok
    assert dist["distance"] == 2
    for p2 in (0.005, 0.01):
        c, clbit = ep.to_stim(text, 1e-3, p2)
        s = c.compile_sampler().sample(200000).astype(np.uint8)
        u = np.random.default_rng(6).random(s.shape)
        s = np.where(s == 1, (u >= 0.04).astype(np.uint8), (u < 0.02).astype(np.uint8))
        col = {q: s[:, clbit[mh.IDX[q]]] for q in mh.QUBITS}
        dets = np.array([np.bitwise_xor.reduce([col[q] for q in d]) for d in mh.detectors()]).T
        obs = np.bitwise_xor.reduce([col[q] for q in mh.observable()])
        acc = ~dets.any(axis=1)
        acc4 = ~dets[:, :4].any(axis=1)
        print(f"   CZ error {p2}: accept {acc.mean():.3f}, postselected error {obs[acc].mean():.4f}; "
              f"without the two-edge check {obs[acc4].mean():.4f}")
    print("all checks passed")


if __name__ == "__main__":
    main()
