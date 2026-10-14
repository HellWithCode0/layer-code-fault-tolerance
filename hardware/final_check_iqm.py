"""Final checks of the IQM Emerald job before submission.

1. Each QASM file parses with Qiskit's own OpenQASM 3 importer.
2. Its gates equal the reference circuit cut_obstruction_hw.build(w), mapped by
   the placement: the two Clifford operators are identical.
3. A noiseless Aer run of the parsed file, read in Qiskit bit order, passes
   every detector with the correct observable (end-to-end with analyze_counts).
4. Every CZ sits on a coupler under both 0-based and 1-based qubit numbers.
5. The exact circuit fault distance of the file is w (CP-SAT on the Stim DEM).
"""
import json
import sys
from pathlib import Path

import stim
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.quantum_info import Clifford
from qiskit_aer import AerSimulator

HERE = Path(__file__).resolve().parent
RUN = __import__("os").environ.get("IQM_RUN", "")   # "" first qubit set, "2" second set
PLACEMENT_FILE = {"": "iqm_robust_placement.json", "2": "iqm_second_placement.json",
                  "3": "iqm_robust_placement.json"}[RUN]   # run 3 repeats run 1's qubits
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scaling_campaign" / "campaign"))
import cut_obstruction_hw as hw
import export_physical as ep
from min_weight_logical import min_weight_logical

place = json.loads((HERE / PLACEMENT_FILE).read_text())[0]
E = {tuple(sorted(e)) for e in json.loads((HERE / "oq_backend_iqm_emerald.json").read_text())
     ["constraint_data"]["topology"]["coupling_map"]}


def unitary_part(qc):
    out = QuantumCircuit(qc.num_qubits)
    for ins in qc.data:
        if ins.operation.name not in ("measure", "barrier"):
            out.append(ins.operation, [out.qubits[qc.find_bit(q).index] for q in ins.qubits])
    return out


for w in (1, 2, 3):
    text = (HERE / f"qasm3_iqm{RUN}_w{w}.qasm").read_text()
    qc = qasm3.loads(text)
    n = qc.num_qubits
    # 2. reference circuit mapped onto the same physical qubits
    ref = QuantumCircuit(n)
    for ins in hw.build(w).data:
        if ins.operation.name in ("measure", "barrier"):
            continue
        ref.append(ins.operation, [ref.qubits[place[hw.QUBITS[hw.build(w).find_bit(q).index]]] for q in ins.qubits])
    same = Clifford(unitary_part(qc)) == Clifford(ref)
    # 3. noiseless end-to-end
    counts = AerSimulator(method="stabilizer").run(transpile(qc, AerSimulator(method="stabilizer")),
                                                   shots=4000).result().get_counts()
    a = hw.analyze_counts(w, counts)
    # 4. couplers under both numberings
    pairs = [tuple(int(x) for x in ln[4:-1].replace("$", "").split(", ")) for ln in text.splitlines() if ln.startswith("cz ")]
    ok0 = all(tuple(sorted(p)) in E for p in pairs)
    ok1 = all(tuple(sorted((p[0] - 1, p[1] - 1))) in E for p in pairs)
    # measured qubit k must be QUBITS[k]
    meas = [ln for ln in text.splitlines() if "measure" in ln]
    order_ok = all(m == f"c[{i}] = measure ${place[q]};" for i, (m, q) in enumerate(zip(meas, hw.QUBITS)))
    # 5. exact circuit fault distance
    c, clbit = ep.to_stim(text, 1e-3, 1e-3)
    phys = sorted({int(x) for x in __import__("re").findall(r"\$(\d+)", text)})
    noisy = stim.Circuit()
    for inst in c:
        if inst.name == "M":
            noisy.append("X_ERROR", inst.targets_copy(), 1e-3)
        noisy.append(inst)
        if inst.name == "R":
            noisy.append("X_ERROR", phys, 1e-3)
    m = len(clbit)
    rec = lambda q: stim.target_rec(clbit[hw.IDX[q]] - m)
    for d in hw.detectors(w):
        noisy.append("DETECTOR", [rec(q) for q in d])
    noisy.append("OBSERVABLE_INCLUDE", [rec(q) for q in hw.observable(w)], 0)
    dist = min_weight_logical(noisy.detector_error_model(decompose_errors=False), 0, time_limit=60)
    print(f"w={w}: parses with Qiskit {qc.num_qubits}q | equals reference circuit: {same} | "
          f"noiseless accept {a['accept_rate']:.3f} errors {a['postselected_failures']} | "
          f"{len(pairs)} CZ on couplers 0-based {ok0} 1-based {ok1} | measure order {order_ok} | "
          f"fault distance {dist['distance']} ({dist['status']})")
    assert same and a["accept_rate"] == 1.0 and a["postselected_failures"] == 0 and ok0 and ok1 and order_ok
    assert dist["distance"] == w
print("all checks passed")
