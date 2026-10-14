"""Compact QASM for providers that need contiguous qubits: only the qubits a width uses."""
import json
from pathlib import Path
from qiskit import QuantumCircuit, qasm2
import cut_obstruction_hw as hw

HERE = Path(__file__).resolve().parent
maps = {}
for w in (1, 2, 3):
    names = hw.DATA + hw.edges_for(w) + hw.ANC
    idx = {q: i for i, q in enumerate(names)}
    qc = QuantumCircuit(len(names), len(names))
    for v in hw.DATA:
        a = idx[f"x_{v}"]
        qc.h(a)
        qc.cx(a, idx[v])
        for e in [e for e in hw.edges_for(w) if v in hw.EDGES[e]]:
            qc.cx(a, idx[e])
        qc.h(a)
    qc.barrier()
    qc.measure(range(len(names)), range(len(names)))
    (HERE / f"qasm_compact_w{w}.qasm").write_text(qasm2.dumps(qc))
    maps[w] = names
(HERE / "compact_maps.json").write_text(json.dumps(maps, indent=1))
print({w: len(n) for w, n in maps.items()})
