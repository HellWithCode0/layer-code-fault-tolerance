"""Place the repeated-rounds circuits (rounds_hw.py) on IQM Emerald and check them.

Two copies on disjoint qubits (one unused coupler between them); every placement
stays SWAP-free if the device reads the qubit numbers as 1-based.  R = 1 and
R = 2 use the same qubits (in R = 1 the round-two ancillas stay idle).
Clbits: copy k, qubit i of rounds_hw.QUBITS -> c[15 k + i].

Checks: Qiskit parses the file; its gates equal the reference circuit
rounds_hw.build(R) on the same qubits (Clifford); a noiseless Aer run passes
every detector with the correct reported bit; every CZ is on a coupler under
both numberings; measured qubit order is right; the exact circuit fault
distance of the reported bit is R (CP-SAT); and rates under IQM-like noise.
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
import rounds_hw as rh
from embed_iqm import chip
from min_weight_logical import min_weight_logical

N = len(rh.QUBITS)
E = {tuple(sorted(e)) for e in json.loads((HERE / "oq_backend_iqm_emerald.json").read_text())
     ["constraint_data"]["topology"]["coupling_map"]}


def robust(p):
    return min(p.values()) >= 1 and all(tuple(sorted((p[a] - 1, p[t] - 1))) in E for a, t in rh.gate_pairs())


def placements():
    g = rx.PyGraph()
    idx = {q: g.add_node(q) for q in rh.QUBITS}
    for a, t in rh.gate_pairs():
        g.add_edge(idx[a], idx[t], None)
    seen, out = set(), []
    for m in rx.vf2_mapping(chip("emerald"), g, subgraph=True, induced=False, call_limit=None):
        p = {rh.QUBITS[hq]: gq for gq, hq in m.items()}
        key = frozenset(p.items())
        if key in seen or not robust(p):
            continue
        seen.add(key)
        out.append(p)
    return out


def pick_two(ps):
    ch = chip("emerald")
    for i, A in enumerate(ps):
        sa = set(A.values())
        for B in ps[i + 1:]:
            sb = set(B.values())
            if sa & sb or any(ch.has_edge(x, y) for x in sa for y in sb):
                continue
            return {"A": A, "B": B}
    return {"A": ps[0]}


def qasm(R, copies):
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{N * len(copies)}] c;"]

    def cx(p, a, t):
        assert tuple(sorted((p[a], p[t]))) in E
        lines.extend([f"h ${p[t]};", f"cz ${p[a]}, ${p[t]};", f"h ${p[t]};"])

    for p in copies.values():
        for v in rh.DATA:
            lines.append(f"h ${p[v]};")
        for r in range(1, R + 1):
            for s, v in rh.STARS[r].items():
                lines.append(f"h ${p[s]};")
                cx(p, s, v)
                cx(p, s, rh.EDGE)
                lines.append(f"h ${p[s]};")
        for c, (x, y) in rh.CHECKS.items():
            lines.append(f"h ${p[c]};")
            cx(p, c, x)
            cx(p, c, y)
            lines.append(f"h ${p[c]};")
    allq = [p[q] for p in copies.values() for q in rh.QUBITS]
    assert len(set(allq)) == len(allq)
    lines.append("barrier " + ", ".join(f"${q}" for q in allq) + ";")
    for k, p in enumerate(copies.values()):
        for i, q in enumerate(rh.QUBITS):
            lines.append(f"c[{N * k + i}] = measure ${p[q]};")
    return "\n".join(lines) + "\n"


def unitary_part(qc):
    out = QuantumCircuit(qc.num_qubits)
    for ins in qc.data:
        if ins.operation.name not in ("measure", "barrier"):
            out.append(ins.operation, [out.qubits[qc.find_bit(q).index] for q in ins.qubits])
    return out


def to_stim(text, p1=0.0, p2=0.0, pm=0.0):
    """h / cz / final measure; X_ERROR before each measurement models readout flips."""
    c, clbit = stim.Circuit(), {}
    phys = sorted({int(x) for x in re.findall(r"\$(\d+)", text)})
    c.append("R", phys)
    if pm:
        c.append("X_ERROR", phys, pm)                 # preparation error
    for ln in text.splitlines():
        if m := re.fullmatch(r"h \$(\d+);", ln):
            c.append("H", [int(m[1])])
            if p1:
                c.append("DEPOLARIZE1", [int(m[1])], p1)
        elif m := re.fullmatch(r"cz \$(\d+), \$(\d+);", ln):
            c.append("CZ", [int(m[1]), int(m[2])])
            if p2:
                c.append("DEPOLARIZE2", [int(m[1]), int(m[2])], p2)
        elif m := re.fullmatch(r"c\[(\d+)\] = measure \$(\d+);", ln):
            clbit[int(m[1])] = len(clbit)
            if pm:
                c.append("X_ERROR", [int(m[2])], pm)
            c.append("M", [int(m[2])])
    return c, clbit


def with_detectors(c, clbit, R, copy):
    c = c.copy()
    m = len(clbit)
    rec = lambda q: stim.target_rec(clbit[N * copy + rh.IDX[q]] - m)
    for d in rh.detectors(R):
        c.append("DETECTOR", [rec(q) for q in d])
    c.append("OBSERVABLE_INCLUDE", [rec(q) for q in rh.observable(R)], 0)
    return c


def main():
    # Two disjoint copies (no pair is fully separated; the chosen pair touches through one
    # unused coupler).  Found by searching the 2221 distinct qubit sets of placements().
    copies = json.loads((HERE / "iqm_rounds_two_copies.json").read_text())
    for p in copies.values():
        assert robust(p)
    print(f"using {len(copies)} copies: {json.dumps(copies)}")
    (HERE / "iqm_rounds_placement.json").write_text(json.dumps(copies))
    rep = {"copies": copies, "checks": {}, "prediction": {}}
    for R in (1, 2):
        text = qasm(R, copies)
        (HERE / f"qasm3_iqm_rounds_R{R}.qasm").write_text(text)
        qc = qasm3.loads(text)
        ref = QuantumCircuit(qc.num_qubits)
        b = rh.build(R)
        for p in copies.values():
            for ins in b.data:
                if ins.operation.name in ("measure", "barrier"):
                    continue
                ref.append(ins.operation, [ref.qubits[p[rh.QUBITS[b.find_bit(q).index]]] for q in ins.qubits])
        same = Clifford(unitary_part(qc)) == Clifford(ref)
        counts = AerSimulator(method="stabilizer").run(transpile(qc, AerSimulator(method="stabilizer")),
                                                       shots=4000).result().get_counts()
        ok_noiseless = True
        for k in range(len(copies)):
            sub = {}
            for s, cnt in counts.items():
                s = s.replace(" ", "")[::-1][N * k:N * (k + 1)][::-1]
                sub[s] = sub.get(s, 0) + cnt
            a = rh.analyze_counts(sub, R)
            ok_noiseless &= a["accept_rate"] == 1.0 and a["postselected_failures"] == 0
        pairs = [tuple(int(x) for x in ln[4:-1].replace("$", "").split(", ")) for ln in text.splitlines() if ln.startswith("cz ")]
        ok0 = all(tuple(sorted(p)) in E for p in pairs)
        ok1 = all(tuple(sorted((p[0] - 1, p[1] - 1))) in E for p in pairs)
        meas = [ln for ln in text.splitlines() if "measure" in ln]
        order_ok = all(meas[N * k + i] == f"c[{N * k + i}] = measure ${p[q]};"
                       for k, p in enumerate(copies.values()) for i, q in enumerate(rh.QUBITS))
        c, clbit = to_stim(text, 1e-3, 1e-3, 1e-3)
        dists = [min_weight_logical(with_detectors(c, clbit, R, k).detector_error_model(decompose_errors=False),
                                    0, time_limit=60)["distance"] for k in range(len(copies))]
        print(f"R={R}: parses {qc.num_qubits}q | equals reference: {same} | noiseless all copies correct: "
              f"{ok_noiseless} | {len(pairs)} CZ on couplers 0-based {ok0} 1-based {ok1} | measure order "
              f"{order_ok} | reported-bit fault distance per copy {dists}")
        assert same and ok_noiseless and ok0 and ok1 and order_ok and all(d == R for d in dists)
        rep["checks"][R] = dict(cz=len(pairs), fault_distance=dists)
        for p2 in (0.005, 0.01):
            c, clbit = to_stim(text, 1e-3, p2, 0.0)
            s = c.compile_sampler().sample(200000).astype(np.uint8)
            u = np.random.default_rng(9).random(s.shape)
            s = np.where(s == 1, (u >= 0.04).astype(np.uint8), (u < 0.02).astype(np.uint8))
            col = {q: s[:, clbit[rh.IDX[q]]] for q in rh.QUBITS}
            dets = np.array([np.bitwise_xor.reduce([col[q] for q in d]) for d in rh.detectors(R)]).T
            obs = np.bitwise_xor.reduce([col[q] for q in rh.observable(R)])
            acc = ~dets.any(axis=1)
            rep["prediction"][f"R={R} p2={p2}"] = dict(accept=float(acc.mean()), error=float(obs[acc].mean()))
            print(f"   CZ error {p2}: accept {acc.mean():.3f}, reported-bit error {obs[acc].mean():.4f}")
    (HERE / "rounds_plan.json").write_text(json.dumps(rep, indent=1))
    print("all checks passed")


if __name__ == "__main__":
    main()
