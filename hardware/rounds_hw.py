#!/usr/bin/env python3
"""Hardware test of repeated rounds (the R term in min{d_X~, d_Z, R}).

Two three-qubit phase-flip blocks (X checks X0X1, X1X2; logical X = X0, logical
Z = Z0Z1Z2, so d_Z = 3) start in |+++>, so the joint logical X_a X_b is +1.  It
is measured by gauging on one edge qubit e (prepared in |0>) joining the ports
a0 and b0: the vertex stars are A_a0 = X_a0 X_e and A_b0 = X_b0 X_e, whose
product is X_a0 X_b0.  Each star outcome alone is random; their product, the
reported bit, must be 0.

The paper: a single wrong star outcome flips the reported bit, so the stars are
measured in R rounds and a wrong outcome that is not repeated shows up as a
change between rounds.  Here R = 1 or 2.  Every round uses fresh star
ancillas (the platform does not allow mid-circuit measurement), and every
qubit is measured once at the end.  After the rounds, the block X checks are
measured once with their own ancillas; they catch a phase error on a port
qubit, which would otherwise flip both rounds alike.

Detectors: the four block checks; for R = 2 also the round-to-round
comparisons of each star.  Observable: s_a1 xor s_b1 (round-one product).
Prediction: circuit fault distance of the reported bit 1 for R = 1, 2 for R = 2.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

from qiskit import QuantumCircuit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cut_obstruction_hw import wilson  # noqa: E402

DATA = ["a0", "a1", "a2", "b0", "b1", "b2"]
EDGE = "e"
STARS = {1: {"sa1": "a0", "sb1": "b0"}, 2: {"sa2": "a0", "sb2": "b0"}}
CHECKS = {"ca01": ("a0", "a1"), "ca12": ("a1", "a2"), "cb01": ("b0", "b1"), "cb12": ("b1", "b2")}
QUBITS = DATA + [EDGE] + [s for r in (1, 2) for s in STARS[r]] + list(CHECKS)   # 15
IDX = {q: i for i, q in enumerate(QUBITS)}


def gate_pairs():
    """All (ancilla, target) pairs used by R = 2 (R = 1 uses a subset)."""
    out = []
    for r in (1, 2):
        for s, v in STARS[r].items():
            out += [(s, v), (s, EDGE)]
    for c, (x, y) in CHECKS.items():
        out += [(c, x), (c, y)]
    return out


def build(R):
    qc = QuantumCircuit(len(QUBITS), len(QUBITS), name=f"R{R}")
    for v in DATA:
        qc.h(IDX[v])
    for r in range(1, R + 1):
        for s, v in STARS[r].items():
            a = IDX[s]
            qc.h(a)
            qc.cx(a, IDX[v])
            qc.cx(a, IDX[EDGE])
            qc.h(a)
    for c, (x, y) in CHECKS.items():
        a = IDX[c]
        qc.h(a)
        qc.cx(a, IDX[x])
        qc.cx(a, IDX[y])
        qc.h(a)
    qc.barrier()
    qc.measure(range(len(QUBITS)), range(len(QUBITS)))
    return qc


def detectors(R):
    dets = [[c] for c in CHECKS]
    if R == 2:
        dets += [["sa1", "sa2"], ["sb1", "sb2"]]
    return dets


def observable(R):
    return ["sa1", "sb1"]


def analyze_counts(counts, R, dets=None):
    """counts in Qiskit order (clbit i is character -1-i)."""
    dets = detectors(R) if dets is None else dets
    obs = observable(R)
    bit = lambda s, q: int(s[-1 - IDX[q]])
    n = acc = fail = 0
    for shot, c in counts.items():
        shot = shot.replace(" ", "")
        n += c
        if any(sum(bit(shot, q) for q in d) % 2 for d in dets):
            continue
        acc += c
        fail += c * (sum(bit(shot, q) for q in obs) % 2)
    return dict(shots=n, accepted=acc, accept_rate=acc / n, postselected_failures=fail,
                postselected_error=fail / max(acc, 1), postselected_error_wilson95=wilson(fail, acc))


if __name__ == "__main__":
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="stabilizer")
    for R in (1, 2):
        a = analyze_counts(sim.run(transpile(build(R), sim), shots=4000).result().get_counts(), R)
        assert a["accept_rate"] == 1.0 and a["postselected_failures"] == 0, a
        print(f"R={R}: noiseless, all detectors silent and reported bit correct in {a['shots']} shots")
