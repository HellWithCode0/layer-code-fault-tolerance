#!/usr/bin/env python3
"""Hardware test of edge multiplicity (Supplemental Corollary "edge multiplicity").

Same two three-qubit blocks as cut_obstruction_hw.py, joined by a bridge of
width one, but the single bridge edge (a0, b0) carries m0 = 2 parallel
ancillas, br0 and br0p.  Every wall between the blocks now contains both, so
the cut obstruction allows deformed X distance 2 instead of 1: a single X
fault on br0 flips the observable Z_a0 Z_b0 Z(br0) but also the new two-edge
cycle check Z(br0) Z(br0p), so it is detected.  Prediction: the postselected
error falls from the width-one level (about 2.4% on IQM Emerald) to the
width-two level (about 0.1% to 0.2%), although the bridge still has one edge.
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
EDGES = {"ea01": ("a0", "a1"), "ea12": ("a1", "a2"), "eb01": ("b0", "b1"),
         "eb12": ("b1", "b2"), "br0": ("a0", "b0"), "br0p": ("a0", "b0")}
ANC = [f"x_{v}" for v in DATA]
QUBITS = DATA + list(EDGES) + ANC          # 18 qubits
IDX = {q: i for i, q in enumerate(QUBITS)}


def targets(v):
    """CNOT order of the star at v.  An X fault on the ancilla between two
    CNOTs spreads X to the remaining targets.  If br0 and br0p came last (or
    first), one such fault would put X on both, which flips the observable and
    leaves the two-edge check silent: circuit distance 1.  Splitting them,
    br0 first and br0p third, makes every such suffix detectable."""
    ts = [v] + [e for e in EDGES if v in EDGES[e]]
    if "br0" in ts:
        block_edge = next(e for e in ts if e not in (v, "br0", "br0p"))
        ts = ["br0", v, "br0p", block_edge]
    return ts


def build():
    qc = QuantumCircuit(len(QUBITS), len(QUBITS), name="w1m2")
    for v in DATA:
        a = IDX[f"x_{v}"]
        qc.h(a)
        for t in targets(v):
            qc.cx(a, IDX[t])
        qc.h(a)
    qc.barrier()
    qc.measure(range(len(QUBITS)), range(len(QUBITS)))
    return qc


def detectors():
    return [["a0", "a1", "ea01"], ["a1", "a2", "ea12"], ["b0", "b1", "eb01"],
            ["b1", "b2", "eb12"], ["br0", "br0p"]]


def observable():
    return ["a0", "b0", "br0"]


def faults():
    return DATA + list(EDGES)


def syndrome_of(flips, dets):
    return tuple(sum(q in d for q in flips) % 2 for d in dets)


def decoder_table(max_weight=3):
    dets, obs, table = detectors(), observable(), {}
    for k in range(max_weight + 1):
        for sub in itertools.combinations(faults(), k):
            s = syndrome_of(sub, dets)
            if s not in table:
                table[s] = sum(q in obs for q in sub) % 2
    return table


def analyze_counts(counts, dets=None):
    """counts in Qiskit order (clbit i is character -1-i).  With dets given
    (a subset of the checks), only postselection is reported."""
    full = dets is None
    dets = detectors() if full else dets
    obs, table = observable(), decoder_table()
    bit = lambda s, q: int(s[-1 - IDX[q]])
    n = acc = acc_fail = dec_fail = 0
    for shot, c in counts.items():
        shot = shot.replace(" ", "")
        s = tuple(sum(bit(shot, q) for q in d) % 2 for d in dets)
        o = sum(bit(shot, q) for q in obs) % 2
        n += c
        if not any(s):
            acc += c
            acc_fail += c * o
        if full:
            corr = table.get(s)
            dec_fail += c * (1 if corr is None else (o ^ corr))
    return dict(shots=n, accepted=acc, accept_rate=acc / n, postselected_failures=acc_fail,
                postselected_error=acc_fail / max(acc, 1), postselected_error_wilson95=wilson(acc_fail, acc),
                decoded_failures=dec_fail, decoded_error=dec_fail / n)


if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator(method="stabilizer")
    a = analyze_counts(sim.run(transpile(build(), sim), shots=4000).result().get_counts())
    assert a["accept_rate"] == 1.0 and a["postselected_failures"] == 0, a
    best = None
    for k in range(1, 4):
        for sub in itertools.combinations(faults(), k):
            if not any(syndrome_of(sub, detectors())) and sum(q in observable() for q in sub) % 2:
                best = k
                break
        if best:
            break
    print(f"noiseless: all detectors silent, observable correct; lightest undetected data/edge logical: {best}")
