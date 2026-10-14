#!/usr/bin/env python3
"""IBM test of repeated rounds with mid-circuit measurement and reset.

The paper: the reported bit of a gauging measurement has fault distance at
least min{d_X~, d_Z, R}.  A single wrong vertex-star outcome flips the
reported product, so the stars are measured in R rounds and a wrong outcome
that is not repeated shows up as a change between rounds; the block's own
checks catch phase errors, up to the block's Z distance.

Blocks: two three-qubit phase-flip codes (X checks X0X1, X1X2; logical X = X0,
logical Z = Z0Z1Z2, d_Z = 3) in |+++>, so X_a X_b = +1.  Gauging graph: one
edge qubit e in |0> between ports a0 and b0; stars A_a0 = X_a0 X_e and
A_b0 = X_b0 X_e.  Every round measures the two stars and the four block
checks with ancillas that are measured and reset (fresh rounds).  The 13
qubits form a path, so the circuit needs no SWAP on heavy-hex:
    a2 - ca12 - a1 - ca01 - a0 - sa - e - sb - b0 - cb01 - b1 - cb12 - b2
Detectors: every block-check outcome (deterministic 0) and, for r >= 2, each
star outcome against the previous round.  Observable (reported bit): the
round-one product sa xor sb.  Prediction: fault distance min{R, 3}, so
1, 2, 3, 3 for R = 1, 2, 3, 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

from qiskit import QuantumCircuit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PATH = ["a2", "ca12", "a1", "ca01", "a0", "sa", "e", "sb", "b0", "cb01", "b1", "cb12", "b2"]
ANCS = PATH[1::2]                       # ca12, ca01, sa, sb, cb01, cb12
STARS = ["sa", "sb"]
CHECKS = ["ca12", "ca01", "cb01", "cb12"]
DATA = ["a2", "a1", "a0", "b0", "b1", "b2"]
EDGE = "e"
LEFT = {a: PATH[PATH.index(a) - 1] for a in ANCS}
RIGHT = {a: PATH[PATH.index(a) + 1] for a in ANCS}
NQ, NB = len(PATH), len(ANCS)


def q(k, name):
    return NQ * k + PATH.index(name)


def clbit(k, r, anc, R):
    """copy k, round r (1-based), ancilla name."""
    return k * R * NB + (r - 1) * NB + ANCS.index(anc)


def build(R, copies=2, resets=1):
    """resets: number of reset operations on each ancilla between rounds (1 in the
    first IBM run; 2 in the follow-up that tests whether failed resets explain
    the persistence of ancilla errors across rounds)."""
    qc = QuantumCircuit(NQ * copies, copies * R * NB, name=f"rounds_R{R}_reset{resets}")
    for k in range(copies):
        for v in DATA:
            qc.h(q(k, v))
    qc.barrier()
    for r in range(1, R + 1):
        for k in range(copies):
            for a in ANCS:
                qc.h(q(k, a))
        for side in (LEFT, RIGHT):
            for k in range(copies):
                for a in ANCS:
                    qc.cx(q(k, a), q(k, side[a]))
        for k in range(copies):
            for a in ANCS:
                qc.h(q(k, a))
        for k in range(copies):
            for a in ANCS:
                qc.measure(q(k, a), clbit(k, r, a, R))
        if r < R:
            for _ in range(resets):
                for k in range(copies):
                    for a in ANCS:
                        qc.reset(q(k, a))
        qc.barrier()
    return qc


def detectors(R, k, include_star_comparisons=True):
    dets = [[clbit(k, r, c, R)] for r in range(1, R + 1) for c in CHECKS]
    if include_star_comparisons:
        dets += [[clbit(k, r, s, R), clbit(k, r - 1, s, R)] for r in range(2, R + 1) for s in STARS]
    return dets


def observable(R, k):
    return [clbit(k, 1, "sa", R), clbit(k, 1, "sb", R)]


def analyze_bits(bits, R, k, include_star_comparisons=True):
    """bits: bool/uint8 array (shots, num_clbits), column i = clbit i."""
    import numpy as np
    from cut_obstruction_hw import wilson
    b = np.asarray(bits, dtype=np.uint8)
    dets = detectors(R, k, include_star_comparisons)
    fired = np.zeros(b.shape[0], dtype=bool)
    for d in dets:
        fired |= (b[:, d].sum(axis=1) % 2).astype(bool)
    obs = b[:, observable(R, k)].sum(axis=1) % 2
    acc = ~fired
    n_acc, n_fail = int(acc.sum()), int(obs[acc].sum())
    return dict(shots=int(b.shape[0]), accepted=n_acc, accept_rate=n_acc / b.shape[0],
                postselected_failures=n_fail, postselected_error=n_fail / max(n_acc, 1),
                postselected_error_wilson95=wilson(n_fail, n_acc))


if __name__ == "__main__":
    import numpy as np
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="stabilizer")
    for R in (1, 2, 3, 4):
        qc = build(R)
        res = sim.run(transpile(qc, sim), shots=2000, memory=True).result()
        mem = res.get_memory()
        bits = np.array([[int(s[-1 - i]) for i in range(qc.num_clbits)] for s in mem], dtype=np.uint8)
        for k in range(2):
            a = analyze_bits(bits, R, k)
            assert a["accept_rate"] == 1.0 and a["postselected_failures"] == 0, (R, k, a)
        print(f"R={R}: noiseless, both copies: all detectors silent, reported bit correct")
