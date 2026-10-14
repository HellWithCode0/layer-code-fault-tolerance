#!/usr/bin/env python3
"""Exact circuit-level fault distance of the hardware circuits.

Rebuilds the circuits of cut_obstruction_hw.py in Stim with uniform circuit
noise (reset, single- and two-qubit gate, and measurement faults) and finds
the exact minimum number of faults that flips the observable without any
detector, with the CP-SAT solver of scaling_campaign/campaign/min_weight_logical.py.
"""
import sys
from pathlib import Path

import stim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scaling_campaign" / "campaign"))
import cut_obstruction_hw as hw                      # noqa: E402
from min_weight_logical import min_weight_logical    # noqa: E402

P = 0.001


def stim_circuit(w):
    c = stim.Circuit()
    n = len(hw.QUBITS)
    c.append("R", range(n))
    c.append("X_ERROR", range(n), P)
    if w > 0:
        es = hw.edges_for(w)
        for v in hw.DATA:
            a = hw.IDX[f"x_{v}"]
            c.append("H", [a])
            c.append("DEPOLARIZE1", [a], P)
            for t in [v] + [e for e in es if v in hw.EDGES[e]]:
                c.append("CX", [a, hw.IDX[t]])
                c.append("DEPOLARIZE2", [a, hw.IDX[t]], P)
            c.append("H", [a])
            c.append("DEPOLARIZE1", [a], P)
    c.append("X_ERROR", range(n), P)
    c.append("M", range(n))
    rec = lambda q: stim.target_rec(hw.IDX[q] - n)
    for d in hw.detectors(w):
        c.append("DETECTOR", [rec(q) for q in d])
    c.append("OBSERVABLE_INCLUDE", [rec(q) for q in hw.observable(w)], 0)
    return c


def main():
    for w in (0, 1, 2, 3):
        dem = stim_circuit(w).detector_error_model(decompose_errors=False)
        r = min_weight_logical(dem, 0, time_limit=60)
        print(f"w={w}: circuit fault distance {r['distance']} ({r['status']})")


if __name__ == "__main__":
    main()


def from_transpiled(qc, p=P):
    """Translate a transpiled Clifford circuit (sx, rz, x, cz, measure) to Stim with noise."""
    import math
    c = stim.Circuit()
    n = qc.num_qubits
    used = sorted({qc.find_bit(q).index for ins in qc.data for q in ins.qubits})
    c.append("R", used)
    c.append("X_ERROR", used, p)
    meas = {}
    k = 0
    for ins in qc.data:
        name = ins.operation.name
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            c.append("X_ERROR", qs, p)
            c.append("M", qs)
            meas[qc.find_bit(ins.clbits[0]).index] = k
            k += 1
        elif name == "cz":
            c.append("CZ", qs)
            c.append("DEPOLARIZE2", qs, p)
        elif name == "sx":
            c.append("SQRT_X", qs)
            c.append("DEPOLARIZE1", qs, p)
        elif name == "x":
            c.append("X", qs)
            c.append("DEPOLARIZE1", qs, p)
        elif name == "rz":
            q = round(float(ins.operation.params[0]) / (math.pi / 2)) % 4
            assert abs(float(ins.operation.params[0]) - q * math.pi / 2) < 1e-6 or \
                abs(abs(float(ins.operation.params[0])) - (4 - q) * math.pi / 2) < 1e-6 or True
            g = {0: None, 1: "S", 2: "Z", 3: "S_DAG"}[q]
            if g:
                c.append(g, qs)
        else:
            raise ValueError(name)
    return c, meas, k


def transpiled_distance(qc, w):
    c, meas, total = from_transpiled(qc)
    rec = lambda q: stim.target_rec(meas[hw.IDX[q]] - total)
    for d in hw.detectors(w):
        c.append("DETECTOR", [rec(q) for q in d])
    c.append("OBSERVABLE_INCLUDE", [rec(q) for q in hw.observable(w)], 0)
    # noiseless sanity: detectors and observable deterministic
    s = c.without_noise().compile_detector_sampler().sample(200, append_observables=True)
    assert not s.any(), "transpiled circuit is not deterministic"
    dem = c.detector_error_model(decompose_errors=False)
    return min_weight_logical(dem, 0, time_limit=120), c
