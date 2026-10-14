"""Predicted rates for the transpiled circuits under a simple Heron-like noise model.

Two-qubit depolarizing 3e-3, one-qubit 3e-4, preparation flip 5e-3, readout
flip 1.5e-2 (typical published ibm_fez medians; not fitted to this run).
"""
import json
import sys
from pathlib import Path

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw
import circuit_distance_check as cdc

P2, P1, PREP, MEAS = 3e-3, 3e-4, 5e-3, 1.5e-2


def noisy(qc):
    import math
    c = stim.Circuit()
    used = sorted({qc.find_bit(q).index for ins in qc.data for q in ins.qubits})
    c.append("R", used)
    c.append("X_ERROR", used, PREP)
    meas, k = {}, 0
    for ins in qc.data:
        name = ins.operation.name
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            c.append("X_ERROR", qs, MEAS)
            c.append("M", qs)
            meas[qc.find_bit(ins.clbits[0]).index] = k
            k += 1
        elif name == "cz":
            c.append("CZ", qs)
            c.append("DEPOLARIZE2", qs, P2)
        elif name in ("sx", "x"):
            c.append({"sx": "SQRT_X", "x": "X"}[name], qs)
            c.append("DEPOLARIZE1", qs, P1)
        elif name == "rz":
            q = round(float(ins.operation.params[0]) / (math.pi / 2)) % 4
            g = {0: None, 1: "S", 2: "Z", 3: "S_DAG"}[q]
            if g:
                c.append(g, qs)
    return c, meas, k


def main(shots=200000):
    be = hw.service().backend(hw.BACKEND)
    circs, init = hw.transpiled(be)
    out = {}
    for w, qc in circs.items():
        c, meas, k = noisy(qc)
        bits = c.compile_sampler().sample(shots)
        col = {q: bits[:, meas[hw.IDX[q]]] for q in hw.QUBITS}
        dets = np.array([np.bitwise_xor.reduce([col[q] for q in d]) for d in hw.detectors(w)]).T
        obs = np.bitwise_xor.reduce([col[q] for q in hw.observable(w)])
        table = hw.decoder_table(w)
        acc = ~dets.any(axis=1)
        corr = np.array([table.get(tuple(int(x) for x in s), 1) for s in dets])
        out[w] = dict(accept_rate=float(acc.mean()),
                      postselected_error=float(obs[acc].mean()),
                      decoded_error=float((obs ^ corr).mean()))
        print(f"predicted w={w}: accept {out[w]['accept_rate']:.3f}, postselected error "
              f"{out[w]['postselected_error']:.5f}, decoded error {out[w]['decoded_error']:.5f}", flush=True)
    (HERE / "prediction.json").write_text(json.dumps(dict(model=dict(p2=P2, p1=P1, prep=PREP, meas=MEAS),
                                                          shots=shots, result=out), indent=1))


if __name__ == "__main__":
    main()
