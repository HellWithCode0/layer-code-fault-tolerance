"""Rigetti Cepheus circuits with fixed physical qubits and no SWAPs.

The first Rigetti run let the compiler place and route the qubits.  The
compiler added SWAP gates (see diagnose_rigetti.py), so the circuits became
about seven times deeper and the detectors saw almost pure noise.

Here every gate is written on a coupler of the 12 x 9 grid, so no SWAP is
needed, and the job is submitted with disable_qubit_rewiring = true.  Each
circuit holds two copies of the experiment in two separate regions of the chip
(rows 1 to 5 and rows 7 to 11), so one job gives two independent samples and
shows whether the region matters.

Classical bits: c[0..18] is copy A and c[19..37] is copy B, each in the order
of cut_obstruction_hw.QUBITS.  The script also parses the QASM back into Stim,
checks that the noiseless circuit leaves every detector silent and the
observable correct, and predicts the rates under Rigetti-like noise.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw
from diagnose_rigetti import embedding, readout_zero

COPIES = {"A": embedding(row=1, col=0), "B": embedding(row=7, col=4)}
N = len(hw.QUBITS)


def coupling():
    b = json.loads((HERE / "oq_backend_cepheus.json").read_text())
    return {tuple(sorted(e)) for e in b["constraint_data"]["topology"]["coupling_map"]}


def qasm(w):
    E = coupling()
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{N * len(COPIES)}] c;"]
    for place in COPIES.values():
        es = hw.edges_for(w)
        for v in hw.DATA:
            a = place[f"x_{v}"]
            lines.append(f"h ${a};")
            for t in [v] + [e for e in es if v in hw.EDGES[e]]:
                p = place[t]
                assert tuple(sorted((a, p))) in E, (v, t, a, p)
                lines += [f"h ${p};", f"cz ${a}, ${p};", f"h ${p};"]
            lines.append(f"h ${a};")
    allq = [place[q] for place in COPIES.values() for q in hw.QUBITS]
    assert len(set(allq)) == len(allq) and 8 not in allq
    lines.append("barrier " + ", ".join(f"${q}" for q in allq) + ";")
    for k, place in enumerate(COPIES.values()):
        for i, q in enumerate(hw.QUBITS):
            lines.append(f"c[{k * N + i}] = measure ${place[q]};")
    return "\n".join(lines) + "\n"


def to_stim(text, p1=0.0, p2=0.0):
    """Parse the QASM above back into Stim (h, cz, measure only)."""
    c, clbit = stim.Circuit(), {}
    phys = sorted({int(x) for x in re.findall(r"\$(\d+)", text)})
    c.append("R", phys)
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
            c.append("M", [int(m[2])])
    return c, clbit


def rates(w, text, p1, p2, r0, r1, shots=200000):
    c, clbit = to_stim(text, p1, p2)
    s = c.compile_sampler().sample(shots).astype(np.uint8)
    if r0 or r1:
        u = np.random.default_rng(3).random(s.shape)
        s = np.where(s == 1, (u >= r1).astype(np.uint8), (u < r0).astype(np.uint8))
    table = hw.decoder_table(w)
    out = {}
    for k, name in enumerate(COPIES):
        col = {q: s[:, clbit[k * N + i]] for i, q in enumerate(hw.QUBITS)}
        dets = np.array([np.bitwise_xor.reduce([col[q] for q in d]) for d in hw.detectors(w)]).T
        obs = np.bitwise_xor.reduce([col[q] for q in hw.observable(w)])
        acc = ~dets.any(axis=1)
        corr = np.array([table.get(tuple(int(x) for x in r), 1) for r in dets])
        out[name] = dict(accept=float(acc.mean()), postselected_error=float(obs[acc].mean()),
                         decoded_error=float((obs ^ corr).mean()),
                         cz=text.count("cz "), detector_rates=[round(float(x), 3) for x in dets.mean(0)])
    return out


def main():
    r0 = float(np.mean(list(readout_zero().values())))
    rep = {"copies": COPIES, "readout_zero_mean": r0, "prediction": {}}
    for w in (1, 2, 3):
        text = qasm(w)
        (HERE / f"qasm3_phys_w{w}.qasm").write_text(text)
        ideal = rates(w, text, 0, 0, 0, 0, shots=20000)
        assert all(v["accept"] == 1.0 and v["postselected_error"] == 0.0 for v in ideal.values()), ideal
        print(f"w={w}: {text.count('cz ')} CZ gates for two copies; noiseless: detectors silent, observable correct")
        for p2 in (0.005, 0.01):
            pr = rates(w, text, 1e-3, p2, r0, 3 * r0)
            rep["prediction"][f"w={w} p2={p2}"] = pr
            for name, v in pr.items():
                print(f"   p2={p2} copy {name}: accept {v['accept']:.3f}, postselected error "
                      f"{v['postselected_error']:.4f}, decoded error {v['decoded_error']:.4f}")
    (HERE / "phys_plan.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
