"""Why did the Rigetti Cepheus run fail?  Three checks.

1. Readout.  The w = 0 control measures idle qubits in |0>; its flip rates give
   the |0> readout error of each qubit.
2. Fit.  The bare circuit (no SWAPs) is simulated in Stim with that readout
   error and a two-qubit depolarizing rate p2.  The p2 that reproduces the
   observed detector violation rates is compared with the published median
   (about 0.5%).
3. Routing.  The circuits are transpiled onto the Cepheus coupling map (12 x 9
   square grid, qubit 8 dead) twice: once with a fixed embedding that needs no
   SWAPs, once with the naive row-major layout (virtual qubit i on physical qubit 9 + i),
   which is what a compiler without a good placement starts from.
"""
import json
import sys
from pathlib import Path

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw

MAPS = json.loads((HERE / "compact_maps.json").read_text())


def bits(w):
    d = json.loads((HERE / f"oq_output_w{w}.json").read_text())
    names = hw.QUBITS if w == 0 else MAPS[str(w)]
    rows, cnt = [], []
    for s, c in d.items():
        s = s.replace(" ", "")
        rows.append([int(s[-1 - i]) for i in range(len(names))])
        cnt.append(c)
    return names, np.array(rows), np.array(cnt)


def observed_rates(w):
    names, B, C = bits(w)
    idx = {q: i for i, q in enumerate(names)}
    return [float(((B[:, [idx[q] for q in d]].sum(1) % 2) * C).sum() / C.sum()) for d in hw.detectors(w)]


def readout_zero():
    names, B, C = bits(0)
    p = (B * C[:, None]).sum(0) / C.sum()
    return {q: float(p[i]) for i, q in enumerate(names)}


def bare(w, p2, r0, r1, p1=1e-3):
    """Bare circuit, CX = H CZ H on the target, asymmetric readout."""
    c = stim.Circuit()
    n = len(hw.QUBITS)
    c.append("R", range(n))
    es = hw.edges_for(w)
    for v in hw.DATA:
        a = hw.IDX[f"x_{v}"]
        c.append("H", [a])
        c.append("DEPOLARIZE1", [a], p1)
        for t in [v] + [e for e in es if v in hw.EDGES[e]]:
            c.append("CX", [a, hw.IDX[t]])
            c.append("DEPOLARIZE2", [a, hw.IDX[t]], p2)
            c.append("DEPOLARIZE1", [hw.IDX[t]], 2 * p1)
        c.append("H", [a])
        c.append("DEPOLARIZE1", [a], p1)
    # asymmetric readout: |1> -> 0 with r1 (amplitude damping like), |0> -> 1 with r0
    c.append("M", range(n))
    s = c.compile_sampler().sample(200000).astype(np.uint8)
    rng = np.random.default_rng(1)
    u = rng.random(s.shape)
    s = np.where(s == 1, (u >= r1).astype(np.uint8), (u < r0).astype(np.uint8))
    return [float(np.bitwise_xor.reduce([s[:, hw.IDX[q]] for q in d]).mean()) for d in hw.detectors(w)]


def embedding(row=1, col=1, width=9):
    """SWAP-free placement on the square grid: ancilla rows at row+1 and row+3."""
    g = lambda r, c: (row + r) * width + (col + c)
    place = {"a0": g(0, 0), "a1": g(0, 2), "a2": g(0, 4),
             "x_a0": g(1, 0), "ea01": g(1, 1), "x_a1": g(1, 2), "ea12": g(1, 3), "x_a2": g(1, 4),
             "br0": g(2, 0), "br1": g(2, 2), "br2": g(2, 4),
             "x_b0": g(3, 0), "eb01": g(3, 1), "x_b1": g(3, 2), "eb12": g(3, 3), "x_b2": g(3, 4),
             "b0": g(4, 0), "b1": g(4, 2), "b2": g(4, 4)}
    return place


def routing():
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    b = json.loads((HERE / "oq_backend_cepheus.json").read_text())
    cmap = CouplingMap(b["constraint_data"]["topology"]["coupling_map"])
    cmap.make_symmetric()
    out = {}
    emb = embedding()
    for w in (1, 2, 3):
        qc = hw.build(w)
        good = transpile(qc, coupling_map=cmap, basis_gates=["cz", "rx", "rz"],
                         initial_layout=[emb[q] for q in hw.QUBITS], optimization_level=1, seed_transpiler=1)
        # naive row-major placement on qubits 9..27 (qubit 8 is dead)
        triv = transpile(qc, coupling_map=cmap, basis_gates=["cz", "rx", "rz"],
                         initial_layout=list(range(9, 9 + len(hw.QUBITS))),
                         optimization_level=1, seed_transpiler=1)
        out[w] = dict(cz_swap_free=int(good.count_ops().get("cz", 0)),
                      cz_trivial_layout=int(triv.count_ops().get("cz", 0)),
                      depth_swap_free=good.depth(), depth_trivial_layout=triv.depth(),
                      circuits=(good, triv))
    return out


def transpiled_sim(w, qc, p2, r0, r1, p1=1e-3, shots=200000):
    """Stim run of a transpiled (cz, rx, rz) circuit with gate noise and asymmetric readout."""
    import math
    c = stim.Circuit()
    used = sorted({qc.find_bit(q).index for ins in qc.data for q in ins.qubits})
    c.append("R", used)
    clbit_of = {}
    order = []
    for ins in qc.data:
        name = ins.operation.name
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            clbit_of[qc.find_bit(ins.clbits[0]).index] = len(order)
            order.append(qs[0])
            c.append("M", qs)
        elif name == "cz":
            c.append("CZ", qs)
            c.append("DEPOLARIZE2", qs, p2)
        elif name in ("rx", "rz"):
            k = round(float(ins.operation.params[0]) / (math.pi / 2)) % 4
            g = {"rx": {1: "SQRT_X", 2: "X", 3: "SQRT_X_DAG"}, "rz": {1: "S", 2: "Z", 3: "S_DAG"}}[name].get(k)
            if g:
                c.append(g, qs)
                if name == "rx":
                    c.append("DEPOLARIZE1", qs, p1)
    s = c.compile_sampler().sample(shots).astype(np.uint8)
    u = np.random.default_rng(2).random(s.shape)
    s = np.where(s == 1, (u >= r1).astype(np.uint8), (u < r0).astype(np.uint8))
    col = {q: s[:, clbit_of[hw.IDX[q]]] for q in hw.QUBITS}
    return [float(np.bitwise_xor.reduce([col[q] for q in d]).mean()) for d in hw.detectors(w)]


def main():
    rep = {"readout_zero": readout_zero()}
    r0 = float(np.mean(list(rep["readout_zero"].values())))
    rep["mean_readout_zero"] = r0
    print(f"mean |0> readout error (w=0 control): {r0:.4f}; worst: "
          f"{max(rep['readout_zero'].items(), key=lambda kv: kv[1])}")
    rep["observed"] = {w: observed_rates(w) for w in (1, 2, 3)}
    rep["fit"] = {}
    for r1 in (r0, 3 * r0):
        for p2 in (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12):
            for w in (1, 2, 3):
                sim = bare(w, p2, r0, r1)
                rep["fit"][f"r1={r1:.3f} p2={p2} w={w}"] = sim
            print(f"r1={r1:.3f} p2={p2:5.3f}: mean detector rate "
                  + "  ".join(f"w{w} sim {np.mean(rep['fit'][f'r1={r1:.3f} p2={p2} w={w}']):.3f}"
                              f" / obs {np.mean(rep['observed'][w]):.3f}" for w in (1, 2, 3)), flush=True)
    rep["routing"] = routing()
    rep["routed_sim"] = {}
    for w, r in rep["routing"].items():
        good, triv = r.pop("circuits")
        print(f"w={w}: CZ gates {r['cz_swap_free']} with the SWAP-free placement, "
              f"{r['cz_trivial_layout']} with the naive placement (depth {r['depth_swap_free']} vs {r['depth_trivial_layout']})")
        for p2 in (0.005, 0.01):
            for tag, qc in (("swap_free", good), ("naive", triv)):
                sim = transpiled_sim(w, qc, p2, r0, 3 * r0)
                rep["routed_sim"][f"w={w} p2={p2} {tag}"] = sim
                print(f"   p2={p2}: {tag:9s} mean detector rate {np.mean(sim):.3f} "
                      f"(observed {np.mean(rep['observed'][w]):.3f})", flush=True)
    (HERE / "rigetti_diagnosis.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
