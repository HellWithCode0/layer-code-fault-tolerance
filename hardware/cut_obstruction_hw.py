#!/usr/bin/env python3
"""Hardware test of the cut obstruction for a joint logical measurement.

Two small blocks are joined by a gauging graph and the joint logical
X_a X_b is measured, as in Sec. III B of the manuscript.  The bridge between
the blocks has w = 1, 2 or 3 edges.  Proposition "Cut obstruction" says that
X on the bridge edges is a logical operator of the deformed code equivalent to
X_a, so the deformed X distance is at most w.  On hardware this shows up as
follows.  The blocks start in |0_a 0_b>, the measurement leaves the Bell pair
with Z_a Z_b = +1, and an X fault on the bridge flips Z_a Z_b.  With w = 1 a
single fault on the bridge edge (including its readout error) flips the
observable without triggering any detector.  With w >= 2 such a fault flips a
cycle check and is detected.

Blocks.  Each block is a three-qubit code with Z checks Z0Z1, Z1Z2, X logical
X0X1X2 (a membrane of weight three, the stand-in for a slab port) and Z
logical Z0.  The deformed X distance is exactly w for w = 1, 2, 3.

Graph.  One vertex per data qubit; edges (a0,a1), (a1,a2), (b0,b1), (b1,b2);
bridges (a_i,b_i) for i < w.  An ancilla qubit sits on every edge, prepared in
|0>.  Each vertex star A_v = X(data_v) X(edges at v) is measured once with its
own ancilla, which is read out only at the end, so the circuit needs no
mid-circuit measurement.  At the end every data and edge qubit is measured in
Z.  Detectors: the deformed Z checks Z_x Z_y Z(e_xy) and the cycle checks of
the bridge squares.  Observable: Z_a0 Z_b0 Z(e_a0b0), the deformed Z_a Z_b.

A control circuit measures the same data qubits without any gauging.  All
circuits use the same physical qubits.

Usage:
  python cut_obstruction_hw.py check            # noiseless and noisy simulation
  python cut_obstruction_hw.py run --shots N    # submit one job to ibm_fez
  python cut_obstruction_hw.py analyze JOB_ID   # fetch, decode, report

The IBM key is read from the file named by the environment variable
IBM_TOKEN_FILE and is never printed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

from qiskit import QuantumCircuit, transpile

HERE = Path(__file__).resolve().parent
BACKEND = "ibm_fez"

# ------------------------------------------------------------------ qubits
DATA = ["a0", "a1", "a2", "b0", "b1", "b2"]
EDGES = {"ea01": ("a0", "a1"), "ea12": ("a1", "a2"), "eb01": ("b0", "b1"),
         "eb12": ("b1", "b2"), "br0": ("a0", "b0"), "br1": ("a1", "b1"),
         "br2": ("a2", "b2")}
ANC = [f"x_{v}" for v in DATA]
QUBITS = DATA + list(EDGES) + ANC          # 19 virtual qubits, same for all
IDX = {q: i for i, q in enumerate(QUBITS)}


def edges_for(w):
    return ["ea01", "ea12", "eb01", "eb12"] + [f"br{i}" for i in range(w)]


def build(w):
    """Circuit for bridge width w (w = 0 is the control without gauging)."""
    qc = QuantumCircuit(len(QUBITS), len(QUBITS), name=f"w{w}")
    if w > 0:
        es = edges_for(w)
        for v in DATA:
            a = IDX[f"x_{v}"]
            # An X fault on the ancilla between two CNOTs spreads X to the
            # remaining targets; modulo A_v this is X on a subset of the star.
            inc = [e for e in es if v in EDGES[e]]
            qc.h(a)
            qc.cx(a, IDX[v])
            for e in inc:
                qc.cx(a, IDX[e])
            qc.h(a)
    qc.barrier()
    qc.measure(range(len(QUBITS)), range(len(QUBITS)))
    return qc


def detectors(w):
    """Lists of qubit names whose Z outcomes must have even parity."""
    if w == 0:
        return [["a0", "a1"], ["a1", "a2"], ["b0", "b1"], ["b1", "b2"]]
    dets = [["a0", "a1", "ea01"], ["a1", "a2", "ea12"],
            ["b0", "b1", "eb01"], ["b1", "b2", "eb12"]]
    if w >= 2:
        dets.append(["ea01", "br1", "eb01", "br0"])
    if w >= 3:
        dets.append(["ea12", "br2", "eb12", "br1"])
    return dets


def observable(w):
    return ["a0", "b0"] if w == 0 else ["a0", "b0", "br0"]


def faults(w):
    """Single X flips (or readout flips) of every measured data/edge qubit."""
    qs = DATA + ([] if w == 0 else edges_for(w))
    return qs


# ---------------------------------------------------------------- decoding
def bit(shot, q):
    # qiskit bitstrings are little-endian: clbit i is the i-th character from the right
    return int(shot[-1 - IDX[q]])


def syndrome_of(flips, dets):
    return tuple(sum(q in d for q in flips) % 2 for d in dets)


def decoder_table(w, max_weight=3):
    """Minimum-weight correction for every syndrome (ties: first in order)."""
    dets, obs, fl = detectors(w), observable(w), faults(w)
    table = {}
    for k in range(max_weight + 1):
        for sub in itertools.combinations(fl, k):
            s = syndrome_of(sub, dets)
            if s not in table:
                table[s] = sum(q in obs for q in sub) % 2
    return table


def analyze_counts(w, counts):
    dets, obs = detectors(w), observable(w)
    table = decoder_table(w)
    n = acc = acc_fail = dec_fail = 0
    for shot, c in counts.items():
        shot = shot.replace(" ", "")
        s = tuple(sum(bit(shot, q) for q in d) % 2 for d in dets)
        o = sum(bit(shot, q) for q in obs) % 2
        n += c
        if not any(s):
            acc += c
            acc_fail += c * o
        corr = table.get(s)
        dec_fail += c * (1 if corr is None else (o ^ corr))
    return dict(w=w, shots=n, accepted=acc, accept_rate=acc / n,
                postselected_failures=acc_fail,
                postselected_error=acc_fail / max(acc, 1),
                postselected_error_wilson95=wilson(acc_fail, acc),
                decoded_failures=dec_fail, decoded_error=dec_fail / n,
                decoded_error_wilson95=wilson(dec_fail, n))


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


# ------------------------------------------------------------- IBM access
def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = Path(os.environ["IBM_TOKEN_FILE"]).read_text().strip()
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)


def transpiled(backend, widths=(0, 1, 2, 3), seed=7):
    """Transpile w=3 first, then reuse its physical qubits for every width."""
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)
    big = pm.run(build(3))
    init = list(big.layout.initial_index_layout(filter_ancillas=True))[:len(QUBITS)]
    pm2 = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                       initial_layout=init, seed_transpiler=seed)
    out = {w: (big if w == 3 else pm2.run(build(w))) for w in widths}
    return out, init


def summary(circ):
    ops = circ.count_ops()
    return dict(depth=circ.depth(), cz=int(ops.get("cz", 0)),
                ops={k: int(v) for k, v in ops.items()})


# ------------------------------------------------------------------ modes
def mode_check(shots):
    from qiskit_aer import AerSimulator
    ideal = AerSimulator(method="stabilizer")
    report = {}
    for w in (0, 1, 2, 3):
        res = ideal.run(transpile(build(w), ideal), shots=shots).result().get_counts()
        a = analyze_counts(w, res)
        assert a["accepted"] == a["shots"] and a["postselected_failures"] == 0, a
        report[f"ideal_w{w}"] = a
        print(f"noiseless w={w}: all detectors silent, observable correct in {a['shots']} shots")
    # weight of the lightest undetected logical, from the fault list
    for w in (1, 2, 3):
        dets, obs = detectors(w), observable(w)
        best = None
        for k in range(1, 5):
            for sub in itertools.combinations(faults(w), k):
                if not any(syndrome_of(sub, dets)) and sum(q in obs for q in sub) % 2:
                    best = k
                    break
            if best:
                break
        report[f"lightest_undetected_logical_w{w}"] = best
        print(f"w={w}: lightest undetected logical among data/edge X faults has weight {best}")
    return report


def mode_noisy(shots):
    """Simulate the transpiled circuits with the backend's own noise model."""
    from qiskit_aer import AerSimulator
    svc = service()
    be = svc.backend(BACKEND)
    circs, init = transpiled(be)
    sim = AerSimulator.from_backend(be)
    rep = {"physical_qubits": init}
    for w, c in circs.items():
        counts = sim.run(c, shots=shots).result().get_counts()
        a = analyze_counts(w, counts)
        a.update(summary(c))
        rep[f"noisy_w{w}"] = a
        print(f"noisy sim w={w}: accept {a['accept_rate']:.3f}, postselected error "
              f"{a['postselected_error']:.4f}, decoded error {a['decoded_error']:.4f}, "
              f"depth {a['depth']}, cz {a['cz']}")
    return rep


def mode_plan():
    """Transpile for the backend and report the circuits, without submitting."""
    svc = service()
    be = svc.backend(BACKEND)
    circs, init = transpiled(be)
    rep = dict(backend=BACKEND, pending_jobs=be.status().pending_jobs,
               physical_qubits=dict(zip(QUBITS, init)),
               circuits={w: summary(c) for w, c in circs.items()})
    print(json.dumps(rep, indent=1))
    return rep


def mode_run(shots):
    from qiskit_ibm_runtime import SamplerV2
    svc = service()
    be = svc.backend(BACKEND)
    circs, init = transpiled(be)
    sampler = SamplerV2(mode=be)
    sampler.options.default_shots = shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    job = sampler.run([circs[w] for w in (0, 1, 2, 3)])
    meta = dict(job_id=job.job_id(), backend=BACKEND, shots=shots, widths=[0, 1, 2, 3],
                physical_qubits=init, qubit_names=QUBITS,
                circuits={w: summary(c) for w, c in circs.items()})
    (HERE / "hw_job.json").write_text(json.dumps(meta, indent=1))
    print("submitted job", job.job_id())
    return meta


def mode_analyze(job_id):
    svc = service()
    job = svc.job(job_id)
    res = job.result()
    meta = json.loads((HERE / "hw_job.json").read_text())
    out = dict(job_id=job_id, backend=meta["backend"], physical_qubits=meta["physical_qubits"])
    try:
        out["usage_seconds"] = job.usage()
    except Exception:
        out["usage_seconds"] = None
    raw = {}
    for w, pub in zip(meta["widths"], res):
        counts = pub.data.c.get_counts()
        raw[str(w)] = counts
        out[f"w{w}"] = analyze_counts(w, counts)
    (HERE / "hw_counts.json").write_text(json.dumps(raw))
    (HERE / "hw_results.json").write_text(json.dumps(out, indent=1))
    for w in meta["widths"]:
        a = out[f"w{w}"]
        print(f"hardware w={w}: shots {a['shots']}, accept {a['accept_rate']:.3f}, "
              f"postselected error {a['postselected_error']:.5f} {a['postselected_error_wilson95']}, "
              f"decoded error {a['decoded_error']:.5f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "noisy", "plan", "run", "analyze"])
    ap.add_argument("job", nargs="?")
    ap.add_argument("--shots", type=int, default=4000)
    a = ap.parse_args()
    if a.mode == "check":
        r = mode_check(a.shots)
        (HERE / "sim_check.json").write_text(json.dumps(r, indent=1))
    elif a.mode == "noisy":
        r = mode_noisy(a.shots)
        (HERE / "sim_noisy.json").write_text(json.dumps(r, indent=1))
    elif a.mode == "plan":
        (HERE / "plan.json").write_text(json.dumps(mode_plan(), indent=1))
    elif a.mode == "run":
        mode_run(a.shots)
    else:
        mode_analyze(a.job)


if __name__ == "__main__":
    main()
