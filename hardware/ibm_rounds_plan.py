"""Plan, compile and verify the IBM repeated-rounds job (ibm_rounds_hw.py) on ibm_kingston.

1. Layout: every 13-qubit simple path of the coupling graph is scored with the
   live calibration (CZ errors of its 12 couplers, readout errors of its six
   ancillas, dephasing of its seven other qubits during a measure-and-reset
   window).  Copy A is the best path; copy B the best path that shares no qubit
   and no coupler with A.
2. Compile: preset pass manager (level 1) with that initial layout, then ALAP
   scheduling and XX dynamical decoupling on idle windows.  The compiled
   circuits are saved (QPY) and are exactly what the submit script sends.
3. Verify the compiled circuits: no SWAP (CZ count = 12 per round per copy),
   physical qubits as planned, noiseless Aer run with every detector silent and
   the reported bit correct, and exact circuit fault distance (CP-SAT on a
   Stim model of the compiled circuit, every operation and idle window noisy):
   expected 1, 2, 3, 3 for R = 1, 2, 3, 4.
4. Predict rates from the calibration (Stim, per-gate errors, readout flips,
   T1/T2 idle channels) and estimate the QPU time per shot.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import stim
from qiskit import qpy
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scaling_campaign" / "campaign"))
import ibm_rounds_hw as rh
from min_weight_logical import min_weight_logical

BACKEND = "ibm_kingston"
ROUNDS = (1, 2, 3, 4)
CAL = json.loads((HERE / f"ibm2_calibration_{BACKEND}.json").read_text())


def cz_err(a, b):
    return CAL["cz"].get(f"{a},{b}", CAL["cz"].get(f"{b},{a}"))


def paths13():
    adj = {}
    for a, b in CAL["coupling"]:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    out = []

    def dfs(p):
        if len(p) == rh.NQ:
            if p[0] < p[-1]:                      # each path once
                out.append(list(p))
            return
        for nb in adj.get(p[-1], ()):
            if nb not in p:
                p.append(nb)
                dfs(p)
                p.pop()
    for s in adj:
        dfs([s])
    return out, adj


def score(p, window):
    s = 0.0
    for a, b in zip(p, p[1:]):
        e = cz_err(a, b)
        if e is None or e >= 0.05:
            return None
        s += e
    for i, phys in enumerate(p):
        if i % 2:                                  # ancilla
            r = CAL["readout"].get(str(phys))
            if r is None or r >= 0.1:
                return None
            s += r
        else:
            t2 = CAL["t2"].get(str(phys))
            if not t2:
                return None
            s += window / t2
    return s


def choose_layout():
    window = CAL["measure_duration"] + (CAL["reset_duration"] or CAL["measure_duration"])
    ps, adj = paths13()
    scored = sorted((sc, p) for p in ps if (sc := score(p, window)) is not None)
    best_a = scored[0]
    A = set(best_a[1])
    for sc, p in scored[1:]:
        if A & set(p) or any(nb in A for x in p for nb in adj[x]):
            continue
        return [best_a, (sc, p)], len(ps)
    raise RuntimeError("no second disjoint path")


def to_stim(qc, p=None, cal=None, dt=None):
    """Compiled circuit -> Stim.  With p: uniform noise p on every operation and
    idle window (for fault distance).  With cal: calibrated noise."""
    c = stim.Circuit()
    nq = qc.num_qubits
    c.append("R", range(nq))
    rec = {}
    nmeas = 0
    for ins in qc.data:
        name = ins.operation.name
        qs = [qc.find_bit(x).index for x in ins.qubits]
        if name == "barrier":
            continue
        if name == "rz":
            k = round(float(ins.operation.params[0]) / (math.pi / 2)) % 4
            assert abs(float(ins.operation.params[0]) - k * math.pi / 2) < 1e-6 or \
                abs(abs(float(ins.operation.params[0])) - abs(round(float(ins.operation.params[0]) / (math.pi / 2)) * math.pi / 2)) < 1e-6
            g = {1: "S", 2: "Z", 3: "S_DAG"}.get(k)
            if g:
                c.append(g, qs)
        elif name in ("sx", "x"):
            c.append("SQRT_X" if name == "sx" else "X", qs)
            pe = p if p is not None else CAL["sx"].get(str(qs[0]), 3e-4) if cal else 0
            if pe:
                c.append("DEPOLARIZE1", qs, pe)
        elif name == "cz":
            c.append("CZ", qs)
            pe = p if p is not None else (cz_err(*qs) or 3e-3) if cal else 0
            if pe:
                c.append("DEPOLARIZE2", qs, pe)
        elif name == "measure":
            pe = p if p is not None else CAL["readout"].get(str(qs[0]), 0.015) if cal else 0
            c.append("M", qs, pe) if pe else c.append("M", qs)
            rec[qc.find_bit(ins.clbits[0]).index] = nmeas
            nmeas += 1
        elif name == "reset":
            c.append("R", qs)
            pe = p if p is not None else CAL["readout"].get(str(qs[0]), 0.015) if cal else 0
            if pe:
                c.append("X_ERROR", qs, pe)
        elif name == "delay":
            if p is not None:
                c.append("DEPOLARIZE1", qs, p)
            elif cal:
                t = ins.operation.duration * dt if ins.operation.unit == "dt" else float(ins.operation.duration)
                t1 = CAL["t1"].get(str(qs[0]), 150e-6)
                t2 = min(CAL["t2"].get(str(qs[0]), 100e-6), 2 * t1)
                px = (1 - math.exp(-t / t1)) / 4
                pz = max(0.0, (1 - math.exp(-t / t2)) / 2 - px)
                c.append("PAULI_CHANNEL_1", qs, [px, px, pz])
        else:
            raise ValueError(f"unexpected operation {name}")
    return c, rec, nmeas


def add_dets(c, rec, nmeas, R, k):
    c = c.copy()
    r_ = lambda cb: stim.target_rec(rec[cb] - nmeas)
    for d in rh.detectors(R, k):
        c.append("DETECTOR", [r_(x) for x in d])
    c.append("OBSERVABLE_INCLUDE", [r_(x) for x in rh.observable(R, k)], 0)
    return c


def main():
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_aer import AerSimulator
    svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                               token=Path(os.environ["IBM_TOKEN_FILE"]).read_text().strip())
    be = svc.backend(BACKEND)
    dt = be.target.dt
    (sa, A), (sb, B), = choose_layout()[0]
    layout = A + B
    print(f"layout from {BACKEND} calibration: copy A {A} (score {sa:.4f}), copy B {B} (score {sb:.4f})")
    pm = generate_preset_pass_manager(optimization_level=1, backend=be, initial_layout=layout, seed_transpiler=11)
    pm.scheduling = PassManager([ALAPScheduleAnalysis(target=be.target),
                                 PadDynamicalDecoupling(target=be.target, dd_sequence=[XGate(), XGate()])])
    plan = dict(backend=BACKEND, layout=dict(A=A, B=B), rounds={})
    compiled = []
    for R in ROUNDS:
        qc = pm.run(rh.build(R))
        compiled.append(qc)
        ops = qc.count_ops()
        cz = int(ops.get("cz", 0))
        # physical qubit of each virtual qubit
        fl = qc.layout.final_index_layout(filter_ancillas=True)
        layout_ok = list(fl) == layout
        # noiseless Aer on the compiled circuit
        res = AerSimulator(method="stabilizer").run(qc, shots=2000, memory=True).result()
        bits = np.array([[int(s[-1 - i]) for i in range(qc.num_clbits)] for s in res.get_memory()], dtype=np.uint8)
        noiseless_ok = all(rh.analyze_bits(bits, R, k)["accept_rate"] == 1.0 and
                           rh.analyze_bits(bits, R, k)["postselected_failures"] == 0 for k in (0, 1))
        # exact fault distance per copy
        cu, rec, nm = to_stim(qc, p=1e-3)
        dist = [min_weight_logical(add_dets(cu, rec, nm, R, k).detector_error_model(decompose_errors=False),
                                   0, time_limit=120)["distance"] for k in (0, 1)]
        # calibrated prediction
        cc, rec, nm = to_stim(qc, cal=True, dt=dt)
        s = cc.compile_sampler().sample(200000).astype(np.uint8)
        bits_c = np.zeros((s.shape[0], qc.num_clbits), dtype=np.uint8)
        for cb, m in rec.items():
            bits_c[:, cb] = s[:, m]
        pred = [rh.analyze_bits(bits_c, R, k) for k in (0, 1)]
        dur = qc.estimate_duration(be.target, unit="s")
        print(f"R={R}: CZ {cz} (expected {12 * R * 2}) | layout as planned {layout_ok} | noiseless correct "
              f"{noiseless_ok} | fault distance per copy {dist} (expected {min(R, 3)}) | duration {1e6 * dur:.1f} us")
        for k in (0, 1):
            print(f"   predicted copy {'AB'[k]}: accept {pred[k]['accept_rate']:.3f}, reported-bit error "
                  f"{100 * pred[k]['postselected_error']:.3f}%")
        assert cz == 24 * R and layout_ok and noiseless_ok and all(d == min(R, 3) for d in dist)
        plan["rounds"][R] = dict(cz=cz, depth=qc.depth(), duration_s=dur, fault_distance=dist,
                                 predicted=[dict(accept=x["accept_rate"], error=x["postselected_error"]) for x in pred])
    with open(HERE / "ibm2_rounds_circuits.qpy", "wb") as f:
        qpy.dump(compiled, f)
    rep = float(CAL["rep_delay_us"] or 250e-6)
    rep = rep if rep < 1 else rep * 1e-6
    plan["rep_delay_s"] = rep
    (HERE / "ibm2_rounds_plan.json").write_text(json.dumps(plan, indent=1))
    print("compiled circuits saved to ibm2_rounds_circuits.qpy; all checks passed")


if __name__ == "__main__":
    main()
