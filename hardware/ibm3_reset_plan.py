"""Plan, compile and verify the IBM reset follow-up (third IBM key, ibm_kingston).

Question: in the first IBM rounds run, ancilla errors persisted across rounds
far more often than independent errors allow, which weakened the gain from
repeated rounds.  If failed resets cause this, a second reset between rounds
should cut the persistence and the reported-bit error.  If leakage causes it,
a second reset (measure and conditional X) cannot help.

Circuits, all in one job on the same qubits: R = 1 (no reset), and R = 2, 3
each with one reset and with two resets between rounds.  Layout, compilation
and checks as in ibm_rounds_plan.py, with today's calibration: no SWAP,
planned physical qubits, noiseless Aer correct, exact fault distance R (CP-SAT
on the compiled circuit), ISA validation, timing rules.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scaling_campaign" / "campaign"))
import ibm_rounds_hw as rh
import ibm_rounds_plan as rp
from min_weight_logical import min_weight_logical

BACKEND = "ibm_kingston"
rp.CAL = json.loads((HERE / f"ibm3_calibration_{BACKEND}.json").read_text())   # today's calibration
CIRCUITS = [("R1", 1, 1), ("R2_single", 2, 1), ("R3_single", 3, 1), ("R2_double", 2, 2), ("R3_double", 3, 2)]


def main():
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.utils.validations import validate_isa_circuits
    from qiskit_aer import AerSimulator
    svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                               token=Path(os.environ["IBM_TOKEN_FILE"]).read_text().strip())
    be = svc.backend(BACKEND)
    t, dt = be.target, be.target.dt
    (sa, A), (sb, B) = rp.choose_layout()[0]
    layout = A + B
    run1 = json.loads((HERE / "ibm2_rounds_plan.json").read_text())["layout"]
    print(f"layout from today's calibration: A {A} ({sa:.4f}), B {B} ({sb:.4f}); "
          f"first run used A {run1['A']}, B {run1['B']}")
    pm = generate_preset_pass_manager(optimization_level=1, backend=be, initial_layout=layout, seed_transpiler=11)
    pm.scheduling = PassManager([ALAPScheduleAnalysis(target=t),
                                 PadDynamicalDecoupling(target=t, dd_sequence=[XGate(), XGate()])])
    plan = dict(backend=BACKEND, layout=dict(A=A, B=B), circuits={})
    compiled = []
    for label, R, resets in CIRCUITS:
        qc = pm.run(rh.build(R, resets=resets))
        compiled.append(qc)
        ops = qc.count_ops()
        cz, nreset = int(ops.get("cz", 0)), int(ops.get("reset", 0))
        layout_ok = list(qc.layout.final_index_layout(filter_ancillas=True)) == layout
        res = AerSimulator(method="stabilizer").run(qc, shots=2000, memory=True).result()
        bits = np.array([[int(s[-1 - i]) for i in range(qc.num_clbits)] for s in res.get_memory()], dtype=np.uint8)
        noiseless_ok = all(rh.analyze_bits(bits, R, k)["accept_rate"] == 1.0 and
                           rh.analyze_bits(bits, R, k)["postselected_failures"] == 0 for k in (0, 1))
        cu, rec, nm = rp.to_stim(qc, p=1e-3)
        dist = [min_weight_logical(rp.add_dets(cu, rec, nm, R, k).detector_error_model(decompose_errors=False),
                                   0, time_limit=120)["distance"] for k in (0, 1)]
        cc, rec, nm = rp.to_stim(qc, cal=True, dt=dt)
        s = cc.compile_sampler().sample(200000).astype(np.uint8)
        bc = np.zeros((s.shape[0], qc.num_clbits), dtype=np.uint8)
        for cb, m in rec.items():
            bc[:, cb] = s[:, m]
        pred = [rh.analyze_bits(bc, R, k) for k in (0, 1)]
        delays = [ins.operation for ins in qc.data if ins.operation.name == "delay"]
        timing_ok = all(op.unit == "dt" and op.duration % t.granularity == 0 and op.duration >= t.min_length
                        for op in delays)
        dur = qc.estimate_duration(t, unit="s")
        print(f"{label}: CZ {cz} (expected {24 * R}) | resets {nreset} (expected {12 * (R - 1) * resets}) | "
              f"layout {layout_ok} | noiseless {noiseless_ok} | fault distance {dist} (expected {R}) | "
              f"timing {timing_ok} | {1e6 * dur:.1f} us | predicted error A {100 * pred[0]['postselected_error']:.3f}% "
              f"B {100 * pred[1]['postselected_error']:.3f}%")
        assert cz == 24 * R and nreset == 12 * (R - 1) * resets and layout_ok and noiseless_ok
        assert all(d == R for d in dist) and timing_ok
        plan["circuits"][label] = dict(R=R, resets=resets, duration_s=dur, fault_distance=dist,
                                       predicted=[x["postselected_error"] for x in pred])
    validate_isa_circuits(compiled, t)
    print("ISA validation passed")
    with open(HERE / "ibm3_reset_circuits.qpy", "wb") as f:
        qpy.dump(compiled, f)
    rep = float(rp.CAL["rep_delay_us"] or 250e-6)
    plan["rep_delay_s"] = rep if rep < 1 else rep * 1e-6
    plan["labels"] = [c[0] for c in CIRCUITS]
    (HERE / "ibm3_reset_plan.json").write_text(json.dumps(plan, indent=1))
    print("saved ibm3_reset_circuits.qpy and ibm3_reset_plan.json; all checks passed")


if __name__ == "__main__":
    main()
