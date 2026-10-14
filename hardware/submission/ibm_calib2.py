"""Read-only: calibration medians of the Heron backends, saved for layout choice."""
import json, os, statistics as st
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService
tok = Path(os.environ["IBM_TOKEN_FILE"]).read_text().strip()
svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)
hw = Path(r"C:\kw\r\hardware")
for name in ("ibm_marrakesh", "ibm_kingston", "ibm_fez"):
    b = svc.backend(name)
    t = b.target
    cz = [p.error for q, p in t["cz"].items() if p is not None and p.error is not None]
    ro = [p.error for q, p in t["measure"].items() if p is not None and p.error is not None]
    mdur = [p.duration for q, p in t["measure"].items() if p is not None and p.duration]
    rdur = [p.duration for q, p in t["reset"].items() if p is not None and p.duration] if "reset" in t.operation_names else []
    t2 = [t.qubit_properties[i].t2 for i in range(b.num_qubits) if t.qubit_properties[i] and t.qubit_properties[i].t2]
    t1 = [t.qubit_properties[i].t1 for i in range(b.num_qubits) if t.qubit_properties[i] and t.qubit_properties[i].t1]
    cfg = b.configuration()
    print(f"{name}: pending {b.status().pending_jobs} | median CZ error {st.median(cz):.4f} | median readout error "
          f"{st.median(ro):.4f} | median T1 {1e6*st.median(t1):.0f} us T2 {1e6*st.median(t2):.0f} us | measure "
          f"{1e6*st.median(mdur):.2f} us reset {1e6*st.median(rdur) if rdur else float('nan'):.2f} us | "
          f"default rep_delay {getattr(cfg, 'default_rep_delay', None)} us")
    props = dict(
        cz={f"{q[0]},{q[1]}": p.error for q, p in t["cz"].items() if p is not None and p.error is not None},
        readout={str(q[0]): p.error for q, p in t["measure"].items() if p is not None and p.error is not None},
        t1={str(i): t.qubit_properties[i].t1 for i in range(b.num_qubits) if t.qubit_properties[i] and t.qubit_properties[i].t1},
        t2={str(i): t.qubit_properties[i].t2 for i in range(b.num_qubits) if t.qubit_properties[i] and t.qubit_properties[i].t2},
        sx={str(q[0]): p.error for q, p in t["sx"].items() if p is not None and p.error is not None},
        measure_duration=st.median(mdur), reset_duration=st.median(rdur) if rdur else None,
        rep_delay_us=getattr(cfg, "default_rep_delay", None),
        coupling=[list(e) for e in t.build_coupling_map().get_edges()])
    (hw / f"ibm2_calibration_{name}.json").write_text(json.dumps(props))
