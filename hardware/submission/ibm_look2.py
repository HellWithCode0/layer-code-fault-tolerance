"""Read-only look at the second IBM account: backends, queues, dynamic-circuit support, usage."""
import os
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService
tok = Path(os.environ["IBM_TOKEN_FILE"]).read_text().strip()
svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)
try:
    print("usage:", svc.usage())
except Exception as e:
    print("usage: unavailable", type(e).__name__, str(e)[:200])
for b in svc.backends(operational=True, simulator=False):
    st = b.status()
    cfg = b.configuration()
    ops = set(b.target.operation_names)
    print(f"{b.name:16s} qubits {b.num_qubits:3d} pending {st.pending_jobs:5d} "
          f"processor {getattr(cfg, 'processor_type', {})} reset {'reset' in ops} "
          f"if_else {'if_else' in ops} measure {'measure' in ops} basis {sorted(ops & {'cz','ecr','sx','rz','x'})}")
