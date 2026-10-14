"""Connect to IBM Quantum, report backends, ibm_fez status and usage. Never prints the key."""
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService

tok = Path(__file__).with_name("ibm_token.txt").read_text().strip()
svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)
for b in svc.backends():
    st = b.status()
    print(f"{b.name:14s} qubits={b.num_qubits:4d} operational={st.operational} pending={st.pending_jobs}")
try:
    print("usage:", svc.usage())
except Exception as e:
    print("usage: unavailable:", type(e).__name__, str(e)[:200])
fez = svc.backend("ibm_fez")
print("fez basis:", fez.operation_names)
print("instance:", getattr(svc, "active_account", lambda: {})().get("instance", "?")[:12] + "...")
