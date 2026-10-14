"""Read-only: save the constraint data of every listed backend."""
import json
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import SchedulerClient
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
auth = ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / "oq_client_id.txt").read_text().strip(),
    client_secret=(here / "oq_key.txt").read_text().strip()))
sc = SchedulerClient(auth=auth)
try:
    for code in ("iqm:garnet", "iqm:emerald", "ionq:forte-1", "ionq:forte-ent"):
        b = sc.get_backend_class(code)
        (hw / f"oq_backend_{code.replace(':', '_')}.json").write_text(json.dumps(b, indent=1))
        cd = b["constraint_data"]
        cm = cd.get("topology", {}).get("coupling_map", [])
        print(code, "| qubits", cd.get("n_qubits"), "| couplers", len(cm), "| native", [g["name"] for g in cd.get("native_ops", [])],
              "| features", cd.get("features"), "| limits", cd.get("limits"),
              "| noise", json.dumps(cd.get("noise"))[:200], "| qubit_props", len(cd.get("qubit_properties") or []),
              "| config", list(b.get("configuration_schema", {}).get("properties", {})))
finally:
    sc.close()
