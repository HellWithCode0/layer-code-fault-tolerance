"""Free: quotes the Emerald circuits at several shot counts and tests whether
validation rejects a CZ between unconnected qubits (0 and 2) with rewiring off.
Creates NO job."""
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient
from openquantum_sdk.models import JobPreparationCreate
from openquantum_sdk.utils import poll_for_status
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
auth = ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / "oq_client_id.txt").read_text().strip(),
    client_secret=(here / "oq_key.txt").read_text().strip()))
mg, sc = ManagementClient(auth=auth), SchedulerClient(auth=auth)
CASES = [("qasm3_iqm_invalid_probe.qasm", 100), ("qasm3_iqm_w3.qasm", 10000), ("qasm3_iqm_w3.qasm", 5000),
         ("qasm3_iqm_w3.qasm", 3000), ("qasm3_iqm_w3.qasm", 1000), ("qasm3_iqm_w1.qasm", 5000)]
try:
    org = next(o for o in mg.list_user_organizations().organizations if o.name == "org_B")
    print("credits before:", mg.get_credit_balance(org.id).spark_credits)
    for f, shots in CASES:
        up = sc.upload_job_input(file_content=(hw / f).read_bytes())
        prep = sc.prepare_job(JobPreparationCreate(
            organization_id=org.id, backend_class_id="iqm:emerald", name=f"quote {f}",
            upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=shots,
            configuration_data={"disable_qubit_rewiring": True}))
        res = poll_for_status(
            get_status_fn=lambda pid: ((r := sc.get_preparation_result(pid)).status in ("Completed", "Failed"), r),
            resource_id=prep.id, interval=2.0, timeout=300)
        pub = next((p.price for p in res.quote if p.name.startswith("Public")), None) if res.status == "Completed" else None
        print(f"{f:30s} {shots:6d} shots: {res.status} public plan {pub} credits {(res.message or '')[:300]!r}")
    print("credits after:", mg.get_credit_balance(org.id).spark_credits)
finally:
    mg.close()
    sc.close()
