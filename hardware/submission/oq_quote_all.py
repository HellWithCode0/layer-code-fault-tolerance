"""Free quotes only: prepares jobs, prints prices, creates NO job. Checks credits before and after."""
import sys
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient
from openquantum_sdk.models import JobPreparationCreate
from openquantum_sdk.utils import poll_for_status
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
auth = ClientCredentials(client_id=(here / "oq_client_id.txt").read_text().strip(),
                         client_secret=(here / "oq_key.txt").read_text().strip())
auth = ClientCredentialsAuth(creds=auth)
mg, sc = ManagementClient(auth=auth), SchedulerClient(auth=auth)
CASES = [("ionq:forte-1", "qasm_compact_w3.qasm", 1000), ("ionq:forte-1", "qasm_compact_w3.qasm", 5000),
         ("ionq:forte-1", "qasm_compact_w1.qasm", 1000),
         ("ionq:forte-ent", "qasm_compact_w3.qasm", 1000),
         ("iqm:emerald", "qasm_compact_w3.qasm", 10000), ("iqm:garnet", "qasm_compact_w3.qasm", 10000)]
try:
    orgs = mg.list_user_organizations().organizations
    org = next(o for o in orgs if o.name == "org_B")
    print("quoting in org", org.name, "credits before:", mg.get_credit_balance(org.id).spark_credits)
    for be, f, shots in CASES:
        try:
            up = sc.upload_job_input(file_content=(hw / f).read_bytes())
            prep = sc.prepare_job(JobPreparationCreate(
                organization_id=org.id, backend_class_id=be, name=f"quote {be} {f}",
                upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=shots, configuration_data={}))
            res = poll_for_status(
                get_status_fn=lambda pid: ((r := sc.get_preparation_result(pid)).status in ("Completed", "Failed"), r),
                resource_id=prep.id, interval=2.0, timeout=300)
            if res.status != "Completed":
                print(f"{be:15s} {f:22s} {shots:6d} shots: {res.status} {(res.message or '')[:200]!r}")
                continue
            plans = "; ".join(f"{p.name} {p.price}" + "".join(f" ({q.name} +{q.price_increase})" for q in p.queue_priorities)
                              for p in res.quote)
            print(f"{be:15s} {f:22s} {shots:6d} shots: {plans}")
        except Exception as e:
            print(f"{be} {f} {shots}: error {type(e).__name__}: {str(e)[:200]}")
    print("credits after:", mg.get_credit_balance(org.id).spark_credits)
finally:
    mg.close()
    sc.close()
