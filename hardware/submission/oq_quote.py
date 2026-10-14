"""Ask every accepting Open Quantum backend for a quote for the w=3 circuit.
Prepares jobs only (quotes); never calls create_job, so no credits are spent."""
import sys
from pathlib import Path

from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient
from openquantum_sdk.models import JobPreparationCreate
from openquantum_sdk.utils import poll_for_status

here = Path(__file__).parent
auth = ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / "oq_client_id.txt").read_text().strip(),
    client_secret=(here / "oq_key.txt").read_text().strip()))
mg, sc = ManagementClient(auth=auth), SchedulerClient(auth=auth)
qasm = Path(r"C:\kw\r\hardware\qasm_w3.qasm").read_bytes()
shots_list = [int(s) for s in sys.argv[1:]] or [1000]
try:
    org = mg.list_user_organizations().organizations[0]
    for be in ["ionq:forte-1", "iqm:garnet", "iqm:emerald", "rigetti:cepheus-1-108q"]:
        for shots in shots_list:
            try:
                up = sc.upload_job_input(file_content=qasm)
                prep = sc.prepare_job(JobPreparationCreate(
                    organization_id=org.id, backend_class_id=be, name=f"quote {be} {shots}",
                    upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=shots,
                    configuration_data={}))
                res = poll_for_status(
                    get_status_fn=lambda pid: ((r := sc.get_preparation_result(pid)).status in ("Completed", "Failed"), r),
                    resource_id=prep.id, interval=2.0, timeout=180)
                if res.status == "Failed":
                    print(f"{be:24s} shots={shots:6d}  preparation failed: {res.message}")
                    continue
                plans = "; ".join(
                    f"{p.name}: {p.price}" + "".join(f" [{q.name} +{q.price_increase}]" for q in p.queue_priorities)
                    for p in res.quote)
                print(f"{be:24s} shots={shots:6d}  {plans}")
            except Exception as e:
                print(f"{be:24s} shots={shots:6d}  error: {type(e).__name__}: {str(e)[:200]}")
finally:
    mg.close()
    sc.close()
