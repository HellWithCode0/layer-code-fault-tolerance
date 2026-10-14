"""Submit the four cut-obstruction circuits to Rigetti Cepheus via Open Quantum.

Approved by the user: 4 jobs x 10,000 shots, public plan, standard queue,
at most 3 credits each (12 total).  Each job is re-quoted first; if any quote
exceeds 3 credits the script stops before creating that job."""
import json
from pathlib import Path

from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient
from openquantum_sdk.models import JobCreate, JobPreparationCreate
from openquantum_sdk.utils import poll_for_status

BACKEND, SHOTS, MAX_CREDITS = "rigetti:cepheus-1-108q", 10000, 3
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
auth = ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / "oq_client_id.txt").read_text().strip(),
    client_secret=(here / "oq_key.txt").read_text().strip()))
mg, sc = ManagementClient(auth=auth), SchedulerClient(auth=auth)
prev = json.loads((hw / "oq_jobs.json").read_text()) if (hw / "oq_jobs.json").exists() else {}
jobs = {int(k): v for k, v in prev.get("jobs", {}).items()}
try:
    org = mg.list_user_organizations().organizations[0]
    print("credits before:", mg.get_credit_balance(org.id))
    for w in (1, 2, 3):
        up = sc.upload_job_input(file_content=(hw / f"qasm_compact_w{w}.qasm").read_bytes())
        prep = sc.prepare_job(JobPreparationCreate(
            organization_id=org.id, backend_class_id=BACKEND, name=f"cut obstruction w={w}",
            upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=SHOTS,
            configuration_data={}))
        res = poll_for_status(
            get_status_fn=lambda pid: ((r := sc.get_preparation_result(pid)).status in ("Completed", "Failed"), r),
            resource_id=prep.id, interval=2.0, timeout=300)
        if res.status == "Failed":
            raise RuntimeError(f"preparation failed for w={w}: {res.message}")
        plan = next(p for p in res.quote if p.name.startswith("Public"))
        prio = next(q for q in plan.queue_priorities if q.name.startswith("Standard"))
        cost = plan.price + prio.price_increase
        if cost > MAX_CREDITS:
            raise RuntimeError(f"w={w} quote {cost} credits exceeds the approved {MAX_CREDITS}; stopping")
        job = sc.create_job(JobCreate(job_preparation_id=prep.id,
                                      execution_plan_id=plan.execution_plan_id,
                                      queue_priority_id=prio.queue_priority_id))
        jobs[w] = dict(job_id=job.id, cost=cost, status=job.status)
        print(f"w={w}: submitted job {job.id} for {cost} credits")
    print("credits after:", mg.get_credit_balance(org.id))
finally:
    (hw / "oq_jobs.json").write_text(json.dumps(dict(backend=BACKEND, shots=SHOTS, jobs=jobs), indent=1))
    mg.close()
    sc.close()
