"""Submit the edge-multiplicity circuit (qasm3_iqm_m2.qasm) to IQM Emerald.
Approved by the user on 2026-10-13 ("do your 4 credits pick"): 3000 shots,
at most 4 credits, org "org_A" (org_B shows 0).  Quotes first and
stops if the price exceeds the cap."""
import json
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient
from openquantum_sdk.models import JobCreate, JobPreparationCreate
from openquantum_sdk.utils import poll_for_status
SHOTS, CAP, ORG = 3000, 4, "org_A"
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
auth = ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / "oq_client_id.txt").read_text().strip(),
    client_secret=(here / "oq_key.txt").read_text().strip()))
mg, sc = ManagementClient(auth=auth), SchedulerClient(auth=auth)
try:
    org = next(o for o in mg.list_user_organizations().organizations if o.name == ORG)
    print("org", org.name, "credits before:", mg.get_credit_balance(org.id).spark_credits)
    up = sc.upload_job_input(file_content=(hw / "qasm3_iqm_m2.qasm").read_bytes())
    prep = sc.prepare_job(JobPreparationCreate(
        organization_id=org.id, backend_class_id="iqm:emerald", name="edge multiplicity m0=2",
        upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=SHOTS,
        configuration_data={"disable_qubit_rewiring": True}))
    res = poll_for_status(
        get_status_fn=lambda pid: ((r := sc.get_preparation_result(pid)).status in ("Completed", "Failed"), r),
        resource_id=prep.id, interval=2.0, timeout=300)
    if res.status == "Failed":
        raise RuntimeError(f"preparation failed: {res.message}")
    plan = next(p for p in res.quote if p.name.startswith("Public"))
    prio = next(q for q in plan.queue_priorities if q.name.startswith("Standard"))
    cost = plan.price + prio.price_increase
    print("quote", cost, "credits for", SHOTS, "shots")
    if cost > CAP:
        raise RuntimeError(f"quote {cost} exceeds the approved {CAP}; stopping")
    job = sc.create_job(JobCreate(job_preparation_id=prep.id, execution_plan_id=plan.execution_plan_id,
                                  queue_priority_id=prio.queue_priority_id))
    (hw / "oq_iqm_m2_job.json").write_text(json.dumps(dict(job_id=job.id, cost=cost, shots=SHOTS, org=ORG), indent=1))
    print("submitted job", job.id, "for", cost, "credits")
    print("credits after:", mg.get_credit_balance(org.id).spark_credits)
finally:
    mg.close()
    sc.close()
