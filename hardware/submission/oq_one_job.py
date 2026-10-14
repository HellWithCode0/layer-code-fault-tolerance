"""Prepare (free) and optionally submit ONE Emerald job.
Usage: OQ_CRED=<suffix> oq_one_job.py FILE SHOTS CAP --org=NAME --out=TAG [--submit]
Without --submit it only validates and quotes.  With --submit it creates the job
if the quote is within CAP and writes hardware/oq_<TAG>_job.json (refuses if it exists)."""
import json
import os
import sys
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient
from openquantum_sdk.models import JobCreate, JobPreparationCreate
from openquantum_sdk.utils import poll_for_status
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
CRED = os.environ.get("OQ_CRED", "")
FILE, SHOTS, CAP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ORG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--org=")), None)
TAG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
SUBMIT = "--submit" in sys.argv
if not ORG or not TAG:
    sys.exit("--org and --out are required")
jobfile = hw / f"oq_{TAG}_job.json"
if SUBMIT and jobfile.exists():
    sys.exit(f"{jobfile.name} exists; refusing to submit twice")
auth = ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / f"oq_client_id{CRED}.txt").read_text().strip(),
    client_secret=(here / f"oq_key{CRED}.txt").read_text().strip()))
mg, sc = ManagementClient(auth=auth), SchedulerClient(auth=auth)
try:
    org = next(o for o in mg.list_user_organizations().organizations if o.name == ORG)
    print("org", org.name, "credits before:", mg.get_credit_balance(org.id).spark_credits)
    up = sc.upload_job_input(file_content=(hw / FILE).read_bytes())
    prep = sc.prepare_job(JobPreparationCreate(
        organization_id=org.id, backend_class_id="iqm:emerald", name=TAG,
        upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=SHOTS,
        configuration_data={"disable_qubit_rewiring": True}))
    res = poll_for_status(
        get_status_fn=lambda pid: ((r := sc.get_preparation_result(pid)).status in ("Completed", "Failed"), r),
        resource_id=prep.id, interval=2.0, timeout=300)
    print("preparation", res.status, repr((res.message or "")[:400]))
    if res.status != "Completed":
        sys.exit(1)
    plan = next(p for p in res.quote if p.name.startswith("Public"))
    prio = next(q for q in plan.queue_priorities if q.name.startswith("Standard"))
    cost = plan.price + prio.price_increase
    print("quote", cost, "credits for", SHOTS, "shots")
    if SUBMIT:
        if cost > CAP:
            sys.exit(f"quote {cost} exceeds cap {CAP}; not submitted")
        job = sc.create_job(JobCreate(job_preparation_id=prep.id, execution_plan_id=plan.execution_plan_id,
                                      queue_priority_id=prio.queue_priority_id))
        jobfile.write_text(json.dumps(dict(job_id=job.id, cost=cost, shots=SHOTS, org=ORG, file=FILE,
                                           cred=CRED), indent=1))
        print("submitted job", job.id, "for", cost, "credits")
    print("credits after:", mg.get_credit_balance(org.id).spark_credits)
finally:
    mg.close()
    sc.close()
