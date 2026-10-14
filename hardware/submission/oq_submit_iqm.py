"""Submit the SWAP-free Emerald circuits (export_iqm.py), widths 1, 2, 3.

Approved by the user on 2026-10-13: run 1 (first qubit set, widths 1-3, 5000 shots, cap 5, org_B);
run 2 ("1."): second qubit set, widths 1 and 2, 5000 shots, cap 5, org_B;
run 3 ("Do 1."): run 1 qubits and files, widths 2 and 3, 12000 shots, cap 12, org_C.
Usage: IQM_RUN=<tag> oq_submit_iqm.py SHOTS MAX_CREDITS_PER_JOB --org=NAME [--widths=2,3] [--quote-only]
IQM_RUN and --org are required; a width already in this run's job file is refused.
Each job is quoted first; if a quote exceeds the cap the script stops before
creating that job.  --quote-only prepares and prints prices and creates nothing."""
import json
import sys
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient
from openquantum_sdk.models import JobCreate, JobPreparationCreate
from openquantum_sdk.utils import poll_for_status

import os
BACKEND = "iqm:emerald"
if "IQM_RUN" not in os.environ:
    sys.exit("set IQM_RUN (\"\" for run 1, 2, 3, ...)")
RUN = os.environ["IQM_RUN"]
ORG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--org=")), None)
if ORG is None:
    sys.exit("name the account to charge with --org=NAME")
WIDTHS = [int(x) for x in next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--widths=")), "1,2,3").split(",")]
SHOTS, CAP = int(sys.argv[1]), int(sys.argv[2])
QUOTE_ONLY = "--quote-only" in sys.argv
here = Path(__file__).parent
CRED = __import__("os").environ.get("OQ_CRED", "")   # "" newest key, "_account3" etc. for older ones
hw = Path(r"C:\kw\r\hardware")
auth = ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / f"oq_client_id{CRED}.txt").read_text().strip(),
    client_secret=(here / f"oq_key{CRED}.txt").read_text().strip()))
mg, sc = ManagementClient(auth=auth), SchedulerClient(auth=auth)
JOBS_FILE = hw / f"oq_iqm{RUN}_jobs.json"
jobs = {int(k): v for k, v in json.loads(JOBS_FILE.read_text())["jobs"].items()} if JOBS_FILE.exists() else {}
n_before = len(jobs)
if not QUOTE_ONLY and any(w in jobs for w in WIDTHS):
    sys.exit(f"{JOBS_FILE.name} already has widths {sorted(jobs)}; refusing to resubmit (use a new IQM_RUN)")
try:
    org = next(o for o in mg.list_user_organizations().organizations if o.name == ORG)
    print("org", org.name, "credits before:", mg.get_credit_balance(org.id).spark_credits)
    for w in WIDTHS:
        up = sc.upload_job_input(file_content=(hw / f"qasm3_iqm{RUN}_w{w}.qasm").read_bytes())
        prep = sc.prepare_job(JobPreparationCreate(
            organization_id=org.id, backend_class_id=BACKEND, name=f"cut obstruction iqm{RUN} w={w}",
            upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=SHOTS,
            configuration_data={"disable_qubit_rewiring": True}))
        res = poll_for_status(
            get_status_fn=lambda pid: ((r := sc.get_preparation_result(pid)).status in ("Completed", "Failed"), r),
            resource_id=prep.id, interval=2.0, timeout=300)
        if res.status == "Failed":
            raise RuntimeError(f"preparation failed for w={w}: {res.message}")
        plan = next(p for p in res.quote if p.name.startswith("Public"))
        prio = next(q for q in plan.queue_priorities if q.name.startswith("Standard"))
        cost = plan.price + prio.price_increase
        print(f"w={w}: quote {cost} credits for {SHOTS} shots")
        if QUOTE_ONLY:
            continue
        if cost > CAP:
            raise RuntimeError(f"w={w} quote {cost} exceeds the approved {CAP}; stopping")
        job = sc.create_job(JobCreate(job_preparation_id=prep.id,
                                      execution_plan_id=plan.execution_plan_id,
                                      queue_priority_id=prio.queue_priority_id))
        jobs[w] = dict(job_id=job.id, cost=cost, status=job.status, shots=SHOTS, org=ORG)
        print(f"w={w}: submitted job {job.id} for {cost} credits")
    print("credits after:", mg.get_credit_balance(org.id).spark_credits)
finally:
    if len(jobs) > n_before:
        JOBS_FILE.write_text(json.dumps(dict(backend=BACKEND, shots=SHOTS, jobs=jobs), indent=1))
    mg.close()
    sc.close()
