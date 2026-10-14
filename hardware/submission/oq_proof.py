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
    job = sc.get_job(json.loads((hw / "oq_jobs.json").read_text())["jobs"]["1"]["job_id"])
    def show(o, ind=0):
        for k, v in o.items():
            if isinstance(v, dict):
                print(" " * ind + k + ":"); show(v, ind + 2)
            else:
                s = str(v); print(" " * ind + f"{k}: {s[:160]}")
    show(job.extra["proof"])
    try:
        print(sc.get_backend_class("rigetti:cepheus-1-108q").keys())
    except Exception as e:
        print("backend class:", type(e).__name__, str(e)[:200])
finally:
    sc.close()
