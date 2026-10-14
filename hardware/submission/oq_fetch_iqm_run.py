"""One-shot: fetch finished outputs of oq_iqm<RUN>_jobs.json (never overwrites)."""
import json, os, time
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import SchedulerClient
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
CRED, RUN = os.environ.get("OQ_CRED", ""), os.environ["IQM_RUN"]
for attempt in range(5):
    try:
        sc = SchedulerClient(auth=ClientCredentialsAuth(creds=ClientCredentials(
            client_id=(here / f"oq_client_id{CRED}.txt").read_text().strip(),
            client_secret=(here / f"oq_key{CRED}.txt").read_text().strip())))
        for w, j in json.loads((hw / f"oq_iqm{RUN}_jobs.json").read_text())["jobs"].items():
            job = sc.get_job(j["job_id"])
            out = hw / f"oq_iqm{RUN}_output_w{w}.json"
            print(f"w={w}: {job.status} {(job.message or '')[:200]}")
            if job.status == "Completed" and not out.exists():
                out.write_text(json.dumps(sc.download_job_output(job)))
                print("   saved", out.name)
        sc.close()
        break
    except Exception as e:
        print("network error, retrying:", type(e).__name__)
        time.sleep(15)
