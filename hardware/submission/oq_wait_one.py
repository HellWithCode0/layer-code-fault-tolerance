"""Wait for one job (hardware/oq_<TAG>_job.json) and save hardware/oq_<TAG>_output.json.
Uses the key recorded in the job file.  Retries network errors."""
import json
import sys
import time
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import SchedulerClient
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
TAG = sys.argv[1]
meta = json.loads((hw / f"oq_{TAG}_job.json").read_text())
out = hw / f"oq_{TAG}_output.json"
def client():
    c = meta.get("cred", "")
    return SchedulerClient(auth=ClientCredentialsAuth(creds=ClientCredentials(
        client_id=(here / f"oq_client_id{c}.txt").read_text().strip(),
        client_secret=(here / f"oq_key{c}.txt").read_text().strip())))
sc, last = client(), None
while True:
    try:
        job = sc.get_job(meta["job_id"])
        if job.status != last:
            print(time.strftime("%H:%M:%S"), TAG, job.status, (job.message or "")[:300], flush=True)
            last = job.status
        if job.status in ("Completed", "Failed", "Canceled"):
            if job.status == "Completed" and job.output_data_url and not out.exists():
                out.write_text(json.dumps(sc.download_job_output(job)))
                print("saved", out.name, flush=True)
            elif out.exists():
                print("WARNING", out.name, "exists; not overwritten", flush=True)
            break
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "network error, retrying:", type(e).__name__, flush=True)
        try:
            sc.close()
        except Exception:
            pass
        time.sleep(20)
        sc = client()
        continue
    time.sleep(20)
sc.close()
