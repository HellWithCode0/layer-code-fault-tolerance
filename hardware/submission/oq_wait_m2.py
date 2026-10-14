"""Wait for the edge-multiplicity job; save its output. Retries network errors."""
import json
import time
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import SchedulerClient
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
job_id = json.loads((hw / "oq_iqm_m2_job.json").read_text())["job_id"]
def client():
    return SchedulerClient(auth=ClientCredentialsAuth(creds=ClientCredentials(
        client_id=(here / "oq_client_id.txt").read_text().strip(),
        client_secret=(here / "oq_key.txt").read_text().strip())))
sc, last = client(), None
while True:
    try:
        job = sc.get_job(job_id)
        if job.status != last:
            print(time.strftime("%H:%M:%S"), job.status, (job.message or "")[:150], flush=True)
            last = job.status
        if job.status in ("Completed", "Failed", "Canceled"):
            if job.status == "Completed":
                (hw / "oq_iqm_m2_output.json").write_text(json.dumps(sc.download_job_output(job)))
                print("saved oq_iqm_m2_output.json", flush=True)
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
    time.sleep(30)
sc.close()
