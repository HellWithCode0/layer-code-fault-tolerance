"""Poll the Open Quantum jobs in oq_jobs.json; download each output when it completes.
Network errors are retried, so a dropped connection does not stop the watch."""
import json
import time
import traceback
from pathlib import Path

from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import SchedulerClient

here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")


def client():
    auth = ClientCredentialsAuth(creds=ClientCredentials(
        client_id=(here / "oq_client_id.txt").read_text().strip(),
        client_secret=(here / "oq_key.txt").read_text().strip()))
    return SchedulerClient(auth=auth)


meta = json.loads((hw / "oq_jobs.json").read_text())
last, sc, finished = {}, client(), set()
while len(finished) < len(meta["jobs"]):
    try:
        for w, j in meta["jobs"].items():
            if w in finished:
                continue
            job = sc.get_job(j["job_id"])
            if last.get(w) != job.status:
                print(time.strftime("%H:%M:%S"), f"w={w}", job.status, (job.message or "")[:120], flush=True)
                last[w] = job.status
            if job.status in ("Completed", "Failed", "Canceled"):
                out = hw / f"oq_output_w{w}.json"
                if job.status == "Completed" and job.output_data_url and not out.exists():
                    out.write_text(json.dumps(sc.download_job_output(job)))
                    print(f"  saved {out.name}", flush=True)
                finished.add(w)
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "network error, retrying:", type(e).__name__, flush=True)
        try:
            sc.close()
        except Exception:
            pass
        time.sleep(30)
        sc = client()
        continue
    time.sleep(60)
sc.close()
print("all Open Quantum jobs finished", flush=True)
