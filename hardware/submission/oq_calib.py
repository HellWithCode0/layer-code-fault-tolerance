"""Read-only: fetch job metadata and the calibration snapshot of each Rigetti job."""
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
meta = json.loads((hw / "oq_jobs.json").read_text())
try:
    for w, j in meta["jobs"].items():
        job = sc.get_job(j["job_id"])
        print(f"w={w} status={job.status} msg={(job.message or '')[:200]!r}")
        print("   extra keys:", {k: (v if not isinstance(v, (dict, list)) else type(v).__name__) for k, v in job.extra.items()})
        if job.calibration_data_url:
            cal = sc.download_job_calibration(job)
            (hw / f"oq_calibration_w{w}.json").write_text(json.dumps(cal))
            print("   calibration saved, top keys:", list(cal)[:20] if isinstance(cal, dict) else type(cal))
        else:
            print("   no calibration snapshot")
finally:
    sc.close()
