"""Read-only: print the platform's recorded input hash of each run-1 Emerald job
(account 2 credentials) next to the hash of our local file."""
import hashlib
import json
import re
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import SchedulerClient
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
sc = SchedulerClient(auth=ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / "oq_client_id_account2.txt").read_text().strip(),
    client_secret=(here / "oq_key_account2.txt").read_text().strip())))
try:
    jobs = json.loads((hw / "oq_iqm_jobs.json").read_text())["jobs"]
    for w, j in sorted(jobs.items()):
        job = sc.get_job(j["job_id"])
        stmt = job.extra.get("proof", {}).get("statement", "")
        m = re.search(r"input_sha256=([0-9a-f]+)", stmt)
        local = hashlib.sha256((hw / f"qasm3_iqm_w{w}.qasm").read_bytes()).hexdigest()
        print(f"w={w}: platform {m[1] if m else None}\n      local    {local}  match={bool(m) and m[1] == local}")
finally:
    sc.close()
