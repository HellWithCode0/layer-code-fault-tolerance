"""Validate and quote the SWAP-free physical-qubit circuits. Creates NO job and spends no credits.
Checks the credit balance before and after to confirm that."""
import json
from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient
from openquantum_sdk.models import JobPreparationCreate
from openquantum_sdk.utils import poll_for_status

BACKEND, SHOTS = "rigetti:cepheus-1-108q", 10000
here = Path(__file__).parent
hw = Path(r"C:\kw\r\hardware")
auth = ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / "oq_client_id.txt").read_text().strip(),
    client_secret=(here / "oq_key.txt").read_text().strip()))
mg, sc = ManagementClient(auth=auth), SchedulerClient(auth=auth)
preps = {}
try:
    org = mg.list_user_organizations().organizations[0]
    print("credits before:", mg.get_credit_balance(org.id))
    for w in (1, 2, 3):
        up = sc.upload_job_input(file_content=(hw / f"qasm3_phys_w{w}.qasm").read_bytes())
        prep = sc.prepare_job(JobPreparationCreate(
            organization_id=org.id, backend_class_id=BACKEND, name=f"cut obstruction phys w={w}",
            upload_endpoint_id=up, job_subcategory_id="phys:oth", shots=SHOTS,
            configuration_data={"disable_qubit_rewiring": True}))
        res = poll_for_status(
            get_status_fn=lambda pid: ((r := sc.get_preparation_result(pid)).status in ("Completed", "Failed"), r),
            resource_id=prep.id, interval=2.0, timeout=300)
        print(f"w={w}: preparation {res.status} {(res.message or '')[:300]!r} config={res.configuration_data}")
        if res.status == "Completed":
            for p in res.quote:
                print(f"   plan {p.name}: {p.price} credits; priorities " +
                      ", ".join(f"{q.name} +{q.price_increase}" for q in p.queue_priorities))
            preps[w] = prep.id
    print("credits after:", mg.get_credit_balance(org.id))
finally:
    (hw / "oq_phys_prep.json").write_text(json.dumps(dict(backend=BACKEND, shots=SHOTS, preparations=preps), indent=1))
    mg.close()
    sc.close()
