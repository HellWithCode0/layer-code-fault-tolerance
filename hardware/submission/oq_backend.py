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
    b = sc.get_backend_class("rigetti:cepheus-1-108q")
    (hw / "oq_backend_cepheus.json").write_text(json.dumps(b, indent=1))
finally:
    sc.close()
