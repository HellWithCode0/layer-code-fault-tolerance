from pathlib import Path
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient
here = Path(__file__).parent
CRED = __import__("os").environ.get("OQ_CRED", "")   # "" newest key, "_account3" etc. for older ones
mg = ManagementClient(auth=ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / f"oq_client_id{CRED}.txt").read_text().strip(),
    client_secret=(here / f"oq_key{CRED}.txt").read_text().strip())))
try:
    for o in mg.list_user_organizations().organizations:
        print(o.name, mg.get_credit_balance(o.id).spark_credits)
finally:
    mg.close()
