"""Read-only look at the Open Quantum account: orgs, credits, providers, backends, job categories.
Never prints the secret; submits nothing."""
from pathlib import Path

from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient

here = Path(__file__).parent
auth = ClientCredentialsAuth(creds=ClientCredentials(
    client_id=(here / "oq_client_id.txt").read_text().strip(),
    client_secret=(here / "oq_key.txt").read_text().strip()))
mg, sc = ManagementClient(auth=auth), SchedulerClient(auth=auth)
try:
    orgs = mg.list_user_organizations().organizations
    for o in orgs:
        try:
            bal = mg.get_credit_balance(o.id)
        except Exception as e:
            bal = f"unavailable ({type(e).__name__})"
        print(f"org: {o.name}  credits: {bal}")
    print("\nproviders:")
    for p in mg.list_providers(limit=50).providers:
        print(f"  {p.name} ({p.short_code})")
    print("\nbackend classes:")
    for b in mg.list_backend_classes(limit=100).backend_classes:
        print(f"  {b.short_code or b.id:28s} {b.name:32s} type={b.type} status={b.status} "
              f"accepting={b.accepting_jobs} queue={b.queue_depth}")
    print("\njob categories:")
    for c in sc.get_job_categories(limit=50).categories:
        subs = sc.get_job_subcategories(c.id, limit=50).categories
        print(f"  {c.short_code or c.id}: {c.name} -> " + ", ".join(f"{s.short_code}" for s in subs))
finally:
    mg.close()
    sc.close()
