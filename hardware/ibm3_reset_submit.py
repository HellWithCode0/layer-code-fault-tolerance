"""Submit, watch and analyze the IBM reset follow-up on ibm_kingston (third key).

Approved by the user on 2026-10-14 ("10 minutes usage you can use as much as
you want").  Sends the verified compiled circuits of ibm3_reset_circuits.qpy
as 100,000-shot pieces interleaved in time, so drift affects every circuit
alike.  Planned QPU time about 390 s of the 600 s quota; max_execution_time
590 s.  Sampler-side decoupling and twirling are off (the circuits carry their
own XX decoupling).

Usage:  submit | watch JOB_ID     (IBM_TOKEN_FILE names the key file)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import qpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BACKEND = "ibm_kingston"
SHOTS = {"R1": 100_000, "R2_single": 300_000, "R3_single": 400_000, "R2_double": 300_000, "R3_double": 400_000}
PIECE = 100_000


def pieces():
    lists = {l: [PIECE] * (n // PIECE) + ([n % PIECE] if n % PIECE else []) for l, n in SHOTS.items()}
    total, order, used = sum(len(v) for v in lists.values()), [], {l: 0 for l in lists}
    for _ in range(total):
        l = min((l for l in lists if used[l] < len(lists[l])), key=lambda l: ((used[l] + 1) / len(lists[l]), l))
        order.append((l, lists[l][used[l]]))
        used[l] += 1
    return order


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=Path(os.environ["IBM_TOKEN_FILE"]).read_text().strip())


def estimate_seconds(plan):
    return sum(n * (plan["rep_delay_s"] + plan["circuits"][l]["duration_s"]) for l, n in SHOTS.items())


def submit():
    from qiskit_ibm_runtime import SamplerV2
    plan = json.loads((HERE / "ibm3_reset_plan.json").read_text())
    est = estimate_seconds(plan)
    print(f"estimated QPU time {est:.0f} s")
    assert est < 520
    with open(HERE / "ibm3_reset_circuits.qpy", "rb") as f:
        circs = dict(zip(plan["labels"], qpy.load(f)))
    svc = service()
    left = svc.usage().get("usage_remaining_seconds", 0)
    print("quota remaining:", left)
    assert left >= 590
    order = pieces()
    sampler = SamplerV2(mode=svc.backend(BACKEND))
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.max_execution_time = 590
    job = sampler.run([(circs[l], None, n) for l, n in order])
    (HERE / "ibm3_reset_job.json").write_text(json.dumps(dict(job_id=job.job_id(), backend=BACKEND, pieces=order,
                                                              shots=SHOTS, estimated_seconds=est,
                                                              layout=plan["layout"]), indent=1))
    print("submitted job", job.job_id(), "with", len(order), "pieces")


def watch(job_id):
    import ibm3_reset_analyze as ra
    last, st = None, ""
    while True:
        try:
            st = str(service().job(job_id).status())
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "network error, retrying:", type(e).__name__, flush=True)
            time.sleep(60)
            continue
        if st != last:
            print(time.strftime("%H:%M:%S"), "IBM", st, flush=True)
            last = st
        if st.endswith(("DONE", "ERROR", "CANCELLED")):
            break
        time.sleep(60)
    if not st.endswith("DONE"):
        try:
            print("job message:", service().job(job_id).error_message())
        except Exception:
            pass
        return
    meta = json.loads((HERE / "ibm3_reset_job.json").read_text())
    for _ in range(10):
        try:
            job = service().job(job_id)
            res = job.result()
            break
        except Exception as e:
            print("download failed, retrying:", type(e).__name__, flush=True)
            time.sleep(60)
    by = {}
    for (l, n), pub in zip(meta["pieces"], res):
        arr = pub.data.c.to_bool_array(order="little").astype(np.uint8)
        assert arr.shape[0] == n
        by.setdefault(l, []).append(arr)
    save = {}
    for l, arrs in by.items():
        b = np.concatenate(arrs)
        save[f"b_{l}"] = np.packbits(b, axis=1, bitorder="little")
        save[f"n_{l}"] = np.array(b.shape[1])
    np.savez_compressed(HERE / "ibm3_reset_bits.npz", **save)
    try:
        meta["usage_seconds"] = job.usage()
        (HERE / "ibm3_reset_job.json").write_text(json.dumps(meta, indent=1))
        print("QPU seconds used:", meta["usage_seconds"])
    except Exception:
        pass
    print("saved ibm3_reset_bits.npz", flush=True)
    ra.main()


if __name__ == "__main__":
    {"submit": lambda: submit(), "watch": lambda: watch(sys.argv[2]),
     "pieces": lambda: print(pieces())}[sys.argv[1]]()
