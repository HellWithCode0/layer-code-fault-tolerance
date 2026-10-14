"""Submit, watch and analyze the IBM repeated-rounds job on ibm_kingston.

Approved by the user on 2026-10-13 ("do a GOOD run ... you can use the 10
minutes entirely").  Sends the verified compiled circuits of
ibm2_rounds_circuits.qpy (R = 1, 2, 3 only) as 100,000-shot pieces,
interleaved in time.  Planned QPU time about 467 s of the 600 s quota;
max_execution_time 590 s.  Sampler-side dynamical decoupling and twirling are
off (the circuits already carry their own XX decoupling).

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
SHOTS = {1: 200_000, 2: 550_000, 3: 1_050_000}
PIECE = 100_000


def pieces():
    """Interleaved list of (R, shots) with at most PIECE shots each."""
    lists = {R: [PIECE] * (n // PIECE) + ([n % PIECE] if n % PIECE else []) for R, n in SHOTS.items()}
    total = sum(len(v) for v in lists.values())
    order, used = [], {R: 0 for R in lists}
    for step in range(total):
        # pick the round count that is furthest behind its share
        R = min((R for R in lists if used[R] < len(lists[R])),
                key=lambda R: (used[R] + 1) / len(lists[R]))
        order.append((R, lists[R][used[R]]))
        used[R] += 1
    return order


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=Path(os.environ["IBM_TOKEN_FILE"]).read_text().strip())


def estimate_seconds(plan):
    rep = plan["rep_delay_s"]
    return sum(n * (rep + plan["rounds"][str(R)]["duration_s"]) for R, n in SHOTS.items())


def submit():
    from qiskit_ibm_runtime import SamplerV2
    plan = json.loads((HERE / "ibm2_rounds_plan.json").read_text())
    est = estimate_seconds(plan)
    print(f"estimated QPU time {est:.0f} s")
    assert est < 520, "estimate too close to the 600 s quota"
    with open(HERE / "ibm2_rounds_circuits.qpy", "rb") as f:
        circs = {R: c for R, c in zip((1, 2, 3, 4), qpy.load(f))}
    svc = service()
    usage = svc.usage()
    print("quota remaining:", usage.get("usage_remaining_seconds"))
    assert usage.get("usage_remaining_seconds", 0) >= 590
    be = svc.backend(BACKEND)
    order = pieces()
    sampler = SamplerV2(mode=be)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.max_execution_time = 590
    job = sampler.run([(circs[R], None, n) for R, n in order])
    meta = dict(job_id=job.job_id(), backend=BACKEND, pieces=order, shots=SHOTS, estimated_seconds=est,
                layout=plan["layout"])
    (HERE / "ibm2_rounds_job.json").write_text(json.dumps(meta, indent=1))
    print("submitted job", job.job_id(), "with", len(order), "pieces:", order)


def watch(job_id):
    import ibm_rounds_analyze as ra
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
        time.sleep(90)
    if not st.endswith("DONE"):
        try:
            print("job message:", service().job(job_id).error_message())
        except Exception:
            pass
        return
    meta = json.loads((HERE / "ibm2_rounds_job.json").read_text())
    for attempt in range(10):
        try:
            job = service().job(job_id)
            res = job.result()
            break
        except Exception as e:
            print("download failed, retrying:", type(e).__name__, flush=True)
            time.sleep(60)
    by_R = {}
    for (R, n), pub in zip(meta["pieces"], res):
        arr = pub.data.c.to_bool_array(order="little").astype(np.uint8)
        assert arr.shape[0] == n
        by_R.setdefault(R, []).append(arr)
    save = {}
    for R, arrs in by_R.items():
        b = np.concatenate(arrs)
        save[f"R{R}"] = np.packbits(b, axis=1, bitorder="little")
        save[f"n{R}"] = np.array(b.shape[1])
    np.savez_compressed(HERE / "ibm2_rounds_bits.npz", **save)
    try:
        meta["usage_seconds"] = job.usage()
        (HERE / "ibm2_rounds_job.json").write_text(json.dumps(meta, indent=1))
        print("QPU seconds used:", meta["usage_seconds"])
    except Exception:
        pass
    print("saved ibm2_rounds_bits.npz", flush=True)
    ra.main()


if __name__ == "__main__":
    if sys.argv[1] == "submit":
        submit()
    elif sys.argv[1] == "watch":
        watch(sys.argv[2])
    elif sys.argv[1] == "pieces":
        print(pieces())
