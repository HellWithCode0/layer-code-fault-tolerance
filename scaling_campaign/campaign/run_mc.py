#!/usr/bin/env python3
"""Monte Carlo campaign: joint logical measurement on growing layer-code blocks.

Usage (from the package root):
    python campaign/run_mc.py --profile heavy
    python campaign/run_mc.py --profile standard --hours 6

Every finished batch is appended to results/mc_batches.csv at once, so the
run can be stopped at any time (Ctrl+C, kill, a closed notebook, a reboot)
and started again with the same command.  It then continues where it stopped,
never repeats a seed, and keeps the original time budget (the start time is
stored in results/budget_<profile>.json; --fresh-budget starts a new one).
Batch seeds are (BASE_SEED + crc32("task_id|batch")) mod 2^63 and are
written in the CSV.

Before sampling, every distinct geometry of the profile is validated
(commuting deformed checks, 2k - 1 logical qubits, deterministic noiseless
circuit in both bases); the result is cached in results/preflight.json.

Scheduling: each worker is given one task at a time for a session of several
batches (about --session-minutes of work), so the decoder is built once per
session.  The next task is always the unfinished one with the least progress,
so rare-event tasks share the remaining time evenly.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import traceback
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from profiles import BASE_SEED, PROFILES, task_id  # noqa: E402

FIELDS = ["task_id", "k", "s", "interface", "width", "mult", "basis", "p", "rounds", "decoder",
          "batch", "seed", "shots", "errors", "obs_names", "obs_errors", "masks", "seconds", "worker"]


def batch_seed(tid, b):
    return (BASE_SEED + zlib.crc32(f"{tid}|{b}".encode())) % (2 ** 63)


def geometry_key(t):
    return (t["k"], t["s"], t["interface"], t["width"], t["mult"], t["rounds"])


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------

def build(t):
    import two_block_family as F
    from decoders import BpLsd
    tb = F.TwoBlock(k=t["k"], s=t["s"], interface=t["interface"], width=max(1, t["width"]),
                    mult=t["mult"])
    R = t["rounds"]
    circ, names = F.build_circuit(tb, t["p"], t["basis"], R, R, R, idle=t["interface"] == "idle")
    dem = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    sectors = None if t["decoder"] == "full" else ({0, 2} if t["basis"] == "X" else {1, 3})
    comp = BpLsd(sectors).compile_decoder_for_dem(dem=dem)
    del dem
    return circ, comp, names


def run_batch(t, circ, comp, names, b, shots):
    import numpy as np
    tid = task_id(t)
    seed = batch_seed(tid, b)
    t0 = time.time()
    dets, obs = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True,
                                                                bit_packed=True)
    pred = comp.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    no = len(names)
    mism = np.unpackbits(pred ^ obs, axis=1, bitorder="little")[:, :no].astype(bool)
    bad = mism.any(axis=1)
    masks = {}
    for row in mism[bad]:
        key = "".join("1" if v else "0" for v in row)
        masks[key] = masks.get(key, 0) + 1
    return dict(task_id=tid, k=t["k"], s=t["s"], interface=t["interface"], width=t["width"],
                mult=t["mult"], basis=t["basis"], p=t["p"], rounds=t["rounds"],
                decoder=t["decoder"], batch=b, seed=seed, shots=shots, errors=int(bad.sum()),
                obs_names=" ".join(names), obs_errors=" ".join(str(int(v)) for v in mism.sum(axis=0)),
                masks=json.dumps(masks, sort_keys=True), seconds=round(time.time() - t0, 3))


def worker_main(wid, in_q, out_q, parent_pid):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(HERE))
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # the main process decides when to stop
    cache_id, cache = None, None
    while True:
        try:
            msg = in_q.get(timeout=60)
        except queue.Empty:
            if os.getppid() != parent_pid:
                return                                 # the main process is gone
            continue
        if msg is None:
            return
        seq, t, batches = msg
        tid = task_id(t)
        try:
            if cache_id != tid:
                cache, cache_id = None, None
                cache = build(t)
                cache_id = tid
            circ, comp, names = cache
            for (b, shots) in batches:
                row = run_batch(t, circ, comp, names, b, shots)
                row["worker"] = wid
                out_q.put(("row", wid, seq, row))
        except Exception:
            out_q.put(("error", wid, seq, (tid, traceback.format_exc())))
        out_q.put(("idle", wid, seq, None))


def preflight_one(g):
    sys.path.insert(0, str(HERE))
    import two_block_family as F
    k, s, iface, width, mult, R = g
    try:
        tb = F.TwoBlock(k=k, s=s, interface=iface, width=max(1, width), mult=mult)
        ok, info = (True, {}) if iface == "idle" else tb.check_deformed_code()
        for basis in "XZ":
            c, _ = F.build_circuit(tb, 0.001, basis, R, R, R, idle=iface == "idle")
            ok = ok and F.noiseless_check(c, 16)
        return g, bool(ok), json.dumps(info)
    except Exception:
        return g, False, traceback.format_exc()


# --------------------------------------------------------------------------
# main side
# --------------------------------------------------------------------------

def load_existing(path, say):
    done, bad = {}, 0
    if not path.exists():
        return done
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                b = int(r["batch"])
                shots, errors, sec = int(r["shots"]), int(r["errors"]), float(r["seconds"])
                d = done.setdefault(r["task_id"], dict(shots=0, errors=0, last=-1, sec=0.0, seen=set()))
                if b in d["seen"]:
                    continue                           # never count a batch twice
                d["seen"].add(b)
                d["shots"] += shots
                d["errors"] += errors
                d["last"] = max(d["last"], b)
                d["sec"] += sec
            except (TypeError, ValueError, KeyError):
                bad += 1
    if bad:
        say(f"skipped {bad} unreadable row(s) in {path.name} (for example a line cut by a reboot)")
    return done


def take_lock(res):
    """One runner per results folder (Linux/macOS).  Returns the open lock file."""
    fh = open(res / ".run_mc.lock", "w")
    try:
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        pass                                           # Windows: no lock
    except OSError:
        print("Another run_mc.py is already writing to this results folder. "
              "Stop it first (see README), or wait for it to finish.", flush=True)
        sys.exit(2)
    return fh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=sorted(PROFILES))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget (default: profile)")
    ap.add_argument("--fresh-budget", action="store_true", help="start the time budget from now")
    ap.add_argument("--session-minutes", type=float, default=10.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "mc_batches.csv"))
    ap.add_argument("--only", default="", help="substring filter on task ids (optional)")
    args = ap.parse_args()

    prof = PROFILES[args.profile]()
    hours = args.hours if args.hours is not None else prof["hours"]
    tasks = [t for t in prof["tasks"] if args.only in task_id(t)]
    out = Path(args.out)
    res = out.parent
    res.mkdir(parents=True, exist_ok=True)
    lock = take_lock(res)
    log = open(out.with_suffix(".log"), "a")

    def say(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    # time budget that survives restarts
    bfile = res / f"budget_{args.profile}.json"
    start = time.time()
    if bfile.exists() and not args.fresh_budget:
        try:
            b = json.loads(bfile.read_text())
            start = float(b["start"])
            if args.hours is None:
                hours = float(b["hours"])
        except Exception:
            pass
    bfile.write_text(json.dumps(dict(start=start, hours=hours, profile=args.profile)))
    deadline = start + hours * 3600

    state = {}
    existing = load_existing(out, say)
    for t in tasks:
        tid = task_id(t)
        e = existing.get(tid, dict(shots=0, errors=0, last=-1, sec=0.0))
        state[tid] = dict(t=t, shots=e["shots"], errors=e["errors"], next=e["last"] + 1,
                          sec=e["sec"], inflight=0, failed=False)

    def finished(st):
        t = st["t"]
        return st["failed"] or st["errors"] >= t["max_errors"] or st["shots"] >= t["max_shots"]

    def progress(st):
        t = st["t"]
        return max(st["errors"] / t["max_errors"], (st["shots"] + st["inflight"]) / t["max_shots"])

    say(f"profile={args.profile} tasks={len(tasks)} workers={args.workers} budget={hours} h "
        f"(left {max(0.0, deadline - time.time()) / 3600:.1f} h) "
        f"already_done={sum(finished(s) for s in state.values())}")

    def on_term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ctx = mp.get_context("spawn")

    # preflight: validate every distinct geometry once
    pfile = res / "preflight.json"
    pf = {}
    if pfile.exists():
        try:
            pf = json.loads(pfile.read_text())
        except Exception:
            pf = {}
    todo = sorted({geometry_key(st["t"]) for st in state.values()} - {tuple(json.loads(k)) for k in pf})
    if todo:
        say(f"preflight: validating {len(todo)} geometries")
        with ctx.Pool(min(args.workers, len(todo))) as pool:
            for g, ok, info in pool.imap_unordered(preflight_one, todo):
                pf[json.dumps(list(g))] = dict(ok=ok, info=info)
                if not ok:
                    say(f"preflight FAILED for {g}: {info}")
        pfile.write_text(json.dumps(pf, indent=1))
    for st in state.values():
        if not pf.get(json.dumps(list(geometry_key(st["t"]))), {}).get("ok", False):
            st["failed"] = True
    nbad = sum(1 for v in pf.values() if not v["ok"])
    say(f"preflight: {len(pf) - nbad} geometries valid, {nbad} invalid (their tasks are skipped)")

    new_file = not out.exists() or out.stat().st_size == 0
    if not new_file:
        with open(out, "rb") as fb:
            fb.seek(-1, os.SEEK_END)
            cut = fb.read(1) != b"\n"
        if cut:
            with open(out, "a") as fa:
                fa.write("\n")                         # never glue a row onto a cut line
    fh = open(out, "a", newline="")
    wtr = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        wtr.writeheader()
        fh.flush()

    out_q = ctx.Queue()
    workers, in_qs = [None] * args.workers, [None] * args.workers

    def start_worker(w):
        q = ctx.Queue()
        p = ctx.Process(target=worker_main, args=(w, q, out_q, os.getpid()), daemon=True)
        p.start()
        workers[w], in_qs[w] = p, q

    for w in range(args.workers):
        start_worker(w)
    idle = set(range(args.workers))
    assigned = {}                                      # w -> [seq, task_id, shots left]
    crashes = {}
    seq_counter = [0]
    t_start = time.time()
    last_report = 0.0

    def assign(w):
        cands = [st for st in state.values() if not finished(st)
                 and st["shots"] + st["inflight"] < st["t"]["max_shots"]]
        if not cands:
            return False
        st = min(cands, key=progress)
        t = st["t"]
        if st["shots"] == 0:
            nb = 1                                     # calibration batch
        else:
            per_batch = st["sec"] / st["shots"] * t["batch"]
            nb = max(1, int(args.session_minutes * 60 / max(per_batch, 1.0)))
            if st["errors"] > 0:                       # do not overshoot max_errors by much
                need = (t["max_errors"] - st["errors"]) * st["shots"] / st["errors"] * 1.2
                nb = min(nb, max(1, -(-int(need) // t["batch"])))
        room = t["max_shots"] - st["shots"] - st["inflight"]
        nb = max(1, min(nb, -(-room // t["batch"])))
        batches = []
        for _ in range(nb):
            batches.append((st["next"], t["batch"]))
            st["next"] += 1
        st["inflight"] += nb * t["batch"]
        seq_counter[0] += 1
        assigned[w] = [seq_counter[0], task_id(t), nb * t["batch"]]
        in_qs[w].put((seq_counter[0], t, batches))
        idle.discard(w)
        return True

    try:
        while True:
            if time.time() < deadline:
                for w in sorted(idle):
                    if not assign(w):
                        break
            if not assigned:
                break
            try:
                kind, w, seq, payload = out_q.get(timeout=30)
            except queue.Empty:
                kind = None
            current = kind is not None and w in assigned and assigned[w][0] == seq
            if kind == "row":
                st = state[payload["task_id"]]
                wtr.writerow(payload)                  # data are kept even from a stale session
                fh.flush()
                st["shots"] += payload["shots"]
                st["errors"] += payload["errors"]
                st["sec"] += payload["seconds"]
                if current:
                    st["inflight"] = max(0, st["inflight"] - payload["shots"])
                    assigned[w][2] -= payload["shots"]
            elif kind == "error":
                tid, tb = payload
                state[tid]["failed"] = True
                say(f"ERROR in task {tid}:\n{tb}")
            elif kind == "idle" and current:
                _, tid, left = assigned.pop(w)
                state[tid]["inflight"] = max(0, state[tid]["inflight"] - left)
                idle.add(w)
            for w2 in list(assigned):
                if not workers[w2].is_alive():
                    _, tid, left = assigned.pop(w2)
                    st = state[tid]
                    st["inflight"] = max(0, st["inflight"] - left)
                    crashes[tid] = crashes.get(tid, 0) + 1
                    if crashes[tid] >= 2:
                        st["failed"] = True
                    say(f"worker {w2} died during task {tid} (out of memory?); restarting it"
                        + ("; task skipped after two crashes" if st["failed"] else ""))
                    start_worker(w2)
                    idle.add(w2)
            if time.time() - last_report > 300:
                last_report = time.time()
                nd = sum(finished(s) for s in state.values())
                shots = sum(s["shots"] for s in state.values())
                left = max(0.0, deadline - time.time()) / 3600
                say(f"tasks finished {nd}/{len(state)}  total shots {shots:,}  "
                    f"budget left {left:.1f} h  busy workers {len(assigned)}")
            if time.time() > deadline + 3600:
                say("hard stop one hour after the budget; unfinished sessions are dropped")
                break
    except KeyboardInterrupt:
        say("stopped by signal; all finished batches are saved")
    finally:
        for q in in_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        fh.close()
        nd = sum(finished(s) for s in state.values())
        say(f"exit: tasks finished {nd}/{len(state)}, this session {(time.time() - t_start) / 3600:.2f} h")
        log.close()
        lock.close()


if __name__ == "__main__":
    main()
