"""Run profiles for run_mc.py.

A task is one (block family, spacing, interface, basis, noise strength,
decoder) point.  Every task runs until it has max_errors failed shots, or
max_shots shots, or the wall-clock budget of the profile runs out; unfinished
tasks share the remaining time evenly.  Batch sizes are fixed per size, and
every batch has a recorded seed, so any batch can be reproduced exactly.

Rounds per phase are R = 2s + 1, at least both block distances
(d_X = 2s + 1, d_Z = 2s), so that time-like faults are never the shortest
failure: the reported bit is limited by d_Z = 2s, the measured pair's Z
sector by d_X = 2s + 1 (thick grid) or by the bridge width (thin bridge).
"""
from __future__ import annotations

BASE_SEED = 20260927

# shots per batch, by spacing (fixed so that seeds reproduce the data exactly)
BATCH = {2: 2000, 3: 600, 4: 300, 5: 150}
MAX_SHOTS = {2: 2_000_000, 3: 600_000, 4: 200_000, 5: 80_000}


def task(k, s, interface, basis, p, width=1, mult=1, decoder="sector", max_errors=200,
         max_shots=None, batch=None):
    return dict(k=k, s=s, interface=interface, width=width if interface == "bridge" else 0,
                mult=mult, basis=basis, p=p, rounds=2 * s + 1, decoder=decoder,
                max_errors=max_errors, max_shots=max_shots or MAX_SHOTS[s],
                batch=batch or BATCH[s])


def task_id(t):
    return (f"k{t['k']}_s{t['s']}_{t['interface']}_w{t['width']}_m{t['mult']}_{t['basis']}"
            f"_p{t['p']:g}_R{t['rounds']}_{t['decoder']}")


def _grid(ks, ss, ifaces, ps, **kw):
    out = []
    for k in ks:
        for s in ss:
            for iface in ifaces:
                for p in ps:
                    for basis in ("X", "Z"):
                        out.append(task(k, s, iface, basis, p, **kw))
    return out


def smoke():
    ts = _grid([2], [2], ["thick", "bridge", "idle"], [0.002], max_shots=400, batch=200)
    return dict(hours=0.25, tasks=ts)


def standard():
    ts = _grid([2], [2, 3, 4], ["thick", "bridge", "idle"], [0.001, 0.002, 0.003, 0.004])
    return dict(hours=12, tasks=ts)


def heavy():
    ts = _grid([2], [2, 3, 4, 5], ["thick", "bridge", "idle"],
               [0.0005, 0.001, 0.002, 0.003, 0.004, 0.006])
    # more spectators: [[6,4,2]] blocks (K = 4 per block)
    ts += _grid([4], [2, 3, 4], ["thick", "bridge"], [0.001, 0.002, 0.003])
    # edge multiplicity 2 on the spanning grid
    ts += _grid([2], [2, 3, 4], ["thick"], [0.001, 0.002, 0.003], mult=2)
    # thin interfaces of width 2 and 3 at larger sizes
    for w in (2, 3):
        ts += _grid([2], [3, 4], ["bridge"], [0.002], width=w)
    return dict(hours=36, tasks=ts)


PROFILES = dict(smoke=smoke, standard=standard, heavy=heavy)
