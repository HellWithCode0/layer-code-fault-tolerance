#!/usr/bin/env python3
"""Exact minimum-weight undetectable logical faults of a detector error model.

For a stim DetectorErrorModel, the fault distance for observable k is

    min |e|  subject to  H e = 0  (every detector silent),  L_k e = 1,

where H is the detector-mechanism incidence matrix and L_k the observable row.
The problem is solved with the CP-SAT solver of OR-Tools using native XOR
constraints.  The solver returns OPTIMAL (exact distance) or FEASIBLE with a
proven lower bound when the time limit is reached; both numbers are reported.
Mechanisms with identical symptoms are merged, which does not change the
minimum.
"""
from __future__ import annotations

import stim
from ortools.sat.python import cp_model


def dem_mechanisms(dem: stim.DetectorErrorModel, keep_detectors=None):
    """Mechanisms (detector set, observable set), merged by symptom.

    keep_detectors: optional set of detector indices; the others are dropped.
    For a CSS circuit with CSS-closed Pauli noise, dropping the detectors of
    the opposite sector does not change the minimum weight of an
    undetectable fault flipping an observable of one sector, because the
    same-sector part of every fault is itself a fault.
    """
    seen = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = frozenset(t.val for t in inst.targets_copy() if t.is_relative_detector_id())
        if keep_detectors is not None:
            dets = frozenset(d for d in dets if d in keep_detectors)
        obs = frozenset(t.val for t in inst.targets_copy() if t.is_logical_observable_id())
        # separators (^) are absent because decompose_errors=False
        key = (dets, obs)
        if key not in seen and (dets or obs):
            seen[key] = inst.args_copy()[0]
    return [(sorted(d), sorted(o)) for (d, o) in seen]


def min_weight_logical(dem: stim.DetectorErrorModel, observable: int,
                       time_limit: float = 120.0, workers: int = 4, keep_detectors=None):
    mech = dem_mechanisms(dem, keep_detectors)
    mech = [(d, o) for (d, o) in mech if d or observable in o]
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"e{j}") for j in range(len(mech))]
    one = m.NewConstant(1)
    det_to = {}
    obs_to = []
    for j, (dets, obs) in enumerate(mech):
        for d in dets:
            det_to.setdefault(d, []).append(x[j])
        if observable in obs:
            obs_to.append(x[j])
    if not obs_to:
        return dict(status="NO_MECHANISM", distance=None, lower_bound=None, mechanisms=len(mech))
    for d, lits in det_to.items():
        m.AddBoolXOr(lits + [one])          # parity 0
    m.AddBoolXOr(obs_to)                    # parity 1
    m.Minimize(sum(x))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    st = solver.Solve(m)
    name = solver.StatusName(st)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = [j for j in range(len(mech)) if solver.Value(x[j])]
        return dict(status=name, distance=int(round(solver.ObjectiveValue())),
                    lower_bound=int(round(solver.BestObjectiveBound())),
                    mechanisms=len(mech), witness=[mech[j] for j in sol])
    return dict(status=name, distance=None, lower_bound=int(round(solver.BestObjectiveBound())),
                mechanisms=len(mech))


def circuit_distances(circuit: stim.Circuit, names, time_limit=120.0, sector_coord=None,
                      sector_values=None, only=None):
    """sector_coord/sector_values: keep only detectors whose coordinate
    number `sector_coord` lies in `sector_values` (CSS sector reduction)."""
    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    keep = None
    if sector_coord is not None:
        coords = dem.get_detector_coordinates()
        keep = {d for d, c in coords.items() if len(c) > sector_coord and int(c[sector_coord]) in sector_values}
    out = {}
    for k, nm in enumerate(names):
        if only is not None and nm not in only:
            continue
        out[nm] = min_weight_logical(dem, k, time_limit, keep_detectors=keep)
    return out
