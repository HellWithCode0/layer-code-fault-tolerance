#!/usr/bin/env python3
"""Numerical Stim validation harness for
"Geometry-preserving fault-tolerant computation with three-dimensional layer codes".

What this script tests
----------------------
1) The circuit-level SI1000 path-strip/logical-Z gauging primitive using
   deterministic Stim DETECTOR/OBSERVABLE circuits and MWPM.
2) A finite Canonical Architecture Schedule (CAS) on growing HGP(rep_L,rep_L)
   CSS instances, and optionally on the paper's truncated WB-incidence instances.
3) Two CAS protocols:
     legacy  : reproduces the manuscript's finite CAS numerical structure
               (idle memory -> slab-Z/frame bookkeeping -> final memory).
     stress  : stronger numerical stress test that replaces the bookkeeping-only
               slab step by an explicit circuit-level gauged-strip stage and counts
               a wrong protected slab measurement as schedule failure.

Important scope
---------------
This is a FINITE numerical stress test, not a proof of the asymptotic threshold
or the full theorem. The "stress" CAS is stage-composed: each protected stage has
its own Stim detector model and is joined at the logical-effect level. It is not
a CAD-complete continuous detector error model of every Williamson-Baspin defect
transition. That distinction is intentional and is written into every output.

The default HP family is self-contained. To run the WB family, place the paper's
`wb_layer_code_instance.py` beside this script (included in the companion bundle).

Recommended environment
-----------------------
Python >= 3.11
stim >= 1.16
pymatching >= 2.4
numpy >= 2.1
matplotlib >= 3.8

Example
-------
  python paper_stim_validation.py --preset quick
  python paper_stim_validation.py --preset paper --families hp wb --protocols legacy stress
  python paper_stim_validation.py --experiment primitive --preset paper

Outputs are written under ./stim_validation_results/ by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pymatching
import stim

SEED_DEFAULT = 20260918
SCOPE_LEGACY = (
    "finite SI1000 CAS reproduction: idle R-round CSS memory -> slab-Z/frame "
    "bookkeeping -> final R-round CSS memory; Stim DEM+MWPM; NOT a CAD-complete "
    "Williamson-Baspin architecture DEM; NOT an asymptotic threshold proof"
)
SCOPE_STRESS = (
    "finite SI1000 stage-composed CAS stress test: idle R-round CSS memory -> "
    "explicit gauged logical-Z strip R rounds -> final R-round CSS memory; "
    "each stage uses Stim DEM+MWPM and a wrong protected slab effect counts as "
    "schedule failure; NOT a continuous transition-detector CAD model and NOT "
    "an asymptotic threshold proof"
)
SCOPE_PRIMITIVE = (
    "finite circuit-level SI1000 logical-Z path-strip gauging primitive on rotated "
    "surface-code instances; Stim DEM+MWPM; NOT a layer-code/full-architecture threshold"
)


# -----------------------------------------------------------------------------
# GF(2) utilities and a self-contained HGP(rep_L, rep_L) growing family.
# -----------------------------------------------------------------------------

def gf2_rank(a: np.ndarray) -> int:
    a = (np.asarray(a, dtype=np.uint8).copy() & 1)
    if a.size == 0:
        return 0
    m, n = a.shape
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, m) if a[i, c]), None)
        if piv is None:
            continue
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        for i in range(m):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        r += 1
        if r == m:
            break
    return r


def nullspace_gf2(h: np.ndarray) -> list[np.ndarray]:
    a = (np.asarray(h, dtype=np.uint8).copy() & 1)
    m, n = a.shape
    r = 0
    pivots: dict[int, int] = {}
    for c in range(n):
        piv = next((i for i in range(r, m) if a[i, c]), None)
        if piv is None:
            continue
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        for i in range(m):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        pivots[c] = r
        r += 1
        if r == m:
            break
    out = []
    for f in [c for c in range(n) if c not in pivots]:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for c, pr in pivots.items():
            v[c] = a[pr, f]
        out.append(v)
    return out


def in_rowspace(h: np.ndarray, v: np.ndarray) -> bool:
    h = np.asarray(h, dtype=np.uint8)
    v = np.asarray(v, dtype=np.uint8).reshape(1, -1)
    if h.size == 0:
        return not bool(v.any())
    return gf2_rank(np.vstack([h, v])) == gf2_rank(h)


def dedup_rows(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.uint8)
    if h.size == 0:
        return h
    seen = set()
    rows = []
    for row in h:
        if not row.any():
            continue
        key = row.tobytes()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row.copy())
    return np.vstack(rows) if rows else np.zeros((0, h.shape[1]), dtype=np.uint8)


def reduce_mod_rows(vecs: Iterable[np.ndarray], h: np.ndarray, n: int) -> list[np.ndarray]:
    cur = np.asarray(h, dtype=np.uint8).copy() if np.asarray(h).size else np.zeros((0, n), dtype=np.uint8)
    kept: list[np.ndarray] = []
    for vv in vecs:
        v = np.asarray(vv, dtype=np.uint8).reshape(1, -1)
        r0 = gf2_rank(cur)
        trial = np.vstack([cur, v])
        if gf2_rank(trial) > r0:
            kept.append(v.ravel())
            cur = trial
    return kept


def repetition_check(L: int) -> np.ndarray:
    if L < 2:
        raise ValueError("L must be >=2")
    h = np.zeros((L - 1, L), dtype=np.uint8)
    for i in range(L - 1):
        h[i, i] = 1
        h[i, i + 1] = 1
    return h


def hgp(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h1 = np.asarray(h1, dtype=np.uint8)
    h2 = np.asarray(h2, dtype=np.uint8)
    r1, n1 = h1.shape
    r2, n2 = h2.shape
    hx = np.hstack([
        np.kron(h1, np.eye(n2, dtype=np.uint8)),
        np.kron(np.eye(r1, dtype=np.uint8), h2.T),
    ]) & 1
    hz = np.hstack([
        np.kron(np.eye(n1, dtype=np.uint8), h2),
        np.kron(h1.T, np.eye(r2, dtype=np.uint8)),
    ]) & 1
    hx = dedup_rows(hx)
    hz = dedup_rows(hz)
    if np.any((hx @ hz.T) & 1):
        raise RuntimeError("HGP construction is not CSS")
    return hx, hz


def min_logical(hx: np.ndarray, hz: np.ndarray, kind: str) -> np.ndarray:
    """Return a low-weight nontrivial logical representative.

    For the small HGP(rep,rep) sizes used here, a nullspace basis plus pairwise
    combinations is sufficient to recover the standard weight-L logical.
    """
    if kind.upper() == "Z":
        candidates = [v for v in nullspace_gf2(hx) if v.any() and not in_rowspace(hz, v)]
        stabilizers = hz
    elif kind.upper() == "X":
        candidates = [v for v in nullspace_gf2(hz) if v.any() and not in_rowspace(hx, v)]
        stabilizers = hx
    else:
        raise ValueError(kind)
    if not candidates:
        raise RuntimeError(f"no {kind} logical found")
    candidates.sort(key=lambda v: int(v.sum()))
    basis = candidates[: min(10, len(candidates))]
    best = basis[0].copy()
    best_w = int(best.sum())
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            v = basis[i] ^ basis[j]
            w = int(v.sum())
            if w and w < best_w and not in_rowspace(stabilizers, v):
                best, best_w = v, w
    return best.astype(np.uint8)


@dataclass(frozen=True)
class CodeInstance:
    family: str
    size: int
    hx: np.ndarray
    hz: np.ndarray
    logz: np.ndarray
    n: int
    d_upper: int
    rounds: int
    note: str


def build_hp_instance(L: int, rounds_scale: int = 1) -> CodeInstance:
    hx, hz = hgp(repetition_check(L), repetition_check(L))
    k = hx.shape[1] - gf2_rank(hx) - gf2_rank(hz)
    if k != 1:
        raise RuntimeError(f"expected HGP(rep_{L},rep_{L}) k=1, got k={k}")
    logz = min_logical(hx, hz, "Z")
    logx = min_logical(hx, hz, "X")
    d_upper = min(int(logz.sum()), int(logx.sum()))
    return CodeInstance(
        family="hp_surface",
        size=L,
        hx=hx,
        hz=hz,
        logz=logz,
        n=hx.shape[1],
        d_upper=d_upper,
        rounds=max(2, rounds_scale * d_upper),
        note="HGP(rep_L,rep_L), self-contained algebraic layer-code substrate",
    )


def build_wb_instance(L: int, rounds_scale: int = 1) -> CodeInstance:
    try:
        from wb_layer_code_instance import build_repetition_layer_code, quasiconcat_and_kernel_logicals
    except Exception as exc:
        raise RuntimeError(
            "WB family requested but wb_layer_code_instance.py is unavailable. "
            "Place it beside this script or use --families hp."
        ) from exc
    hx, hz, meta = build_repetition_layer_code(L)
    logs = quasiconcat_and_kernel_logicals(hx, hz, meta)
    if not (logs["Z_is_logical"] and logs["X_is_logical"]):
        raise RuntimeError(f"WB L={L}: logical audit failed")
    d_upper = min(int(logs["Z_weight"]), int(logs["X_weight"]))
    return CodeInstance(
        family="wb_truncated",
        size=L,
        hx=np.asarray(hx, dtype=np.uint8),
        hz=np.asarray(hz, dtype=np.uint8),
        logz=np.asarray(logs["Z"], dtype=np.uint8),
        n=int(meta["n"]),
        d_upper=d_upper,
        rounds=max(2, rounds_scale * d_upper),
        note="truncated WB-incidence test family; not a CAD reconstruction of the full artwork",
    )


# -----------------------------------------------------------------------------
# Surface-code extraction and generic logical-Z strip gauging.
# -----------------------------------------------------------------------------

def rotated_surface_code(d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract Hx, Hz and low-weight logicals from Stim's rotated memory circuit."""
    from collections import defaultdict

    circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=1)
    coords: dict[int, tuple[float, ...]] = {}
    for inst in circ:
        if inst.name == "QUBIT_COORDS":
            coords[inst.targets_copy()[0].value] = tuple(inst.gate_args_copy())
    data = sorted(q for q, (x, y) in coords.items() if int(x) % 2 == 1 and int(y) % 2 == 1)
    data_set = set(data)
    data_index = {q: i for i, q in enumerate(data)}
    n = len(data)

    h_targets = set()
    seen_r = False
    for inst in circ:
        if inst.name == "R":
            seen_r = True
        if seen_r and inst.name == "H":
            h_targets.update(t.value for t in inst.targets_copy())
            break

    cx_pairs = []
    phase = 0
    for inst in circ:
        if inst.name == "H" and phase == 0:
            phase = 1
            continue
        if inst.name == "H" and phase == 1:
            phase = 2
            continue
        if phase == 1 and inst.name == "CX":
            ts = [t.value for t in inst.targets_copy()]
            cx_pairs.extend((ts[i], ts[i + 1]) for i in range(0, len(ts), 2))

    supp: dict[int, set[int]] = defaultdict(set)
    for a, b in cx_pairs:
        ma, mb = a not in data_set, b not in data_set
        if ma and not mb:
            supp[a].add(b)
        elif mb and not ma:
            supp[b].add(a)

    hx_rows, hz_rows = [], []
    for m, s in supp.items():
        row = np.zeros(n, dtype=np.uint8)
        for dq in s:
            row[data_index[dq]] = 1
        (hx_rows if m in h_targets else hz_rows).append(row)
    hx = dedup_rows(np.vstack(hx_rows))
    hz = dedup_rows(np.vstack(hz_rows))
    if np.any((hx @ hz.T) & 1):
        raise RuntimeError("Stim-extracted rotated code failed CSS check")
    logx = min(reduce_mod_rows(nullspace_gf2(hz), hx, n), key=lambda v: int(v.sum()))
    logz = min(reduce_mod_rows(nullspace_gf2(hx), hz, n), key=lambda v: int(v.sum()))
    return hx, hz, logx.astype(np.uint8), logz.astype(np.uint8)


def gauge_logical_z_strip(
    hx: np.ndarray, hz: np.ndarray, logz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Attach a thin algebraic strip that gauges a pure-Z logical support.

    This is the same path-strip construction used by the paper's finite gauging
    experiment. For an arbitrary CSS logical Z, even overlap with every X check
    lets us terminate/check the strip locally at the binary-algebra level.
    """
    hx = np.asarray(hx, dtype=np.uint8)
    hz = np.asarray(hz, dtype=np.uint8)
    logz = np.asarray(logz, dtype=np.uint8)
    n = hx.shape[1]
    support = [int(i) for i in np.flatnonzero(logz)]
    if not support:
        raise ValueError("empty logical-Z support")
    if np.any((hx @ logz) & 1):
        raise ValueError("provided logZ does not commute with Hx")

    m = len(support)
    nt = n + m
    hx_g = np.hstack([hx, np.zeros((hx.shape[0], m), dtype=np.uint8)])
    hz_g = np.hstack([hz, np.zeros((hz.shape[0], m), dtype=np.uint8)])
    pos = {q: j for j, q in enumerate(support)}

    new_z = []
    for j, q in enumerate(support):
        row = np.zeros(nt, dtype=np.uint8)
        row[q] = 1
        row[n + j] = 1
        new_z.append(row)

    new_x = []
    for j in range(m - 1):
        row = np.zeros(nt, dtype=np.uint8)
        row[n + j] = 1
        row[n + j + 1] = 1
        new_x.append(row)

    hx_def = []
    for row0 in hx_g:
        row = row0.copy()
        ov = [q for q in support if row[q]]
        if len(ov) % 2:
            raise RuntimeError("logical-Z commutation parity unexpectedly odd")
        # Pair overlap endpoints onto the strip. This keeps the deformed X row local
        # only when the supplied support is geometrically slab/path compatible; here
        # we audit the algebraic finite circuit, not a full CAD embedding.
        for a, b in zip(ov[0::2], ov[1::2]):
            row[n + pos[a]] ^= 1
            row[n + pos[b]] ^= 1
        hx_def.append(row)

    hx_def = dedup_rows(np.vstack(hx_def + new_x))
    hz_def = dedup_rows(np.vstack([hz_g] + new_z))
    # Drop residual anticommuting generators (same cleanup as manuscript
    # gauge_logical_Z_strip). Bare strip XX edges anticommute with ZZ joints.
    keep_x = [i for i, row in enumerate(hx_def) if not np.any((hz_def @ row) & 1)]
    hx_def = hx_def[keep_x] if keep_x else hx_def[:0].reshape(0, hx_def.shape[1])
    keep_z = [i for i, row in enumerate(hz_def) if not np.any((hx_def @ row) & 1)]
    hz_def = hz_def[keep_z] if keep_z else hz_def[:0].reshape(0, hz_def.shape[1])
    if np.any((hx_def @ hz_def.T) & 1):
        raise RuntimeError("gauged strip is not CSS after commuting-row cleanup")

    meas = np.zeros(nt, dtype=np.uint8)
    meas[:n] = logz
    if in_rowspace(hz_def, meas):
        # The protected effect should be nontrivial modulo stabilizers. If a specific
        # finite truncation makes this stabilizer-trivial, use a low-weight Z logical
        # of the deformed code and report the substitution in metadata.
        zlogs = reduce_mod_rows(nullspace_gf2(hx_def), hz_def, nt)
        if not zlogs:
            raise RuntimeError("gauged code has no nontrivial Z logical/effect")
        meas = min(zlogs, key=lambda v: int(v.sum())).astype(np.uint8)
        effect_note = "fallback_low_weight_deformed_Z_logical"
    else:
        effect_note = "input_logZ_product_on_data"

    meta = {
        "n_base": n,
        "n_strip": m,
        "n_total": nt,
        "effect_weight": int(meas.sum()),
        "effect_note": effect_note,
        "k_deformed": int(nt - gf2_rank(hx_def) - gf2_rank(hz_def)),
    }
    return hx_def, hz_def, meas, meta


# -----------------------------------------------------------------------------
# Stim circuit construction, decoding, and statistics.
# -----------------------------------------------------------------------------

def si1000_params(p: float) -> dict[str, float]:
    return {
        "p2": float(p),
        "p1": float(p) / 10.0,
        "p_meas": 5.0 * float(p),
        "p_reset": 2.0 * float(p),
        "p_idle": float(p) / 10.0,
        "p_idle_wait": 2.0 * float(p),
    }


def build_css_z_memory_circuit(
    hx: np.ndarray,
    hz: np.ndarray,
    logz: np.ndarray,
    rounds: int,
    p: float,
) -> stim.Circuit:
    """CSS Z-memory/protected-effect circuit with SI1000-like noise.

    Data start in |0>. Z stabilizers therefore have known first-round parity;
    X stabilizers are allowed a random initial eigenvalue and only get temporal
    difference detectors from round 2 onward.
    """
    hx = np.asarray(hx, dtype=np.uint8)
    hz = np.asarray(hz, dtype=np.uint8)
    logz = np.asarray(logz, dtype=np.uint8)
    n = hx.shape[1]
    if hz.shape[1] != n or logz.size != n:
        raise ValueError("dimension mismatch")
    if np.any((hx @ hz.T) & 1):
        raise ValueError("non-CSS matrices")
    if np.any((hx @ logz) & 1):
        raise ValueError("logZ/effect anticommutes with an X check")

    mz, mx = hz.shape[0], hx.shape[0]
    data = list(range(n))
    az = list(range(n, n + mz))
    ax = list(range(n + mz, n + mz + mx))
    pr = si1000_params(p)
    c = stim.Circuit()

    # Coordinates are schematic; detector geometry is carried by parity, not these labels.
    for q in data:
        c.append("QUBIT_COORDS", [q], [float(q), 0.0])
    for i, q in enumerate(az):
        c.append("QUBIT_COORDS", [q], [float(i), 1.0])
    for i, q in enumerate(ax):
        c.append("QUBIT_COORDS", [q], [float(i), 2.0])

    c.append("R", data + az + ax)
    if pr["p_reset"]:
        c.append("X_ERROR", data + az + ax, pr["p_reset"])
    c.append("TICK")

    stride = mz + mx
    for r in range(rounds):
        # Z checks.
        if pr["p_idle_wait"]:
            c.append("DEPOLARIZE1", data, pr["p_idle_wait"])
        for i, row in enumerate(hz):
            a = az[i]
            for q in np.flatnonzero(row):
                q = int(q)
                c.append("CX", [q, a])
                if pr["p2"]:
                    c.append("DEPOLARIZE2", [q, a], pr["p2"])
            if pr["p_meas"]:
                c.append("X_ERROR", [a], pr["p_meas"])
            c.append("MR", [a])
            if r == 0:
                c.append("DETECTOR", [stim.target_rec(-1)], [0.0, float(i), float(r)])
            else:
                c.append(
                    "DETECTOR",
                    [stim.target_rec(-1), stim.target_rec(-(1 + stride))],
                    [0.0, float(i), float(r)],
                )
        if pr["p_reset"] and az:
            c.append("X_ERROR", az, pr["p_reset"])
        c.append("TICK")
        if pr["p_idle"]:
            c.append("DEPOLARIZE1", data, pr["p_idle"])

        # X checks.
        if pr["p_idle_wait"]:
            c.append("DEPOLARIZE1", data, pr["p_idle_wait"])
        for i, row in enumerate(hx):
            a = ax[i]
            c.append("H", [a])
            if pr["p1"]:
                c.append("DEPOLARIZE1", [a], pr["p1"])
            for q in np.flatnonzero(row):
                q = int(q)
                c.append("CX", [a, q])
                if pr["p2"]:
                    c.append("DEPOLARIZE2", [a, q], pr["p2"])
            c.append("H", [a])
            if pr["p1"]:
                c.append("DEPOLARIZE1", [a], pr["p1"])
            if pr["p_meas"]:
                c.append("X_ERROR", [a], pr["p_meas"])
            c.append("MR", [a])
            if r > 0:
                c.append(
                    "DETECTOR",
                    [stim.target_rec(-1), stim.target_rec(-(1 + stride))],
                    [1.0, float(i), float(r)],
                )
        if pr["p_reset"] and ax:
            c.append("X_ERROR", ax, pr["p_reset"])
        c.append("TICK")
        if pr["p_idle"]:
            c.append("DEPOLARIZE1", data, pr["p_idle"])

    # Final destructive Z readout closes the Z-check world lines.
    if pr["p_meas"]:
        c.append("X_ERROR", data, pr["p_meas"])
    c.append("M", data)
    for i, row in enumerate(hz):
        # Last Z check is before the final round's X checks: mx records later than it.
        targets = [stim.target_rec(-(n + mx + mz - i))]
        for q in np.flatnonzero(row):
            targets.append(stim.target_rec(-(n - int(q))))
        c.append("DETECTOR", targets, [0.0, float(i), float(rounds)])
    obs = [stim.target_rec(-(n - int(q))) for q in np.flatnonzero(logz)]
    if not obs:
        raise ValueError("empty observable")
    c.append("OBSERVABLE_INCLUDE", obs, 0)
    return c


@dataclass
class CompiledStage:
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    matching: pymatching.Matching
    approximate_dem: bool
    decomposition_error: str | None


def compile_stage(circuit: stim.Circuit, allow_approx: bool) -> CompiledStage:
    """Compile a graphlike MWPM decoder, optionally allowing Stim decomposition gaps.

    Approximate mode is deliberately surfaced in the output. It is useful for the
    truncated-WB diagnostic but should not be mistaken for an exact hypergraph decoder.
    """
    err = None
    try:
        dem = circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
        )
        approx = False
    except Exception as exc:
        if not allow_approx:
            raise
        err = f"{type(exc).__name__}: {exc}"
        dem = circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
            ignore_decomposition_failures=True,
        )
        approx = True
    matching = pymatching.Matching.from_detector_error_model(dem)
    return CompiledStage(circuit, dem, matching, approx, err)


def sample_stage(stage: CompiledStage, shots: int, seed: int) -> np.ndarray:
    sampler = stage.circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    pred = stage.matching.decode_batch(dets)
    pred = np.asarray(pred, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    if pred.shape != obs.shape:
        raise RuntimeError(f"decoder/observable shape mismatch: {pred.shape} vs {obs.shape}")
    return np.any(pred ^ obs, axis=1)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def stage_stats(mask: np.ndarray) -> tuple[int, float, float, float]:
    k = int(np.count_nonzero(mask))
    p, lo, hi = wilson(k, len(mask))
    return k, p, lo, hi


# -----------------------------------------------------------------------------
# Experiments.
# -----------------------------------------------------------------------------

def simulate_primitive(d: int, p: float, shots: int, seed: int, allow_approx: bool) -> dict:
    hx, hz, _lx, lz = rotated_surface_code(d)
    ghx, ghz, effect, gmeta = gauge_logical_z_strip(hx, hz, lz)
    circuit = build_css_z_memory_circuit(ghx, ghz, effect, rounds=d, p=p)
    stage = compile_stage(circuit, allow_approx=allow_approx)
    mask = sample_stage(stage, shots, seed)
    fail, rate, lo, hi = stage_stats(mask)
    return {
        "experiment": "gauging_primitive",
        "family": "rotated_surface_code",
        "protocol": "explicit_gauged_strip",
        "size": d,
        "n_base": hx.shape[1],
        "n_simulated": ghx.shape[1],
        "d_upper": d,
        "rounds": d,
        "p": p,
        "shots": shots,
        "logical_failures": fail,
        "logical_error_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "idle_failures": -1,
        "gauge_failures": fail,
        "final_failures": -1,
        "gauge_rate": rate,
        "dem_detectors": int(stage.dem.num_detectors),
        "dem_errors": int(stage.dem.num_errors),
        "approximate_dem": stage.approximate_dem,
        "decomposition_error": stage.decomposition_error or "",
        "gauge_effect_weight": gmeta["effect_weight"],
        "scope": SCOPE_PRIMITIVE,
    }


def simulate_cas(
    inst: CodeInstance,
    p: float,
    shots: int,
    seed: int,
    protocol: str,
    allow_approx: bool,
) -> dict:
    base_circuit = build_css_z_memory_circuit(inst.hx, inst.hz, inst.logz, inst.rounds, p)
    base = compile_stage(base_circuit, allow_approx=allow_approx)
    idle = sample_stage(base, shots, seed)
    final = sample_stage(base, shots, seed + 104729)

    gauge = np.zeros(shots, dtype=bool)
    gauge_meta = {"n_total": inst.n, "effect_weight": int(inst.logz.sum())}
    gauge_stage = None
    if protocol == "stress":
        ghx, ghz, effect, gauge_meta = gauge_logical_z_strip(inst.hx, inst.hz, inst.logz)
        gauge_circuit = build_css_z_memory_circuit(ghx, ghz, effect, inst.rounds, p)
        gauge_stage = compile_stage(gauge_circuit, allow_approx=allow_approx)
        gauge = sample_stage(gauge_stage, shots, seed + 7919)
        joint = idle | gauge | final
        scope = SCOPE_STRESS
    elif protocol == "legacy":
        # Manuscript-style finite CAS: slab-Z/frame is tracked but the published
        # failure metric is set by the two protected memory/readout stages.
        joint = idle | final
        scope = SCOPE_LEGACY
    else:
        raise ValueError(protocol)

    fail, rate, lo, hi = stage_stats(joint)
    idle_f, idle_r, _, _ = stage_stats(idle)
    final_f, final_r, _, _ = stage_stats(final)
    gauge_f, gauge_r, _, _ = stage_stats(gauge)
    approx = base.approximate_dem or bool(gauge_stage and gauge_stage.approximate_dem)
    decomp_msgs = [m for m in [base.decomposition_error, getattr(gauge_stage, "decomposition_error", None)] if m]

    return {
        "experiment": "cas",
        "family": inst.family,
        "protocol": protocol,
        "size": inst.size,
        "n_base": inst.n,
        "n_simulated": int(gauge_meta["n_total"] if protocol == "stress" else inst.n),
        "d_upper": inst.d_upper,
        "rounds": inst.rounds,
        "p": p,
        "shots": shots,
        "logical_failures": fail,
        "logical_error_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "idle_failures": idle_f,
        "gauge_failures": gauge_f if protocol == "stress" else -1,
        "final_failures": final_f,
        "idle_rate": idle_r,
        "gauge_rate": gauge_r if protocol == "stress" else -1.0,
        "final_rate": final_r,
        "dem_detectors": int(base.dem.num_detectors),
        "dem_errors": int(base.dem.num_errors),
        "gauge_dem_detectors": int(gauge_stage.dem.num_detectors) if gauge_stage else -1,
        "gauge_dem_errors": int(gauge_stage.dem.num_errors) if gauge_stage else -1,
        "approximate_dem": approx,
        "decomposition_error": " || ".join(decomp_msgs),
        "gauge_effect_weight": int(gauge_meta["effect_weight"]),
        "scope": scope,
        "instance_note": inst.note,
    }


# -----------------------------------------------------------------------------
# Reporting.
# -----------------------------------------------------------------------------

def monotone_report(rows: list[dict], key_fields: tuple[str, ...]) -> str:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(r[k] for k in key_fields)
        groups.setdefault(key, []).append(r)
    lines = []
    for key, sub in sorted(groups.items(), key=lambda kv: str(kv[0])):
        lines.append(" | ".join(f"{k}={v}" for k, v in zip(key_fields, key)))
        sizes = sorted({int(r["size"]) for r in sub})
        by_p: dict[float, dict[int, dict]] = {}
        for r in sub:
            by_p.setdefault(float(r["p"]), {})[int(r["size"])] = r
        mono = []
        wsep = []
        for p in sorted(by_p):
            if p == 0:
                continue
            m = by_p[p]
            if not all(s in m for s in sizes):
                continue
            rates = [m[s]["logical_error_rate"] for s in sizes]
            ok = all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))
            if ok:
                mono.append(p)
                if all(m[b]["wilson_hi"] < m[a]["wilson_lo"] for a, b in zip(sizes, sizes[1:])):
                    wsep.append(p)
            lines.append(
                f"  p={p:.5g}: " + ", ".join(f"{s}={m[s]['logical_error_rate']:.5g}" for s in sizes)
                + ("  MONO" if ok else "")
            )
        lines.append(f"  monotone low-error points: {mono or 'none'}")
        lines.append(f"  Wilson-separated monotone points: {wsep or 'none'}")
        if mono:
            later = [p for p in sorted(by_p) if p > max(mono) and p not in mono]
            if later:
                lines.append(f"  finite-size crossing/break window: ~[{max(mono):.5g}, {min(later):.5g}]")
        lines.append("")
    return "\n".join(lines)


def save_csv(rows: list[dict], path: Path) -> None:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_rows(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped: {exc}")
        return

    # One figure per experiment/family/protocol to keep interpretation clean.
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["experiment"], r["family"], r["protocol"]), []).append(r)
    for (experiment, family, protocol), sub in groups.items():
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        for size in sorted({int(r["size"]) for r in sub}):
            s = sorted([r for r in sub if int(r["size"]) == size and float(r["p"]) > 0], key=lambda r: float(r["p"]))
            if not s:
                continue
            xs = [r["p"] for r in s]
            ys = [max(r["logical_error_rate"], 1e-7) for r in s]
            ylo = [max(r["logical_error_rate"] - r["wilson_lo"], 0.0) for r in s]
            yhi = [max(r["wilson_hi"] - r["logical_error_rate"], 0.0) for r in s]
            ax.errorbar(xs, ys, yerr=[ylo, yhi], marker="o", capsize=2, label=f"size={size}")
        ax.set_yscale("log")
        ax.set_xlabel("SI1000 two-qubit parameter p")
        ax.set_ylabel("Logical / protected-effect failure rate")
        ax.set_title(f"{experiment}: {family} / {protocol}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        stem = out_dir / f"{experiment}__{family}__{protocol}"
        fig.savefig(stem.with_suffix(".png"), dpi=220)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)


def dump_circuit(out_dir: Path, name: str, stage: CompiledStage) -> None:
    (out_dir / f"{name}.stim").write_text(str(stage.circuit), encoding="utf-8")
    (out_dir / f"{name}.dem").write_text(str(stage.dem), encoding="utf-8")


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def choose_preset(name: str) -> tuple[list[float], int]:
    if name == "quick":
        return [0.0, 0.001, 0.002, 0.003, 0.004, 0.005], 800
    if name == "paper":
        return [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010], 3000
    if name == "deep":
        return [0.0, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.006, 0.008, 0.010], 10000
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preset", choices=["quick", "paper", "deep"], default="quick")
    ap.add_argument("--experiment", choices=["primitive", "cas", "all"], default="all")
    ap.add_argument("--families", nargs="+", choices=["hp", "wb"], default=["hp"])
    ap.add_argument("--protocols", nargs="+", choices=["legacy", "stress"], default=["legacy", "stress"])
    ap.add_argument("--hp-sizes", default="3,5,7")
    ap.add_argument("--wb-sizes", default="3,4,5,6")
    ap.add_argument("--primitive-distances", default="3,5,7")
    ap.add_argument("--rounds-scale", type=int, choices=[1, 2], default=1)
    ap.add_argument("--shots", type=int, default=None, help="override preset shot count per point")
    ap.add_argument("--p-grid", default=None, help="comma-separated p values; overrides preset")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("stim_validation_results"))
    ap.add_argument(
        "--allow-approx-dem",
        action="store_true",
        help="if graphlike DEM decomposition fails, use Stim ignore_decomposition_failures=True and flag the result",
    )
    ap.add_argument("--dump-example-circuits", action="store_true")
    args = ap.parse_args()

    p_grid, preset_shots = choose_preset(args.preset)
    if args.p_grid:
        p_grid = parse_floats(args.p_grid)
    shots = args.shots or preset_shots
    hp_sizes = parse_ints(args.hp_sizes)
    wb_sizes = parse_ints(args.wb_sizes)
    prim_sizes = parse_ints(args.primitive_distances)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()

    metadata = {
        "timestamp_epoch": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "stim_version": getattr(stim, "__version__", "unknown"),
        "pymatching_version": getattr(pymatching, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "seed": args.seed,
        "preset": args.preset,
        "shots_per_point": shots,
        "p_grid": p_grid,
        "rounds_scale": args.rounds_scale,
        "allow_approx_dem": args.allow_approx_dem,
        "scope": "finite numerical validation only; no asymptotic theorem is inferred from crossings",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=== Stim paper validation ===")
    print(json.dumps({k: metadata[k] for k in ["stim_version", "pymatching_version", "numpy_version", "seed", "preset", "shots_per_point"]}, indent=2))
    print(f"output: {out_dir}")

    seed_counter = int(args.seed)

    if args.experiment in ("primitive", "all"):
        print("\n[1] Explicit path-strip gauging primitive")
        for d in prim_sizes:
            for p in p_grid:
                rec = simulate_primitive(d, p, shots if p > 0 else max(250, shots // 4), seed_counter, args.allow_approx_dem)
                seed_counter += 1009
                rows.append(rec)
                print(
                    f"primitive d={d} p={p:.5g} pL={rec['logical_error_rate']:.5g} "
                    f"[{rec['wilson_lo']:.5g},{rec['wilson_hi']:.5g}] "
                    f"fail={rec['logical_failures']}/{rec['shots']} approxDEM={rec['approximate_dem']}"
                )
                if p == 0 and rec["logical_failures"] != 0:
                    raise RuntimeError(f"NOISELESS FAILURE in primitive d={d}")

    if args.experiment in ("cas", "all"):
        print("\n[2] Canonical Architecture Schedule")
        instances: list[CodeInstance] = []
        if "hp" in args.families:
            instances.extend(build_hp_instance(L, args.rounds_scale) for L in hp_sizes)
        if "wb" in args.families:
            instances.extend(build_wb_instance(L, args.rounds_scale) for L in wb_sizes)

        for inst in instances:
            k = inst.n - gf2_rank(inst.hx) - gf2_rank(inst.hz)
            print(
                f"instance {inst.family} size={inst.size}: n={inst.n} k={k} "
                f"d_upper={inst.d_upper} R={inst.rounds} wt(logZ)={int(inst.logz.sum())}"
            )
            for protocol in args.protocols:
                for p in p_grid:
                    nshots = shots if p > 0 else max(250, shots // 4)
                    rec = simulate_cas(inst, p, nshots, seed_counter, protocol, args.allow_approx_dem)
                    seed_counter += 1009
                    rows.append(rec)
                    print(
                        f"  {protocol:6s} p={p:.5g} pL={rec['logical_error_rate']:.5g} "
                        f"[{rec['wilson_lo']:.5g},{rec['wilson_hi']:.5g}] "
                        f"idle={rec['idle_failures']} gauge={rec['gauge_failures']} "
                        f"final={rec['final_failures']} / {nshots} approxDEM={rec['approximate_dem']}"
                    )
                    if p == 0 and rec["logical_failures"] != 0:
                        raise RuntimeError(
                            f"NOISELESS FAILURE in {inst.family} size={inst.size} protocol={protocol}"
                        )

    save_csv(rows, out_dir / "stim_validation_results.csv")
    report = monotone_report(rows, ("experiment", "family", "protocol"))
    (out_dir / "crossing_report.txt").write_text(report, encoding="utf-8")
    plot_rows(rows, out_dir)

    if args.dump_example_circuits:
        print("\n[dump] writing representative .stim/.dem files")
        d = prim_sizes[0]
        hx, hz, _lx, lz = rotated_surface_code(d)
        ghx, ghz, effect, _ = gauge_logical_z_strip(hx, hz, lz)
        c = build_css_z_memory_circuit(ghx, ghz, effect, rounds=d, p=0.001)
        st = compile_stage(c, allow_approx=args.allow_approx_dem)
        dump_circuit(out_dir, f"primitive_d{d}_p0p001", st)
        if "hp" in args.families:
            inst = build_hp_instance(hp_sizes[0], args.rounds_scale)
            c = build_css_z_memory_circuit(inst.hx, inst.hz, inst.logz, inst.rounds, 0.001)
            st = compile_stage(c, allow_approx=args.allow_approx_dem)
            dump_circuit(out_dir, f"hp_size{inst.size}_memory_p0p001", st)

    metadata["elapsed_seconds"] = time.time() - t0
    metadata["n_rows"] = len(rows)
    metadata["approximate_dem_rows"] = int(sum(bool(r.get("approximate_dem")) for r in rows))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== finite-size report ===")
    print(report)
    print(f"wrote: {out_dir / 'stim_validation_results.csv'}")
    print(f"wrote: {out_dir / 'crossing_report.txt'}")
    print(f"elapsed: {metadata['elapsed_seconds']:.1f}s")
    print(
        "\nInterpretation rule: treat monotone/crossing behavior as finite-size evidence only. "
        "If the stress protocol disagrees with the legacy protocol, that is scientifically useful: "
        "it means the explicit protected-measurement circuit changes the finite-volume scaling and "
        "the manuscript's numerical claim should be scoped accordingly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
