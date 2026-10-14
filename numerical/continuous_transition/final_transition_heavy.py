#!/usr/bin/env python3
"""Final high-value numerical falsification suite for
"Geometry-preserving fault-tolerant computation with three-dimensional layer codes".

This campaign is intended to be run *after* the stage-composed SI1000 campaigns.
Its main new test is a single continuous Stim circuit that switches

    base CSS memory -> algebraically gauged strip -> base CSS memory

without resetting the persistent data block between stages.  Transition detectors
are generated from exact operator identities between the pre-/post-switch check
bases.  In the X sector, each old check is equated to the corresponding modified
check times a product of strip checks; in the Z sector, persistent checks are
matched directly.  The script verifies these GF(2) identities before building the
circuit and requires an exact graphlike Stim detector error model for headline runs.

This is a much stronger *finite falsification test* of the code-deformation logic
than independent stage composition, but it is still not a microscopic CAD-complete
Williamson-Baspin T3/T4 implementation and cannot numerically prove an asymptotic
threshold theorem.

Recommended on an 8-core Xeon / 16 GB RAM host:

    pip install -r requirements-known-good.txt
    python final_transition_heavy.py --preset heavy --workers 4

For a multi-day final campaign:

    python final_transition_heavy.py --preset brutal --workers 4

Resume is automatic from per-point JSON checkpoints.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / "final_transition_results"
MASTER_SEED = 20260918


def stable_seed(*parts: Any, base: int = MASTER_SEED) -> int:
    h = hashlib.sha256((str(base) + "|" + "|".join(map(str, parts))).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = (z/den)*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, max(0.0, c-h), min(1.0, c+h)


def parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def import_deps():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import stim  # type: ignore
    import pymatching  # type: ignore
    import paper_stim_validation as psv  # type: ignore
    return stim, pymatching, psv


# -----------------------------------------------------------------------------
# Exact algebraic strip switch with transition identities.
# -----------------------------------------------------------------------------

def detailed_gauge(psv, hx: np.ndarray, hz: np.ndarray, logz: np.ndarray) -> dict:
    hx = np.asarray(hx, dtype=np.uint8)
    hz = np.asarray(hz, dtype=np.uint8)
    logz = np.asarray(logz, dtype=np.uint8)
    n = hx.shape[1]
    support = [int(i) for i in np.flatnonzero(logz)]
    if not support:
        raise ValueError("empty logical support")
    if np.any((hx @ logz) & 1):
        raise ValueError("logical Z anticommutes with Hx")

    m = len(support)
    nt = n + m
    pos = {q: j for j, q in enumerate(support)}

    # Persistent Z checks are exactly the old Z rows padded with zeros.
    hz_old_pad = np.hstack([hz, np.zeros((hz.shape[0], m), dtype=np.uint8)])
    new_z = []
    for j, q in enumerate(support):
        row = np.zeros(nt, dtype=np.uint8)
        row[q] = 1
        row[n+j] = 1
        new_z.append(row)
    new_z = np.vstack(new_z) if new_z else np.zeros((0, nt), dtype=np.uint8)

    # Strip X checks X_j X_{j+1}.
    strip_x = []
    for j in range(m-1):
        row = np.zeros(nt, dtype=np.uint8)
        row[n+j] = 1
        row[n+j+1] = 1
        strip_x.append(row)
    strip_x = np.vstack(strip_x) if strip_x else np.zeros((0, nt), dtype=np.uint8)

    # One modified X row for every old X row, preserving row correspondence.
    hx_old_pad = np.hstack([hx, np.zeros((hx.shape[0], m), dtype=np.uint8)])
    modified = []
    transition_paths: list[list[int]] = []
    for old in hx_old_pad:
        row = old.copy()
        ov_qubits = [q for q in support if row[q]]
        if len(ov_qubits) % 2:
            raise RuntimeError("odd overlap with logical-Z support")
        rel = np.zeros(max(0, m-1), dtype=np.uint8)
        for qa, qb in zip(ov_qubits[0::2], ov_qubits[1::2]):
            a, b = pos[qa], pos[qb]
            row[n+a] ^= 1
            row[n+b] ^= 1
            lo, hi = sorted((a, b))
            if hi > lo:
                rel[lo:hi] ^= 1
        modified.append(row)
        transition_paths.append([int(j) for j in np.flatnonzero(rel)])
    modified = np.vstack(modified) if modified else np.zeros((0, nt), dtype=np.uint8)

    # CSS memory X-stabilizers = deformed/modified rows only. Bare strip XX
    # anticommute with new ZZ joints; keep them separately and measure only on
    # entry/exit rounds that omit those joints.
    hx_g = modified.copy()
    hz_g = np.vstack([hz_old_pad, new_z]) if len(new_z) else hz_old_pad.copy()
    strip_x_mat = strip_x

    if np.any((hx_g @ hz_g.T) & 1):
        raise RuntimeError("detailed gauged code is not CSS")

    # Audit every transition operator identity exactly over GF(2).
    for i in range(hx.shape[0]):
        rhs = modified[i].copy()
        for j in transition_paths[i]:
            rhs ^= strip_x_mat[j]
        if not np.array_equal(rhs, hx_old_pad[i]):
            raise RuntimeError(f"X transition identity failed for row {i}")
    for i in range(hz.shape[0]):
        if not np.array_equal(hz_g[i], hz_old_pad[i]):
            raise RuntimeError(f"Z transition identity failed for row {i}")

    return {
        "n_base": n,
        "n_strip": m,
        "support": support,
        "hx_g": hx_g,
        "hz_g": hz_g,
        "hz_old_pad": hz_old_pad,
        "strip_x": strip_x_mat,
        "n_base_x": hx.shape[0],
        "n_base_z": hz.shape[0],
        "n_strip_x": int(strip_x_mat.shape[0]),
        "n_new_z": int(new_z.shape[0]),
        "transition_paths": transition_paths,
    }


def noise_params(p: float, profile: str) -> dict[str, float]:
    # Baseline follows the earlier paper harness.
    mult = {
        "baseline": (1.0, 1.0, 1.0, 1.0, 1.0),
        "meas2x":   (1.0, 2.0, 1.0, 1.0, 1.0),
        "idle2x":   (1.0, 1.0, 2.0, 1.0, 1.0),
        "twoq1p5x": (1.5, 1.0, 1.0, 1.0, 1.0),
        "reset2x":  (1.0, 1.0, 1.0, 2.0, 1.0),
    }
    if profile not in mult:
        raise ValueError(profile)
    m2, mm, mi, mr, m1 = mult[profile]
    return {
        "p2": min(0.499, p * m2),
        "p1": min(0.499, p/10 * m1),
        "p_meas": min(0.499, 5*p * mm),
        "p_reset": min(0.499, 2*p * mr),
        "p_idle": min(0.499, p/10 * mi),
        "p_idle_wait": min(0.499, 2*p * mi),
    }


class RecBook:
    def __init__(self, stim_mod):
        self.stim = stim_mod
        self.count = 0

    def one(self) -> int:
        idx = self.count
        self.count += 1
        return idx

    def rec(self, absolute_idx: int):
        return self.stim.target_rec(absolute_idx - self.count)


def build_continuous_switch_circuit(
    hx: np.ndarray,
    hz: np.ndarray,
    logz: np.ndarray,
    size: int,
    p: float,
    pre_rounds: int,
    gauge_rounds: int,
    post_rounds: int,
    profile: str,
):
    stim, _pym, psv = import_deps()
    g = detailed_gauge(psv, hx, hz, logz)
    hxg = g["hx_g"]
    hzg = g["hz_g"]
    hz_old_pad = g["hz_old_pad"]
    strip_x = g["strip_x"]
    n = g["n_base"]
    m = g["n_strip"]
    rz, rx = hz.shape[0], hx.shape[0]
    rgz, rgx = hzg.shape[0], hxg.shape[0]
    rsx = int(strip_x.shape[0])
    pr = noise_params(p, profile)

    # Qubit allocation.
    D = list(range(n))
    S = list(range(n, n+m))
    q = n + m
    BZ = list(range(q, q+rz)); q += rz
    BX = list(range(q, q+rx)); q += rx
    GZ = list(range(q, q+rgz)); q += rgz
    GX = list(range(q, q+rgx)); q += rgx
    SX = list(range(q, q+rsx)); q += rsx

    c = stim.Circuit()
    rb = RecBook(stim)
    for i, qq in enumerate(D):
        c.append("QUBIT_COORDS", [qq], [float(i), 0.0])
    for j, qq in enumerate(S):
        c.append("QUBIT_COORDS", [qq], [float(j), 0.5])

    def reset(qubits: list[int]):
        if not qubits:
            return
        c.append("R", qubits)
        if pr["p_reset"]:
            c.append("X_ERROR", qubits, pr["p_reset"])
        c.append("TICK")

    def idle_noise(data: list[int], wait: bool):
        pp = pr["p_idle_wait"] if wait else pr["p_idle"]
        if pp and data:
            c.append("DEPOLARIZE1", data, pp)

    def meas_z(rows: np.ndarray, anc: list[int], data_map: list[int]) -> list[int]:
        out = []
        idle_noise(data_map, True)
        for i, row in enumerate(rows):
            a = anc[i]
            for col in np.flatnonzero(row):
                dq = data_map[int(col)]
                c.append("CX", [dq, a])
                if pr["p2"]:
                    c.append("DEPOLARIZE2", [dq, a], pr["p2"])
            if pr["p_meas"]:
                c.append("X_ERROR", [a], pr["p_meas"])
            c.append("MR", [a])
            out.append(rb.one())
        if pr["p_reset"] and anc:
            c.append("X_ERROR", anc, pr["p_reset"])
        c.append("TICK")
        idle_noise(data_map, False)
        return out

    def meas_x(rows: np.ndarray, anc: list[int], data_map: list[int]) -> list[int]:
        out = []
        idle_noise(data_map, True)
        for i, row in enumerate(rows):
            a = anc[i]
            c.append("H", [a])
            if pr["p1"]:
                c.append("DEPOLARIZE1", [a], pr["p1"])
            for col in np.flatnonzero(row):
                dq = data_map[int(col)]
                c.append("CX", [a, dq])
                if pr["p2"]:
                    c.append("DEPOLARIZE2", [a, dq], pr["p2"])
            c.append("H", [a])
            if pr["p1"]:
                c.append("DEPOLARIZE1", [a], pr["p1"])
            if pr["p_meas"]:
                c.append("X_ERROR", [a], pr["p_meas"])
            c.append("MR", [a])
            out.append(rb.one())
        if pr["p_reset"] and anc:
            c.append("X_ERROR", anc, pr["p_reset"])
        c.append("TICK")
        idle_noise(data_map, False)
        return out

    def detector(abs_indices: list[int], xyz: tuple[float,float,float]):
        c.append("DETECTOR", [rb.rec(i) for i in abs_indices], list(xyz))

    # Persistent data begins in |0>; base syndrome ancillas are reset.
    reset(D + BZ + BX)
    prev_z = prev_x = None
    for r in range(pre_rounds):
        z = meas_z(hz, BZ, D)
        x = meas_x(hx, BX, D)
        if r == 0:
            for i, mi in enumerate(z):
                detector([mi], (0.0, float(i), float(r)))
        else:
            for i in range(rz):
                detector([prev_z[i], z[i]], (0.0, float(i), float(r)))
            for i in range(rx):
                detector([prev_x[i], x[i]], (1.0, float(i), float(r)))
        prev_z, prev_x = z, x

    # ENTRY: introduce strip + gauged extraction ancillas. Persistent data is NOT reset.
    # Strip XX is measured only on entry/exit rounds that omit new ZZ joints (CSS).
    reset(S + GZ + GX + SX)
    all_data = D + S
    gz_prev = gx_prev = sx_prev = None
    for gr in range(gauge_rounds):
        is_edge = (gr == 0) or (gr == gauge_rounds - 1)
        if is_edge:
            # Persistent Z only (no new ZZ joints) + modified X + strip XX.
            gz = meas_z(hz_old_pad, GZ[:rz], all_data)
            gx = meas_x(hxg, GX, all_data)
            sx = meas_x(strip_x, SX, all_data) if rsx else []
        else:
            gz = meas_z(hzg, GZ, all_data)
            gx = meas_x(hxg, GX, all_data)
            sx = sx_prev
        if gr == 0:
            for i in range(rz):
                detector([prev_z[i], gz[i]], (2.0, float(i), 0.0))
            # old_i = modified_i * product(strip_x[path])
            for i in range(rx):
                inds = [prev_x[i], gx[i]]
                inds += [sx[j] for j in g["transition_paths"][i]]
                detector(inds, (3.0, float(i), 0.0))
        else:
            # Memory detectors on CSS gauged stabilizers. Edge rounds only have rz Z rows.
            z_cmp = rz if is_edge else rgz
            for i in range(z_cmp):
                if gz_prev is not None and i < len(gz_prev):
                    detector([gz_prev[i], gz[i]], (2.0, float(i), float(gr)))
            for i in range(rgx):
                detector([gx_prev[i], gx[i]], (3.0, float(i), float(gr)))
        gz_prev, gx_prev, sx_prev = gz, gx, sx

    # EXIT: reset base syndrome ancillas, not persistent data. Strip is discarded.
    reset(BZ + BX)
    prev_z = prev_x = None
    for r in range(post_rounds):
        z = meas_z(hz, BZ, D)
        x = meas_x(hx, BX, D)
        if r == 0:
            for i in range(rz):
                detector([gz_prev[i], z[i]], (4.0, float(i), 0.0))
            for i in range(rx):
                inds = [x[i], gx_prev[i]]
                inds += [sx_prev[j] for j in g["transition_paths"][i]]
                detector(inds, (5.0, float(i), 0.0))
        else:
            for i in range(rz):
                detector([prev_z[i], z[i]], (4.0, float(i), float(r)))
            for i in range(rx):
                detector([prev_x[i], x[i]], (5.0, float(i), float(r)))
        prev_z, prev_x = z, x

    # Destructive Z readout closes persistent Z-check world lines and defines logical Z.
    if pr["p_meas"]:
        c.append("X_ERROR", D, pr["p_meas"])
    c.append("M", D)
    final_m = [rb.one() for _ in D]
    for i, row in enumerate(hz):
        inds = [prev_z[i]] + [final_m[int(col)] for col in np.flatnonzero(row)]
        detector(inds, (6.0, float(i), 0.0))
    obs = [rb.rec(final_m[int(col)]) for col in np.flatnonzero(logz)]
    c.append("OBSERVABLE_INCLUDE", obs, 0)

    meta = {
        "size": size,
        "n_base": n,
        "n_strip": m,
        "n_total_data_during_gauge": n+m,
        "base_x_checks": rx,
        "base_z_checks": rz,
        "gauged_x_checks": rgx,
        "gauged_z_checks": rgz,
        "pre_rounds": pre_rounds,
        "gauge_rounds": gauge_rounds,
        "post_rounds": post_rounds,
        "noise_profile": profile,
        "transition_identity_audit": True,
        "scope": "continuous algebraic base->gauged-strip->base code switch; not microscopic WB CAD T3/T4",
    }
    return c, meta


@dataclass(frozen=True)
class Task:
    size: int
    p: float
    gauge_scale: int
    profile: str
    min_shots: int
    max_shots: int
    batch: int
    target_failures: int
    rel_ci: float
    abs_ci: float
    noiseless_shots: int
    seed: int

    @property
    def key(self) -> str:
        ps = f"{self.p:.7g}".replace(".", "p")
        return f"L{self.size}__p{ps}__G{self.gauge_scale}d__{self.profile}"


def stop(task: Task, failures: int, shots: int) -> str | None:
    if task.p == 0:
        return "noiseless" if shots >= task.noiseless_shots else None
    if shots < task.min_shots:
        return None
    rate, lo, hi = wilson(failures, shots)
    half = (hi-lo)/2
    if failures >= task.target_failures and half <= max(task.abs_ci, task.rel_ci*max(rate, 1e-12)):
        return "precision"
    if shots >= task.max_shots:
        return "max_shots"
    return None


def run_task(task: Task, checkpoint_dir: str) -> dict:
    checkpoint = Path(checkpoint_dir) / f"{task.key}.json"
    if checkpoint.exists():
        old = json.loads(checkpoint.read_text())
        if old.get("complete"):
            return old

    stim, pymatching, psv = import_deps()
    inst = psv.build_hp_instance(task.size, rounds_scale=1)
    pre = max(2, inst.d_upper)
    post = max(2, inst.d_upper)
    gauge = max(2, task.gauge_scale * inst.d_upper)
    circ, meta = build_continuous_switch_circuit(
        inst.hx, inst.hz, inst.logz, task.size, task.p,
        pre_rounds=pre, gauge_rounds=gauge, post_rounds=post, profile=task.profile,
    )

    # Headline results require exact graphlike detector decomposition.
    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    shots = 0
    failures = 0
    batches = 0
    t0 = time.time()
    while True:
        reason = stop(task, failures, shots)
        if reason:
            break
        todo = task.noiseless_shots - shots if task.p == 0 else min(task.batch, task.max_shots - shots)
        if todo <= 0:
            break
        seed = stable_seed(task.key, batches, base=task.seed)
        sampler = circ.compile_detector_sampler(seed=seed)
        det, obs = sampler.sample(shots=todo, separate_observables=True)
        pred = matching.decode_batch(det)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        bad = np.any(pred != obs, axis=1)
        failures += int(np.count_nonzero(bad))
        shots += int(todo)
        batches += 1

    rate, lo, hi = wilson(failures, shots)
    rec = {
        "complete": True,
        **asdict(task),
        **meta,
        "shots": shots,
        "failures": failures,
        "logical_error_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "stop_reason": stop(task, failures, shots) or "budget",
        "detectors": int(circ.num_detectors),
        "observables": int(circ.num_observables),
        "circuit_qubits": int(circ.num_qubits),
        "dem_errors": int(dem.num_errors),
        "exact_graphlike_dem": True,
        "elapsed_s": time.time()-t0,
    }
    if task.p == 0 and failures:
        rec["FATAL_NOISELESS_FAILURE"] = True
    atomic_json(checkpoint, rec)
    return rec


def crossing_interp(rows: list[dict], a: int, b: int) -> float | None:
    pts = {}
    for r in rows:
        if int(r["size"]) in (a,b) and float(r["p"]) > 0:
            pts.setdefault(float(r["p"]), {})[int(r["size"])] = float(r["logical_error_rate"])
    prev = None
    for p in sorted(pts):
        if a not in pts[p] or b not in pts[p]:
            continue
        delta = pts[p][a] - pts[p][b]
        if prev is not None and delta * prev[1] <= 0 and delta != prev[1]:
            p0, d0 = prev
            return p0 + (p-p0) * (-d0)/(delta-d0)
        prev = (p, delta)
    return None


def report(rows: list[dict], out: Path) -> None:
    lines = [
        "# Final continuous-transition numerical report",
        "",
        "This report summarizes finite circuit-level falsification tests. It does not numerically prove the asymptotic theorem and the transition model is algebraic rather than a microscopic Williamson-Baspin CAD T3/T4 circuit.",
        "",
    ]
    for profile in sorted({r["profile"] for r in rows}):
        for gs in sorted({int(r["gauge_scale"]) for r in rows if r["profile"] == profile}):
            sub = [r for r in rows if r["profile"] == profile and int(r["gauge_scale"]) == gs]
            sizes = sorted({int(r["size"]) for r in sub})
            lines += [f"## profile={profile}, gauged rounds={gs}d", ""]
            byp = {}
            for r in sub:
                byp.setdefault(float(r["p"]), {})[int(r["size"])] = r
            mono, separated = [], []
            for p in sorted(byp):
                m = byp[p]
                if not all(s in m for s in sizes):
                    continue
                rates = [float(m[s]["logical_error_rate"]) for s in sizes]
                ok = all(rates[i] > rates[i+1] for i in range(len(rates)-1)) if p > 0 else True
                sep = p > 0 and ok and all(float(m[sizes[i+1]]["wilson_hi"]) < float(m[sizes[i]]["wilson_lo"]) for i in range(len(sizes)-1))
                if p > 0 and ok: mono.append(p)
                if sep: separated.append(p)
                lines.append(f"- p={p:.6g}: " + ", ".join(f"L{s}={m[s]['logical_error_rate']:.6g}" for s in sizes) + ("  MONO" if ok and p>0 else "") + ("  WILSON-SEPARATED" if sep else ""))
            x = crossing_interp(sub, min(sizes), max(sizes)) if len(sizes) >= 2 else None
            lines += [f"- monotone points: {mono or 'none'}", f"- Wilson-separated monotone points: {separated or 'none'}", f"- smallest-vs-largest linear crossing diagnostic: {x if x is not None else 'none'}", ""]
    (out / "FINAL_CONTINUOUS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    for profile in sorted({r["profile"] for r in rows}):
        for gs in sorted({int(r["gauge_scale"]) for r in rows if r["profile"] == profile}):
            sub = [r for r in rows if r["profile"] == profile and int(r["gauge_scale"]) == gs and float(r["p"]) > 0]
            if not sub:
                continue
            fig, ax = plt.subplots(figsize=(5.2, 3.7))
            for size in sorted({int(r["size"]) for r in sub}):
                rr = sorted([r for r in sub if int(r["size"]) == size], key=lambda x: float(x["p"]))
                xs = [float(r["p"]) for r in rr]
                ys = [max(float(r["logical_error_rate"]), 1e-8) for r in rr]
                ylo = [max(float(r["logical_error_rate"])-float(r["wilson_lo"]),0) for r in rr]
                yhi = [max(float(r["wilson_hi"])-float(r["logical_error_rate"]),0) for r in rr]
                ax.errorbar(xs, ys, yerr=[ylo,yhi], marker="o", capsize=2, label=f"L={size}")
            ax.set_yscale("log")
            ax.set_xlabel("SI1000 two-qubit parameter p")
            ax.set_ylabel("End-to-end logical failure rate")
            ax.set_title(f"Continuous base→gauge→base, {profile}, gauge={gs}d")
            ax.grid(True, which="both", alpha=.25)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            stem = out / f"continuous__{profile}__G{gs}d"
            fig.savefig(stem.with_suffix(".png"), dpi=220)
            fig.savefig(stem.with_suffix(".pdf"))
            plt.close(fig)


def preset(name: str):
    if name == "sanity":
        return dict(
            sizes=[3,5], p=[0.0,0.0025,0.0045], gauge_scales=[1], profiles=["baseline"],
            min_shots=1000, max_shots=3000, batch=1000, target_failures=50, rel_ci=.35, abs_ci=.01, noiseless=500,
        )
    if name == "heavy":
        return dict(
            sizes=[3,5,7,9,11],
            p=[0.0,0.0015,0.0025,0.0035,0.0040,0.00425,0.0045,0.00475,0.0050,0.0055],
            gauge_scales=[1,2,3], profiles=["baseline"],
            min_shots=8000, max_shots=80000, batch=4000, target_failures=300, rel_ci=.15, abs_ci=.0025, noiseless=2000,
        )
    if name == "brutal":
        return dict(
            sizes=[3,5,7,9,11,13],
            p=[0.0,0.0010,0.0015,0.0020,0.0025,0.0030,0.0035,0.00375,0.0040,0.0042,0.0044,0.0046,0.0048,0.0050,0.00525,0.0055,0.0060],
            gauge_scales=[1,2,3,4], profiles=["baseline"],
            min_shots=15000, max_shots=200000, batch=5000, target_failures=600, rel_ci=.10, abs_ci=.0015, noiseless=4000,
        )
    if name == "robustness":
        return dict(
            sizes=[3,5,7,9,11],
            p=[0.0,0.0025,0.0035,0.0040,0.0045,0.0050],
            gauge_scales=[2], profiles=["baseline","meas2x","idle2x","twoq1p5x","reset2x"],
            min_shots=8000, max_shots=60000, batch=4000, target_failures=300, rel_ci=.15, abs_ci=.0025, noiseless=1500,
        )
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preset", choices=["sanity","heavy","brutal","robustness"], default="sanity")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--p-grid", default=None)
    ap.add_argument("--gauge-scales", default=None)
    ap.add_argument("--profiles", default=None, help="comma-separated: baseline,meas2x,idle2x,twoq1p5x,reset2x")
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    args = ap.parse_args()

    cfg = preset(args.preset)
    if args.sizes: cfg["sizes"] = parse_ints(args.sizes)
    if args.p_grid: cfg["p"] = parse_floats(args.p_grid)
    if args.gauge_scales: cfg["gauge_scales"] = parse_ints(args.gauge_scales)
    if args.profiles: cfg["profiles"] = [x.strip() for x in args.profiles.split(",") if x.strip()]

    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    ck = out / "checkpoints"; ck.mkdir(exist_ok=True)
    meta = {
        "preset": args.preset, "config": cfg, "python": sys.version, "platform": platform.platform(),
        "workers": args.workers, "seed": args.seed,
        "scope": "continuous algebraic transition-detector falsification; not microscopic WB CAD T3/T4 and not asymptotic threshold proof",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    tasks = [Task(size=L,p=p,gauge_scale=gs,profile=prof,
                  min_shots=cfg["min_shots"],max_shots=cfg["max_shots"],batch=cfg["batch"],
                  target_failures=cfg["target_failures"],rel_ci=cfg["rel_ci"],abs_ci=cfg["abs_ci"],
                  noiseless_shots=cfg["noiseless"],seed=args.seed)
             for prof in cfg["profiles"] for gs in cfg["gauge_scales"] for L in cfg["sizes"] for p in cfg["p"]]

    print(f"Final continuous-transition campaign: {len(tasks)} points, workers={args.workers}")
    rows: list[dict] = []
    fatal = False
    with cf.ProcessPoolExecutor(max_workers=max(1,args.workers)) as ex:
        futs = {ex.submit(run_task, t, str(ck)): t for t in tasks}
        for fut in cf.as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result(); rows.append(r)
                print(f"{t.key}: {r['failures']}/{r['shots']} pL={r['logical_error_rate']:.6g} [{r['wilson_lo']:.6g},{r['wilson_hi']:.6g}] {r['stop_reason']}")
                if r.get("FATAL_NOISELESS_FAILURE"):
                    fatal = True
            except Exception as exc:
                fatal = True
                print(f"FAILED {t.key}: {exc}")
                traceback.print_exc()

    # Reload all complete checkpoints so resumed runs produce complete aggregate output.
    rows = []
    for pth in sorted(ck.glob("*.json")):
        try:
            r = json.loads(pth.read_text())
            if r.get("complete"): rows.append(r)
        except Exception:
            pass
    rows.sort(key=lambda r: (r["profile"], int(r["gauge_scale"]), int(r["size"]), float(r["p"])))
    write_csv(out / "all_points.csv", rows)
    report(rows, out)
    make_plots(rows, out)

    # Hard failure if a noiseless point failed or if expected results are missing.
    if any(float(r["p"]) == 0 and int(r["failures"]) != 0 for r in rows):
        fatal = True
    (out / "STATUS.txt").write_text(
        ("FAIL\n" if fatal else "PASS\n") +
        "PASS means scripts completed and all sampled noiseless points had zero logical failures. It is not a proof of the paper theorem.\n",
        encoding="utf-8"
    )
    return 2 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
