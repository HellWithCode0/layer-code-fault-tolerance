#!/usr/bin/env python3
"""Circuit-level SI1000 MWPM for path-strip gauging of rotated SC logical Z.

Builds a DETERMINISTIC Stim circuit realizing the deformed surface-code /
lattice-surgery path-strip from gauging_rectangle_mwpm.gauge_logical_Z_strip
as a Z-basis memory of the deformed CSS code at d in {3,5,7}:

  - Data + strip qubits carry the deformed Hx/Hz stabilizers.
  - Each round: Z- then X-stabilizer extraction with dedicated measure qubits,
    MR, and DETECTOR annotations comparing consecutive rounds (stim pattern).
  - Final data Z measurements; detectors reconcile with last Z-stabilizers;
    OBSERVABLE_INCLUDE is the deformed logical Z supported on data+strip
    (commutes with final Z stabilizers; no ancilla-measure records in the
    observable — this was the root cause of prior non-determinism).

Noise: superconducting-inspired SI1000 (Gidney & Bacon / Honeycomb; tqec/stim
ratios): DEPOLARIZE2(p) after CX, DEPOLARIZE1(p/10) after H and idle,
measurement flip 5p, reset X_ERROR 2p, extra DEPOLARIZE1(2p) on data while
waiting for measure/reset.  Applied by inserting channels after each gate and
around MR.

Decode: circuit.detector_error_model(decompose_errors=True) +
pymatching.Matching.from_detector_error_model; Monte Carlo via
stim.CompiledDetectorSampler.

If SI1000 circuit construction fails determinism or DEM decomposition after
serious attempts, falls back to uniform depolarizing circuit-level noise on the
same deterministic strip memory circuit (still an upgrade over phenomenological
spacetime) and records the failure mode.

SCOPE: circuit-level gauging-primitive (deformed path-strip Z-memory) threshold
estimate under SI1000 (or documented fallback); NOT a layer-code or full
architecture schedule threshold.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim

from gauging_rectangle_mwpm import (
    gauge_logical_Z_strip,
    gf2_rank,
    nullspace_gf2,
    reduce_mod_rows,
    rotated_surface_code,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SEED = 20260918
DISTANCES = (3, 5, 7)
# Physical p grid for SI1000 (two-qubit parameter).
P_GRID = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
SCOPE_SI1000 = (
    "circuit-level gauging-primitive (deformed path-strip Z-memory) under SI1000 "
    "with DEM+MWPM; NOT a layer-code or full architecture schedule threshold"
)
SCOPE_FALLBACK = (
    "circuit-level gauging-primitive (deformed path-strip Z-memory) under uniform "
    "depolarizing circuit noise (SI1000 blocked — see failure_mode); NOT a "
    "layer-code or full architecture schedule threshold"
)


def si1000_params(p: float) -> dict:
    """Standard SI1000 channel strengths from two-qubit parameter p."""
    return {
        "p2": p,
        "p1": p / 10.0,
        "p_meas_flip": 5.0 * p,
        "p_reset": 2.0 * p,
        "p_idle_wait": 2.0 * p,
        "p_idle": p / 10.0,
    }


def logical_Z_deformed(Hx, Hz, meas):
    """Logical Z of the deformed strip: the ZZ-joint measurement operator.

    `meas` from gauge_logical_Z_strip is supported on data logZ plus matching
    strip qubits and commutes with Hx (X-stabilizers). Prefer it when valid;
    else min-weight ker(Hx)/row(Hz).
    """
    n = Hx.shape[1]
    zlogs = reduce_mod_rows(nullspace_gf2(Hx), Hz, n)
    v = np.asarray(meas, dtype=int)

    def is_log_z(u):
        return bool(np.all((Hx @ u) % 2 == 0) and gf2_rank(np.vstack([Hz, u])) > gf2_rank(Hz))

    if is_log_z(v):
        return v.astype(np.uint8), "strip_ZZ_joints"
    if zlogs:
        best = min(zlogs, key=lambda u: int(u.sum()))
        return best.astype(np.uint8), "min_weight_ker"
    raise RuntimeError("no logical Z found on deformed strip")


def _append_gate(circuit: stim.Circuit, name: str, targets, p_after: float | None, channel: str):
    circuit.append(name, targets)
    if p_after and p_after > 0:
        circuit.append(channel, targets, p_after)


def build_strip_memory_circuit(
    d: int,
    rounds: int,
    noise: str,
    p: float,
) -> tuple[stim.Circuit, dict]:
    """Deterministic deformed-strip Z-memory Stim circuit with detectors/observable."""
    Hx0, Hz0, logX0, logZ0 = rotated_surface_code(d)
    Hx, Hz, meas, meta = gauge_logical_Z_strip(Hx0, Hz0, logZ0)
    n = meta["n_tot"]
    logZ, log_note = logical_Z_deformed(Hx, Hz, meas)
    assert int(np.sum((Hx @ logZ) % 2)) == 0

    # Measure qubits: one per Z-check then one per X-check
    n_z = Hz.shape[0]
    n_x = Hx.shape[0]
    z_meas = list(range(n, n + n_z))
    x_meas = list(range(n + n_z, n + n_z + n_x))
    n_tot_q = n + n_z + n_x

    if noise == "si1000":
        pr = si1000_params(p)
    elif noise == "uniform_depolarizing":
        pr = {
            "p2": p,
            "p1": p,
            "p_meas_flip": p,
            "p_reset": p,
            "p_idle_wait": p,
            "p_idle": p,
        }
    elif noise == "none":
        pr = {k: 0.0 for k in ("p2", "p1", "p_meas_flip", "p_reset", "p_idle_wait", "p_idle")}
    else:
        raise ValueError(noise)

    circ = stim.Circuit()
    for q in range(n_tot_q):
        # Place data on a line; measure qubits above.
        if q < n:
            circ.append("QUBIT_COORDS", [q], [float(q), 0.0])
        else:
            circ.append("QUBIT_COORDS", [q], [float(q - n), 1.0])

    # Reset all qubits into |0> (Z-basis codespace + logical Z = +1 if logZ even weight...
    # For CSS Z-memory, |0>^n is stabilized by Z-stabilizers; logical Z eigenvalue
    # is +1 when logZ has even parity on |0>, which it does (all zeros).
    data_and_strip = list(range(n))
    all_meas = z_meas + x_meas
    circ.append("R", data_and_strip + all_meas)
    if pr["p_reset"] > 0:
        circ.append("X_ERROR", data_and_strip + all_meas, pr["p_reset"])
    circ.append("TICK")

    def measure_z_stabilizers(round_idx: int):
        """Measure Hz rows into z_meas qubits with CX schedule + SI1000."""
        # Idle-wait depolarization on data while measure qubits are being used
        if pr["p_idle_wait"] > 0:
            circ.append("DEPOLARIZE1", data_and_strip, pr["p_idle_wait"])
        for ci in range(n_z):
            m = z_meas[ci]
            supp = [j for j in range(n) if Hz[ci, j]]
            for q in supp:
                circ.append("CX", [q, m])
                if pr["p2"] > 0:
                    circ.append("DEPOLARIZE2", [q, m], pr["p2"])
        # Measurement flip then MR
        if pr["p_meas_flip"] > 0:
            circ.append("X_ERROR", z_meas, pr["p_meas_flip"])
        circ.append("MR", z_meas)
        if pr["p_reset"] > 0:
            circ.append("X_ERROR", z_meas, pr["p_reset"])

    def measure_x_stabilizers(round_idx: int):
        if pr["p_idle_wait"] > 0:
            circ.append("DEPOLARIZE1", data_and_strip, pr["p_idle_wait"])
        for ci in range(n_x):
            m = x_meas[ci]
            supp = [j for j in range(n) if Hx[ci, j]]
            circ.append("H", [m])
            if pr["p1"] > 0:
                circ.append("DEPOLARIZE1", [m], pr["p1"])
            for q in supp:
                circ.append("CX", [m, q])
                if pr["p2"] > 0:
                    circ.append("DEPOLARIZE2", [m, q], pr["p2"])
            circ.append("H", [m])
            if pr["p1"] > 0:
                circ.append("DEPOLARIZE1", [m], pr["p1"])
        if pr["p_meas_flip"] > 0:
            circ.append("X_ERROR", x_meas, pr["p_meas_flip"])
        circ.append("MR", x_meas)
        if pr["p_reset"] > 0:
            circ.append("X_ERROR", x_meas, pr["p_reset"])

    # rec indexing: each MR of k qubits appends k records.
    # After a full round we have produced n_z + n_x measurement records.
    # Track cumulative measurements for detector relative indices.
    # Stim rec[-1] is most recent.
    #
    # Round structure records per round: first n_z (Z), then n_x (X).
    # Detectors for Z-checks and X-checks separately.

    for r in range(rounds):
        if pr["p_idle"] > 0:
            circ.append("DEPOLARIZE1", data_and_strip, pr["p_idle"])
        circ.append("TICK")
        measure_z_stabilizers(r)
        circ.append("TICK")
        measure_x_stabilizers(r)
        circ.append("TICK")

        # Detectors (stim Z-memory pattern):
        #  - Round 0: ONLY Z-stabilizer detectors (deterministic on |0>^n).
        #    X-stabilizer outcomes are random on Z-basis product states — do NOT
        #    declare detectors for them on the first round; they set the frame.
        #  - Round r>0: Z and X detectors vs previous round.
        # After this round's MR, records are:
        #   rec[-(n_x)] ... rec[-1] = X measurements
        #   rec[-(n_x+n_z)] ... rec[-(n_x+1)] = Z measurements
        if r == 0:
            for i in range(n_z):
                circ.append(
                    "DETECTOR",
                    [stim.target_rec(-(n_x + n_z - i))],
                    [0.0, float(i), float(r)],
                )
        else:
            stride = n_z + n_x
            for i in range(n_z):
                cur = -(n_x + n_z - i)
                prev = cur - stride
                circ.append(
                    "DETECTOR",
                    [stim.target_rec(cur), stim.target_rec(prev)],
                    [0.0, float(i), float(r)],
                )
            for i in range(n_x):
                cur = -(n_x - i)
                prev = cur - stride
                circ.append(
                    "DETECTOR",
                    [stim.target_rec(cur), stim.target_rec(prev)],
                    [1.0, float(i), float(r)],
                )

    # Final data measurement (Z basis) — observable from data+strip only
    if pr["p_meas_flip"] > 0:
        circ.append("X_ERROR", data_and_strip, pr["p_meas_flip"])
    circ.append("M", data_and_strip)
    # Final Z-stabilizer detectors: each Hz row vs product of data measurements
    # After M, rec[-n + j] = data qubit j (j=0..n-1), i.e. rec[-(n-j)]
    # Last Z-stabilizer measurements are further back: after final M (n records),
    # the last round's Z meas start at rec[-(n + n_x + n_z)] ...
    for ci in range(n_z):
        targets = []
        # last round Z measurement for check ci:
        # order of last round records before final M: [Z0..Znz-1, X0..Xnx-1]
        # so Z_ci is rec[-(n + n_x + n_z - ci)]
        targets.append(stim.target_rec(-(n + n_x + n_z - ci)))
        for j in range(n):
            if Hz[ci, j]:
                targets.append(stim.target_rec(-(n - j)))
        circ.append("DETECTOR", targets, [0.0, float(ci), float(rounds)])

    # Observable: logical Z on data+strip — ONLY data measurement records
    obs_targets = []
    for j in range(n):
        if logZ[j]:
            obs_targets.append(stim.target_rec(-(n - j)))
    if not obs_targets:
        raise RuntimeError("empty observable")
    circ.append("OBSERVABLE_INCLUDE", obs_targets, 0)

    info = {
        "d": d,
        "n_data": meta["n_data"],
        "n_anc_strip": meta["n_anc"],
        "n_tot_data_strip": n,
        "n_z_checks": n_z,
        "n_x_checks": n_x,
        "n_qubits_circuit": n_tot_q,
        "rounds": rounds,
        "noise": noise,
        "p": p,
        "logZ_note": log_note,
        "logZ_weight": int(logZ.sum()),
        "k_deformed": int(n - gf2_rank(Hx) - gf2_rank(Hz)),
    }
    return circ, info


def assert_deterministic_observable(d: int, rounds: int, shots: int = 200) -> None:
    """p=0 circuit must yield zero detection events and zero observable flips."""
    circ, info = build_strip_memory_circuit(d, rounds, noise="none", p=0.0)
    # Also verify stim can build a DEM (no noise => empty-ish but valid)
    sampler = circ.compile_detector_sampler()
    dets, obs = sampler.sample(shots, separate_observables=True)
    if dets.any() or obs.any():
        raise RuntimeError(
            f"non-deterministic noiseless circuit at d={d}: "
            f"det_flips={int(dets.sum())} obs_flips={int(obs.sum())} info={info}"
        )


def try_build_noisy(d: int, rounds: int, noise: str, p: float):
    circ, info = build_strip_memory_circuit(d, rounds, noise=noise, p=p)
    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    return circ, dem, matching, info


def shot_budget(p: float, d: int) -> int:
    base = 20000 if p <= 0.003 else (12000 if p <= 0.006 else 8000)
    if d == 7:
        base = max(4000, base // 2)
    if d == 5:
        base = max(6000, (base * 3) // 4)
    return int(base)


def simulate(d: int, p: float, shots: int, noise: str, seed: int) -> dict:
    rounds = d
    circ, dem, matching, info = try_build_noisy(d, rounds, noise, p)
    sampler = circ.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    # obs shape (shots, n_obs); matching.decode_batch returns predicted obs flips
    preds = matching.decode_batch(dets)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    obs_u = obs.astype(np.uint8)
    pred_u = preds.astype(np.uint8)[:, : obs_u.shape[1]]
    fails = int(np.sum(np.any(pred_u != obs_u, axis=1)))
    rate, lo, hi = wilson_interval(fails, shots)
    scope = SCOPE_SI1000 if noise == "si1000" else SCOPE_FALLBACK
    return {
        "d": d,
        "p": float(p),
        "shots": shots,
        "rounds": rounds,
        "seed": seed,
        "noise_model": noise,
        "logical_failures": fails,
        "logical_error_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "n_detectors": circ.num_detectors,
        "n_dem_errors": dem.num_errors,
        "n_qubits": info["n_qubits_circuit"],
        "n_tot_data_strip": info["n_tot_data_strip"],
        "k_deformed": info["k_deformed"],
        "logZ_weight": info["logZ_weight"],
        "logZ_note": info["logZ_note"],
        "scope": scope,
    }


def plot_results(rows, out_stem: Path, noise: str):
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    markers = {3: "o", 5: "s", 7: "^"}
    for d in sorted({r["d"] for r in rows}):
        sub = sorted([r for r in rows if r["d"] == d], key=lambda r: r["p"])
        xs = [r["p"] for r in sub]
        ys = [max(r["logical_error_rate"], 1e-6) for r in sub]
        ylo = [max(r["logical_error_rate"] - r["wilson_lo"], 0.0) for r in sub]
        yhi = [max(r["wilson_hi"] - r["logical_error_rate"], 0.0) for r in sub]
        ax.errorbar(
            xs, ys, yerr=[ylo, yhi], fmt=f"{markers.get(d, 'o')}-",
            capsize=2, label=rf"$d={d}$",
        )
    xlab = r"SI1000 two-qubit parameter $p$" if noise == "si1000" else r"Uniform depolarizing $p$"
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"Logical error rate $p_L$")
    ax.set_yscale("log")
    title = "Gauging-strip circuit MWPM"
    if noise == "si1000":
        title += " (SI1000)"
    else:
        title += " (uniform circ.)"
    title += "\n(not arch. threshold)"
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def crossing_lines(rows) -> str:
    by_p = {}
    for r in rows:
        by_p.setdefault(r["p"], {})[r["d"]] = r["logical_error_rate"]
    mono = []
    lines = []
    for p in sorted(by_p):
        m = by_p[p]
        if not all(d in m for d in DISTANCES):
            continue
        lines.append(
            f"  p={p:.4f}: " + ", ".join(f"d{d}={m[d]:.4g}" for d in DISTANCES)
        )
        if m[3] > m[5] > m[7]:
            mono.append(p)
    if mono:
        lines.insert(0, f"threshold-like ordering d3>d5>d7 at p in {mono}")
    else:
        lines.insert(0, "NO full d3>d5>d7 monotonic crossing on this grid (report finite-size data honestly)")
    return "\n".join(lines)


def main() -> int:
    print(f"SEED={SEED}")
    failure_mode = ""
    noise_used = "si1000"

    # --- Phase 1: determinism checks (noiseless) ---
    print("Phase 1: noiseless determinism checks...")
    try:
        for d in DISTANCES:
            assert_deterministic_observable(d, rounds=d, shots=300)
            print(f"  d={d}: PASS noiseless deterministic observable/detectors")
    except Exception as e:
        failure_mode = f"noiseless determinism failure: {type(e).__name__}: {e}"
        print("FAIL", failure_mode)
        # Try to continue debugging / fallback path below
        noise_used = "uniform_depolarizing"

    # --- Phase 2: SI1000 DEM build ---
    if not failure_mode:
        print("Phase 2: SI1000 DEM build + decode smoke...")
        try:
            for d in DISTANCES:
                circ, dem, matching, info = try_build_noisy(d, d, "si1000", 0.001)
                # smoke sample
                samp = circ.compile_detector_sampler(seed=SEED)
                dets, obs = samp.sample(200, separate_observables=True)
                preds = matching.decode_batch(dets)
                print(
                    f"  d={d}: qubits={info['n_qubits_circuit']} dets={circ.num_detectors} "
                    f"dem_errors={dem.num_errors} smoke_obs_rate={obs.mean():.3g}"
                )
        except Exception as e:
            failure_mode = f"SI1000 DEM/decode failure: {type(e).__name__}: {e}"
            print("FAIL", failure_mode)
            noise_used = "uniform_depolarizing"

    if failure_mode:
        print(f"Falling back to uniform depolarizing circuit-level noise. Reason: {failure_mode}")
        # Re-check determinism already done; rebuild path uses uniform
        try:
            for d in DISTANCES:
                assert_deterministic_observable(d, rounds=d, shots=100)
                circ, dem, matching, info = try_build_noisy(d, d, "uniform_depolarizing", 0.001)
                print(f"  fallback d={d}: dem_errors={dem.num_errors} OK")
        except Exception as e:
            print(f"FATAL: even fallback failed: {type(e).__name__}: {e}")
            (ROOT / "gauging_rectangle_si1000_mwpm_FAILURE.txt").write_text(
                failure_mode + f"\nfallback also failed: {e}\n", encoding="utf-8"
            )
            return 1

    # --- Phase 3: Monte Carlo sweep ---
    print(f"Phase 3: Monte Carlo sweep with noise_model={noise_used}")
    rows = []
    t0 = time.time()
    seed_i = SEED
    for d in DISTANCES:
        for p in P_GRID:
            shots = shot_budget(p, d)
            # unique seed per point for reproducibility of the whole table
            rec = simulate(d, p, shots, noise_used, seed=seed_i)
            seed_i += 1
            if failure_mode:
                rec["si1000_failure_mode"] = failure_mode
            else:
                rec["si1000_failure_mode"] = ""
            rows.append(rec)
            print(
                f"d={d} p={p:.4f}: p_L={rec['logical_error_rate']:.4g} "
                f"[{rec['wilson_lo']:.4g},{rec['wilson_hi']:.4g}] "
                f"fails={rec['logical_failures']}/{shots}"
            )
    print(f"elapsed {time.time()-t0:.1f}s")

    out = ROOT / "gauging_rectangle_si1000_mwpm.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_results(rows, FIG / "gauging_rectangle_si1000_mwpm", noise_used)
    report = crossing_lines(rows)
    print("crossing scan:\n" + report)
    (ROOT / "gauging_rectangle_si1000_crossing.txt").write_text(
        f"noise_model={noise_used}\nfailure_mode={failure_mode}\n{report}\n",
        encoding="utf-8",
    )
    print(f"wrote {out.name} and figures/gauging_rectangle_si1000_mwpm.pdf/.png")
    print(f"SCOPE: {rows[0]['scope']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
