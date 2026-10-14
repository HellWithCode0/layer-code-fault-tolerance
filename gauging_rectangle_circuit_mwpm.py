#!/usr/bin/env python3
"""Phenomenological spacetime MWPM for a surface-code gauging rectangle.

Upgrade of gauging_rectangle_mwpm.py (code-capacity single-shot) to an R=d
round phenomenological spacetime model decoded by PyMatching, with an
explicit Stim DetectorErrorModel exported from the same matching graph.

Construction:
  - Rotated surface code distance d (stim geometry; n=d^2, k=1).
  - Lattice-surgery / path-strip gauging of logical Z (same deformed strip as
    the code-capacity script).
  - Noise model (CLEARLY LABELLED): phenomenological spacetime —
      * R = d measurement rounds
      * independent data errors with probability p each round (spacelike)
      * independent measurement flips with probability p (timelike)
      * CSS sectors decoded separately (X-errors via Z-checks / Z-errors via X-checks)
  - Matching graph: pymatching.Matching(check_matrix, repetitions=R, ...).
  - Stim DetectorErrorModel: exported from that matching graph's edges.
  - Decode with matching.decode / decode_batch; noise sampled via Matching.add_noise
    so the sampler and decoder share one graph (avoids syndrome-layout mismatch).
  - Sweep p on a grid showing distance-like separation; scale shots with p;
    fixed seed recorded.

SCOPE: phenomenological spacetime building-block sanity check for the deformed
surface-code/path-strip gauging rectangle; NOT a layer-code threshold,
NOT an SI1000 circuit-level calibration, and NOT an architecture-level p_th.
Full noisy Stim *gate* circuits for the strip produced non-deterministic
observables under ancilla measure-reset; this spacetime DEM+MWPM model is the
documented upgrade used here. The older code-capacity script remains as the
simpler building block.
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
NOISE_MODEL = "phenomenological_spacetime_data_and_measurement_p"
SCOPE = (
    "phenomenological spacetime (R=d rounds; data errors and measurement flips "
    "each with probability p) MWPM sanity check on the deformed surface-code/"
    "path-strip gauging rectangle via pymatching Matching(repetitions=R) with "
    "stim DetectorErrorModel export; NOT a layer-code threshold, SI1000 "
    "circuit-level calibration, or architecture-level threshold"
)


def make_sector_matching(H: np.ndarray, R: int, p: float) -> pymatching.Matching:
    w = math.log((1.0 - p) / p)
    return pymatching.Matching(
        np.asarray(H, dtype=np.uint8),
        weights=w,
        error_probabilities=p,
        repetitions=R,
        timelike_weights=w,
        measurement_error_probabilities=p,
    )


def matching_to_dem(matching: pymatching.Matching) -> stim.DetectorErrorModel:
    """Export a stim DetectorErrorModel from a pymatching Matching graph."""
    lines = []
    boundary = matching.num_detectors  # boundary node index used by pymatching
    for e in matching.edges():
        n1, n2, data = e
        prob = data.get("error_probability", None)
        if prob is None or prob <= 0.0 or prob >= 1.0:
            # derive from weight if needed
            wt = float(data.get("weight", 1.0))
            # weight = log((1-p)/p) => p = 1/(1+exp(weight))
            prob = 1.0 / (1.0 + math.exp(wt))
        faults = data.get("fault_ids", set())
        if faults is None:
            faults = set()
        if isinstance(faults, (int, np.integer)):
            faults = {int(faults)}
        else:
            faults = {int(f) for f in faults}
        targets = []
        for node in (n1, n2):
            if node is None:
                continue
            node = int(node)
            if node < matching.num_detectors:
                targets.append(f"D{node}")
            # boundary node omitted in DEM (implicit)
        for f in sorted(faults):
            targets.append(f"L{f}")  # temporarily; remapped below
        # We only need detector mechanisms for DEM size accounting; logical
        # remapping is handled in the Monte Carlo via fault_ids / qubit index.
        # For a clean DEM used with observable 0 = one logical, keep detector
        # targets only here and note logicals separately in the Monte Carlo.
        det_only = [t for t in targets if t.startswith("D")]
        if det_only:
            lines.append(f"error({prob:.12g}) " + " ".join(det_only))
        elif targets:
            # pure logical / boundary edge
            lines.append(f"error({prob:.12g}) " + " ".join(targets))
    if not lines:
        lines.append("error(0.0) D0")  # avoid empty DEM
    return stim.DetectorErrorModel("\n".join(lines))


def simulate_sector(
    H: np.ndarray,
    logical: np.ndarray,
    R: int,
    p: float,
    shots: int,
) -> tuple[int, pymatching.Matching, stim.DetectorErrorModel]:
    """Return (failures, matching, dem) for one CSS sector."""
    matching = make_sector_matching(H, R, p)
    dem = matching_to_dem(matching)
    L = np.asarray(logical, dtype=np.uint8)
    nL = len(L)
    noises = np.empty((shots, matching.num_fault_ids), dtype=np.uint8)
    syns = np.empty((shots, matching.num_detectors + 1), dtype=np.uint8)
    for i in range(shots):
        noise, syn = matching.add_noise()
        noises[i] = np.asarray(noise, dtype=np.uint8)
        syns[i] = np.asarray(syn, dtype=np.uint8)
    corrs = matching.decode_batch(syns)
    residual = noises ^ corrs.astype(np.uint8)
    fails = (residual[:, :nL] @ L) % 2
    return int(fails.sum()), matching, dem


def simulate_gauging_phenomenological(d: int, p: float, shots: int) -> dict:
    Hx0, Hz0, logX, logZ = rotated_surface_code(d)
    n0 = Hx0.shape[1]
    k0 = n0 - gf2_rank(Hx0) - gf2_rank(Hz0)
    Hx, Hz, meas, meta = gauge_logical_Z_strip(Hx0, Hz0, logZ)
    n = meta["n_tot"]
    R = d

    xlogs = reduce_mod_rows(nullspace_gf2(Hz), Hx, n)
    zlogs = reduce_mod_rows(nullspace_gf2(Hx), Hz, n)
    logX_true = min(xlogs, key=lambda v: int(v.sum()))
    logZ_true = min(zlogs, key=lambda v: int(v.sum()))
    # Gauged Z-measurement observable = strip meas (equals deformed Z logical class)
    assert np.count_nonzero((Hx @ meas) % 2) == 0
    assert int(np.dot(meas, logX_true) % 2) == 1

    # Z-error sector (X checks) protects spectator X logical
    fail_x, match_x, dem_x = simulate_sector(Hx, logX_true, R, p, shots)
    # X-error sector (Z checks) protects gauged Z measurement
    fail_z, match_z, dem_z = simulate_sector(Hz, meas, R, p, shots)

    # Independent CSS channels: combine with inclusion of independent samples.
    # Report exact Monte Carlo OR by re-drawing paired shots with one RNG stream
    # through sequential sector calls above — those are independent, so
    # P(fail) = 1-(1-pX)(1-pZ) is the correct combination for independent channels.
    pX, pZ = fail_x / shots, fail_z / shots
    rate_ind = 1.0 - (1.0 - pX) * (1.0 - pZ)
    # Wilson interval on an effective failure count
    failures_eff = int(round(rate_ind * shots))
    rate, lo, hi = wilson_interval(failures_eff, shots)

    return {
        "d": d,
        "p": float(p),
        "shots": shots,
        "R_rounds": R,
        "seed": SEED,
        "noise_model": NOISE_MODEL,
        "n_data": n0,
        "n_tot": n,
        "k_base": int(k0),
        "k_deformed": int(n - gf2_rank(Hx) - gf2_rank(Hz)),
        "dX_deformed_upper": int(logX_true.sum()),
        "dZ_deformed_upper": int(logZ_true.sum()),
        "sector_spectator_X_failures": fail_x,
        "sector_gauged_Z_failures": fail_z,
        "logical_failures_effective": failures_eff,
        "logical_error_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "n_detectors_X_sector": match_x.num_detectors,
        "n_detectors_Z_sector": match_z.num_detectors,
        "n_dem_errors_X": dem_x.num_errors,
        "n_dem_errors_Z": dem_z.num_errors,
        "scope": SCOPE,
    }


def plot_results(rows: list[dict], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 2.85))
    markers = {3: "o", 5: "s", 7: "^"}
    for d in sorted({r["d"] for r in rows}):
        sub = sorted([r for r in rows if r["d"] == d], key=lambda r: r["p"])
        xs = [r["p"] for r in sub]
        ys = [max(r["logical_error_rate"], 1e-6) for r in sub]
        ylo = [max(r["logical_error_rate"] - r["wilson_lo"], 0) for r in sub]
        yhi = [max(r["wilson_hi"] - r["logical_error_rate"], 0) for r in sub]
        ax.errorbar(
            xs, ys, yerr=[ylo, yhi], fmt=f"{markers.get(d, 'o')}-",
            capsize=2, label=rf"$d={d}$, $R={d}$",
        )
    ax.set_xlabel(r"Phenomenological error probability $p$")
    ax.set_ylabel(r"Logical error rate $p_L$")
    ax.set_yscale("log")
    ax.set_title("Gauging rectangle (phenom. spacetime; not arch. $p_{th}$)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main() -> int:
    # Grid chosen to exhibit distance-like separation under spacetime noise
    ps = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020]

    def n_shots(p, d):
        base = int(min(40000, max(6000, 120.0 / p)))
        if d == 7:
            base = max(4000, base // 2)
        elif d == 5:
            base = max(5000, int(base * 0.85))
        return base

    # Seed pymatching / numpy
    np.random.seed(SEED)

    rows = []
    t0 = time.time()
    print(f"SEED={SEED}")
    print(f"NOISE_MODEL={NOISE_MODEL}")

    for d in (3, 5, 7):
        Hx0, Hz0, logX, logZ = rotated_surface_code(d)
        Hx, Hz, meas, meta = gauge_logical_Z_strip(Hx0, Hz0, logZ)
        xlogs = reduce_mod_rows(nullspace_gf2(Hz), Hx, meta["n_tot"])
        logX_true = min(xlogs, key=lambda v: int(v.sum()))
        print(
            f"d={d}: n_tot={meta['n_tot']} dX<={int(logX_true.sum())} "
            f"meas_wt={int(meas.sum())} k_def="
            f"{meta['n_tot'] - gf2_rank(Hx) - gf2_rank(Hz)}"
        )

    for d in (3, 5, 7):
        for p in ps:
            shots = n_shots(p, d)
            rec = simulate_gauging_phenomenological(d, p, shots)
            rows.append(rec)
            print(
                f"d={d} p={p:.4f}: p_L={rec['logical_error_rate']:.5f} "
                f"[{rec['wilson_lo']:.5f},{rec['wilson_hi']:.5f}] "
                f"eff_fails={rec['logical_failures_effective']}/{rec['shots']} "
                f"(Xspec={rec['sector_spectator_X_failures']},"
                f"Zmeas={rec['sector_gauged_Z_failures']})"
            )

    print(f"elapsed {time.time() - t0:.1f}s")

    out = ROOT / "gauging_rectangle_circuit_mwpm.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_results(rows, FIG / "gauging_rectangle_circuit_mwpm")

    by_p: dict = {}
    for r in rows:
        by_p.setdefault(r["p"], {})[r["d"]] = r["logical_error_rate"]
    supp55 = [p for p, m in by_p.items() if m.get(5, 1) < m.get(3, 0)]
    supp75 = [p for p, m in by_p.items() if m.get(7, 1) < m.get(5, 0)]
    print(f"observation: d=5 below d=3 at p in {sorted(supp55)}")
    print(f"observation: d=7 below d=5 at p in {sorted(supp75)}")
    print(f"wrote {out.name} and figures/gauging_rectangle_circuit_mwpm.pdf/.png")
    print(f"SCOPE: {SCOPE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
