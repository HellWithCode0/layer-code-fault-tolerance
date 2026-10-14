#!/usr/bin/env python3
"""Gap 7: Coherent / leakage exploratory probes vs SI1000 Pauli baseline.

- Coherent over-rotation on rotated-SC Z-memory: compare Pauli-twirled
  effective noise vs coherent ZZ/X small rotations via Stim.
- Leakage toy: mark leaked data qubits as hard errors / transport to neighbors.

SCOPE: exploratory numerical probes; FT theorem remains Pauli local-stochastic.
"""
from __future__ import annotations
import csv, math
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
SEED = 20260918
DISTANCES = (3, 5)
P_GRID = [0.001, 0.002, 0.003, 0.005, 0.008]
COHERENT_THETA = [0.0, 0.05, 0.10, 0.20]  # radians over-rotation
LEAK_RATES = [0.0, 0.001, 0.003, 0.01]
SCOPE = (
    "exploratory coherent-overrotation and leakage-toy probes on SC memory / "
    "CAS stage proxy; FT theorem still Pauli local-stochastic"
)

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k/n; den = 1+z*z/n
    c = (p+z*z/(2*n))/den; h=(z/den)*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, max(0.,c-h), min(1.,c+h)

def si1000_rotated_memory(d, p, rounds=None, coherent_theta=0.0, leak_p=0.0):
    """Build Stim circuit for rotated surface code Z memory with optional extras."""
    if rounds is None:
        rounds = d
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p/10,
        before_measure_flip_probability=5*p,
        after_reset_flip_probability=2*p,
    )
    # Append coherent over-rotation probe as approximate Pauli channel via
    # stim's CORRELATED / or as additional Z errors with p_eff ~ theta^2/4
    # Honest approach: convert coherent angle to effective Z probability for
    # Monte Carlo comparison, AND also run a circuit with RX/RZ over-rotations
    # on data between rounds when theta>0 (stim supports RZ).
    if coherent_theta > 0:
        # Build a modified circuit: inject RZ(theta) on all data qubits each round
        # as a coherent probe, then compare to Pauli-twirl p_z = sin^2(theta/2)
        # For Stim tableau sampling of detectors we need Pauli noise; so we add
        # Z_ERROR with p_twirl = sin(theta/2)**2 on data each round as the
        # Pauli-twirled image of RZ(theta), and separately record the coherent
        # angle. True coherent simulation of detectors is outside Stim Pauli MC;
        # we report Pauli-twirl proxy + note limitation.
        p_twirl = float(np.sin(coherent_theta / 2.0) ** 2)
        # Append extra Z_ERROR instructions by rewriting: easiest is to add
        # after_clifford style by regenerating with higher p, but cleaner:
        # post-process by OR-ing additional independent Z flips in analysis.
        circ = (circ, p_twirl)  # marker tuple
    if leak_p > 0 and not isinstance(circ, tuple):
        # Leakage toy handled in sampling loop, not circuit rewrite
        pass
    return circ

def sample_memory(d, p, shots, seed, coherent_theta=0.0, leak_p=0.0, twirl=True):
    rounds = d
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d, rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p/10,
        before_measure_flip_probability=5*p,
        after_reset_flip_probability=2*p,
    )
    p_twirl = float(np.sin(coherent_theta / 2.0) ** 2) if coherent_theta > 0 else 0.0
    # If coherent and twirl: add Z_ERROR(p_twirl) on data qubits once per round
    # by increasing before_round_data_depolarization approximately.
    # More honest: regenerate with extra depolarization approximating twirl.
    if coherent_theta > 0 and twirl:
        p_eff = min(0.5, p/10 + p_twirl)
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d, rounds=rounds,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p_eff,
            before_measure_flip_probability=5*p,
            after_reset_flip_probability=2*p,
        )
        note = f"pauli_twirl_of_RZ(theta={coherent_theta})"
    elif coherent_theta > 0 and not twirl:
        # "Coherent" probe: use same twirl channel but label as coherent_proxy;
        # Stim cannot sample non-Pauli detector outcomes. Document limitation.
        p_eff = min(0.5, p/10 + p_twirl)
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d, rounds=rounds,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p_eff,
            before_measure_flip_probability=5*p,
            after_reset_flip_probability=2*p,
        )
        note = f"coherent_RZ_proxy_via_twirl(theta={coherent_theta}); Stim Pauli MC limitation"
    else:
        circ = base
        note = "si1000_like_baseline"

    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circ.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    # Leakage toy: randomly flip additional observables with leak transport
    if leak_p > 0:
        rng = np.random.default_rng(seed + 17)
        # Each shot: with prob leak_p per data qubit per round, inject logical-ish damage
        # Toy: flip observable with probability 1-(1-leak_p)^(d*rounds*alpha)
        n_data = d * d
        p_shot_leak = 1.0 - (1.0 - leak_p) ** (n_data * rounds * 0.25)
        leak_flips = rng.random(shots) < p_shot_leak
        obs = obs.copy()
        obs[:, 0] ^= leak_flips
        note = note + f"+leakage_toy(p={leak_p},p_shot={p_shot_leak:.4g})"

    preds = np.array([matching.decode(dets[i]) for i in range(shots)], dtype=np.uint8)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    fail = int(np.sum(np.any(preds != obs, axis=1)))
    rate, lo, hi = wilson(fail, shots)
    return fail, rate, lo, hi, note, p_twirl

def main():
    rows = []
    # Baseline + coherent
    for d in DISTANCES:
        for p in P_GRID:
            shots = 8000 if p <= 0.002 else (5000 if p <= 0.005 else 3000)
            for theta in COHERENT_THETA:
                fail, rate, lo, hi, note, p_tw = sample_memory(
                    d, p, shots, SEED + d*100 + int(p*1e4), coherent_theta=theta, twirl=True)
                rows.append(dict(
                    probe="coherent_twirl", d=d, p=p, theta=theta, leak_p=0.0,
                    p_twirl=p_tw, shots=shots, logical_failures=fail,
                    logical_error_rate=rate, wilson_lo=lo, wilson_hi=hi,
                    note=note, scope=SCOPE, seed=SEED,
                ))
                print(f"coherent d={d} p={p} theta={theta}: p_L={rate:.4g} ({note})")
            for lp in LEAK_RATES:
                if lp == 0.0:
                    continue
                fail, rate, lo, hi, note, p_tw = sample_memory(
                    d, p, shots, SEED + 999 + d*100, leak_p=lp, coherent_theta=0.0)
                rows.append(dict(
                    probe="leakage_toy", d=d, p=p, theta=0.0, leak_p=lp,
                    p_twirl=0.0, shots=shots, logical_failures=fail,
                    logical_error_rate=rate, wilson_lo=lo, wilson_hi=hi,
                    note=note, scope=SCOPE, seed=SEED,
                ))
                print(f"leak d={d} p={p} leak={lp}: p_L={rate:.4g}")

    out = ROOT / "noise_beyond_pauli_probes.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # Figure: coherent degradation at fixed p
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    p_fix = 0.003
    for d in DISTANCES:
        sub = [r for r in rows if r["probe"]=="coherent_twirl" and r["p"]==p_fix and r["d"]==d]
        sub = sorted(sub, key=lambda r: r["theta"])
        axes[0].plot([r["theta"] for r in sub], [max(r["logical_error_rate"],1e-6) for r in sub],
                     "o-", label=rf"$d={d}$")
    axes[0].set_xlabel(r"coherent $\theta$ (twirl proxy)")
    axes[0].set_ylabel(r"$p_L$"); axes[0].set_yscale("log")
    axes[0].set_title(rf"Coherent probe at $p={p_fix}$")
    axes[0].grid(True, which="both", alpha=0.25); axes[0].legend(frameon=False, fontsize=7)

    for d in DISTANCES:
        sub = [r for r in rows if r["probe"]=="leakage_toy" and r["p"]==p_fix and r["d"]==d]
        # include baseline theta0 leak0
        base = [r for r in rows if r["probe"]=="coherent_twirl" and r["p"]==p_fix and r["d"]==d and r["theta"]==0.0]
        sub = sorted(sub, key=lambda r: r["leak_p"])
        xs = [0.0] + [r["leak_p"] for r in sub]
        ys = ([base[0]["logical_error_rate"]] if base else [np.nan]) + [r["logical_error_rate"] for r in sub]
        axes[1].plot(xs, [max(y,1e-6) for y in ys], "s-", label=rf"$d={d}$")
    axes[1].set_xlabel("leakage toy rate"); axes[1].set_ylabel(r"$p_L$"); axes[1].set_yscale("log")
    axes[1].set_title(rf"Leakage toy at $p={p_fix}$")
    axes[1].grid(True, which="both", alpha=0.25); axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG/"noise_beyond_pauli_probes.pdf"); fig.savefig(FIG/"noise_beyond_pauli_probes.png", dpi=240)
    plt.close(fig)
    print(f"wrote {out.name}; SCOPE: {SCOPE}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
