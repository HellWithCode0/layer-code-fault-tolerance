#!/usr/bin/env python3
"""Gap 1/2/3: Circuit-level CSS syndrome-extraction on WB layer-code instances.

Builds a deterministic Stim SE circuit (ancilla per check, CNOT spread, measure/reset)
with SI1000-like noise, samples measurement records for R=d_upper rounds, and decodes
with BP-OSD on the classical spacetime parity-check matrix (circuit-level noise,
phenomenological PCM). Also writes a Stim circuit excerpt with consecutive-round
DETECTOR annotations where Stim accepts them.

Mini CAS-on-WB: circuit idle memory + noisy logical-Z product measurement stage.

SCOPE: circuit-level SE on truncated WB-incidence instances L in {3,4,5,6};
NOT CAD artwork; architecture p_th only if a clear L-crossing appears.
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
import stim
from ldpc import BpOsdDecoder

from wb_layer_code_instance import (
    build_repetition_layer_code,
    quasiconcat_and_kernel_logicals,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SEED = 20260918
L_VALUES = (3, 4, 5, 6)
P_GRID = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
OSD_ORDER = {3: 5, 4: 6, 5: 6, 6: 6}
MAX_ITER = 30
SCOPE = (
    "circuit-level CSS SE Stim schedule on truncated WB L in {3,4,5,6} with "
    "SI1000-like noise + spacetime BP-OSD; NOT CAD artwork DEM; "
    "NOT asymptotic architecture p_th unless L-crossing observed"
)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def si1000(p):
    return dict(p2=p, p1=p / 10.0, p_meas=5 * p, p_reset=2 * p, p_idle=p / 10.0)


def shot_budget(p, L):
    if p == 0.0:
        return 300
    if p <= 0.002:
        base = 500
    elif p <= 0.005:
        base = 350
    elif p <= 0.01:
        base = 250
    else:
        base = 180
    if L >= 6:
        base = max(60, base // 3)
    elif L == 5:
        base = max(100, base // 2)
    elif L == 4:
        base = max(140, (base * 2) // 3)
    return int(base)


def spacetime_check_matrix(H, R):
    m, n = H.shape
    Hst = np.zeros((R * m, R * n + R * m), dtype=np.uint8)
    for r in range(R):
        for i in range(m):
            row = r * m + i
            Hst[row, r * n : (r + 1) * n] = H[i]
            Hst[row, R * n + r * m + i] = 1
            if r > 0:
                Hst[row, R * n + (r - 1) * m + i] = 1
    return Hst


def make_decoder(H, p, osd_order):
    return BpOsdDecoder(
        np.asarray(H, dtype=np.uint8),
        error_rate=float(max(p, 1e-6)),
        max_iter=MAX_ITER,
        bp_method="minimum_sum",
        osd_method="osd_cs",
        osd_order=int(osd_order),
    )


def append_SE_round(c, data, Hx, Hz, ax, az, pr):
    """One SE round: Z-stabs (Hz) then X-stabs (Hx). Returns (#Z meas, #X meas)."""
    for i, row in enumerate(Hz):
        a = az[i]
        c.append("R", [a])
        if pr["p_reset"] > 0:
            c.append("X_ERROR", [a], pr["p_reset"])
        for q in np.flatnonzero(row):
            q = int(q)
            c.append("CX", [data[q], a])
            if pr["p2"] > 0:
                c.append("DEPOLARIZE2", [data[q], a], pr["p2"])
        if pr["p_meas"] > 0:
            c.append("X_ERROR", [a], pr["p_meas"])
        c.append("MR", [a])
    c.append("TICK")
    if pr["p_idle"] > 0:
        c.append("DEPOLARIZE1", data, pr["p_idle"])
    for i, row in enumerate(Hx):
        a = ax[i]
        c.append("R", [a])
        if pr["p_reset"] > 0:
            c.append("X_ERROR", [a], pr["p_reset"])
        c.append("H", [a])
        if pr["p1"] > 0:
            c.append("DEPOLARIZE1", [a], pr["p1"])
        for q in np.flatnonzero(row):
            q = int(q)
            c.append("CX", [a, data[q]])
            if pr["p2"] > 0:
                c.append("DEPOLARIZE2", [a, data[q]], pr["p2"])
        c.append("H", [a])
        if pr["p1"] > 0:
            c.append("DEPOLARIZE1", [a], pr["p1"])
        if pr["p_meas"] > 0:
            c.append("X_ERROR", [a], pr["p_meas"])
        c.append("MR", [a])
    c.append("TICK")
    if pr["p_idle"] > 0:
        c.append("DEPOLARIZE1", data, pr["p_idle"])
    return Hz.shape[0], Hx.shape[0]


def build_measurement_circuit(Hx, Hz, R, p, noise="si1000", basis="Z"):
    """SE circuit returning only measurement records (no DETECTORs).

    Record layout per shot:
      for r in range(R): mz Z-anc + mx X-anc
      then n data bits (Z or X basis)
    """
    Hx = np.asarray(Hx, dtype=np.uint8)
    Hz = np.asarray(Hz, dtype=np.uint8)
    n = Hx.shape[1]
    n_xstab, n_zstab = Hx.shape[0], Hz.shape[0]
    pr = si1000(p) if noise == "si1000" else dict(
        p2=p, p1=p, p_meas=p, p_reset=p, p_idle=p
    )
    c = stim.Circuit()
    data = list(range(n))
    az = list(range(n, n + n_zstab))
    ax = list(range(n + n_zstab, n + n_zstab + n_xstab))

    c.append("R", data)
    if pr["p_reset"] > 0:
        c.append("X_ERROR", data, pr["p_reset"])
    if basis == "X":
        c.append("H", data)
        if pr["p1"] > 0:
            c.append("DEPOLARIZE1", data, pr["p1"])
    c.append("TICK")
    if pr["p_idle"] > 0:
        c.append("DEPOLARIZE1", data, pr["p_idle"])

    for _ in range(R):
        append_SE_round(c, data, Hx, Hz, ax, az, pr)

    if basis == "X":
        c.append("H", data)
        if pr["p1"] > 0:
            c.append("DEPOLARIZE1", data, pr["p1"])
    if pr["p_meas"] > 0:
        c.append("X_ERROR", data, pr["p_meas"])
    c.append("M", data)
    return c, dict(n=n, mz=n_zstab, mx=n_xstab, R=R, basis=basis, noise=noise)


def build_detector_circuit_excerpt(Hx, Hz, R, p=0.001):
    """Stim circuit with consecutive-round DETECTORs only (documentation / DEM stats)."""
    Hx = np.asarray(Hx, dtype=np.uint8)
    Hz = np.asarray(Hz, dtype=np.uint8)
    n = Hx.shape[1]
    mz, mx = Hz.shape[0], Hx.shape[0]
    pr = si1000(p)
    c = stim.Circuit()
    data = list(range(n))
    az = list(range(n, n + mz))
    ax = list(range(n + mz, n + mz + mx))

    c.append("R", data)
    c.append("TICK")
    # Round 0
    append_SE_round(c, data, Hx, Hz, ax, az, pr)
    for r in range(1, R):
        # Z then X with detectors vs previous full round (offset mz+mx)
        for i, row in enumerate(Hz):
            a = az[i]
            c.append("R", [a])
            if pr["p_reset"] > 0:
                c.append("X_ERROR", [a], pr["p_reset"])
            for q in np.flatnonzero(row):
                q = int(q)
                c.append("CX", [data[q], a])
                if pr["p2"] > 0:
                    c.append("DEPOLARIZE2", [data[q], a], pr["p2"])
            if pr["p_meas"] > 0:
                c.append("X_ERROR", [a], pr["p_meas"])
            c.append("MR", [a])
            c.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-(1 + mz + mx))])
        c.append("TICK")
        if pr["p_idle"] > 0:
            c.append("DEPOLARIZE1", data, pr["p_idle"])
        for i, row in enumerate(Hx):
            a = ax[i]
            c.append("R", [a])
            if pr["p_reset"] > 0:
                c.append("X_ERROR", [a], pr["p_reset"])
            c.append("H", [a])
            if pr["p1"] > 0:
                c.append("DEPOLARIZE1", [a], pr["p1"])
            for q in np.flatnonzero(row):
                q = int(q)
                c.append("CX", [a, data[q]])
                if pr["p2"] > 0:
                    c.append("DEPOLARIZE2", [a, data[q]], pr["p2"])
            c.append("H", [a])
            if pr["p1"] > 0:
                c.append("DEPOLARIZE1", [a], pr["p1"])
            if pr["p_meas"] > 0:
                c.append("X_ERROR", [a], pr["p_meas"])
            c.append("MR", [a])
            c.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-(1 + mz + mx))])
        c.append("TICK")
        if pr["p_idle"] > 0:
            c.append("DEPOLARIZE1", data, pr["p_idle"])
    # Final data + observable (Z_L filled by caller via include)
    c.append("M", data)
    return c, dict(n=n, mz=mz, mx=mx, R=R)


def simulate_circuit_memory(L, p, shots, noise="si1000", seed=SEED):
    """Z-basis idle memory under circuit-level SE noise.

    Init |0>^n is Z-stabilized (Z_L=+1) but not X-stabilized. X-stab rounds
    establish a random Pauli frame; only consecutive-round X-stab *detectors*
    diagnose Z errors. Final Z_L is read from data after applying the decoded
    Z correction. Noiseless runs must give p_L=0.
    """
    Hx, Hz, meta = build_repetition_layer_code(L)
    logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
    assert logs["X_is_logical"] and logs["Z_is_logical"]
    n = meta["n"]
    d_upper = min(logs["X_weight"], logs["Z_weight"])
    R = max(2, int(d_upper))
    logZ = logs["Z"].astype(np.uint8)
    order = OSD_ORDER[L]

    circ, info = build_measurement_circuit(Hx, Hz, R, p, noise=noise, basis="Z")
    raw = circ.compile_sampler(seed=seed).sample(shots)
    mz, mx = info["mz"], info["mx"]
    syn_len = R * (mz + mx)

    # Spacetime PCM with R rounds; feed detector syndromes with s[0]=0 and
    # s[r]=m[r]⊕m[r-1] packed as the difference model used by spacetime_check_matrix.
    Hst_z = spacetime_check_matrix(Hx, R)
    prior = float(p) if p > 0 else 1e-6
    dec_z = make_decoder(Hst_z, prior, order)

    fail = 0
    for s in range(shots):
        row = raw[s]
        x_rounds = []
        off = 0
        for r in range(R):
            off += mz  # Z-ancilla outcomes unused for Z-memory logical
            x_rounds.append(np.asarray(row[off : off + mx], dtype=np.uint8))
            off += mx
        data_m = np.asarray(row[syn_len : syn_len + n], dtype=np.uint8)

        # Detector syndrome for spacetime PCM:
        # spacetime_check_matrix expects s[r] = Hx e[r] + mu[r] + mu[r-1],
        # which matches measured detector d[r]=m[r]⊕m[r-1] when e are per-round.
        syn_z = np.zeros(R * mx, dtype=np.uint8)
        for r in range(R):
            if r == 0:
                syn_z[r * mx : (r + 1) * mx] = 0  # frame reference; no absolute syndrome
            else:
                syn_z[r * mx : (r + 1) * mx] = (x_rounds[r] ^ x_rounds[r - 1]) % 2

        cz = np.asarray(dec_z.decode(syn_z), dtype=np.uint8)
        corr_z = np.zeros(n, dtype=np.uint8)
        for r in range(R):
            corr_z ^= cz[r * n : (r + 1) * n]

        bits = (data_m ^ corr_z) % 2
        if int((bits @ logZ) % 2) != 0:
            fail += 1

    rate, lo, hi = wilson(fail, shots)
    return dict(
        L=L,
        n=n,
        R=R,
        d_upper=d_upper,
        p=float(p),
        shots=shots,
        seed=seed,
        noise_model=noise,
        logical_failures=fail,
        logical_error_rate=rate,
        wilson_lo=lo,
        wilson_hi=hi,
        n_spacetime_checks_Z=int(Hst_z.shape[0]),
        n_spacetime_vars_Z=int(Hst_z.shape[1]),
        osd_order=order,
        decoder="ldpc.BpOsdDecoder_spacetime_circuitSE_detectors",
        schedule="circuit_SE_memory_Z",
        scope=SCOPE,
        note="circuit_noise_Xstab_detectors_ZL",
    )


def simulate_wb_mini_cas(L, p, shots, seed=SEED):
    Hx, Hz, meta = build_repetition_layer_code(L)
    logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
    d_upper = min(logs["X_weight"], logs["Z_weight"])
    R = max(2, int(d_upper))
    recA = simulate_circuit_memory(L, p, shots, noise="si1000", seed=seed)
    rng = np.random.default_rng(seed + 17)
    logZ = logs["Z"].astype(np.uint8)
    failB = 0
    dec = make_decoder(Hx, max(p, 1e-6), OSD_ORDER[L])
    for _ in range(shots):
        flips = (rng.random(meta["n"]) < p).astype(np.uint8)
        ehat = np.asarray(dec.decode((Hx @ flips) % 2), dtype=np.uint8)
        res = (flips ^ ehat) % 2
        if int((res @ logX) % 2) != 0:
            failB += 1
    pA = float(recA["logical_error_rate"])
    pB = failB / shots if shots else 1.0
    p_or = pA + pB - pA * pB
    rate, lo, hi = wilson(int(round(p_or * shots)), shots)
    return dict(
        L=L,
        n=meta["n"],
        R=R,
        d_upper=d_upper,
        p=float(p),
        shots=shots,
        seed=seed,
        noise_model="si1000_circuit_plus_pheno_logicalZ",
        logical_failures=int(round(p_or * shots)),
        logical_error_rate=p_or,
        wilson_lo=lo,
        wilson_hi=hi,
        stageA_pL=pA,
        stageB_pL=pB,
        decoder="BPOSD_circuit_memory_plus_logicalZ",
        schedule="mini_CAS_WB_memory_plus_logicalZ",
        scope=SCOPE,
        note="stretch_Gap2_CAS_on_WB",
    )


def crossing_report(rows, label):
    by_p = {}
    for r in rows:
        by_p.setdefault(r["p"], {})[r["L"]] = r["logical_error_rate"]
    lines = [label]
    mono = []
    for p in sorted(by_p):
        if p == 0.0:
            continue
        m = by_p[p]
        ls = sorted(m)
        if len(ls) < 2:
            continue
        seq = [m[L] for L in ls]
        if len(ls) >= 3 and all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)):
            mono.append(p)
        pairs = [f"L{b}<L{a}" for a, b in zip(ls, ls[1:]) if m[b] < m[a]]
        lines.append(
            f"  p={p:.4g}: "
            + ", ".join(f"L{L}={m[L]:.4g}" for L in ls)
            + (f"  [{', '.join(pairs)}]" if pairs else "  [no larger-L wins]")
        )
    if mono:
        lines.insert(1, f"threshold-like ordering at p in {mono}")
    else:
        lines.insert(1, "NO clear L-crossing on this circuit-level window")
    return "\n".join(lines)


def plot_rows(rows, stem, title, ylabel):
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    markers = {3: "o", 4: "s", 5: "^", 6: "D"}
    for L in sorted({r["L"] for r in rows}):
        sub = sorted([r for r in rows if r["L"] == L and r["p"] > 0], key=lambda r: r["p"])
        if not sub:
            continue
        xs = [r["p"] for r in sub]
        ys = [max(r["logical_error_rate"], 1e-6) for r in sub]
        ax.plot(xs, ys, f"{markers.get(L, 'o')}-", label=rf"$L={L}$")
    ax.set_xlabel(r"SI1000-like circuit $p$")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def emit_samples():
    Hx, Hz, meta = build_repetition_layer_code(3)
    logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
    R = max(2, min(logs["X_weight"], logs["Z_weight"]))
    circ_m, info = build_measurement_circuit(Hx, Hz, R, 0.001, basis="Z")
    circ_d, _ = build_detector_circuit_excerpt(Hx, Hz, R, 0.001)
    # append observable on detector circuit
    n = info["n"]
    logZ = logs["Z"]
    obs = [stim.target_rec(-(n - int(q))) for q in np.flatnonzero(logZ)]
    circ_d.append("OBSERVABLE_INCLUDE", obs, 0)
    (ROOT / "wb_layer_circuit_dem_sample.txt").write_text(
        f"# measurement circuit L=3 n={n} R={R} mz={info['mz']} mx={info['mx']}\n"
        + "\n".join(str(circ_m).splitlines()[:60])
        + "\n...\n\n# detector excerpt (consecutive rounds)\n"
        + "\n".join(str(circ_d).splitlines()[:40])
        + "\n...\n"
    )
    # DEM stats attempt
    row = dict(
        L=3,
        n=n,
        R=R,
        mz=info["mz"],
        mx=info["mx"],
        measurement_circuit_lines=len(str(circ_m).splitlines()),
        detector_circuit_lines=len(str(circ_d).splitlines()),
        scope=SCOPE,
    )
    try:
        dem = circ_d.detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=True
        )
        row["n_detectors"] = dem.num_detectors
        row["n_observables"] = dem.num_observables
        row["dem_status"] = "ok"
    except Exception as ex:
        row["n_detectors"] = -1
        row["n_observables"] = 1
        row["dem_status"] = f"nondeterministic_or_error:{str(ex)[:120]}"
    with (ROOT / "wb_layer_circuit_dem_stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    return row


def alternate_note():
    path = ROOT / "wb_layer_alternate_distance_note.txt"
    path.write_text(
        "Alternate growing-distance note (Gap 2/3)\n"
        "Current truncated WB family: d_upper=min(wt X_L, wt Z_L) in {2,3,4,5} for L=3..6\n"
        "because open-boundary condensation of weight-2 modes and fixed 3-rep Y_DATA width\n"
        "cap the Z logical. A longer-corridor / larger-L construction with fewer condensed\n"
        "boundaries could raise d_upper while remaining WB-incidence-derived, but would\n"
        "change k or introduce residual boundary logicals — not shipped as a new family here.\n"
        "Circuit-level SE numerics use the audited truncated instances (honest scope).\n"
    )
    return path


def main():
    t0 = time.time()
    print(f"SEED={SEED}")
    print(f"SCOPE={SCOPE}")
    stats = emit_samples()
    print("sample/stats:", stats)
    alternate_note()

    print("Noiseless sanity L=3...")
    san = simulate_circuit_memory(3, 0.0, 200, seed=SEED)
    print(
        f"  p0 p_L={san['logical_error_rate']:.4g} "
        f"fails={san['logical_failures']}/{san['shots']}"
    )

    rows = []
    cas_rows = []
    for L in L_VALUES:
        Hx, Hz, meta = build_repetition_layer_code(L)
        logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
        d_upper = min(logs["X_weight"], logs["Z_weight"])
        print(f"L={L} n={meta['n']} d_upper={d_upper}")
        for p in P_GRID:
            sh = shot_budget(p, L)
            if p == 0.0:
                sh = min(sh, 120)
            rec = simulate_circuit_memory(
                L, p, sh, noise="si1000", seed=SEED + L * 1009 + int(p * 1e6)
            )
            rows.append(rec)
            print(
                f"  L={L} p={p}: p_L={rec['logical_error_rate']:.4g} "
                f"fails={rec['logical_failures']}/{rec['shots']}"
            )
        for p in (0.001, 0.005, 0.01):
            sh = max(40, shot_budget(p, L) // 2)
            crec = simulate_wb_mini_cas(L, p, sh, seed=SEED + 5000 + L)
            cas_rows.append(crec)
            print(f"  CAS L={L} p={p}: p_L={crec['logical_error_rate']:.4g}")

    out = ROOT / "wb_layer_circuit_dem.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    plot_rows(
        rows,
        FIG / "wb_layer_circuit_dem",
        "WB circuit-level SE memory",
        r"$p_L$ (spacetime BP-OSD)",
    )
    report = crossing_report(rows, "Gap1/2/3 circuit-level WB SE")
    p0s = [r for r in rows if r["p"] == 0.0]
    report += "\nNoiseless p=0:\n" + "\n".join(
        f"  L={r['L']}: p_L={r['logical_error_rate']:.4g} "
        f"fails={r['logical_failures']}/{r['shots']}"
        for r in p0s
    )
    (ROOT / "wb_layer_circuit_dem_crossing.txt").write_text(report + "\n")
    print(report)

    cout = ROOT / "wb_layer_mini_cas.csv"
    ckeys = sorted({k for r in cas_rows for k in r.keys()})
    with cout.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ckeys, extrasaction="ignore")
        w.writeheader()
        w.writerows(cas_rows)
    plot_rows(
        cas_rows,
        FIG / "wb_layer_mini_cas",
        "Mini CAS on WB (memory+logical Z)",
        r"$p_L$ (OR stages)",
    )
    crep = crossing_report(cas_rows, "Mini CAS on WB")
    (ROOT / "wb_layer_mini_cas_crossing.txt").write_text(crep + "\n")
    print(crep)

    # Architecture note update
    (ROOT / "wb_layer_architecture_pth_note.txt").write_text(
        "Gap 2 — Architecture p_th on actual layer-code instances\n"
        + report
        + "\n\nCircuit-level SE schedule (this campaign) + prior phenomenological schedule.\n"
        "If no crossing: STATUS=PARTIAL/OPEN due to truncation-limited d_upper "
        "(d_upper=2,3,4,5 for L=3..6).\n"
    )

    print(f"elapsed {time.time() - t0:.1f}s")
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
