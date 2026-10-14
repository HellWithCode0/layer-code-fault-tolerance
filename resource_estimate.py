#!/usr/bin/env python3
"""Concrete space-time resource model for the layer-code architecture (review point 4).

The architecture theorem is existential: it fixes exponents, not constants.
This script turns its accounting into one explicit estimate in which every
constant is a named, adjustable assumption (ASSUMPTIONS, overridable on the
command line and echoed into the report), and compares it with a
surface-code / lattice-surgery baseline at equal target failure probability.

This version replaces an earlier estimate in which a referee found three
reproducible errors; audit_old_model() recomputes them from the old formulas:
  (A) the old "quadratic" law q -> 35 q^2 was labelled as the paper's proven
      always-output bound, but the proven bound is C_dist q^2 and the exact
      always-output RM15 map starts at 105 q^2;
  (B) the old surface-code serial distance was chosen from a short
      logical-layer duration, not from the serial runtime that was reported;
  (C) the old serial layer-code space never allocated the factory blocks that
      its own timing used.
Here every schedule has an explicit integer allocation of blocks (or tiles)
by role, and its distance f is the least odd f whose memory failure over the
complete runtime of that schedule (factory waiting included) fits the budget.

Distillation models (all three reported, never mixed):
  proved                  : the paper's always-output bound q -> C_dist q^2,
                            C_dist = binom(15 + c_Y, 2);
  exact_always_output     : exact error map of the ideal always-output
                            15-to-1 and Steane 7-to-1 factories with
                            single-error recovery (counts enumerated here);
  postselected_optimistic : standard postselected factories with retries.
                            NOT covered by the paper's theorem.

Architectures / schedules:
  layer code, serial      : the theorem's minimal-space schedule, one
                            depth-first factory per data block;
  layer code, serial_shared: one depth-first factory shared by all data
                            blocks (the analogue of the surface-code serial
                            schedule with one factory unit);
  layer code, throughput  : parallel factories deliver the W resources of a
                            logical layer within layer_slots gadget slots;
  surface code, serial    : one 15-to-1 unit;
  surface code, throughput: enough 15-to-1 units for W resources per 2f rounds.
Layer-code blocks come from an explicit family (balanced input, spacing s=2)
with three distance scenarios: slab_upper (data strings of a slab, not a proved
upper bound), small_instance, proved_lower.

None of these numbers is a hardware prediction.

Outputs (next to this script):
  resource_estimate.csv           one row per noise point x scenario x model x
                                  architecture/schedule;
  resource_estimate.md            assumptions, audit of the old model, tables;
  resource_estimate_summary.json  key numbers (base noise, small_instance) and
                                  the storage crossovers.
Prints a final PASS/FAIL line for the internal consistency checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

ASSUMPTIONS = {
    # ---- target computation ---------------------------------------------
    "W": (100, "logical sites"),
    "T": (1_000_000, "logical layers; one distilled |A> per site per layer (worst case)"),
    "eps": (0.01, "total target failure; eps_mem = eps/2 (memory and gadgets), eps_res = eps/2 (final resources)"),
    # ---- noise points ----------------------------------------------------
    "p_base": (1e-3, "physical error rate, base noise point"),
    "q0_base": (1e-2, "raw injected resource error, base noise point"),
    "p_secondary": (1e-4, "physical error rate, secondary noise point"),
    "q0_secondary": (1e-3, "raw injected resource error, secondary noise point"),
    "p_th": (1e-2, "threshold in the logical failure model (both architectures)"),
    "A": (0.1, "prefactor: failure per logical qubit per round p_L(f) = A (p/p_th)^((f+1)/2)"),
    # ---- distillation ------------------------------------------------------
    "cY_LC": (15, "conditional S-correction slots per 15-to-1 run on layer-code blocks (each consumes one |Y>)"),
    "cY_SC": (0, "conditional S-correction slots per 15-to-1 run in the surface-code unit (auto-corrected)"),
    "max_levels": (12, "more distillation levels than this counts as not converging"),
    # ---- gadget-slot costs (layer code) ----------------------------------
    "slot_rounds": (3, "rounds per gadget slot, in units of f"),
    "G15": (30, "gadget slots per 15-to-1 run on layer-code blocks"),
    "G7": (14, "gadget slots per Steane 7-to-1 run on layer-code blocks"),
    "G_inj": (2, "gadget slots per raw resource (injection plus encoded courier)"),
    "G_consume": (1, "gadget slots to consume one resource into the data"),
    "A_run_logicals": (17, "logical slots held by one 15-to-1 run in the throughput schedule"),
    "Y_run_logicals": (9, "logical slots held by one 7-to-1 run in the throughput schedule"),
    "courier_blocks": (2, "live courier blocks per raw resource (throughput) or per factory (serial), full size"),
    "layer_slots": (2, "gadget slots per logical layer in the throughput-limited layer-code schedule"),
    # ---- layer-code block family -------------------------------------------
    "s": (2, "spacing of the explicit realization"),
    "rate": (0.1, "input code rate: K = max(1, floor(rate N))"),
    "check_frac": (0.45, "m_X = m_Z = round(check_frac N) input checks (round half up)"),
    "check_weight": (8, "input check weight w (supports assumed to span the whole input)"),
    "rel_dist": (0.1, "input relative distance: d_in = ceil(rel_dist N)"),
    "N_min": (8, "smallest input length considered (a weight-w check needs N >= w)"),
    "N_max": (6000, "largest input length tabulated"),
    "anc_LC": (1.0, "syndrome-extraction ancillas per layer-code data qubit"),
    "m0": (2, "edge multiplicity of the gauging grid (full-distance measurement)"),
    "grid_ratio": (1.0, "gauging-grid edges per block qubit, before multiplicity"),
    "cnot_layers_LC": (28, "CNOT layers per layer-code extraction round"),
    # ---- surface-code baseline -------------------------------------------
    "tile_coeff": (2.0, "physical qubits per surface-code tile, in units of f^2"),
    "route_SC": (2, "tiles per logical site (data plus routing)"),
    "layer_rounds_SC": (2, "rounds per logical layer, in units of f"),
    "fac_tiles_SC": (11, "tiles per 15-to-1 unit (Litinski-style)"),
    "fac_rounds_SC": (11, "rounds per 15-to-1 run, in units of f"),
    "cnot_layers_SC": (8, "CNOT/measure layers per surface-code round"),
    # ---- conversion / search -----------------------------------------------
    "t_layer_ns": (50.0, "duration of one CNOT/measure layer in ns"),
    "f_max": (4001, "largest odd distance searched"),
}

SCENARIOS = ("slab_upper", "small_instance", "proved_lower")
MODELS = ("proved", "exact_always_output", "postselected_optimistic")
SCHEDULES = (("layer_code", "serial"), ("layer_code", "serial_shared"), ("layer_code", "throughput"),
             ("surface_code", "serial"), ("surface_code", "throughput"))

RM15_ALWAYS_OUTPUT_EXPECTED = [0, 0, 105, 35, 1260, 168, 4725, 435, 6000, 280, 2835, 105, 420, 0, 15, 1]


# ============================================================ enumeration
def enumerate_hamming_factory(r):
    """Z-error statistics of a distillation factory whose checks form the
    [2^r - 1, 2^r - 1 - r] Hamming parity-check matrix (column j = binary j+1).

    r = 4 is the RM15 15-to-1 factory, r = 3 the Steane 7-to-1 factory.
    Returns (n, always, acc, acc_log), each a list indexed by error weight:
      always : patterns that end logical after minimum-weight single-error
               recovery (residual has zero syndrome, logical iff odd weight);
      acc    : zero-syndrome (accepted) patterns;
      acc_log: accepted patterns that are logical (odd weight).
    """
    n = 2 ** r - 1
    e = np.arange(2 ** n, dtype=np.int64)
    syn = np.zeros_like(e)
    wt = np.zeros_like(e)
    for j in range(n):
        bit = (e >> j) & 1
        syn ^= bit * (j + 1)
        wt += bit
    # recovery flips bit (syn - 1) when syn != 0, changing the weight by +-1
    res_wt = np.where(syn != 0, wt + 1 - 2 * ((e >> np.maximum(syn - 1, 0)) & 1), wt)
    logical_after = (res_wt % 2) == 1
    always = np.bincount(wt[logical_after], minlength=n + 1).tolist()
    acc = np.bincount(wt[syn == 0], minlength=n + 1).tolist()
    acc_log = np.bincount(wt[(syn == 0) & (wt % 2 == 1)], minlength=n + 1).tolist()
    return n, always, acc, acc_log


def poly(counts, n, q):
    return sum(c * q ** w * (1 - q) ** (n - w) for w, c in enumerate(counts) if c)


# ============================================================ distillation
def distill(model, cY, q0, target, a, fac):
    """Recursion of one distillation model for one architecture.

    Returns levels, q_final, converged, the q sequences, and the tallies per
    final |A>: a_runs (15-to-1 runs), y_runs (7-to-1 runs), inj (raw
    injections), all including retries."""
    n15, al15, acc15, log15 = fac["rm15"]
    n7, al7, acc7, log7 = fac["steane"]
    useY = cY > 0
    qA, qY = [q0], [q0]
    tA = dict(a_runs=0.0, y_runs=0.0, inj=1.0)     # per level-r |A>
    tY = dict(y_runs=0.0, inj=1.0)                  # per level-r |Y>
    pacc_hist = []
    r = 0
    while qA[-1] > target:
        if r + 1 > a["max_levels"]:
            return dict(converged=False, levels=None, q_final=qA[-1], qA=qA, qY=qY,
                        reason=f"more than {a['max_levels']} levels", **tA)
        if model == "proved":
            C = math.comb(15 + cY, 2)
            nA = C * qA[-1] ** 2
            nY = nA                                     # shared family parameter q_r
            pA = pY = 1.0
        else:
            qc = qA[-1] + qY[-1] / 2 - qA[-1] * qY[-1] if useY else qA[-1]
            if model == "exact_always_output":
                nA, pA = poly(al15, n15, qc), 1.0
                nY, pY = poly(al7, n7, qY[-1]), 1.0
            else:
                pA = poly(acc15, n15, qc)
                nA = poly(log15, n15, qc) / pA
                pY = poly(acc7, n7, qY[-1])
                nY = poly(log7, n7, qY[-1]) / pY
        if nA >= qA[-1]:
            return dict(converged=False, levels=None, q_final=qA[-1], qA=qA, qY=qY,
                        reason=f"q_A stops decreasing at level {r + 1} ({qA[-1]:.3g} -> {nA:.3g})", **tA)
        cy = cY if useY else 0
        tA = dict(a_runs=(1 + 15 * tA["a_runs"]) / pA,
                  y_runs=(15 * tA["y_runs"] + cy * tY["y_runs"]) / pA,
                  inj=(15 * tA["inj"] + cy * tY["inj"]) / pA)
        tY = dict(y_runs=(1 + 7 * tY["y_runs"]) / pY, inj=7 * tY["inj"] / pY)
        qA.append(nA)
        qY.append(nY)
        pacc_hist.append((pA, pY))
        r += 1
    return dict(converged=True, levels=r, q_final=qA[-1], qA=qA, qY=qY, pacc=pacc_hist,
                reason="", **tA)


# ============================================================ layer-code family
class Family:
    """Explicit layer-code block family, tabulated over N."""

    def __init__(self, a):
        s, w = a["s"], a["check_weight"]
        N = np.arange(0, a["N_max"] + 1, dtype=np.int64)
        m = np.floor(a["check_frac"] * N + 0.5).astype(np.int64)
        Lx = s * (m + 1)
        Lz = s * (m + 1)
        wid = s * (N - 1)
        data = Lx * (Lz - 1) + (Lx + 1) * Lz
        csheet = wid * (Lz - 1) + (wid - 1) * Lz
        asheet = Lx * (wid + 1) + (Lx + 1) * wid
        self.N = N
        self.n = N * data + m * csheet + m * asheet
        self.K = np.maximum(1, np.floor(a["rate"] * N + 1e-9).astype(np.int64))
        self.Lz = Lz
        d_in = np.ceil(a["rel_dist"] * N - 1e-9).astype(np.int64)
        eta = max(1, w // 2)
        self.d = {"slab_upper": Lz * d_in,
                  "small_instance": (Lz * d_in) // 2,
                  "proved_lower": Lz * ((d_in + eta - 1) // eta)}
        for sc in SCENARIOS:
            self.d[sc][: a["N_min"]] = 0          # below N_min: not admissible
        self.eta = eta

    def pick(self, scenario, f):
        ok = np.nonzero(self.d[scenario] >= f)[0]
        if len(ok) == 0:
            raise RuntimeError(f"N_max too small for f={f} ({scenario})")
        N = int(ok[0])
        return N, int(self.K[N]), int(self.n[N]), int(self.d[scenario][N])


def convergence_threshold(model, cY, a, fac, lo=1e-5, hi=0.2, iters=40):
    """Largest raw error q0 for which the recursion of `model` still
    decreases at every level down to 1e-15 (bisection on q0)."""
    ok = lambda q0: distill(model, cY, q0, 1e-15, dict(a, max_levels=40), fac)["converged"]
    if not ok(lo):
        return None
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def p_L(a, p, f):
    return a["A"] * (p / a["p_th"]) ** ((f + 1) / 2)


def slots_per_resource(a, dist):
    return a["G15"] * dist["a_runs"] + a["G7"] * dist["y_runs"] + a["G_inj"] * dist["inj"]


def lc_layout(a, fam, scenario, schedule, f, dist):
    """Integer allocation and runtime of a layer-code schedule at distance f."""
    N, K, n, d = fam.pick(scenario, f)
    W, T = a["W"], a["T"]
    if K >= 2:
        Kd, work_per_data = K - 1, 0
    else:
        Kd, work_per_data = 1, 1
    B_data = math.ceil(W / Kd)
    B_work = work_per_data * B_data
    l = dist["levels"]
    sA = slots_per_resource(a, dist)
    if schedule in ("serial", "serial_shared"):
        S_fac = 15 * l + 7 * max(l - 1, 0) + 2
        fac_per = math.ceil(S_fac / K)
        n_fac = B_data if schedule == "serial" else 1
        B_fac = n_fac * fac_per
        B_cour = n_fac * a["courier_blocks"]
        # resources per factory per logical layer: K_d (one factory per data
        # block) or W (one shared factory), produced one after another
        per_factory = Kd if schedule == "serial" else W
        layer_slot_count = per_factory * (sA + a["G_consume"])
        total_slots = T * layer_slot_count
    else:
        V_fac = (math.ceil(a["A_run_logicals"] / K) * a["G15"] * dist["a_runs"]
                 + math.ceil(a["Y_run_logicals"] / K) * a["G7"] * dist["y_runs"])
        V_cour = a["courier_blocks"] * a["G_inj"] * dist["inj"]
        B_fac = math.ceil(W * V_fac / a["layer_slots"])
        B_cour = math.ceil(W * V_cour / a["layer_slots"])
        S_fac = None
        fac_per = None
        layer_slot_count = a["layer_slots"]
        total_slots = T * a["layer_slots"] + sA
    blocks = B_data + B_work + B_fac + B_cour
    per_block = n * (1 + a["anc_LC"] + a["m0"] * a["grid_ratio"])
    rounds = total_slots * a["slot_rounds"] * f
    return dict(N=N, K=K, n=n, d_block=d, K_d=Kd, n_data=B_data, n_working=B_work,
                n_factory=B_fac, n_courier=B_cour, units_total=blocks,
                S_fac=S_fac, factory_blocks_per_data_block=fac_per,
                L_live=blocks * K, peak_qubits=blocks * per_block,
                layer_rounds=layer_slot_count * a["slot_rounds"] * f, rounds=rounds,
                round_s=a["cnot_layers_LC"] * a["t_layer_ns"] * 1e-9)


def sc_layout(a, schedule, f, dist):
    W, T = a["W"], a["T"]
    runs = dist["a_runs"]
    data = W
    routing = W * (a["route_SC"] - 1)
    if schedule == "serial":
        fac = a["fac_tiles_SC"]
        layer_rounds = max(a["layer_rounds_SC"] * f, W * runs * a["fac_rounds_SC"] * f)
    else:
        fac = math.ceil(W * runs * a["fac_tiles_SC"] * a["fac_rounds_SC"] / a["layer_rounds_SC"] - 1e-9)
        layer_rounds = a["layer_rounds_SC"] * f
    tiles = data + routing + fac
    return dict(N=None, K=1, n=int(a["tile_coeff"] * f * f), d_block=f, K_d=1,
                n_data=data, n_working=routing, n_factory=fac, n_courier=0,
                units_total=tiles, S_fac=None, factory_blocks_per_data_block=None,
                L_live=tiles, peak_qubits=tiles * a["tile_coeff"] * f * f,
                layer_rounds=layer_rounds, rounds=T * layer_rounds,
                round_s=a["cnot_layers_SC"] * a["t_layer_ns"] * 1e-9)


def choose_distance(a, p, layout_fn):
    """Least odd f >= 3 with L_live * R_total * p_L(f) <= eps_mem, where the
    allocation and runtime are recomputed for every trial f."""
    eps_mem = a["eps"] / 2
    for f in range(3, a["f_max"] + 1, 2):
        lay = layout_fn(f)
        mem = lay["L_live"] * lay["rounds"] * p_L(a, p, f)
        if mem <= eps_mem:
            lay["f"] = f
            lay["memory_failure"] = mem
            return lay
    raise RuntimeError("no distance up to f_max meets the memory budget")


# ============================================================ storage crossover
def storage_crossover(a, fam, scenario, mode="computation"):
    """Distance at which a layer-code block holds a logical more cheaply than
    the surface code, with consistent accounting on both sides:
      computation : block n (1 + anc_LC + m0 grid_ratio) per K_d data sites
                    (as in the space tables) versus route_SC tiles per site;
      storage     : block n (1 + anc_LC) per K_d sites versus one tile.
    The comparison is not monotone in d (K and n jump with N), so both the
    first d at which the layer code is cheaper and the d from which it stays
    cheaper (up to the largest tabulated distance) are returned."""
    if mode == "computation":
        lc_fac, sc_tiles = 1 + a["anc_LC"] + a["m0"] * a["grid_ratio"], a["route_SC"]
    else:
        lc_fac, sc_tiles = 1 + a["anc_LC"], 1
    dmax = int(fam.d[scenario].max())
    first = last_bad = None
    for d in range(3, dmax + 1):
        N, K, n, dd = fam.pick(scenario, d)
        Kd = K - 1 if K >= 2 else 1
        cheaper = n * lc_fac / Kd < sc_tiles * a["tile_coeff"] * d * d
        if cheaper and first is None:
            first = dict(d=d, N=N, K=K, K_d=Kd, n=n, d_block=dd,
                         lc_per_logical=n * lc_fac / Kd,
                         sc_per_logical=sc_tiles * a["tile_coeff"] * d * d)
        if not cheaper:
            last_bad = d
    if first is None:
        return dict(d=None, d_stable=None)
    first["d_stable"] = None if last_bad == dmax else (last_bad + 1 if last_bad else 3)
    return first


# ============================================================ audit
def audit_old_model(a, fac):
    """Reproduce the referee's three findings from the old script's formulas
    (old defaults: cubic 35 q^3 law, cn = 509, cK = 0.47, reserve = 0.5)."""
    W, T, eps = a["W"], a["T"], a["eps"]
    p, pth, A = a["p_base"], a["p_th"], a["A"]
    pl = lambda f: A * (p / pth) ** ((f + 1) / 2)
    q, lvl = a["q0_base"], 0
    while q > eps / (2 * W * T):
        q, lvl = 35 * q ** 3, lvl + 1
    runs = (15 ** lvl - 1) // 14
    out = {"old_levels_cubic": lvl, "old_runs_per_resource": runs}
    # (B) old SC distance from a 2f-round logical layer on 2W tiles
    f_old = next(f for f in range(3, 400, 2) if W * T * 2 * f * 2 * pl(f) <= eps / 2)
    R_serial = lambda f: T * max(2 * f, W * runs * 11 * f)
    mem = W * R_serial(f_old) * pl(f_old)
    f_need = next(f for f in range(3, 400, 2) if W * R_serial(f) * pl(f) <= eps / 2)
    out.update(B_old_f_sc=f_old, B_old_serial_rounds=R_serial(f_old),
               B_old_memory_sum=mem, B_budget=eps / 2, B_least_f_same_model=f_need)
    # (A) quadratic law
    out.update(A_old_quadratic_coeff=35, A_exact_leading_coeff=fac["rm15"][1][2],
               A_proved_C_dist_LC=math.comb(15 + a["cY_LC"], 2),
               A_proved_C_dist_SC=math.comb(15 + a["cY_SC"], 2))
    # (C) old serial layer-code space
    cn, cK, reserve = 509.0, 0.47, 0.5
    f_lc = next(f for f in range(3, 400, 2)
                if W * T * reserve * max(1.0, cK * f ** 0.5) * runs * 30 * 3 * f * pl(f) <= eps / 2)
    K = max(1.0, cK * f_lc ** 0.5)
    Kd = max(1.0, reserve * K)
    cells = math.ceil(W / Kd)
    old_blocks = cells + 1
    per_fac = math.ceil(16 / K)
    out.update(C_old_f_lc=f_lc, C_old_K=K, C_old_reserved_slots_per_block=K - Kd,
               C_old_blocks_allocated=old_blocks, C_factory_blocks_needed_per_factory=per_fac,
               C_factories_used=cells, C_factory_blocks_needed_total=cells * per_fac,
               # the old allocation had one block beyond the data cells; credit it
               C_factory_blocks_missing=cells * per_fac - (old_blocks - cells))
    return out


# ============================================================ helpers
def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}" if abs(x) < 10 ** 7 else f"{float(x):.2e}"
    if isinstance(x, float):
        if x == 0:
            return "0"
        if math.isnan(x):
            return "-"
        e = math.floor(math.log10(abs(x)))
        return f"{x:.3g}" if -2 <= e < 5 else f"{x:.2e}"
    return str(x)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k, (v, meaning) in ASSUMPTIONS.items():
        if isinstance(v, int) and not isinstance(v, bool):
            typ = lambda x: int(float(x))
        else:
            typ = type(v)
        ap.add_argument(f"--{k}", type=typ, default=v, help=meaning)
    return vars(ap.parse_args())


COLS = ["noise_point", "p", "q0", "scenario", "model", "architecture", "schedule", "converged",
        "note", "levels", "q_final", "W_T_q_final", "f", "N", "K", "K_d", "n", "d_block",
        "n_data", "n_working", "n_factory", "n_courier", "units_total", "unit",
        "factory_blocks_per_data_block", "a_runs_per_resource", "y_runs_per_resource",
        "raw_injections_per_resource", "L_live", "peak_qubits", "layer_rounds", "rounds",
        "seconds", "qubit_seconds", "memory_failure", "eps_mem"]


def main():
    t0 = time.time()
    a = parse_args()
    eps_mem, eps_res = a["eps"] / 2, a["eps"] / 2
    target = eps_res / (a["W"] * a["T"])
    checks = {}

    fac = {"rm15": enumerate_hamming_factory(4), "steane": enumerate_hamming_factory(3)}
    checks["RM15 always-output counts match the stated list"] = fac["rm15"][1] == RM15_ALWAYS_OUTPUT_EXPECTED
    checks["RM15 accepted patterns = 2048 Hamming codewords"] = sum(fac["rm15"][2]) == 2048
    fam = Family(a)
    noise = {"base": (a["p_base"], a["q0_base"]), "secondary": (a["p_secondary"], a["q0_secondary"])}

    rows = []
    for npt, (p, q0) in noise.items():
        dists = {(m, arch): distill(m, a["cY_LC"] if arch == "layer_code" else a["cY_SC"], q0, target, a, fac)
                 for m in MODELS for arch in ("layer_code", "surface_code")}
        sc_cache = {}
        for scen in SCENARIOS:
            for m in MODELS:
                for arch, sched in SCHEDULES:
                    dist = dists[(m, arch)]
                    row = dict(noise_point=npt, p=p, q0=q0, scenario=scen, model=m,
                               architecture=arch, schedule=sched, converged=dist["converged"],
                               note=dist["reason"] + ("; not covered by the paper's theorem"
                                                      if m == "postselected_optimistic" else ""),
                               levels=dist["levels"], q_final=dist["q_final"],
                               unit="block" if arch == "layer_code" else "tile", eps_mem=eps_mem)
                    if dist["converged"]:
                        row["W_T_q_final"] = a["W"] * a["T"] * dist["q_final"]
                        row.update(a_runs_per_resource=dist["a_runs"], y_runs_per_resource=dist["y_runs"],
                                   raw_injections_per_resource=dist["inj"])
                        if arch == "layer_code":
                            fn = lambda f: lc_layout(a, fam, scen, sched, f, dist)
                            lay = choose_distance(a, p, fn)
                        else:
                            fn = lambda f: sc_layout(a, sched, f, dist)
                            key = (m, sched)
                            if key not in sc_cache:
                                sc_cache[key] = choose_distance(a, p, fn)
                            lay = dict(sc_cache[key])
                        # independent minimality evidence: the full layout at f - 2 misses the budget
                        if lay["f"] > 3:
                            lm = fn(lay["f"] - 2)
                            row["_mem_f_minus_2"] = lm["L_live"] * lm["rounds"] * p_L(a, p, lay["f"] - 2)
                        else:
                            row["_mem_f_minus_2"] = math.inf
                        lay["seconds"] = lay["rounds"] * lay["round_s"]
                        lay["qubit_seconds"] = lay["peak_qubits"] * lay["seconds"]
                        row.update({k: lay[k] for k in COLS if k in lay})
                    rows.append(row)

    # ---------------------------------------------------------- checks
    conv = [r for r in rows if r["converged"]]
    # recomputed from the reported L_live, rounds and f rather than trusting memory_failure
    checks["every converged row meets the memory budget"] = all(
        r["L_live"] * r["rounds"] * p_L(a, r["p"], r["f"]) <= eps_mem * (1 + 1e-12) for r in conv)
    checks["every converged row uses the least passing f (f - 2 fails)"] = all(
        r["_mem_f_minus_2"] > eps_mem for r in conv)
    checks["every converged row meets the resource budget"] = all(r["W_T_q_final"] <= eps_res * (1 + 1e-12) for r in conv)
    checks["all allocations are integers"] = all(
        all(isinstance(r[k], (int, np.integer)) for k in ("n_data", "n_working", "n_factory", "n_courier", "units_total"))
        for r in conv)
    # each serial layer-code factory holds its S_fac logical slots
    ok = True
    for r in conv:
        if r["architecture"] == "layer_code" and r["schedule"] == "serial":
            l = r["levels"]
            S = 15 * l + 7 * max(l - 1, 0) + 2
            ok &= r["n_factory"] * r["K"] >= r["n_data"] * S
    checks["serial layer-code factories hold their logical slots"] = ok
    audit = audit_old_model(a, fac)
    checks["audit (C): old serial layer code needs ceil(16/K) = 7 blocks per factory at f = 27"] = (
        audit["C_old_f_lc"] == 27 and audit["C_factory_blocks_needed_per_factory"] == 7
        and audit["C_factory_blocks_missing"] > audit["C_old_blocks_allocated"]) if (a["W"], a["T"], a["eps"], a["p_base"], a["q0_base"]) == (100, 1_000_000, 0.01, 1e-3, 1e-2) else True
    checks["audit (B): old SC serial memory sum 4.048 at f=23, least passing f=31"] = (
        audit["B_old_f_sc"] == 23 and abs(audit["B_old_memory_sum"] - 4.048) < 1e-3
        and audit["B_least_f_same_model"] == 31) if (a["W"], a["T"], a["eps"], a["p_base"], a["q0_base"]) == (100, 1_000_000, 0.01, 1e-3, 1e-2) else True
    checks["audit (A): exact leading coefficient is 105"] = audit["A_exact_leading_coeff"] == 105

    xo = {sc: storage_crossover(a, fam, sc, "computation") for sc in SCENARIOS}
    xo_storage = {sc: storage_crossover(a, fam, sc, "storage") for sc in SCENARIOS}
    thr = {f"{m}|{arch}": convergence_threshold(m, a["cY_LC"] if arch == "layer_code" else a["cY_SC"], a, fac)
           for m in MODELS for arch in ("layer_code", "surface_code")}

    # ---------------------------------------------------------- CSV
    with open(ROOT / "resource_estimate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") if r.get(k) is not None else "" for k in COLS})

    # ---------------------------------------------------------- JSON
    summ = {"target_q_final_per_resource": target, "eps_mem": eps_mem, "eps_res": eps_res,
            "base_small_instance": {}, "storage_crossover": xo,
            "storage_crossover_storage_only": xo_storage, "audit_old_model": audit,
            "q0_convergence_threshold": thr,
            "distillation_nonconvergence": {}}
    for r in rows:
        if r["noise_point"] == "base" and r["scenario"] == "small_instance":
            key = f"{r['model']}|{r['architecture']}|{r['schedule']}"
            summ["base_small_instance"][key] = {k: r.get(k) for k in (
                "converged", "note", "levels", "q_final", "f", "N", "K", "n", "n_data", "n_working",
                "n_factory", "n_courier", "units_total", "peak_qubits", "rounds", "seconds",
                "qubit_seconds", "memory_failure")}
        if not r["converged"]:
            summ["distillation_nonconvergence"][f"{r['noise_point']}|{r['model']}|{r['architecture']}"] = r["note"]
    (ROOT / "resource_estimate_summary.json").write_text(
        json.dumps(summ, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)),
        newline="\n")

    # ---------------------------------------------------------- Markdown
    L = ["# Resource estimate (explicit assumptions, explicit allocations)", "",
         "Every schedule has an integer allocation of blocks (layer code) or tiles (surface code) by role, "
         "and its distance f is the least odd f with L_live x R_total x p_L(f) <= eps_mem, where L_live counts "
         "every live logical qubit (data, working, factory and courier) and R_total is the complete runtime of "
         "that schedule including factory waiting. The model 'postselected_optimistic' is NOT covered by the "
         "paper's theorem.", "",
         "## Assumptions", "", "| assumption | value | meaning |", "|---|---|---|"]
    for k, (_, meaning) in ASSUMPTIONS.items():
        L.append(f"| `{k}` | {a[k]} | {meaning} |")
    L += ["", f"Derived: eps_mem = {eps_mem}, eps_res = {eps_res}, required q_final <= eps_res/(W T) = {target:.3g}. "
          f"Layer-code distance scenarios: slab_upper d = L_z d_in; small_instance d = L_z d_in / 2; "
          f"proved_lower d = L_z ceil(d_in / eta), eta = {fam.eta}.", "",
          "## Distillation maps", "",
          f"- proved: q_(r+1) = C_dist q_r^2 with C_dist = binom(15 + c_Y, 2) = {math.comb(15 + a['cY_LC'], 2)} "
          f"(layer code, c_Y = {a['cY_LC']}) and {math.comb(15 + a['cY_SC'], 2)} (surface code, c_Y = {a['cY_SC']}). "
          "The Steane constant binom(7,2) = 21 is smaller; A and Y share the family parameter q_r.",
          f"- exact_always_output, RM15 malignant counts by weight (enumerated): {fac['rm15'][1]}",
          f"- exact_always_output, Steane 7-to-1 malignant counts by weight (enumerated): {fac['steane'][1]}",
          f"- postselected_optimistic, RM15 accepted counts {fac['rm15'][2]}, accepted logical {fac['rm15'][3]}; "
          f"Steane accepted {fac['steane'][2]}, accepted logical {fac['steane'][3]}.",
          "- Layer-code A factory per-coordinate error q_c = q_A + q_Y/2 - q_A q_Y; surface code q_c = q_A.", "",
          "## Largest raw error q0 at which each recursion converges", "",
          "| model | layer code (c_Y = %d) | surface code (c_Y = %d) |" % (a["cY_LC"], a["cY_SC"]), "|---|---|---|"]
    for m in MODELS:
        L.append(f"| {m} | {fmt(thr[m + '|layer_code'])} | {fmt(thr[m + '|surface_code'])} |")
    L += ["",
          "## Audit of the old model", "",
          f"- (A) old 'quadratic' law 35 q^2 versus the exact always-output leading coefficient "
          f"{audit['A_exact_leading_coeff']} (q_out = 105 q^2 + O(q^3)); the proved bound is C_dist q^2 with "
          f"C_dist = {audit['A_proved_C_dist_LC']} (layer code) or {audit['A_proved_C_dist_SC']} (surface code).",
          f"- (B) old surface-code serial distance f = {audit['B_old_f_sc']} was chosen from a 2f-round layer; the "
          f"serial runtime is {audit['B_old_serial_rounds']:.4g} rounds and W R_total p_L(f) = "
          f"{audit['B_old_memory_sum']:.4g} against a budget of {audit['B_budget']}; the least f passing under the "
          f"same model is {audit['B_least_f_same_model']}.",
          f"- (C) old serial layer code at f = {audit['C_old_f_lc']}: K = {audit['C_old_K']:.3f}, reserve leaves "
          f"{audit['C_old_reserved_slots_per_block']:.2f} logical slots per block, {audit['C_old_blocks_allocated']} "
          f"blocks allocated in total, but each of the {audit['C_factories_used']} factories needs ceil(16/K) = "
          f"{audit['C_factory_blocks_needed_per_factory']} blocks, i.e. {audit['C_factory_blocks_needed_total']} factory "
          f"blocks in total, of which {audit['C_factory_blocks_missing']} were never allocated (the one block beyond "
          "the data cells is credited).", "",
          "## Storage crossover", "",
          "computation: block with extraction ancillas and gauging grid, n (1 + anc_LC + m0 grid_ratio) per K_d "
          "sites, versus route_SC surface-code tiles per site (the accounting of the space tables); "
          "storage: block with extraction ancillas only versus one tile. 'first d' is the least distance at which "
          "the layer code is cheaper; 'stable from' is the distance from which it stays cheaper.", "",
          "| accounting | scenario | first d | stable from | N | K | K_d | n | qubits per logical (LC) | qubits per logical (SC) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for label, table in (("computation", xo), ("storage", xo_storage)):
        for sc, x in table.items():
            if x["d"] is None:
                L.append(f"| {label} | {sc} | none up to N_max | - | - | - | - | - | - | - |")
            else:
                L.append(f"| {label} | {sc} | {x['d']} | {x['d_stable']} | {x['N']} | {x['K']} | {x['K_d']} | "
                         f"{fmt(x['n'])} | {fmt(float(x['lc_per_logical']))} | {fmt(float(x['sc_per_logical']))} |")

    hdr = ("| model | architecture | schedule | converged | levels | q_final | f | N | K | n (block or tile) | "
           "data | working/routing | factory | courier | peak qubits | rounds | seconds | qubit-seconds | memory failure |")
    sep = "|" + "---|" * 19

    def table_row(r):
        if not r["converged"]:
            return (f"| {r['model']} | {r['architecture']} | {r['schedule']} | False | - | {fmt(r['q_final'])} | "
                    f"- | - | - | - | - | - | - | - | - | - | - | - | {r['note'].split(';')[0]} |")
        return (f"| {r['model']} | {r['architecture']} | {r['schedule']} | True | {r['levels']} | "
                f"{fmt(r['q_final'])} | {r['f']} | {fmt(r['N'])} | {r['K']} | {fmt(r['n'])} | {fmt(r['n_data'])} | "
                f"{fmt(r['n_working'])} | {fmt(r['n_factory'])} | {fmt(r['n_courier'])} | "
                f"{fmt(float(r['peak_qubits']))} | {fmt(float(r['rounds']))} | {fmt(float(r['seconds']))} | "
                f"{fmt(float(r['qubit_seconds']))} | {fmt(float(r['memory_failure']))} |")

    main_tables = []
    for npt, (p, q0) in noise.items():
        t = [f"## Main table: {npt} noise point (p = {p}, q0 = {q0}), small_instance scenario", "", hdr, sep]
        t += [table_row(r) for r in rows if r["noise_point"] == npt and r["scenario"] == "small_instance"]
        main_tables += t + [""]
    L += [""] + main_tables
    for npt, (p, q0) in noise.items():
        L += [f"## All scenarios: {npt} noise point (p = {p}, q0 = {q0})", "",
              "Surface-code rows do not depend on the layer-code scenario and repeat across scenarios.", ""]
        for sc in SCENARIOS:
            L += [f"### {sc}", "", hdr, sep]
            L += [table_row(r) for r in rows if r["noise_point"] == npt and r["scenario"] == sc]
            L.append("")
    L += ["## Checks", ""] + [f"- {'ok' if v else 'FAILED'}: {k}" for k, v in checks.items()]
    (ROOT / "resource_estimate.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    # ---------------------------------------------------------- console
    print("Audit of the old model:")
    for k, v in audit.items():
        print(f"  {k} = {fmt(v) if isinstance(v, float) else v}")
    print()
    print("Storage crossover (computation):", {sc: (x.get("d"), x.get("d_stable")) for sc, x in xo.items()})
    print("Storage crossover (storage only):", {sc: (x.get("d"), x.get("d_stable")) for sc, x in xo_storage.items()})
    print()
    print("\n".join(main_tables))
    for k, v in checks.items():
        print(f"{'ok' if v else 'FAILED'}: {k}")
    elapsed = time.time() - t0
    status = "PASS" if all(checks.values()) else "FAIL"
    print(f"{status}: resource_estimate ({len(rows)} rows, {len(conv)} converged, {elapsed:.1f} s)")


if __name__ == "__main__":
    main()
