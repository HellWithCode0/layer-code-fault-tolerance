#!/usr/bin/env python3
"""Gap 5: Efficient practical decoder timing on WB memory/spacetime instances.

SCOPE: empirical polynomial practical scaling on these finite instances;
NOT a complexity proof for asymptotic architecture decoding.
"""
from __future__ import annotations
import csv, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ldpc import BpOsdDecoder
from wb_layer_code_instance import build_repetition_layer_code, quasiconcat_and_kernel_logicals
from wb_layer_spacetime_schedule import spacetime_check_matrix, R_ROUNDS, OSD_ORDER

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
SEED = 20260918
L_VALUES = (3, 4, 5, 6)
P = 0.02
N_TRIALS = 40
SCOPE = (
    "empirical decoder wall-time vs n on WB L in {3,4,5,6} memory and spacetime; "
    "practical BP-OSD scaling demo; NOT asymptotic complexity proof"
)

def time_bposd(H, p, osd_order, syndromes):
    dec = BpOsdDecoder(np.asarray(H, dtype=np.uint8), error_rate=float(p),
                       max_iter=30, bp_method="minimum_sum", osd_method="osd_cs",
                       osd_order=int(osd_order))
    _ = dec.decode(syndromes[0])
    t0 = time.perf_counter()
    for i in range(len(syndromes)):
        _ = dec.decode(syndromes[i])
    return (time.perf_counter() - t0) / len(syndromes)

def try_mwpm(H, syndromes):
    try:
        import pymatching
        matching = pymatching.Matching.from_check_matrix(H, weights=np.ones(H.shape[1]))
        _ = matching.decode(syndromes[0])
        t0 = time.perf_counter()
        for i in range(len(syndromes)):
            _ = matching.decode(syndromes[i])
        return (time.perf_counter() - t0) / len(syndromes), "pymatching.Matching.from_check_matrix"
    except Exception as e:
        return None, f"MWPM_unavailable:{type(e).__name__}"

def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for L in L_VALUES:
        Hx, Hz, meta = build_repetition_layer_code(L)
        logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
        n = meta["n"]; R = R_ROUNDS[L]; order = OSD_ORDER[L]
        syn_mem = np.asarray([((Hx @ (rng.random(n) < P).astype(np.uint8)) % 2) for _ in range(N_TRIALS)], dtype=np.uint8)
        t_mem = time_bposd(Hx, P, order, syn_mem)
        t_mwpm, mwpm_note = try_mwpm(Hx, syn_mem)
        Hst = spacetime_check_matrix(Hx, R)
        n_st = Hst.shape[1]
        syn_st = np.asarray([((Hst @ (rng.random(n_st) < P).astype(np.uint8)) % 2) for _ in range(N_TRIALS)], dtype=np.uint8)
        t_st = time_bposd(Hst, P, order, syn_st)
        rec = dict(L=L, n=n, R=R, n_memory_checks=int(Hx.shape[0]),
                   n_spacetime_vars=int(Hst.shape[1]), n_spacetime_checks=int(Hst.shape[0]),
                   d_upper=min(logs["X_weight"], logs["Z_weight"]), p=P, trials=N_TRIALS,
                   osd_order=order, t_bposd_memory_s=t_mem, t_bposd_spacetime_s=t_st,
                   t_mwpm_memory_s="" if t_mwpm is None else t_mwpm, mwpm_note=mwpm_note,
                   decoder_primary="ldpc.BpOsdDecoder", scope=SCOPE)
        rows.append(rec)
        print(f"L={L} n={n}: t_mem={t_mem*1e3:.2f}ms t_st={t_st*1e3:.2f}ms mwpm={t_mwpm if t_mwpm else mwpm_note}")
    out = ROOT / "wb_decoder_scaling.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    ns = [r["n"] for r in rows]
    ax.plot(ns, [r["t_bposd_memory_s"]*1e3 for r in rows], "o-", label="BP-OSD memory")
    ax.plot(ns, [r["t_bposd_spacetime_s"]*1e3 for r in rows], "s-", label="BP-OSD spacetime")
    if all(r["t_mwpm_memory_s"] != "" for r in rows):
        ax.plot(ns, [float(r["t_mwpm_memory_s"])*1e3 for r in rows], "^-", label="MWPM memory")
    ax.set_xlabel(r"$n$"); ax.set_ylabel("mean decode time (ms)")
    ax.set_title("WB decoder scaling (empirical)"); ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7); fig.tight_layout()
    fig.savefig(FIG/"wb_decoder_scaling.pdf"); fig.savefig(FIG/"wb_decoder_scaling.png", dpi=240)
    plt.close(fig)
    print(f"wrote {out.name}; SCOPE: {SCOPE}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
