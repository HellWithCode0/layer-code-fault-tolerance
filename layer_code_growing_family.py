#!/usr/bin/env python3
"""Plan A: Growing-distance layer-code-inspired CSS family.

Primary construction: hypergraph product (HP) of classical repetition codes
HGP(rep_L, rep_L). This is the algebraic planar-surface / layer-code HP family
with k=1 and code distance d=L (exact on the kernel min-weight upper bound).
It is the standard CSS lift underlying layer-code / HP constructions and is
WB-junction-compatible at the classical-input level (repetition syndromes map
to interlayer / corridor checks in the geometric picture).

Also ships rotated-surface patch instances extracted from Stim (geometric
growing patches) and rectangular HGP(rep_m, rep_d) layer stacks.

Writes Hx/Hz npz per instance + family CSV.

SCOPE: growing-distance layer-code-geometry / HP-surface CSS family for finite-size
threshold hunts; NOT a CAD reconstruction of published Williamson–Baspin artwork.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import stim
from ldpc.codes import rep_code, ring_code

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "growing_family"
OUT_DIR.mkdir(exist_ok=True)

SCOPE = (
    "growing-distance layer-code-geometry / HP-surface CSS family "
    "(HGP of classical repetition codes + Stim rotated patches); "
    "NOT CAD reconstruction of published WB artwork"
)


def gf2_rank(a: np.ndarray) -> int:
    a = (a.copy() % 2).astype(int)
    if a.size == 0:
        return 0
    m, n = a.shape
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if a[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        for i in range(m):
            if i != r and a[i, c]:
                a[i] = (a[i] + a[r]) % 2
        r += 1
    return r


def to_dense(H) -> np.ndarray:
    if hasattr(H, "toarray"):
        H = H.toarray()
    return np.asarray(H, dtype=int) % 2


def nullspace_gf2(H: np.ndarray):
    A = (H.copy() % 2).astype(int)
    m, n = A.shape
    r = 0
    pivots = {}
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] = (A[i] + A[r]) % 2
        pivots[c] = r
        r += 1
        if r == m:
            break
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=int)
        v[f] = 1
        for c, pr in pivots.items():
            v[c] = A[pr, f]
        basis.append(v)
    return basis


def in_rowspace(H: np.ndarray, v: np.ndarray) -> bool:
    if H.size == 0:
        return bool(np.all(v == 0))
    return gf2_rank(np.vstack([H, v.reshape(1, -1)])) == gf2_rank(H)


def dedup_rows(H: np.ndarray) -> np.ndarray:
    if H.size == 0:
        return H
    seen, rows = set(), []
    for row in H:
        if row.sum() == 0:
            continue
        key = tuple(int(x) for x in row.tolist())
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, H.shape[1]), dtype=int)


def min_logicals(Hx: np.ndarray, Hz: np.ndarray):
    """Return min-weight X/Z logical representatives and weights."""
    z_cands = [v for v in nullspace_gf2(Hx) if v.sum() > 0 and not in_rowspace(Hz, v)]
    x_cands = [v for v in nullspace_gf2(Hz) if v.sum() > 0 and not in_rowspace(Hx, v)]
    # Reduce: keep a basis of logicals, take min weight among small linear combos
    def refine(cands, Hstab, max_basis=8):
        if not cands:
            return None, None, 0
        cands = sorted(cands, key=lambda v: int(v.sum()))
        basis = []
        for v in cands:
            if len(basis) >= max_basis:
                break
            trial = basis + [v]
            # linearly independent mod stabilizers approx via stacked matrix
            M = np.vstack(trial + ([Hstab] if Hstab.size else []))
            if gf2_rank(np.vstack(trial)) > gf2_rank(np.vstack(basis) if basis else np.zeros((0, v.size), dtype=int)):
                if not in_rowspace(Hstab, v):
                    basis.append(v)
        if not basis:
            basis = [cands[0]]
        best = min(basis, key=lambda v: int(v.sum()))
        best_wt = int(best.sum())
        # try XOR pairs
        for i in range(len(basis)):
            for j in range(i + 1, len(basis)):
                v = (basis[i] + basis[j]) % 2
                if v.sum() and not in_rowspace(Hstab, v) and int(v.sum()) < best_wt:
                    best, best_wt = v, int(v.sum())
        return best.astype(int), best_wt, len(cands)

    logZ, dZ, nz = refine(z_cands, Hz)
    logX, dX, nx = refine(x_cands, Hx)
    return {
        "X": logX,
        "Z": logZ,
        "X_weight": dX,
        "Z_weight": dZ,
        "n_xlogs": nx,
        "n_zlogs": nz,
        "X_is_logical": logX is not None and bool(np.all((Hz @ logX) % 2 == 0) and not in_rowspace(Hx, logX)),
        "Z_is_logical": logZ is not None and bool(np.all((Hx @ logZ) % 2 == 0) and not in_rowspace(Hz, logZ)),
    }


def hgp(H1, H2):
    H1, H2 = to_dense(H1), to_dense(H2)
    r1, n1 = H1.shape
    r2, n2 = H2.shape
    Hx = np.hstack([
        np.kron(H1, np.eye(n2, dtype=int)),
        np.kron(np.eye(r1, dtype=int), H2.T),
    ]) % 2
    Hz = np.hstack([
        np.kron(np.eye(n1, dtype=int), H2),
        np.kron(H1.T, np.eye(r2, dtype=int)),
    ]) % 2
    Hx, Hz = dedup_rows(Hx), dedup_rows(Hz)
    assert np.all((Hx @ Hz.T) % 2 == 0)
    return Hx.astype(int), Hz.astype(int)


def build_hp_surface(L: int):
    """Plan A primary: HGP(rep_L, rep_L) — planar HP / layer-code algebra, d=L, k=1."""
    Hx, Hz = hgp(rep_code(L), rep_code(L))
    logs = min_logicals(Hx, Hz)
    n = Hx.shape[1]
    rx, rz = gf2_rank(Hx), gf2_rank(Hz)
    k = n - rx - rz
    meta = {
        "family": "hp_surface",
        "construction": "HGP(rep_L, rep_L)",
        "L": L,
        "d_param": L,
        "m_layers": L,
        "n": n,
        "k": k,
        "rank_Hx": rx,
        "rank_Hz": rz,
        "css_ok": 1,
        "d_upper": min(logs["X_weight"], logs["Z_weight"]) if logs["X_weight"] and logs["Z_weight"] else None,
        "X_logical_weight": logs["X_weight"],
        "Z_logical_weight": logs["Z_weight"],
        "scope": SCOPE,
    }
    return Hx, Hz, meta, logs


def build_hp_layer_stack(d: int, m: int):
    """Rectangular layer stack: HGP(rep_m, rep_d) — m classical layers × d in-plane."""
    Hx, Hz = hgp(rep_code(m), rep_code(d))
    logs = min_logicals(Hx, Hz)
    n = Hx.shape[1]
    rx, rz = gf2_rank(Hx), gf2_rank(Hz)
    k = n - rx - rz
    meta = {
        "family": "hp_layer_stack",
        "construction": f"HGP(rep_m={m}, rep_d={d})",
        "L": max(d, m),
        "d_param": d,
        "m_layers": m,
        "n": n,
        "k": k,
        "rank_Hx": rx,
        "rank_Hz": rz,
        "css_ok": 1,
        "d_upper": min(logs["X_weight"], logs["Z_weight"]) if logs["X_weight"] and logs["Z_weight"] else None,
        "X_logical_weight": logs["X_weight"],
        "Z_logical_weight": logs["Z_weight"],
        "scope": SCOPE,
    }
    return Hx, Hz, meta, logs


def build_hp_toric(L: int):
    """Plan B proxy: HGP(ring_L, ring_L) toric, k=2, d=L."""
    Hx, Hz = hgp(ring_code(L), ring_code(L))
    logs = min_logicals(Hx, Hz)
    n = Hx.shape[1]
    rx, rz = gf2_rank(Hx), gf2_rank(Hz)
    k = n - rx - rz
    meta = {
        "family": "hp_toric",
        "construction": "HGP(ring_L, ring_L)",
        "L": L,
        "d_param": L,
        "m_layers": L,
        "n": n,
        "k": k,
        "rank_Hx": rx,
        "rank_Hz": rz,
        "css_ok": 1,
        "d_upper": min(logs["X_weight"], logs["Z_weight"]) if logs["X_weight"] and logs["Z_weight"] else None,
        "X_logical_weight": logs["X_weight"],
        "Z_logical_weight": logs["Z_weight"],
        "scope": SCOPE,
    }
    return Hx, Hz, meta, logs


def build_rotated_surface_stim(d: int):
    """Geometric growing patch: Stim rotated surface, d odd, k=1."""
    assert d % 2 == 1
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=1,
        after_clifford_depolarization=0.0,
    )
    coords = {}
    for op in circ:
        if op.name == "QUBIT_COORDS":
            coords[op.targets_copy()[0].value] = tuple(op.gate_args_copy())
    data_qs = sorted(
        q for q, c in coords.items() if int(c[0]) % 2 == 1 and int(c[1]) % 2 == 1
    )
    meas_qs = sorted(set(coords) - set(data_qs))
    x_meas = set()
    cx_edges = []
    for op in circ:
        if op.name == "H" and not x_meas:
            for t in op.targets_copy():
                x_meas.add(t.value)
        elif op.name == "CX":
            ts = [t.value for t in op.targets_copy()]
            for i in range(0, len(ts), 2):
                cx_edges.append((ts[i], ts[i + 1]))
    z_meas = set(meas_qs) - x_meas
    xsup, zsup = defaultdict(set), defaultdict(set)
    for c, t in cx_edges:
        if c in x_meas and t in data_qs:
            xsup[c].add(t)
        elif t in z_meas and c in data_qs:
            zsup[t].add(c)
        elif c in z_meas and t in data_qs:
            zsup[c].add(t)
        elif t in x_meas and c in data_qs:
            xsup[t].add(c)
    dmap = {q: i for i, q in enumerate(data_qs)}
    n = len(data_qs)

    def mat(sup):
        rows = []
        for mq in sorted(sup):
            r = np.zeros(n, dtype=int)
            for q in sup[mq]:
                if q in dmap:
                    r[dmap[q]] = 1
            if r.sum() >= 2:
                rows.append(r)
        return np.vstack(rows) if rows else np.zeros((0, n), dtype=int)

    Hx, Hz = dedup_rows(mat(xsup)), dedup_rows(mat(zsup))
    assert np.all((Hx @ Hz.T) % 2 == 0)
    logs = min_logicals(Hx, Hz)
    rx, rz = gf2_rank(Hx), gf2_rank(Hz)
    k = n - rx - rz
    meta = {
        "family": "rotated_surface_stim",
        "construction": "stim.rotated_memory_z CSS extract",
        "L": d,
        "d_param": d,
        "m_layers": 1,
        "n": n,
        "k": k,
        "rank_Hx": rx,
        "rank_Hz": rz,
        "css_ok": 1,
        "d_upper": min(logs["X_weight"], logs["Z_weight"]) if logs["X_weight"] and logs["Z_weight"] else None,
        "X_logical_weight": logs["X_weight"],
        "Z_logical_weight": logs["Z_weight"],
        "scope": SCOPE,
    }
    return Hx, Hz, meta, logs


def save_instance(tag: str, Hx, Hz, meta, logs):
    path = OUT_DIR / f"{tag}.npz"
    np.savez_compressed(
        path,
        Hx=Hx.astype(np.uint8),
        Hz=Hz.astype(np.uint8),
        X_logical=(logs["X"] if logs["X"] is not None else np.zeros(Hx.shape[1], dtype=int)).astype(np.uint8),
        Z_logical=(logs["Z"] if logs["Z"] is not None else np.zeros(Hx.shape[1], dtype=int)).astype(np.uint8),
        L=np.int32(meta["L"]),
        n=np.int32(meta["n"]),
        k=np.int32(meta["k"]),
        d_upper=np.int32(meta["d_upper"] or 0),
    )
    return path


def main() -> int:
    rows = []
    # Primary Plan A family: HP-surface L=3,5,7,9
    print("=== Plan A primary: HGP(rep_L, rep_L) ===")
    for L in (3, 5, 7, 9):
        Hx, Hz, meta, logs = build_hp_surface(L)
        tag = f"hp_surface_L{L}"
        save_instance(tag, Hx, Hz, meta, logs)
        ok = meta["css_ok"] and meta["k"] >= 1 and logs["X_is_logical"] and logs["Z_is_logical"]
        print(
            f"  {tag}: n={meta['n']} k={meta['k']} d_upper={meta['d_upper']} "
            f"Xwt={meta['X_logical_weight']} Zwt={meta['Z_logical_weight']} ok={ok}"
        )
        rows.append({**meta, "tag": tag, "ok": int(ok)})

    # Layer stacks (d,m)
    print("=== Plan A stacks: HGP(rep_m, rep_d) ===")
    for d, m in ((3, 3), (5, 3), (5, 5), (7, 3), (7, 5), (7, 7)):
        Hx, Hz, meta, logs = build_hp_layer_stack(d, m)
        tag = f"hp_stack_d{d}_m{m}"
        save_instance(tag, Hx, Hz, meta, logs)
        ok = meta["css_ok"] and meta["k"] >= 1 and logs["X_is_logical"] and logs["Z_is_logical"]
        print(
            f"  {tag}: n={meta['n']} k={meta['k']} d_upper={meta['d_upper']} ok={ok}"
        )
        rows.append({**meta, "tag": tag, "ok": int(ok)})

    # Geometric patches
    print("=== Geometric: Stim rotated surface ===")
    for d in (3, 5, 7, 9):
        Hx, Hz, meta, logs = build_rotated_surface_stim(d)
        tag = f"rot_surface_d{d}"
        save_instance(tag, Hx, Hz, meta, logs)
        ok = meta["css_ok"] and meta["k"] >= 1 and logs["X_is_logical"] and logs["Z_is_logical"]
        print(
            f"  {tag}: n={meta['n']} k={meta['k']} d_upper={meta['d_upper']} ok={ok}"
        )
        rows.append({**meta, "tag": tag, "ok": int(ok)})

    # Plan B toric proxy
    print("=== Plan B: HGP toric ===")
    for L in (3, 4, 5, 6, 7):
        Hx, Hz, meta, logs = build_hp_toric(L)
        tag = f"hp_toric_L{L}"
        save_instance(tag, Hx, Hz, meta, logs)
        ok = meta["css_ok"] and meta["k"] >= 1 and logs["X_is_logical"] and logs["Z_is_logical"]
        print(
            f"  {tag}: n={meta['n']} k={meta['k']} d_upper={meta['d_upper']} ok={ok}"
        )
        rows.append({**meta, "tag": tag, "ok": int(ok)})

    out = ROOT / "layer_code_growing_family.csv"
    fields = [
        "tag", "family", "construction", "L", "d_param", "m_layers", "n", "k",
        "rank_Hx", "rank_Hz", "css_ok", "d_upper", "X_logical_weight",
        "Z_logical_weight", "ok", "scope",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.name} and {OUT_DIR}/")
    # Select primary threshold family: hp_surface L=3,5,7 (growing d, n manageable)
    primary = [r for r in rows if r["family"] == "hp_surface" and r["L"] in (3, 5, 7)]
    print("PRIMARY threshold family (hp_surface):")
    for r in primary:
        print(f"  L={r['L']} n={r['n']} d_upper={r['d_upper']} k={r['k']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
