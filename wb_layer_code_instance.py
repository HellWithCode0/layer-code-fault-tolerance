#!/usr/bin/env python3
"""Finite WB-incidence-derived test instance for 3-qubit Z-repetition.

Truncated finite CSS test instance using audited WB Sec. VI B junction incidences
for classical input S=⟨Z1Z2, Z2Z3⟩ (3 xz data layers + 2 yz check corridors).

Novel ẑ junction SX checks use audited first/last supports from
wb_junction_support_catalog.csv (Sec. VI B items 3 and 5), translated along
junction lines. Ordinary in-plane xz stars (smooth at x-boundaries), yz
corridor stars, and xz/yz plaquettes fill the truncated layers; only
CSS-commuting X candidates are retained. Spurious weight-2 open-boundary
modes of the truncation are condensed as additional local Z stabilizers so
the codespace isolates a single logical pair (k=1), matching the expected
concatenated surface⊗repetition sector.

SCOPE: truncated finite test instance using audited Williamson--Baspin Sec. VI B
junction incidences for the 3-qubit-repetition input; not a literal finite
reconstruction or metric calibration of the complete published lattice artwork.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

_EDGE_DELTA = {
    "xm": ((-1, 0, 0), (0, 0, 0)),
    "xp": ((0, 0, 0), (1, 0, 0)),
    "ym": ((0, -1, 0), (0, 0, 0)),
    "yp": ((0, 0, 0), (0, 1, 0)),
    "zm": ((0, 0, -1), (0, 0, 0)),
    "zp": ((0, 0, 0), (0, 0, 1)),
}
FIRST_Z_SX = ("xm", "xp", "yp", "zm", "zp")
LAST_Z_SX = ("xm", "xp", "ym", "zm", "zp")


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


def _canon_edge(a, b):
    a, b = tuple(int(x) for x in a), tuple(int(x) for x in b)
    return (a, b) if a <= b else (b, a)


def _translate_edge(label: str, origin):
    p0, p1 = _EDGE_DELTA[label]
    ox, oy, oz = origin
    return _canon_edge(
        (ox + p0[0], oy + p0[1], oz + p0[2]),
        (ox + p1[0], oy + p1[1], oz + p1[2]),
    )


class EdgeQubitIndex:
    def __init__(self):
        self._edge_to_i = {}
        self._i_to_edge = []

    def ensure(self, a, b) -> int:
        e = _canon_edge(a, b)
        if e not in self._edge_to_i:
            self._edge_to_i[e] = len(self._i_to_edge)
            self._i_to_edge.append(e)
        return self._edge_to_i[e]

    def get(self, a, b):
        return self._edge_to_i.get(_canon_edge(a, b))

    @property
    def n(self) -> int:
        return len(self._i_to_edge)

    def edge(self, i: int):
        return self._i_to_edge[i]


def remove_zero_and_duplicate_rows(H, labels):
    if H.size == 0:
        return H, labels
    keep, seen = [], set()
    for i, row in enumerate(H):
        if row.sum() == 0:
            continue
        key = tuple(int(x) for x in row.tolist())
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    if not keep:
        return np.zeros((0, H.shape[1]), dtype=int), []
    return H[keep], [labels[i] for i in keep]


def nullspace_gf2(H):
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


def reduce_mod_rows(vecs, H, n):
    cur = H.copy() if H.size else np.zeros((0, n), dtype=int)
    kept = []
    for v in vecs:
        r0 = gf2_rank(cur)
        trial = np.vstack([cur, v.reshape(1, -1)])
        if gf2_rank(trial) > r0:
            kept.append(v)
            cur = trial
    return kept


def in_rowspace(H, v):
    if H.size == 0:
        return np.all(v == 0)
    return gf2_rank(np.vstack([H, v.reshape(1, -1)])) == gf2_rank(H)


def slab_rank_identity(Hx, Hz, S, which):
    n = Hx.shape[1]
    S = list(S)
    Sc = [i for i in range(n) if i not in set(S)]
    rx, rz = gf2_rank(Hx), gf2_rank(Hz)
    k = n - rx - rz

    def cols(H, ii):
        if not ii:
            return np.zeros((H.shape[0], 0), dtype=int)
        return H[:, ii]

    if which == "X":
        lhs = len(S) - gf2_rank(cols(Hz, S)) - rx + gf2_rank(cols(Hx, Sc))
    else:
        lhs = len(S) - gf2_rank(cols(Hx, S)) - rz + gf2_rank(cols(Hz, Sc))
    return lhs, k, lhs == k


def build_repetition_layer_code(L: int = 3):
    """Build Hx, Hz for the truncated 3-rep layer code at size L (>=3)."""
    if L < 3:
        raise ValueError("L>=3 required for interior catalog junction stars")
    Y_DATA = (0, 2, 4)
    XJ = (1, 2) if L == 3 else (1, L - 1)
    assert XJ[0] != XJ[1]

    idx = EdgeQubitIndex()
    for y in Y_DATA:
        for z in range(L + 1):
            for x in range(L):
                idx.ensure((x, y, z), (x + 1, y, z))
        for x in range(L + 1):
            for z in range(L):
                idx.ensure((x, y, z), (x, y, z + 1))
    for ci, xj in enumerate(XJ):
        y0, y1 = Y_DATA[ci], Y_DATA[ci + 1]
        for z in range(L + 1):
            for y in range(y0, y1):
                idx.ensure((xj, y, z), (xj, y + 1, z))
        for y in range(y0 + 1, y1):
            for z in range(L):
                idx.ensure((xj, y, z), (xj, y, z + 1))

    zrows, zmeta = [], []

    def pz(supp, lab):
        if len(supp) < 2:
            return
        r = np.zeros(idx.n, dtype=int)
        for i in supp:
            r[i] = 1
        zrows.append(r)
        zmeta.append(lab)

    for y in Y_DATA:
        for x in range(L):
            for z in range(L):
                e = [
                    idx.get((x, y, z), (x + 1, y, z)),
                    idx.get((x + 1, y, z), (x + 1, y, z + 1)),
                    idx.get((x, y, z + 1), (x + 1, y, z + 1)),
                    idx.get((x, y, z), (x, y, z + 1)),
                ]
                if all(v is not None for v in e):
                    pz(set(e), f"xz_plaq:y{y}:({x},{z})")
    for ci, xj in enumerate(XJ):
        y0, y1 = Y_DATA[ci], Y_DATA[ci + 1]
        for y in range(y0, y1):
            for z in range(L):
                e = [
                    idx.get((xj, y, z), (xj, y + 1, z)),
                    idx.get((xj, y + 1, z), (xj, y + 1, z + 1)),
                    idx.get((xj, y, z + 1), (xj, y + 1, z + 1)),
                    idx.get((xj, y, z), (xj, y, z + 1)),
                ]
                if all(v is not None for v in e):
                    pz(set(e), f"yz_plaq:c{ci}:({y},{z})")
    Hz, zmeta = remove_zero_and_duplicate_rows(np.vstack(zrows), zmeta)

    junc_z = list(range(1, L))
    junc = set()
    for ci, xj in enumerate(XJ):
        for y in (Y_DATA[ci], Y_DATA[ci + 1]):
            for z in junc_z:
                junc.add((xj, y, z))

    xcand, xmeta = [], []

    def px(supp, lab):
        if len(supp) < 2:
            return
        r = np.zeros(idx.n, dtype=int)
        for i in supp:
            r[i] = 1
        xcand.append(r)
        xmeta.append(lab)

    n_junc = 0
    for ci, xj in enumerate(XJ):
        yf, yl = Y_DATA[ci], Y_DATA[ci + 1]
        for z in junc_z:
            for lab, origin, tag in (
                (FIRST_Z_SX, (xj, yf, z), "first_z"),
                (LAST_Z_SX, (xj, yl, z), "last_z"),
            ):
                s, ok = set(), True
                for name in lab:
                    a, b = _translate_edge(name, origin)
                    qi = idx.get(a, b)
                    if qi is None:
                        ok = False
                        break
                    s.add(qi)
                if ok:
                    px(s, f"{tag}:c{ci}:{origin}")
                    n_junc += 1

    # Smooth yz-boundaries (omit ordinary stars at x=0 and x=L)
    for y in Y_DATA:
        for x in range(1, L):
            for z in range(L + 1):
                if (x, y, z) in junc:
                    continue
                s = set()
                for ep in (
                    ((x - 1, y, z), (x, y, z)),
                    ((x, y, z), (x + 1, y, z)),
                    ((x, y, z - 1), (x, y, z)) if z > 0 else None,
                    ((x, y, z), (x, y, z + 1)) if z < L else None,
                ):
                    if ep is None:
                        continue
                    qi = idx.get(*ep)
                    if qi is not None:
                        s.add(qi)
                px(s, f"xz_star:({x},{y},{z})")

    for ci, xj in enumerate(XJ):
        y0, y1 = Y_DATA[ci], Y_DATA[ci + 1]
        for y in range(y0 + 1, y1):
            for z in range(L + 1):
                s = set()
                for ep in (
                    ((xj, y, z - 1), (xj, y, z)) if z > 0 else None,
                    ((xj, y, z), (xj, y, z + 1)) if z < L else None,
                    ((xj, y - 1, z), (xj, y, z)),
                    ((xj, y, z), (xj, y + 1, z)),
                ):
                    if ep is None:
                        continue
                    qi = idx.get(*ep)
                    if qi is not None:
                        s.add(qi)
                px(s, f"yz_star:c{ci}:({xj},{y},{z})")

    Xc = np.vstack(xcand)
    keep = [i for i, row in enumerate(Xc) if np.all((Hz @ row) % 2 == 0)]
    Hx = Xc[keep]
    xmeta = [xmeta[i] for i in keep]
    Hx, xmeta = remove_zero_and_duplicate_rows(Hx, xmeta)

    # Drop qubits uncovered by X (truncated dangling edges)
    cov = Hx.sum(axis=0)
    keep_q = [i for i in range(idx.n) if cov[i] > 0]
    edges = [idx.edge(i) for i in keep_q]
    Hx = Hx[:, keep_q]
    Hz = Hz[:, keep_q]
    Hz, zmeta = remove_zero_and_duplicate_rows(Hz, zmeta)
    Hx, xmeta = remove_zero_and_duplicate_rows(Hx, xmeta)

    n = Hx.shape[1]
    assert np.all((Hx @ Hz.T) % 2 == 0)

    # Condense spurious open-boundary Z modes until k=1
    zlogs = reduce_mod_rows(nullspace_gf2(Hx), Hz, n)
    zlogs_sorted = sorted(zlogs, key=lambda v: (int(v.sum()), tuple(v.tolist())))
    condensed = []
    Hz_work = Hz.copy()
    zmeta_work = list(zmeta)
    target_k = 1
    for v in zlogs_sorted:
        rx, rz = gf2_rank(Hx), gf2_rank(Hz_work)
        k_now = n - rx - rz
        if k_now <= target_k:
            break
        if in_rowspace(Hz_work, v):
            continue
        # only condense very local truncation modes
        if int(v.sum()) > 4:
            continue
        Hz_work = np.vstack([Hz_work, v.reshape(1, -1)])
        zmeta_work.append(f"trunc_boundary_Z_condense:wt{int(v.sum())}")
        condensed.append(v)
        # ensure still CSS
        if not np.all((Hx @ Hz_work.T) % 2 == 0):
            Hz_work = Hz_work[:-1]
            zmeta_work.pop()
            condensed.pop()

    Hz, zmeta = Hz_work, zmeta_work
    rx, rz = gf2_rank(Hx), gf2_rank(Hz)
    k = n - rx - rz

    # If still k>1, condense next-lightest regardless of wt cap
    if k > target_k:
        zlogs = reduce_mod_rows(nullspace_gf2(Hx), Hz, n)
        for v in sorted(zlogs, key=lambda u: int(u.sum())):
            if k <= target_k:
                break
            if in_rowspace(Hz, v):
                continue
            trial = np.vstack([Hz, v.reshape(1, -1)])
            if not np.all((Hx @ trial.T) % 2 == 0):
                continue
            Hz = trial
            zmeta.append(f"trunc_boundary_Z_condense:wt{int(v.sum())}")
            condensed.append(v)
            rz = gf2_rank(Hz)
            k = n - rx - rz

    meta = {
        "name": f"wb_rep3_L{L}",
        "L": L,
        "n": n,
        "Y_DATA": Y_DATA,
        "XJ": XJ,
        "edges": edges,
        "x_meta": xmeta,
        "z_meta": zmeta,
        "n_junction_checks": sum(1 for m in xmeta if m.startswith("first_z") or m.startswith("last_z")),
        "n_condensed_boundary": len(condensed),
        "input": "3-qubit Z-repetition S=<Z1Z2,Z2Z3>",
        "scope": (
            "truncated finite test instance using audited Williamson--Baspin Sec VI B "
            "junction incidences for the 3-qubit-repetition input; not a literal finite "
            "reconstruction or metric calibration of the complete published lattice artwork"
        ),
    }
    return Hx.astype(int), Hz.astype(int), meta


def quasiconcat_and_kernel_logicals(Hx, Hz, meta):
    n = Hx.shape[1]
    edges = meta["edges"]
    L = meta["L"]
    Y_DATA = meta["Y_DATA"]
    XJ = meta["XJ"]

    # Map edge -> column
    etoi = {e: i for i, e in enumerate(edges)}

    def qi(a, b):
        return etoi.get(_canon_edge(a, b))

    # Quasiconcat X: x-paths on all three data layers at mid z + y pairings
    z_mid = L // 2
    x_log = np.zeros(n, dtype=int)
    for y in Y_DATA:
        for x in range(L):
            i = qi((x, y, z_mid), (x + 1, y, z_mid))
            if i is not None:
                x_log[i] ^= 1
    for ci, xj in enumerate(XJ):
        y0, y1 = Y_DATA[ci], Y_DATA[ci + 1]
        for y in range(y0, y1):
            i = qi((xj, y, z_mid), (xj, y + 1, z_mid))
            if i is not None:
                x_log[i] ^= 1

    # Quasiconcat Z: z-path on middle layer away from walls
    x_z = next(x for x in range(L + 1) if x not in XJ)
    z_log = np.zeros(n, dtype=int)
    for z in range(L):
        i = qi((x_z, Y_DATA[1], z), (x_z, Y_DATA[1], z + 1))
        if i is not None:
            z_log[i] ^= 1

    xlogs = reduce_mod_rows(nullspace_gf2(Hz), Hx, n)
    zlogs = reduce_mod_rows(nullspace_gf2(Hx), Hz, n)

    def is_logical_X(v):
        return bool(np.all((Hz @ v) % 2 == 0) and not in_rowspace(Hx, v))

    def is_logical_Z(v):
        return bool(np.all((Hx @ v) % 2 == 0) and not in_rowspace(Hz, v))

    # Prefer quasiconcat if valid; else min-weight kernel reps
    if not is_logical_X(x_log) and xlogs:
        x_log = min(xlogs, key=lambda v: int(v.sum()))
        x_note = "min-weight ker(Hz)/row(Hx)"
    else:
        x_note = "quasiconcatenated" if is_logical_X(x_log) else "invalid"
    if not is_logical_Z(z_log) and zlogs:
        z_log = min(zlogs, key=lambda v: int(v.sum()))
        z_note = "min-weight ker(Hx)/row(Hz)"
    else:
        z_note = "quasiconcatenated" if is_logical_Z(z_log) else "invalid"

    return {
        "X": x_log,
        "Z": z_log,
        "X_is_logical": is_logical_X(x_log),
        "Z_is_logical": is_logical_Z(z_log),
        "X_weight": int(x_log.sum()),
        "Z_weight": int(z_log.sum()),
        "X_note": x_note,
        "Z_note": z_note,
        "n_xlogs": len(xlogs),
        "n_zlogs": len(zlogs),
    }


def region_prefix_y(meta, y_cut):
    return [
        i
        for i, ((a), (b)) in enumerate(
            (meta["edges"][j][0], meta["edges"][j][1]) for j in range(len(meta["edges"]))
        )
        if a[1] <= y_cut or b[1] <= y_cut
    ]


def region_yz_slab(meta, ci):
    Y_DATA, XJ = meta["Y_DATA"], meta["XJ"]
    y0, y1 = Y_DATA[ci], Y_DATA[ci + 1]
    xj = XJ[ci]
    S = []
    for i, (a, b) in enumerate(meta["edges"]):
        if a[1] == b[1] and a[1] in (y0, y1):
            S.append(i)
        elif xj in (a[0], b[0]) and min(a[1], b[1]) >= y0 and max(a[1], b[1]) <= y1:
            S.append(i)
    return sorted(set(S))


def main() -> int:
    chosen = None
    for L in (3, 4):
        Hx, Hz, meta = build_repetition_layer_code(L)
        n = meta["n"]
        rx, rz = gf2_rank(Hx), gf2_rank(Hz)
        k = n - rx - rz
        css = bool(np.all((Hx @ Hz.T) % 2 == 0))
        logs = quasiconcat_and_kernel_logicals(Hx, Hz, meta)
        print(
            f"L={L}: n={n} k={k} rx={rx} rz={rz} CSS={css} "
            f"junctions={meta['n_junction_checks']} condensed={meta['n_condensed_boundary']} "
            f"Xlog={logs['X_is_logical']}(wt{logs['X_weight']},{logs['X_note']}) "
            f"Zlog={logs['Z_is_logical']}(wt{logs['Z_weight']},{logs['Z_note']})"
        )
        if css and k >= 1 and logs["X_is_logical"] and logs["Z_is_logical"]:
            chosen = (Hx, Hz, meta, logs)
            if k == 1:
                break

    if chosen is None:
        print("FAIL: no CSS instance with k>=1 and explicit logicals")
        return 1

    Hx, Hz, meta, logs = chosen
    n, L = meta["n"], meta["L"]
    rx, rz = gf2_rank(Hx), gf2_rank(Hz)
    k = n - rx - rz
    assert np.all((Hx @ Hz.T) % 2 == 0)
    assert k >= 1

    rows = []
    regions = [("all_qubits", list(range(n)))]
    for ci in (0, 1):
        regions.append((f"yz_slab_c{ci}", region_yz_slab(meta, ci)))
    for t, y_cut in enumerate(meta["Y_DATA"]):
        regions.append((f"prefix_y_le_{y_cut}", region_prefix_y(meta, y_cut)))

    for tag, S in regions:
        lx, _, px = slab_rank_identity(Hx, Hz, S, "X")
        lz, _, pz = slab_rank_identity(Hx, Hz, S, "Z")
        rows.append({
            "code_id": meta["name"],
            "n": n,
            "k": k,
            "rank_Hx": rx,
            "rank_Hz": rz,
            "region": tag,
            "S_size": len(S),
            "lhs_X": lx,
            "pass_X": int(px),
            "lhs_Z": lz,
            "pass_Z": int(pz),
            "both_pass": int(px and pz),
            "X_logical_weight": logs["X_weight"],
            "Z_logical_weight": logs["Z_weight"],
            "X_logical_note": logs["X_note"],
            "Z_logical_note": logs["Z_note"],
            "n_junction_checks": meta["n_junction_checks"],
            "n_condensed_boundary": meta["n_condensed_boundary"],
            "scope": meta["scope"],
        })

    out = ROOT / "wb_layer_code_slab_certificate.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    np.savez_compressed(
        ROOT / "wb_layer_code_matrices.npz",
        Hx=Hx.astype(np.uint8),
        Hz=Hz.astype(np.uint8),
        X_logical=logs["X"].astype(np.uint8),
        Z_logical=logs["Z"].astype(np.uint8),
        L=np.int32(L),
        n=np.int32(n),
        k=np.int32(k),
    )

    d_up = min(logs["X_weight"], logs["Z_weight"])
    print("PASS finite WB-incidence-derived test instance")
    print(f"  n={n} k={k} L={L} rx={rx} rz={rz}")
    print(f"  CSS: Hx Hz^T = 0; d_upper <= {d_up} via explicit logicals")
    print(f"  junction SX checks kept: {meta['n_junction_checks']}")
    print(f"  slab both_pass regions: {[r['region'] for r in rows if r['both_pass']]}")
    print(f"  wrote {out.name}, wb_layer_code_matrices.npz")
    print(f"  SCOPE: {meta['scope']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
