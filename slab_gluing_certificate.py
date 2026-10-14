#!/usr/bin/env python3
"""Slab gluing certificate on complete check matrices (revision, review point 1).

Four parts, all exact GF(2) computations, no Monte Carlo:

A. Negative controls on the archived cell certificate
   (wb_exact_cell_patch_audit.py).  Every boundary word, not only the kernel
   words, is fed to the archived solver; the full syndrome of each archived
   witness is then evaluated on every check of its patch, including the
   "boundary_adjacent" checks that the archived test omitted.

B. Complete finite layer codes (layer_code_explicit.py), for a family of
   input CSS codes and spacings, in two junction realizations:
     two_sheet    - the explicit realization used by the revised theorem;
     single_chart - the plain union obtained by reading the six junction
                    cells in one unit-cubic chart (negative control).
   For each code: all checks commute; logical dimension k versus the input K.

C. Symbolic gluing forms.  The prefix-parity operator is written as a sum of
   binary variables (data-string bits x_q and backbone-segment bits p_{j,t}),
   each variable being a fixed edge set.  For every stabilizer the parity of
   its overlap with the operator is a linear form in these variables; the
   script lists every distinct nonzero form.  The operator commutes with all
   stabilizers exactly when every listed form vanishes.  In the two-sheet
   realization the forms are exactly the first/middle/last relations
   x+p=0, p+p'+x=0, p+x=0, each supported on one junction.  The same is done
   for the dual Z slabs.

D. Exhaustive words and complete logical checks.  For single-check inputs
   every incidence word and every backbone assignment is tried (negative
   controls included).  For every input code, every basis logical, and every
   height z0 (resp. column x0), the prefix operator is verified against every
   stabilizer of the complete code, its thickness is recorded, stabilizer
   inputs map to stabilizers, and the X/Z pairing equals the input pairing.

Outputs (CSV) are written next to this script.  Exit status 0 iff every
two_sheet check passes (single_chart failures are the expected negative
control).
"""
from __future__ import annotations

import csv
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from ldpc import mod2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from layer_code_explicit import (E, build_layer_code, gf2_rank, in_rowspace,  # noqa: E402
                                 prefix_bits, support_extent, x_slab, z_slab)

# --------------------------------------------------------------------------
# Input code family
# --------------------------------------------------------------------------

def hamming(r):
    cols = [[(c >> b) & 1 for b in range(r)] for c in range(1, 2 ** r)]
    return np.array(cols, dtype=np.uint8).T


def hgp(H1, H2):
    H1 = np.asarray(H1, np.uint8); H2 = np.asarray(H2, np.uint8)
    m1, n1 = H1.shape; m2, n2 = H2.shape
    HX = np.hstack([np.kron(H1, np.eye(n2, dtype=np.uint8)),
                    np.kron(np.eye(m1, dtype=np.uint8), H2.T)]) % 2
    HZ = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), H2),
                    np.kron(H1.T, np.eye(m2, dtype=np.uint8))]) % 2
    return HX.astype(np.uint8), HZ.astype(np.uint8)


def input_codes():
    rep3 = np.array([[1, 1, 0], [0, 1, 1]])
    h3 = hamming(3)
    rng = np.random.default_rng(20260926)
    codes = [
        ("422", np.ones((1, 4)), np.ones((1, 4))),
        ("642", np.ones((1, 6)), np.ones((1, 6))),
        ("steane_7_1_3", h3, h3),
        ("hamming_15_7_3", hamming(4), hamming(4)),
        ("surface_13_1_3", *hgp(rep3, rep3)),
    ]
    # random hypergraph-product inputs with all check weights >= 2
    made = 0
    while made < 2:
        H1 = rng.integers(0, 2, size=(2, 3))
        H2 = rng.integers(0, 2, size=(2, 3))
        HX, HZ = hgp(H1, H2)
        HX = HX[HX.sum(1) > 0]; HZ = HZ[HZ.sum(1) > 0]
        if HX.shape[0] == 0 or HZ.shape[0] == 0:
            continue
        if HX.sum(1).min() < 2 or HZ.sum(1).min() < 2:
            continue
        K = HX.shape[1] - gf2_rank(HX) - gf2_rank(HZ)
        if K < 1:
            continue
        codes.append((f"random_hgp_{made}", HX, HZ))
        made += 1
    return [(n, np.asarray(a, np.uint8), np.asarray(b, np.uint8)) for n, a, b in codes]


def nullspace(H):
    H = np.asarray(H, np.uint8)
    if H.shape[0] == 0:
        return np.eye(H.shape[1], dtype=np.uint8)
    K = mod2.nullspace(sp.csr_matrix(H))
    return np.asarray(K.todense() if sp.issparse(K) else K, dtype=np.uint8)


# --------------------------------------------------------------------------
# Part A: archived certificate
# --------------------------------------------------------------------------

def part_a(archive: Path):
    rows_words, rows_synd = [], []
    if not (archive / "wb_exact_cell_patch_audit.py").exists():
        print("  [A] archived package not found; skipped")
        return rows_words, rows_synd
    sys.path.insert(0, str(archive))
    import reproducibility_checks as rc          # noqa: E402
    import wb_exact_cell_patch_audit as old      # noqa: E402
    E_named, jdata = rc.wb_junction_support_data()
    named_key = {nm: old.edge_key(*ends) for nm, ends in E_named.items()}
    kernels = {c: set(ws) for c, ws in old.KERNELS.items()}
    for cell, legs in old.OCCUPATION_LEGS.items():
        allw = list(itertools.product((0, 1), repeat=len(legs)))
        saved = old.KERNELS[cell]
        old.KERNELS[cell] = allw
        cat, cert, _, _, _ = old.audit_cell(cell, old.PATCH_RADIUS, E_named, jdata, named_key)
        old.KERNELS[cell] = saved
        for w, r in zip(allw, cert):
            rows_words.append(dict(
                cell=cell, word="".join(map(str, w)), in_kernel=w in kernels[cell],
                archived_solver_status=r["status"], witness=r["witness_edges"]))
        # full syndrome of the archived kernel-word witnesses on every check
        R = old.PATCH_RADIUS
        edges = old.all_edges(R)
        eidx = {e: i for i, e in enumerate(edges)}
        novel = [r for r in jdata if r["cell"] == cell]
        novel_z = {frozenset(named_key[e] for e in r["support"]) for r in novel if r["pauli"] == "Z"}
        stars, plaqs = old.ordinary_stars_plaquettes(R, novel_z)
        pauli = old.PAULI_TYPE[cell]
        leg_keys = {named_key[L] for L in legs}
        checks = []
        if pauli == "X":
            checks += [("novel_" + r["check"], [named_key[e] for e in r["support"]], "novel")
                       for r in novel if r["pauli"] == "Z"]
            checks += [(c["check_id"], list(c["support_keys"]),
                        "boundary_adjacent" if c["support_keys"] & leg_keys else "retained")
                       for c in plaqs]
        else:
            checks += [("novel_" + r["check"], [named_key[e] for e in r["support"]], "novel")
                       for r in novel if r["pauli"] == "X"]
            checks += [(c["check_id"], list(c["support_keys"]),
                        "boundary_adjacent" if c["support_keys"] & leg_keys else "retained")
                       for c in stars]
        for w in saved:
            wd = "".join(map(str, w))
            wit = [r for r in cert if r["kernel_word"] == wd]
            A = np.zeros((len(checks), len(edges)), np.uint8)
            for i, (_, supp, _) in enumerate(checks):
                for e in supp:
                    if e in eidx:
                        A[i, eidx[e]] = 1
            B = np.zeros((len(legs), len(edges)), np.uint8)
            for i, L in enumerate(legs):
                B[i, eidx[named_key[L]]] = 1
            sol = old.gf2_solve(np.vstack([A[[k for k, c in enumerate(checks) if c[2] != "boundary_adjacent"]], B]),
                                np.concatenate([np.zeros(sum(c[2] != "boundary_adjacent" for c in checks), np.uint8),
                                                np.array(w, np.uint8)]))
            syn = (A @ sol) % 2
            bad = [checks[k] for k in np.nonzero(syn)[0]]
            rows_synd.append(dict(
                cell=cell, word=wd, witness_weight=int(sol.sum()),
                syndrome_retained=sum(1 for c in bad if c[2] == "retained"),
                syndrome_boundary_adjacent=sum(1 for c in bad if c[2] == "boundary_adjacent"),
                violated_boundary_checks=" ".join(c[0] for c in bad if c[2] == "boundary_adjacent")))
    return rows_words, rows_synd


# --------------------------------------------------------------------------
# Parts B-D
# --------------------------------------------------------------------------

def x_variables(lc, z0):
    g = lc.geometry
    nin = lc.HX_in.shape[1]
    V = {}
    for q in range(nin):
        v = np.zeros(lc.n, np.uint8)
        for xx in range(g["Lx"] + 1):
            v[lc.q(("D", q), E((xx, g["Yq"][q], z0), (xx, g["Yq"][q], z0 + 1)), False)] ^= 1
        V[f"x{q}"] = v
    zero = np.zeros(nin, np.uint8)
    base = {(jj, tt): 0 for jj, sup in enumerate(g["suppZ"]) for tt in range(len(sup) - 1)}
    for (j, t) in base:
        ov = dict(base); ov[(j, t)] = 1
        V[f"p{j}.{t}"] = x_slab(lc, zero, z0, override=ov)
    return V


def z_variables(lc, x0):
    g = lc.geometry
    nin = lc.HX_in.shape[1]
    V = {}
    for q in range(nin):
        v = np.zeros(lc.n, np.uint8)
        for zz in range(g["Lz"]):
            v[lc.q(("D", q), E((x0, g["Yq"][q], zz), (x0, g["Yq"][q], zz + 1)), False)] ^= 1
        V[f"z{q}"] = v
    zero = np.zeros(nin, np.uint8)
    base = {(ii, tt): 0 for ii, sup in enumerate(g["suppX"]) for tt in range(len(sup) - 1)}
    for (i, t) in base:
        ov = dict(base); ov[(i, t)] = 1
        V[f"r{i}.{t}"] = z_slab(lc, zero, x0, override=ov)
    return V


def linear_forms(V, H):
    names = list(V)
    if not names:
        return Counter()
    M = sp.csr_matrix(np.array([V[n] for n in names], dtype=np.uint8).T.astype(np.int32))
    S = (H.astype(np.int32) @ M).toarray() % 2
    out = Counter()
    for r in range(S.shape[0]):
        f = tuple(names[c] for c in np.nonzero(S[r])[0])
        if f:
            out[f] += 1
    return out


def expected_forms(supports, dvar, bvar):
    """The prefix-parity relations, one per incidence."""
    exp = set()
    for j, sup in enumerate(supports):
        k = len(sup)
        exp.add(tuple(sorted([f"{dvar}{sup[0]}", f"{bvar}{j}.0"])))
        for t in range(1, k - 1):
            exp.add(tuple(sorted([f"{dvar}{sup[t]}", f"{bvar}{j}.{t-1}", f"{bvar}{j}.{t}"])))
        exp.add(tuple(sorted([f"{dvar}{sup[-1]}", f"{bvar}{j}.{k-2}"])))
    return exp


def classify_forms(F, supports, dvar, bvar):
    exp = expected_forms(supports, dvar, bvar)
    got = {tuple(sorted(f)) for f in F}
    extra = sorted(got - exp)
    missing = sorted(exp - got)
    mult = max(F.values()) if F else 0
    return exp, got, extra, missing, mult


def form_kind(f):
    return f"{len(f)}-term"


def complete_code_checks(name, HX, HZ, s, mode):
    t0 = time.time()
    lc = build_layer_code(HX, HZ, spacing=s, junction_mode=mode)
    HXf, HZf = lc.HX, lc.HZ
    anti = int(((HXf.astype(np.int32) @ HZf.T.astype(np.int32)).toarray() % 2).sum())
    K = HX.shape[1] - gf2_rank(HX) - gf2_rank(HZ)
    k = lc.n - gf2_rank(HXf) - gf2_rank(HZf)
    walls_ok = all(w["status"] == "solved" for w in lc.wall_report)
    g = lc.geometry
    rx, rz = gf2_rank(HXf), gf2_rank(HZf)
    row = dict(code=name, spacing=s, mode=mode, n_in=HX.shape[1], K_in=K,
               n_phys=lc.n, n_xchecks=HXf.shape[0], n_zchecks=HZf.shape[0],
               euler_n_minus_checks=lc.n - HXf.shape[0] - HZf.shape[0],
               euler_input=HX.shape[1] - HX.shape[0] - HZ.shape[0],
               x_relations=HXf.shape[0] - rx, x_relations_input=HX.shape[0] - gf2_rank(HX),
               z_relations=HZf.shape[0] - rz, z_relations_input=HZ.shape[0] - gf2_rank(HZ),
               max_check_weight=int(max(HXf.sum(1).max(), HZf.sum(1).max())),
               max_qubit_degree=int(max(HXf.sum(0).max(), HZf.sum(0).max())),
               anticommuting_pairs=anti, k_phys=k, k_equals_K=(k == K),
               wall_lines=len(lc.wall_report), walls_solved=walls_ok)
    forms_rows = []
    # symbolic forms at every height / column
    xbad = zbad = 0
    x_forms_all, z_forms_all = set(), set()
    for z0 in range(g["Lz"]):
        F = linear_forms(x_variables(lc, z0), HZf)
        exp, got, extra, missing, mult = classify_forms(F, g["suppZ"], "x", "p")
        x_forms_all |= got
        ok = (not extra) and (not missing)
        xbad += (not ok)
        forms_rows.append(dict(code=name, spacing=s, mode=mode, orientation="X slab (height z0)",
                               position=z0, n_distinct_forms=len(got), n_expected=len(exp),
                               unexpected_forms=" | ".join("+".join(f) for f in extra[:6]),
                               missing_forms=" | ".join("+".join(f) for f in missing[:6]),
                               max_checks_per_form=mult, status="PASS" if ok else "FAIL"))
    for x0 in range(g["Lx"] + 1):
        F = linear_forms(z_variables(lc, x0), HXf)
        exp, got, extra, missing, mult = classify_forms(F, g["suppX"], "z", "r")
        z_forms_all |= got
        ok = (not extra) and (not missing)
        zbad += (not ok)
        forms_rows.append(dict(code=name, spacing=s, mode=mode, orientation="Z slab (column x0)",
                               position=x0, n_distinct_forms=len(got), n_expected=len(exp),
                               unexpected_forms=" | ".join("+".join(f) for f in extra[:6]),
                               missing_forms=" | ".join("+".join(f) for f in missing[:6]),
                               max_checks_per_form=mult, status="PASS" if ok else "FAIL"))
    row.update(x_heights=g["Lz"], x_heights_forms_ok=g["Lz"] - xbad,
               z_columns=g["Lx"] + 1, z_columns_forms_ok=g["Lx"] + 1 - zbad)

    # explicit logical operators on the complete code
    KX, KZ = nullspace(HZ), nullspace(HX)     # all pure-X / pure-Z input logicals (incl. stabilizers)
    n_x = n_z = n_x_ok = n_z_ok = 0
    max_tz = max_tx = 0
    for z0 in range(g["Lz"]):
        for x in KX:
            v = x_slab(lc, x, z0)
            n_x += 1
            n_x_ok += int(not np.any((HZf @ v) % 2))
            lo, hi = support_extent(lc, v, 2)
            if lo is not None:
                max_tz = max(max_tz, hi - lo)
    for x0 in range(g["Lx"] + 1):
        for z in KZ:
            v = z_slab(lc, z, x0)
            n_z += 1
            n_z_ok += int(not np.any((HXf @ v) % 2))
            lo, hi = support_extent(lc, v, 0)
            if lo is not None:
                max_tx = max(max_tx, hi - lo)
    # pairing and triviality at a few positions (including ones on check layers)
    pair_ok = triv_ok = True
    heights = sorted({0, 1, g["Zi"][0] if g["Zi"] else 0, g["Lz"] - 1})
    cols = sorted({0, g["Xj"][0] if g["Xj"] else 0, g["Lx"]})
    Xin = np.array([x for x in KX])
    Zin = np.array([z for z in KZ])
    want = (Xin.astype(int) @ Zin.T.astype(int)) % 2
    for z0 in heights:
        XS = np.array([x_slab(lc, x, z0) for x in KX])
        for x0 in cols:
            ZS = np.array([z_slab(lc, z, x0) for z in KZ])
            got = (XS.astype(int) @ ZS.T.astype(int)) % 2
            pair_ok &= bool(np.array_equal(got, want))
        # input stabilizers map to stabilizers
        for xs in np.asarray(HX, np.uint8):
            v = x_slab(lc, xs, z0)
            triv_ok &= bool(in_rowspace(HXf, v)[0])
    for x0 in cols:
        for zs in np.asarray(HZ, np.uint8):
            v = z_slab(lc, zs, x0)
            triv_ok &= bool(in_rowspace(HZf, v)[0])
    # the realized logical classes span all K logical pairs
    pr = gf2_rank(want) if want.size else 0
    row.update(x_slab_ops=n_x, x_slab_ops_commuting=n_x_ok, z_slab_ops=n_z,
               z_slab_ops_commuting=n_z_ok, x_slab_z_thickness=max_tz,
               z_slab_x_thickness=max_tx, pairing_equals_input=pair_ok,
               pairing_rank=pr, stabilizer_inputs_trivial=triv_ok,
               seconds=round(time.time() - t0, 1))
    euler_ok = (row["euler_n_minus_checks"] == row["euler_input"]
                and row["x_relations"] == row["x_relations_input"]
                and row["z_relations"] == row["z_relations_input"])
    row["euler_and_relations_match_input"] = euler_ok
    good = (anti == 0 and k == K and euler_ok and walls_ok and xbad == 0 and zbad == 0 and n_x_ok == n_x
            and n_z_ok == n_z and pair_ok and triv_ok and pr == K)
    row["status"] = "PASS" if good else "FAIL"
    return row, forms_rows, lc


def wall_controls():
    """Walls close iff every X-check/Z-check overlap is even (input commutation)."""
    cases = [([[1, 1, 1, 0]], [[0, 1, 1, 1]]), ([[1, 1, 0, 0]], [[0, 1, 1, 0]]),
             ([[1, 1, 1, 1, 0]], [[0, 1, 1, 1, 1]]), ([[1, 1, 1, 1, 1, 1]], [[1, 0, 1, 1, 1, 0]]),
             ([[1, 1, 1, 1, 1, 1]], [[1, 1, 1, 0, 0, 0]])]
    rows = []
    for hx, hz in cases:
        hx, hz = np.array(hx, np.uint8), np.array(hz, np.uint8)
        lc = build_layer_code(hx, hz, spacing=2, require_css=False)
        anti = int(((lc.HX.astype(np.int32) @ lc.HZ.T.astype(np.int32)).toarray() % 2).sum())
        ov = int((hx.astype(int) @ hz.T.astype(int)).sum())
        st = lc.wall_report[0]["status"] if lc.wall_report else "none"
        rows.append(dict(HX="".join(map(str, hx[0])), HZ="".join(map(str, hz[0])), overlap=ov,
                         wall_system=st, anticommuting_pairs=anti,
                         status="PASS" if ((ov % 2 == 0) == (st == "solved") and (anti == 0) == (ov % 2 == 0)) else "FAIL"))
    return rows


def exhaustive_words(kmax=6):
    """Single-check inputs: every incidence word and every backbone assignment."""
    rows = []
    for mode in ("two_sheet", "single_chart"):
        for k in range(2, kmax + 1):
            for orient in ("X", "Z"):
                H = np.ones((1, k), np.uint8)
                empty = np.zeros((0, k), np.uint8)
                if orient == "X":
                    lc = build_layer_code(empty, H, spacing=2, junction_mode=mode)
                    V = x_variables(lc, 0)
                    Hc = lc.HZ
                    dv, bv = "x", "p0."
                else:
                    lc = build_layer_code(H, empty, spacing=2, junction_mode=mode)
                    V = z_variables(lc, 0)
                    Hc = lc.HX
                    dv, bv = "z", "r0."
                names = [f"{dv}{q}" for q in range(k)] + [f"{bv}{t}" for t in range(k - 1)]
                M = np.array([V[nm] for nm in names], np.uint8)
                S = (Hc.astype(np.int32) @ sp.csr_matrix(M.T.astype(np.int32))).toarray() % 2
                accepted = predicted = agree = 0
                total = 0
                for bits in itertools.product((0, 1), repeat=2 * k - 1):
                    b = np.array(bits, np.uint8)
                    zero = not np.any((S.astype(int) @ b) % 2)
                    xw, pw = bits[:k], bits[k:]
                    pred = (sum(xw) % 2 == 0) and list(pw) == prefix_bits(xw)[:-1]
                    total += 1
                    accepted += zero
                    predicted += pred
                    agree += (zero == pred)
                rows.append(dict(mode=mode, slab=orient, check_weight=k, assignments=total,
                                 zero_syndrome=accepted, predicted_by_Hend_Hmid=predicted,
                                 agreement=agree,
                                 status="PASS" if agree == total else "DIFFERS"))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def export_instance(lc, path):
    HX, HZ = lc.HX, lc.HZ
    np.savez_compressed(
        path, HX=HX.toarray().astype(np.uint8), HZ=HZ.toarray().astype(np.uint8),
        qubit_sheet=np.array([f"{k[0][0]}{k[0][1]}" for k in lc.qubits]),
        qubit_edge=np.array([[list(k[1][0]), list(k[1][1])] for k in lc.qubits], dtype=np.int32),
        xcheck_label=np.array([str(c["label"]) for c in lc.xchecks]),
        zcheck_label=np.array([str(c["label"]) for c in lc.zchecks]),
        HX_in=lc.HX_in, HZ_in=lc.HZ_in)


def main(archive=None):
    archive = Path(archive) if archive else ROOT.parent / "source"
    print("[A] archived cell certificate: all boundary words and full syndromes")
    wrows, srows = part_a(archive)
    write_csv(ROOT / "slab_cert_A_archived_all_words.csv", wrows)
    write_csv(ROOT / "slab_cert_A_archived_full_syndrome.csv", srows)
    for cell in sorted({r["cell"] for r in wrows}):
        acc = [r["word"] for r in wrows if r["cell"] == cell and r["archived_solver_status"] == "PASS"]
        odd = [w for w in acc if not any(r["cell"] == cell and r["word"] == w and r["in_kernel"] for r in wrows)]
        print(f"    {cell}: accepts {acc}; outside kernel: {odd}")
    for r in srows:
        if r["syndrome_boundary_adjacent"]:
            print(f"    witness {r['cell']}/{r['word']}: {r['syndrome_boundary_adjacent']} violated boundary-adjacent checks")

    print("[B-D] complete codes")
    crow, frows = [], []
    for name, HX, HZ in input_codes():
        spacings = (2, 3) if HX.shape[1] <= 7 else (2,)
        for s in spacings:
            for mode in ("two_sheet", "single_chart"):
                row, fr, lc = complete_code_checks(name, HX, HZ, s, mode)
                crow.append(row)
                frows.extend(fr)
                print(f"    {name:16s} s={s} {mode:12s} n={row['n_phys']:6d} k={row['k_phys']:3d} "
                      f"K={row['K_in']:2d} anti={row['anticommuting_pairs']} "
                      f"Xforms {row['x_heights_forms_ok']}/{row['x_heights']} "
                      f"Zforms {row['z_columns_forms_ok']}/{row['z_columns']} "
                      f"Xops {row['x_slab_ops_commuting']}/{row['x_slab_ops']} "
                      f"Zops {row['z_slab_ops_commuting']}/{row['z_slab_ops']} "
                      f"pair={row['pairing_equals_input']} {row['status']} ({row['seconds']}s)")
                if name == "422" and s == 2 and mode == "two_sheet":
                    export_instance(lc, ROOT / "explicit_layer_code_422_s2.npz")
                if name == "steane_7_1_3" and s == 2 and mode == "two_sheet":
                    export_instance(lc, ROOT / "explicit_layer_code_steane_s2.npz")
    write_csv(ROOT / "slab_cert_B_complete_codes.csv", crow)
    write_csv(ROOT / "slab_cert_C_forms.csv", frows)

    print("[D] exhaustive incidence words and backbone assignments")
    erows = exhaustive_words(6)
    write_csv(ROOT / "slab_cert_D_exhaustive_words.csv", erows)
    for r in erows:
        print(f"    {r['mode']:12s} {r['slab']} w={r['check_weight']} zero-syndrome "
              f"{r['zero_syndrome']}/{r['assignments']} predicted {r['predicted_by_Hend_Hmid']} {r['status']}")

    print("[E] wall solvability versus input commutation")
    wrows2 = wall_controls()
    write_csv(ROOT / "slab_cert_E_wall_controls.csv", wrows2)
    for r in wrows2:
        print(f"    HX={r['HX']} HZ={r['HZ']} overlap={r['overlap']} walls={r['wall_system']} "
              f"anticommuting={r['anticommuting_pairs']} {r['status']}")

    ok = all(r["status"] == "PASS" for r in crow if r["mode"] == "two_sheet")
    ok &= all(r["status"] == "PASS" for r in wrows2)
    ok &= all(r["status"] == "PASS" for r in erows if r["mode"] == "two_sheet")
    neg = all(r["status"] == "FAIL" for r in crow if r["mode"] == "single_chart")
    print(f"TWO-SHEET CERTIFICATE: {'PASS' if ok else 'FAIL'}; "
          f"single-chart negative control fails as expected: {neg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
