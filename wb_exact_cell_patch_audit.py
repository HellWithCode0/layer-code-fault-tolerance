#!/usr/bin/env python3
"""Exact finite-cell GF(2) stabilizer certificate for WB Sec. VI B first/middle/last cells.

Proof-critical hierarchy
------------------------
  * This script = proof-critical exact cell certificate (A_int p = 0, B p = w).
  * wb_junction_bulk_syndrome_audit (R=2/R=3) = demoted locality stress test.
  * wb_bullet_plus_geometric_audit = SM explanatory cross-check only
    (theorem no longer uses bullet/plus as microscopic primitives).

Method
------
For each cell in {first,middle,last}_{z,x} and each kernel word w of H_end or
H_mid, search a fixed L^∞ radius-R cubic patch for a Pauli p (X on z-family,
Z on x-family) satisfying over F_2:

    A_int p = 0,     B p = w

where A_int stacks all retained_internal + novel_internal checks of the
anticommuting Pauli type, and B's rows are indicator vectors of the designated
occupation-leg edges (boundary_port).  Ordinary checks that intersect a
designated port are classified boundary_adjacent (open onto the ports) and are
NOT rows of A_int.  Ordinary origin star and novel-matching plaquettes are
replaced.  PASS requires additionally that p has zero support on non-port
frontier edges (other_boundary_syndrome_weight := frontier non-port weight).

GF(2) linear algebra: augmented Gaussian elimination (NumPy uint8 bit ops);
one particular solution is returned (free variables set to 0).

No Monte Carlo.  No invented edges beyond the unit-cubic chart of
wb_junction_support_catalog.csv and the ordinary cubic surface-code complex.
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np

import reproducibility_checks as rc

ROOT = Path(__file__).resolve().parent

# Patch radius for the proof-critical solve (stress uses the same classification).
PATCH_RADIUS = 2

OCCUPATION_LEGS = {
    # (ordered to match H_end / H_mid variable order)
    "first_z": ("yp", "xm"),          # (p, x)
    "last_z": ("ym", "xm"),
    "middle_z": ("ym", "yp", "xm"),   # (pL, pR, x); 110 = pass-through
    "first_x": ("zp", "xp"),          # (r, z)
    "last_x": ("zm", "xp"),
    "middle_x": ("zm", "zp", "xp"),
}

KERNELS = {
    "first_z": [(0, 0), (1, 1)],
    "last_z": [(0, 0), (1, 1)],
    "middle_z": [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    "first_x": [(0, 0), (1, 1)],
    "last_x": [(0, 0), (1, 1)],
    "middle_x": [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
}

PAULI_TYPE = {c: ("X" if c.endswith("_z") else "Z") for c in OCCUPATION_LEGS}

SOURCE_META = {
    "first_z": (3, 23),
    "middle_z": (4, 23),
    "last_z": (5, 23),
    "first_x": (6, 24),
    "middle_x": (7, 24),
    "last_x": (8, 24),
}


def edge_key(a, b):
    return tuple(sorted((tuple(a), tuple(b))))


def all_edges(R: int):
    edges = set()
    for v in itertools.product(range(-R, R + 1), repeat=3):
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            w = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
            if all(-R <= c <= R for c in w):
                edges.add(edge_key(v, w))
    return sorted(edges)


def ordinary_stars_plaquettes(R: int, novel_z_keys: set[frozenset]):
    """Ordinary SC stars/plaquettes in the ball; origin star omitted (replaced)."""
    stars, plaquettes = [], []
    for v in itertools.product(range(-R, R + 1), repeat=3):
        if v == (0, 0, 0):
            continue
        inc = []
        for d in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            w = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
            if all(-R <= c <= R for c in w):
                inc.append(edge_key(v, w))
        if len(inc) == 6:
            stars.append({"check_id": f"ord_star{v}", "pauli": "X",
                          "support_keys": frozenset(inc), "loc": v})
    for x, y, z in itertools.product(range(-R, R), range(-R, R), range(-R, R + 1)):
        e = frozenset([
            edge_key((x, y, z), (x + 1, y, z)), edge_key((x, y, z), (x, y + 1, z)),
            edge_key((x + 1, y, z), (x + 1, y + 1, z)),
            edge_key((x, y + 1, z), (x + 1, y + 1, z)),
        ])
        plaquettes.append({"check_id": f"ord_xy({x},{y},{z})", "pauli": "Z",
                           "support_keys": e, "loc": (x, y, z)})
    for x, y, z in itertools.product(range(-R, R), range(-R, R + 1), range(-R, R)):
        e = frozenset([
            edge_key((x, y, z), (x + 1, y, z)), edge_key((x, y, z), (x, y, z + 1)),
            edge_key((x + 1, y, z), (x + 1, y, z + 1)),
            edge_key((x, y, z + 1), (x + 1, y, z + 1)),
        ])
        plaquettes.append({"check_id": f"ord_xz({x},{y},{z})", "pauli": "Z",
                           "support_keys": e, "loc": (x, y, z)})
    for x, y, z in itertools.product(range(-R, R + 1), range(-R, R), range(-R, R)):
        e = frozenset([
            edge_key((x, y, z), (x, y + 1, z)), edge_key((x, y, z), (x, y, z + 1)),
            edge_key((x, y + 1, z), (x, y + 1, z + 1)),
            edge_key((x, y, z + 1), (x, y + 1, z + 1)),
        ])
        plaquettes.append({"check_id": f"ord_yz({x},{y},{z})", "pauli": "Z",
                           "support_keys": e, "loc": (x, y, z)})
    plaquettes = [p for p in plaquettes if p["support_keys"] not in novel_z_keys]
    return stars, plaquettes


def gf2_solve(A: np.ndarray, b: np.ndarray):
    """Solve A x = b over F_2 by augmented Gaussian elimination. Free vars → 0."""
    A = (np.asarray(A, dtype=np.uint8) % 2).copy()
    b = (np.asarray(b, dtype=np.uint8) % 2).reshape(-1).copy()
    m, n = A.shape
    M = np.hstack([A, b.reshape(-1, 1)])
    row = 0
    pivots = []
    for col in range(n):
        piv = None
        for i in range(row, m):
            if M[i, col]:
                piv = i
                break
        if piv is None:
            continue
        M[[row, piv]] = M[[piv, row]]
        for i in range(m):
            if i != row and M[i, col]:
                M[i] ^= M[row]
        pivots.append(col)
        row += 1
        if row == m:
            break
    for i in range(row, m):
        if M[i, n]:
            return None
    x = np.zeros(n, dtype=np.uint8)
    for i, col in enumerate(pivots):
        x[col] = M[i, n]
    return x


def _fmt_endpoints(keys, key_to_name, named_ends):
    parts = []
    for k in keys:
        name = key_to_name.get(k)
        if name and name in named_ends:
            a, b = named_ends[name]
            parts.append(f"{name}:{a}->{b}")
        else:
            parts.append(f"{k[0]}->{k[1]}")
    return ";".join(parts)


def build_cell_catalog_rows(cell: str, R: int, E_named, jdata, named_key):
    """Classify every check / port in the finite patch for one cell."""
    item, fig = SOURCE_META[cell]
    legs = OCCUPATION_LEGS[cell]
    leg_keys = {named_key[L] for L in legs}
    novel = [r for r in jdata if r["cell"] == cell]
    novel_z_keys = {
        frozenset(named_key[e] for e in r["support"])
        for r in novel if r["pauli"] == "Z"
    }
    stars, plaquettes = ordinary_stars_plaquettes(R, novel_z_keys)
    key_to_name = {v: k for k, v in named_key.items()}
    rows = []

    # Boundary ports (occupation legs)
    for i, leg in enumerate(legs):
        rows.append(dict(
            cell=cell, check_id=f"port_{leg}", pauli="—",
            support_edges=leg,
            support_endpoints=_fmt_endpoints([named_key[leg]], key_to_name, E_named),
            role=f"occupation leg {i} (H variable)",
            status="boundary_port",
            boundary_port=leg,
            source_section="VI B",
            source_item=item,
            source_figure=fig,
        ))

    # Novel internal checks
    for r in novel:
        rows.append(dict(
            cell=cell, check_id=f"novel_{r['check']}", pauli=r["pauli"],
            support_edges=" ".join(r["support"]),
            support_endpoints=_fmt_endpoints(
                [named_key[e] for e in r["support"]], key_to_name, E_named),
            role=r["role"],
            status="novel_internal",
            boundary_port="",
            source_section="VI B",
            source_item=item,
            source_figure=fig,
        ))

    # Replaced: origin star (always, for every junction with novel SX)
    origin_edges = [named_key[n] for n in ("xm", "xp", "ym", "yp", "zm", "zp")
                    if n in named_key]
    # For first/last the novel SX is the replacement of the truncated star
    rows.append(dict(
        cell=cell, check_id="replaced_origin_star", pauli="X",
        support_edges="(ordinary origin star; superseded by novel SX)",
        support_endpoints="",
        role="ordinary star at junction vertex replaced by novel SX (WB Remark 9)",
        status="replaced",
        boundary_port="",
        source_section="VI B Remark 9",
        source_item=item,
        source_figure=fig,
    ))
    for r in novel:
        if r["pauli"] == "Z":
            rows.append(dict(
                cell=cell, check_id=f"replaced_plaq_{r['check']}", pauli="Z",
                support_edges=" ".join(r["support"]),
                support_endpoints=_fmt_endpoints(
                    [named_key[e] for e in r["support"]], key_to_name, E_named),
                role="ordinary plaquette superseded by novel SZ (same support)",
                status="replaced",
                boundary_port="",
                source_section="VI B Remark 9",
                source_item=item,
                source_figure=fig,
            ))

    # Ordinary checks: retained_internal vs boundary_adjacent
    for ch in stars + plaquettes:
        touches_port = bool(ch["support_keys"] & leg_keys)
        names = sorted(key_to_name[k] for k in ch["support_keys"] if k in key_to_name)
        anon = [k for k in ch["support_keys"] if k not in key_to_name]
        supp_str = " ".join(names + [f"{a}->{b}" for a, b in anon])
        status = "boundary_adjacent" if touches_port else "retained_internal"
        rows.append(dict(
            cell=cell, check_id=ch["check_id"], pauli=ch["pauli"],
            support_edges=supp_str,
            support_endpoints=_fmt_endpoints(
                list(ch["support_keys"]), key_to_name, E_named),
            role=("ordinary check open onto occupation ports"
                  if touches_port else "ordinary SC check fully internal to patch"),
            status=status,
            boundary_port=" ".join(legs) if touches_port else "",
            source_section="VI A + VI B patch",
            source_item=item,
            source_figure=fig,
        ))
    return rows, novel, novel_z_keys, stars, plaquettes, leg_keys


def audit_cell(cell: str, R: int, E_named, jdata, named_key):
    pauli = PAULI_TYPE[cell]
    legs = OCCUPATION_LEGS[cell]
    cat_rows, novel, novel_z_keys, stars, plaquettes, leg_keys = build_cell_catalog_rows(
        cell, R, E_named, jdata, named_key)
    edges = all_edges(R)
    edge_index = {e: i for i, e in enumerate(edges)}
    n_e = len(edges)
    key_to_name = {v: k for k, v in named_key.items()}

    # A_int rows: novel_internal + retained_internal of anticommuting type
    int_checks = []
    if pauli == "X":
        for r in novel:
            if r["pauli"] == "Z":
                supp = [named_key[e] for e in r["support"]]
                int_checks.append(("novel_Z", supp))
        for ch in plaquettes:
            if ch["support_keys"] & leg_keys:
                continue  # boundary_adjacent
            if not all(e in edge_index for e in ch["support_keys"]):
                continue
            int_checks.append((ch["check_id"], list(ch["support_keys"])))
    else:
        for r in novel:
            if r["pauli"] == "X":
                supp = [named_key[e] for e in r["support"]]
                int_checks.append(("novel_X", supp))
        for ch in stars:
            if ch["support_keys"] & leg_keys:
                continue
            if not all(e in edge_index for e in ch["support_keys"]):
                continue
            int_checks.append((ch["check_id"], list(ch["support_keys"])))

    A_int = np.zeros((len(int_checks), n_e), dtype=np.uint8)
    for i, (_, supp) in enumerate(int_checks):
        for e in supp:
            A_int[i, edge_index[e]] = 1

    B = np.zeros((len(legs), n_e), dtype=np.uint8)
    for i, leg in enumerate(legs):
        B[i, edge_index[named_key[leg]]] = 1

    # Frontier non-port edges (leakage / other-boundary weight)
    frontier_nonport = []
    for e in edges:
        a, b = e
        if max(max(abs(c) for c in a), max(abs(c) for c in b)) == R:
            if e not in leg_keys:
                frontier_nonport.append(e)

    cert_rows = []
    n_pass = n_fail = 0
    for w in KERNELS[cell]:
        word = "".join(map(str, w))
        A_full = np.vstack([A_int, B]) if A_int.size else B.copy()
        b_full = np.concatenate([
            np.zeros(A_int.shape[0], dtype=np.uint8),
            np.asarray(w, dtype=np.uint8),
        ])
        sol = gf2_solve(A_full, b_full)
        if sol is None:
            n_fail += 1
            cert_rows.append(dict(
                cell=cell, kernel_word=word, pauli_type=pauli,
                witness_edges="", witness_weight=0,
                internal_syndrome_weight=-1,
                boundary_signature="", expected_signature=word,
                other_boundary_syndrome_weight=-1,
                support_radius=R, status="FAIL",
                note="no F2 solution in patch",
            ))
            continue
        int_syn = int(np.sum((A_int @ sol) % 2)) if A_int.size else 0
        bsig = "".join(str(int(x)) for x in (B @ sol) % 2)
        other = sum(int(sol[edge_index[e]]) for e in frontier_nonport)
        wit_keys = [edges[i] for i, v in enumerate(sol) if v]
        wit_names = [key_to_name.get(k, f"{k[0]}->{k[1]}") for k in wit_keys]
        ok = (int_syn == 0) and (bsig == word) and (other == 0)
        if ok:
            n_pass += 1
            status = "PASS"
        else:
            n_fail += 1
            status = "FAIL"
        cert_rows.append(dict(
            cell=cell, kernel_word=word, pauli_type=pauli,
            witness_edges="(identity)" if not wit_names else " ".join(wit_names),
            witness_weight=len(wit_names),
            internal_syndrome_weight=int_syn,
            boundary_signature=bsig, expected_signature=word,
            other_boundary_syndrome_weight=other,
            support_radius=R, status=status,
            note=("vacuum" if not wit_names else
                  "exact F2 witness; A_int p=0, B p=w, no frontier leakage"),
        ))
    return cat_rows, cert_rows, n_pass, n_fail, len(int_checks)


def write_provenance(path: Path) -> None:
    path.write_text("""# Exact cell patch catalog — provenance

Source: Williamson–Baspin, *Layer codes*, arXiv:2309.16503v2, Sec. VI B items 3–8
(Figs. 23–24) and Remark 9 (novel checks replace ordinary layer checks).

## Classification (no catalog-touch exemption)

| status | meaning |
|---|---|
| `novel_internal` | Transcribed Sec. VI B novel SX/SZ (from `wb_junction_support_catalog.csv`) |
| `replaced` | Ordinary origin star (and ordinary plaquette of novel-SZ support) superseded per Remark 9 |
| `retained_internal` | Ordinary SC star/plaquette in the L∞ patch whose support is disjoint from designated occupation legs |
| `boundary_port` | Designated occupation-leg edge (H_end / H_mid variable) |
| `boundary_adjacent` | Ordinary SC check that intersects a designated occupation leg (opens onto the ports; not a row of A_int) |

## Occupation ↔ H variables

| cell | legs (ordered) | H | kernel words |
|---|---|---|---|
| first_z / last_z | (backbone, data) | H_end=[1 1] | 00, 11 |
| middle_z | (pL, pR, x) | H_mid=[1 1 1] | 000, 110 (pass-through), 101, 011 |
| first_x / last_x / middle_x | X↔Z dual | same | same |

110 = occupied-backbone pass-through; 101 and 011 = the two attachment orientations.

## What was not done

- No freehand edges beyond the unit-cubic chart and the ordinary cubic complex.
- PASS is exact (internal syndrome 0 + boundary signature = w + no frontier leakage).
- R=2/R=3 exterior-syndrome bulk tests remain as independent stress checks only.
""")


def main(radius: int = PATCH_RADIUS) -> int:
    E_named, jdata = rc.wb_junction_support_data()
    named_key = {n: edge_key(*ends) for n, ends in E_named.items()}

    all_cat = []
    all_cert = []
    total_pass = total_fail = total_words = 0

    for cell in OCCUPATION_LEGS:
        cat, cert, n_pass, n_fail, n_int = audit_cell(
            cell, radius, E_named, jdata, named_key)
        all_cat.extend(cat)
        all_cert.extend(cert)
        total_pass += n_pass
        total_fail += n_fail
        total_words += n_pass + n_fail
        print(f"  {cell}: PASS {n_pass}/{n_pass+n_fail}  A_int_rows={n_int}")

    cat_path = ROOT / "wb_exact_cell_patch_catalog.csv"
    fields = ["cell", "check_id", "pauli", "support_edges", "role", "status",
              "boundary_port", "source_section", "source_item", "source_figure",
              "support_endpoints"]
    with cat_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_cat)

    cert_path = ROOT / "wb_exact_cell_word_certificate.csv"
    cfields = ["cell", "kernel_word", "pauli_type", "witness_edges", "witness_weight",
               "internal_syndrome_weight", "boundary_signature", "expected_signature",
               "other_boundary_syndrome_weight", "support_radius", "status", "note"]
    with cert_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cfields)
        w.writeheader()
        w.writerows(all_cert)

    write_provenance(ROOT / "wb_exact_cell_patch_provenance.md")

    print(f"EXACT CELL CERTIFICATE: PASS {total_pass}/{total_words} fail={total_fail} R={radius}")
    print(f"  -> {cat_path.name} ({len(all_cat)} rows)")
    print(f"  -> {cert_path.name} ({len(all_cert)} rows)")
    nonzero = [r for r in all_cert if r["kernel_word"] not in ("00", "000")]
    nz_pass = sum(1 for r in nonzero if r["status"] == "PASS")
    print(f"  nonzero words: {nz_pass}/{len(nonzero)}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
