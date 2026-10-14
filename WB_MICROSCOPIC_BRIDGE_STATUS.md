# WB microscopic bridge — status (2026-09-19, Europe/Paris)

> **Superseded in the revision (2026-09-26).** The exact-cell certificate described here
> does not establish the boundary relations; see `VERIFICATION.txt` and supplement
> Sec. SII. The slab theorem now rests on the explicit realization of supplement Sec. SI.
> The text below is kept for provenance.

Package: `Katoch_FINAL_SOURCE_2026-09-19`

## Executive summary

The slab theorem’s critical microscopic input is the six Sec. VI B **first/middle/last**
stabilizer cells with boundary matrices **H_end = [1 1]** and **H_mid = [1 1 1]**.
Every kernel word (incl. vacuum) has an **exact finite-patch GF(2) certificate**
(`wb_exact_cell_word_certificate.csv`; **16/16 PASS**, nonzero **10/10**).

Bullet/plus factorization is **SM explanatory cross-check only** — not a theorem primitive.
Axiom (WB-LR) is **RETIRED**. Defensible closure:

> No residual un-audited local-realizability assumption remains beyond the
> explicitly transcribed Williamson–Baspin stabilizer cells.

## Hierarchy

| Layer | Role | Ledger |
|---|---|---|
| Exact cell GF(2) certificate | **Proof-critical** | `wb_exact_cell_word_certificate.csv` |
| R=2 / R=3 exterior-syndrome bulk test | Demoted stress | `wb_junction_bulk_syndrome_audit.csv` |
| Bullet/plus geometric audit | SM cross-check only | `wb_bullet_plus_*` |

## Claim ledger

| ID | Claim | Status |
|---|---|---|
| C1 | GF(2) H_end/H_mid; 110∈ker H_mid | PASS |
| C2 | Bullet/plus kernels (SM only) + H_inc | PASS |
| C3 | Six VI B supports | PASS |
| C4 | Prefix parity words 2..8 | PASS |
| C5 | Chart-local (legacy) | PASS |
| C5b | Bulk exterior (STRESS) | PASS |
| C6 | Refinement moves | PASS |
| C7 | Finite CSS instances | PASS |
| C8 | Combinatorial incidence (no pymatching) | PASS |
| **C9** | **exact_cell_realizability** | **PASS 16/16** |
| **C10** | **slab_inputs_closed** (refuses PASS unless C9) | **PASS** |

Master table: `wb_bridge_claim_status.csv`.

## Theorem primitives

At interior incidence: `(p_{i-1}, p_i, x_i) ∈ ker H_mid` with H_mid=[1 1 1].
Words 110 / 101 / 011 = pass-through / two attachment orientations.

At endpoints: `(p_1, x_1), (p_{k-1}, x_k) ∈ ker H_end` with H_end=[1 1].

## How to re-run (one command)

```bash
cd Katoch_FINAL_SOURCE_2026-09-19
bash run_bridge_verification.sh
# complete pinned-environment workflow:
bash run_verification.sh
```

## Key new files

- `wb_exact_cell_patch_catalog.csv` / `wb_exact_cell_patch_provenance.md`
- `wb_exact_cell_word_certificate.csv`
- `wb_exact_cell_patch_audit.py`
- `wb_catalog_utils.py` (bridge ↔ OA shared; no pymatching)
- `run_bridge_verification.sh`
