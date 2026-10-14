# WB microscopic bridge — dependency map

> **Superseded in the revision (2026-09-26).** The exact-cell certificate described here
> does not establish the boundary relations; see `VERIFICATION.txt` and supplement
> Sec. SII. The slab theorem now rests on the explicit realization of supplement Sec. SI.
> The text below is kept for provenance.

## Chain (published → exact cells → slab theorem)

| Layer | Object | Status | Ledger |
|---|---|---|---|
| Published source | Condensed-e generators → H_end, H_mid | derived | `wb_gf2_derivation_audit.csv` |
| Published source | Sec. VI B items 3–8 novel checks | transcribed | `wb_junction_support_catalog.csv` |
| Exact patch | Novel/retained/replaced/boundary_port/boundary_adjacent | classified | `wb_exact_cell_patch_catalog.csv` |
| **Proof-critical** | Exact GF(2) word certificates (16/16) | **PASS** | `wb_exact_cell_word_certificate.csv` |
| Stress | R=2/R=3 exterior-syndrome bulk | demoted | `wb_junction_bulk_syndrome_audit.csv` |
| SM only | Bullet/plus geometric (4/4 nonzero) | cross-check | `wb_bullet_plus_*` |
| Axiom (WB-LR) | — | **RETIRED** | `wb_bridge_axiom_WB_LR.txt` |
| Lemma | Δ_cell = O(1) | finite audited cells | `lem:wb-cell-locality` |
| **Theorem** | `thm:slab-resolved` | H_end/H_mid + VI B cells + prefix | main text |

**One sentence:** The slab theorem uses only source-transcribed Sec. VI B first/middle/last
stabilizer cells, audited H_end/H_mid kernels (incl. 110 pass-through), and the prefix
construction; no residual un-audited local-realizability assumption remains beyond those cells.

Figure: `figures/wb_bridge_dependency.pdf` (`make_wb_bridge_dependency_figure.py`).
