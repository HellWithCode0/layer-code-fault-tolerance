# Bullet/plus edge-support transcription provenance

Source: Williamson–Baspin, *Layer codes*, arXiv:2309.16503v2 (Nat. Commun. 15, 9528, 2024).
Local PDF: `../wb_source/2309.16503v2.pdf`.

## Identification (not invention)

### Plus (`⊕`, Sec. III E / Fig. 22 abstract; Sec. VI B lattice)

- **Katoch SM table** (and `slab_cell_audit.csv`): plus ↔ “middle line-defect realization in Sec. VI B”.
- **Kernel identity:** \(\mathcal P_m=\{000,110,101,011\}=\ker H_{\rm mid}\).
- **Edge rows:** copied from the already-transcribed Sec. VI B **item 4** (`middle_z`) novel SX (weight 6) and SZ (weight 4) supports in `wb_junction_support_catalog.csv`.
- **Occupation legs:** `(pL,pR,b) ↔ (ym,yp,xm)` — same map as middle_z’s `(pL,pR,x)`.
- **Chart Paulis:** reused from `wb_local_chart_syndrome_audit.csv` middle_z rows (words 110,101,011).

### Bullet (`•`, Sec. III E / Fig. 18; Sec. VI finite lattice)

- **Fig. 18 (p.29):** trijunction of a zy surface-code layer with an xz layer; right panel shows three **m**-strings meeting — two along the grey (zy) sheet (vertical ±z) and one into the red (xz) sheet (+x). Caption: condensed anyons \(\langle e_\ell e_r,\, e_\ell e_{\hat z},\, m_\ell m_r m_{\hat z}\rangle\).
- **Fig. 22–23:** k-body Z-check network; Sec. VI B items 3–5 are the lattice checks realizing that network. Item 3 (first \(\hat z\)-junction) is the finite SX modification at the Fig. 18-type attachment where a yz-layer meets its first xz-layer (Fig. 23).
- **Novel-check row:** Sec. VI B **item 3** SX support `{xm,xp,yp,zm,zp}` (same transcription as `first_z`).
- **Occupation legs for m-word 111:** `(dL,dR,b) ↔ (zm,zp,xp)` — the three Fig. 18 m-legs in the unit-cubic chart with junction vertex at the origin and the first xz-layer in +x (matching item 3’s geometry).
- **Ambiguity:** Fig. 18 does not ink cubic edge labels. Reading the red-leg sign as −x (`xm`) instead of +x (`xp`) also yields a bulk-PASS Pauli `{xm,zm,zp}`. We adopt **+x (`xp`)** to match the published first-junction orientation (xz layer on the +x side of the yz sheet). Both readings are recorded; the audit uses `xp`.
- **Kernel:** \(\mathcal B_m=\{000,111\}\); word 111 ↔ X on all three occupation legs.

## What was *not* done

- No freehand edges beyond the Sec. VI B item 3/4 supports and the Fig. 18 three-leg reading.
- PASS criteria identical to junctions: empty novel syndrome + empty ordinary bulk exterior syndrome (catalog-disjoint stars/plaquettes) at \(L^\infty\) radius 2 (stress radius 3).

## Audit outputs

- `wb_bullet_plus_bulk_syndrome_audit.csv`
- `wb_bullet_plus_chart_syndrome_audit.csv`
