# Exact cell patch catalog — provenance

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
