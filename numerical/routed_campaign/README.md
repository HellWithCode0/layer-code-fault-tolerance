# Routed two-block campaign (Supplemental Sec. "Routed nearest-neighbor extraction and growing inputs")

Joint measurement of logical X1 of two explicit layer-code blocks by grid gauging, with every
check extracted by the routed nearest-neighbor schedule of Supplemental Proposition
`prop:nn-extraction`: one routing ancilla per check, a depth-first SWAP shuttle over the
lexicographically first spanning tree of a reserved routing graph, a data-ancilla CNOT on the
first visit to each contact site, retracing before readout, greedy coloring of overlapping
extraction words, and per-step idle noise on data qubits and on the routing qubits of the active
color class. Spacing s = 2, one ancilla per grid edge, R rounds per phase.

Inputs: `iceberg_422` ([[4,2,2]], n = 224), `band_m2` (checks 111100 and 001111 for both X and
Z, [[6,2,2]], n = 720) and `steane_713` ([[7,1,3]], n = 1792).

The circuits were generated and solved on the author's machine by a separate routed-circuit
generator; this directory holds its outputs.

## Folders

| Folder | Runs | Contents |
|---|---|---|
| `short_schedule_R1_R2/` | 3 October 2026; R = 2 (iceberg, band), R = 1 (Steane) | code parameters, noiseless certification, fault distances, BP+LSD Monte Carlo, the failed matching cross-check, run log, the generator's own summary |
| `rerun_R3_R4/` | 3 to 4 October 2026; R = 3 (iceberg), R = 4 (band) | the same tables, the exhaustive single-mechanism decoder test `exhaustive_iceberg.csv`, witnesses; a snapshot taken while the run was still going (`STATUS.md`) |
| `long_schedule_R5_R7/` | 4 to 5 October 2026; R = 5 (iceberg, complete), R = 7 (band, partial) | certification, fault distances for every observable, witnesses, run log; the band rows are a snapshot of a run still in progress |

## Columns of `fault_distance.csv`

`distance` is the weight of the lightest undetectable pattern found, `lower_bound` the bound
the CP-SAT solver proved, `status` OPTIMAL (equal), FEASIBLE (a gap remains) or UNKNOWN (no
pattern found within the cap). Rows with `query = le6` asked only whether a pattern of weight at
most 6 exists. The `meets` column is the generator's comparison with its own target and is not
used by the paper: for the X-basis observables of the band input it compares with R = 7 or
d_X, while those observables are flipped by Z-type operators; the lightest one that flips X1 has weight 6 there
(`band_spacelike.csv`).

## Witnesses

Each `witnesses/*.json` file lists the detector-error-model mechanisms of the pattern (detector
indices and observable indices). `routed_campaign_check.py` in the package root checks that the
detector symptoms of every pattern cancel, that it flips its target observable, and that its
size equals the reported distance.

## Derived files

- `key_numbers.json`: every number quoted in the paper, written by `python routed_campaign_check.py`.
- `band_spacelike.csv`: lightest Z-type operator of the band block anticommuting with each X
  logical class, written by `python band_spacelike_check.py`.

## What is not here

No failure rates are used from these runs (Supplemental Sec. "Routed nearest-neighbor
extraction and growing inputs"). The band grid and idle Z rows and a long-schedule Steane run
were not finished.
