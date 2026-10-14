# Fault-distance rerun status (snapshot 4 October 2026, ~12:06 PT)

Quanto's orchestrator is still running. This folder is a snapshot, not a finished campaign.

## Done

Iceberg `[[4,2,2]]`, spacing 2, rounds = 3, m = 1:

| interface | Z (`Z1aZ1b`) | X (`m` or idle `X1a`) |
|---|---|---|
| bridge | 1 OPTIMAL | 3 OPTIMAL |
| thick grid | 5 OPTIMAL | 3 OPTIMAL |
| idle | 5 OPTIMAL | witness 10, lower bound 6, FEASIBLE |

Band code, spacing 2, rounds = 4, m = 1:

| interface | Z | X |
|---|---|---|
| bridge | 1 OPTIMAL | 4 OPTIMAL |
| thick grid | witness 17, lower bound 5, FEASIBLE (phase-2 prove of ≤6 unknown) | 4 OPTIMAL |

Geometric cuts: Iceberg bridge 1 / grid 5; band bridge 1 / grid 17. Steane params already built with m_grid = 2, cut = 18, rounds = 9.

## Incomplete

- Steane fault distances: **not started**. Params only.
- Band idle: not in this table.
- Exhaustive Iceberg decoder check: in progress at snapshot time. Partial rows in `exhaustive_iceberg.csv`: bridge Z PASS; idle Z/X and bridge X FAIL under BP+LSD (weight-1 faults below the predicted distance left uncorrected). Thick Z was running.

## Not the asymptotic claim

These inputs are not the balanced-good `K=Θ(N)` family. Monte Carlo from the Oct 3 package is archived under `quanto/original/` and was not the goal of this rerun.
