# Exact commands used for the archived results

Run from this directory with the packages in `requirements-revision.txt`
(`pip install -r requirements-revision.txt`).

| Output | Command |
|---|---|
| `slab_cert_*.csv`, `explicit_layer_code_*.npz` | `python slab_gluing_certificate.py /path/to/Katoch_FINAL_SOURCE_2026-09-26_FIGURES` |
| `explicit_layer_code_distance.csv` | `python explicit_layer_code_distance.py` |
| `distance_lower_bound_check.csv`, `distance_bound_vs_exact.csv` | `python distance_lower_bound_check.py` (after `explicit_layer_code_distance.py`, whose CSV it reads) |
| `frame_propagation_check.csv` | `python frame_propagation_check.py` |
| (console only) | `python glue_example_check.py` |
| `boundary_composition_check.csv` | `python boundary_composition_check.py` (several minutes on one core) |
| `resource_defect_check.csv` | `python resource_defect_check.py` (about 3 minutes) |
| `resource_estimate.csv`, `resource_estimate.md`, `resource_estimate_summary.json` | `python resource_estimate.py` (about one minute) |
| `two_block_distance.csv` | `python two_block_measurement.py --distance --time-limit 180 --out two_block_distance.csv`, then `python tighten_two_block.py --configs thick:0:1,thick:0:2 --observable Z1aZ1b --time-limit 1200` |
| `two_block_mc.csv` | for each interface `c` in `bridge:1:1 bridge:2:1 bridge:3:1 thick:0:1 thick:0:2` and `k` in `0 1 2 3`: `python two_block_measurement.py --configs c --skip-code-distance --shots 250 --p 0.002 --seed $((20260926 + 1000*k)) --out mc_c_k.csv`; then `python merge_two_block_mc.py mc_*.csv --out two_block_mc.csv` (the 20 chunk files are archived in `numerical/two_block_mc/`) |
| `numerical/scaling_campaign/` (two-block scaling campaign, 29 September 2026) | `cd scaling_campaign && bash run_all.sh heavy` on 8 cores (Part A: `python campaign/run_mc.py --profile heavy`, 36 h budget; every batch and its seed is a row of `mc_batches.csv`; see `scaling_campaign/README.md`) |
| `numerical/scaling_campaign/fault_distance.csv` | `cd scaling_campaign && python campaign/run_distances.py fault --spacings 2 3 --time-limit 1800` (Part C of `run_all.sh`, 8 threads). Its `expected` column for the reported bit `m` is `2s = d_Z`; that expectation was wrong, because the weight-`d_Z` Z logicals flip only the spectator (see `reported_bit_spacelike.csv`), and the measured values `2s+1 = R` are the correct ones |
| `numerical/scaling_campaign/code_distance.csv` | `cd scaling_campaign && python campaign/run_distances.py code --spacings 2 --time-limit 7200` (Part B of `run_all.sh`, Steane input, 8 threads) |
| `numerical/scaling_campaign/reported_bit_spacelike.csv` | `python reported_bit_spacelike_check.py` (about one minute) |
| `figures/two_block_scaling.pdf`, `two_block_scaling_table.tex`, `numerical/scaling_campaign/key_numbers.json` | `python make_scaling_figure.py` (reads `numerical/scaling_campaign/mc_batches.csv` and `decoder_check.csv`, the latter from `cd scaling_campaign && python campaign/decoder_check.py --shots 2000`, and the three distance files above; every number in Supplemental Sec. "Scaling with block size" comes from `key_numbers.json`) |
| `numerical/routed_campaign/` (routed two-block campaign, 3 to 5 October 2026) | generated and solved on the author's 8-core machine by a separate routed-circuit generator; the archived tables, logs and witnesses are described in `numerical/routed_campaign/README.md` |
| `numerical/routed_campaign/key_numbers.json` | `python routed_campaign_check.py` (checks the 40 certification rows and the 45 archived witnesses; seconds) |
| `numerical/routed_campaign/band_spacelike.csv` | `python band_spacelike_check.py 1200` (CP-SAT, 8 threads; at most 20 minutes per class) |
| `figures/junction_wall_geometry.pdf` | `python make_junction_figure.py` |
| the other 21 figure files | `python make_publication_figures.py` |

The CP-SAT runs are time-limited, so a slower or busier machine can report a looser bracket
(a smaller proven lower bound) for the entries that are not exact. The two-block distance sweep
and the original Monte Carlo ran concurrently on four cores, and the tightening pass ran alongside
the Monte Carlo.

Seeds. Every Monte Carlo run uses a recorded integer seed: `two_block_measurement.py` adds a
CRC32 offset of the interface label and basis to the base seed `--seed` and writes the seed of
each (interface, basis) run into its CSV; `merge_two_block_mc.py` refuses a repeated seed and
lists all seeds in `two_block_mc.csv`. (The first version of the two-block sampling derived its
seeds from Python's `hash()` of strings, which is salted per process; those runs could not be
regenerated and are replaced.)
