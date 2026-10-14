# Layer-code scaling campaign

Simulations for *Universal fault-tolerant computation on three-dimensional
layer codes at optimal distance* (A. Katoch).  Built 27 September 2026.

**Status (3 October 2026).** This README was written before the run.  All
four parts have run, and their results are in Supplemental Sec. "Scaling with
block size" and in `numerical/scaling_campaign/`.  Two outcomes differ from
the predictions in the table below:

* Part B gave brackets only: d_X between 14 and 20, d_Z between 14 and 19,
  around the theorem's bounds of 18 and 16.  No logical below the bound was found.
* Part C: the reported bit has fault distance R = 2s+1 (exactly 5 at s = 2,
  5 to 7 at s = 3), not d_Z = 2s.  The weight-d_Z Z logicals flip only the
  spectator (`reported_bit_spacelike_check.py`).  Through the spanning grid the
  measured pair is exactly d_X = 5 at s = 2, but only bracketed (4 to 7) at s = 3.

## What this tests, and why it helps the paper

The paper's central claim is about joint logical measurements between two
layer-code blocks:

* a **thick** measurement grid keeps the full code distance;
* a **thin** interface cannot: its distance stays small however large the blocks get.

Right now the paper shows this at circuit level for **one small size only**
(two [[4,2,2]]-input blocks at spacing 2, 1000 shots, one noise strength).
The Supplemental outlook (item i) says a growing-size campaign is the next
step.  This package is that campaign.

| Part | What it runs | What it shows |
|---|---|---|
| A (main) | Full circuit-level Monte Carlo of the joint measurement at spacings s = 2, 3, 4 (and 5 in `heavy`), that is block distances (d_X, d_Z) = (5, 4) up to (11, 10), several noise strengths, thin vs thick vs idle memory, both logical sectors, reported bit, spectators | Thick grid: failure falls as blocks grow. Thin bridge: it does not. Finite-size crossing of the thick-grid curves. |
| B | Exact code distance (CP-SAT) of the explicit realization of the Steane [[7,1,3]] code | Tests the distance theorem on an input with d_in = 3 (the paper only checked d_in = 2). |
| C | Exact circuit fault distances (CP-SAT) at spacing 2 and 3 | Thin bridge of width w has fault distance w; thick grid has the full distance d_X = 2s+1 for the measured pair and d_Z = 2s for the reported bit, now beyond the paper's single size. (Prediction; see the status note above.) |
| D | The paper's full-model decoder and the campaign's decoder on identical shots (s = 2) | Shows the faster decoder does not change the conclusions. |

The circuits are the paper's own: for the paper's parameters the new code
builds **character-for-character identical** circuits to
`two_block_measurement.py` (the smoke test checks this).

**Decoder.** BP+LSD with the paper's settings.  To make larger sizes
affordable it decodes the X and Z sectors separately (standard for CSS
codes; about 60 times faster in our test at s = 2).  All faults are still
simulated; only the decoder uses less information, so results are typically
the same or slightly worse.  Part D checks this on identical shots.

## How to run

You need Linux (Debian 13 is fine), Python 3.11 to 3.13, 8 cores, 16 GB RAM,
and about 2 GB free disk.

### In a Jupyter notebook

1. Upload `layer_code_scaling_campaign.zip` and open `run_campaign.ipynb`
   (after unzipping).
2. Run the cells from top to bottom.  The long run starts in the background,
   so closing the browser tab does not stop it.

### In a terminal

```bash
unzip layer_code_scaling_campaign.zip
cd layer_code_scaling_campaign
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python campaign/smoke_test.py
nohup bash run_all.sh heavy > run.log 2>&1 &
```

If `pip install -r requirements.txt` cannot find an exact version, use
`pip install -r requirements-loose.txt` instead.

### How long it takes (8 cores)

| Profile | Time | Contents |
|---|---|---|
| `smoke` (inside the smoke test) | about 5 min | checks only |
| `standard` | about 12 h | Part A at d = 5, 7, 9 |
| `heavy` (recommended) | about 2 days | Part A at s = 2 to 5, [[6,4,2]] blocks, edge multiplicity 2, wider bridges; then Parts D, C and B |

The time is a budget: when it runs out, every task stops cleanly and you
keep all data collected so far.  A restart keeps the original budget.
Change it with `--hours`, for example
`python campaign/run_mc.py --profile heavy --hours 20`.
`results_bundle.zip` is written right after Part A and again at the very end.

### Stop, check, resume

* See progress: `tail results/mc_batches.log`
* See results so far (safe while it runs): `python campaign/analyze.py`
* Stop: `pkill -f run_all.sh; pkill -f run_mc.py; pkill -f run_distances.py; pkill -f decoder_check.py`
  (the worker processes then exit by themselves within a minute)
* Resume: run the same command again.  It continues where it stopped, never
  reuses a seed, skips the smoke test, and refuses to start if a run is
  already going.

## When it finishes

Run `python make_results_zip.py`.  It writes `results_bundle.zip` with every
CSV, the figures, `SUMMARY.md`, the logs, and your package versions.  Send
that file back; it is all that is needed to add the results to the paper.

## Files

* `campaign/two_block_family.py`: the circuits (generalized from the paper's script)
* `campaign/decoders.py`: BP+LSD decoders
* `campaign/profiles.py`: what each profile runs
* `campaign/run_mc.py`: Part A runner (resumable, one batch per CSV row, recorded seeds)
* `campaign/run_distances.py`: Parts B and C
* `campaign/decoder_check.py`: Part D
* `campaign/analyze.py`: tables, fits, figures, `results/SUMMARY.md`
* `campaign/smoke_test.py`: the five-minute check
* `campaign/layer_code_explicit.py`, `campaign/min_weight_logical.py`: unchanged copies from the paper's package
