> **Superseded in part (second review, 26 September 2026).** This is the response to the
> first review. Its paths `paper-review/revision/code/` and `revision/code/RUNS.md` refer to
> the working tree of that revision; in this package the scripts are in the root and the
> commands in `REVISION_RUNS.md`. Its distance bound (L_x+1) d_in / w_X, its resource numbers
> and crossovers, and its description of the twirl are superseded; see
> `RESPONSE_TO_SECOND_REVIEW.md` and the supplement's Archive section.

# Response to the review and summary of the revision

Manuscript: *Geometry-preserving fault-tolerant computation with three-dimensional layer codes*
(A. Katoch, revision of 26 September 2026).

This note answers the four main points of the review one at a time. For each point it says
whether the point holds, what we found when we checked it, and what changed in the
manuscript, supplement and code. All numbers below come from scripts in
`paper-review/revision/code/`. `revision/code/RUNS.md` gives the exact command for each output.

**Summary.** All four points hold, and two of them go further than the review said.

1. **Slab certificate.** The single-chart transcription of the Williamson–Baspin (WB) junction cells does not
   just lack a certificate. When the cells are glued, they enforce the wrong relation at middle junctions
   (`{000,111}` instead of `ker H_mid`), and the codes they produce do not encode the input
   logical qubits. The slab theorem is now proved for an explicit realization with complete check
   matrices. For that realization we prove, for every CSS input, that it encodes exactly `K` qubits,
   has single-layer slab representatives at every height, and has distance
   `d_X ≥ (L_x+1) d_X^in / w_X` and `d_Z ≥ L_z d_Z^in / w_Z` (so `Θ(N²)` for good inputs).
2. **Frame propagation.** Equations A15b/A15c/S3/S4 were missing the syndrome terms. We restored them,
   corrected the definition of relative effect, and added a lemma. That lemma closes the exact place
   where the missing terms could have broken the composition theorem.
3. **Complete measurement.** We simulated a complete two-block logical measurement on explicit layer-code blocks
   at circuit level, with a noiseless-reference (Choi-state) input that tests the reported bit, both sectors
   and the spectators. With a thin interface, one fault gives an undetectable logical error.
4. **Space–time cost.** The depth bound is now stated prominently. A concrete resource estimate with explicit
   assumptions shows that the architecture's advantage is far from the practical regime with present constants.

The review recommended stating the architecture theorem as conditional if the distance of the
new realization stayed open. We proved it instead (Theorem SI.6), so the universal architecture
theorem is unconditional. Every input to it is now either proved in the paper or cited.

---

## 1. Strengthen the slab certificate

**Does the point hold?** Yes, fully.

**What we checked** (`slab_gluing_certificate.py`, part A):

* We fed every boundary word, not only the kernel words, to the archived solver
  (`wb_exact_cell_patch_audit.py`). The ẑ cells accept `01` and `10` at the endpoints and all eight
  words at middle cells. The x̂ cells reject the odd words. This reproduces the review exactly.
* We evaluated every archived witness on *every* check of its patch, including the omitted
  "boundary_adjacent" checks. Every nonzero word leaves 2–6 violated checks
  (`slab_cert_A_archived_full_syndrome.csv`).

**What we found beyond the review.** The review allowed that the boundary syndrome might
cancel once the patches are glued. It does not. Gluing the single-chart cells gives the plain union of the
sheets, with shared line edges and merged stars. At a middle junction that union condenses every
`e` pair but only the fourfold `m` product, so a slab operator must satisfy `p_{t-1} = p_t = x_t`.
On complete codes (part B):

| input | K | k of the single-chart code |
|---|---|---|
| [[4,2,2]] | 2 | 4 |
| [[6,4,2]] | 4 | 6 |
| Steane [[7,1,3]] | 1 | 10 |
| Hamming [[15,7,3]] | 7 | 29 |
| surface [[13,1,3]] | 1 | 15 |
| random HGP (2 codes) | 2 | 20, 18 |

In every case no prefix-parity slab operator commutes with all checks. The old slab theorem therefore
rested on a transcription that does not realize the WB layer code.

**What changed.**

* **An explicit realization** (supplement Sec. SI A; `layer_code_explicit.py`). The sheets own distinct
  qubits. A middle Z junction adds the check sheet's edge to the data star, and the data line edge
  to the check face on one side. First and last Z junctions share the line. X junctions add the data
  z-edge to the check star, and the check's line edge to the data face above. Where an X-check sheet
  crosses a Z-check sheet, one-qubit "wall" additions are solved from the commutation conditions.
  Every check lies within unit distance of one lattice vertex and has weight ≤ 6. The condensed anyons at each junction are exactly
  the WB condensation data behind `H_end` and `H_mid`. The transcription's "novel" S^Z entries are exactly the faces
  that carry the other sheet's line edge here. This suggests the published cells use two sheets. We do not
  claim that the realization is literally identical to the drawings.
* **Proofs that hold for every CSS input**, not only for finite checks:
  * *Lemma SI.1 (walls).* Along each crossing line, the commutation equations form a path with both
    ends fixed. They are solvable, and then uniquely, **iff** `|supp X_i ∩ supp Z_j|` is even. So the
    point defects close precisely because the input commutes.
  * *Proposition SI.2 (k = K).* By an Euler count, each data sheet contributes +1 to `n − #X − #Z` and
    each check sheet −1. The relation spaces of the lattice checks inject into those of the input checks.
    Together with the K slab pairs, this gives `k = K`.
  * *Theorem SI.3 (slabs).* For every face, the overlap parity with the prefix operator is either 0 or
    exactly one first/middle/last relation of a single incidence. It follows that for every height
    `z0`, every pure-X class has a representative on the single layer of z-edges spanning
    `[z0, z0+1]`, and dually for Z in a plane `x = x0`. The thickness is `t0 = 1`, and no admissibility
    condition is needed.
* **Certificate on complete check matrices** (Sec. SII; `slab_cert_*.csv`). We ran seven inputs
  (n up to 10,936) at spacings 2 and 3 and checked that:
  * all checks commute and every wall system is solved;
  * the Euler count and the relation dimensions match the proof, and `k = K`;
  * the per-check linear forms are exactly the prefix relations at every height and column;
  * every slab operator (in total 564 X and 650 Z) commutes with every stabilizer;
  * stabilizer inputs map to stabilizers, and the pairing equals the input pairing with rank K.

  We also exhausted every incidence word and backbone assignment for checks of weight 2–6 (up to 2,048
  assignments each). The zero-syndrome set is exactly the `2^{k−1}` prefix words, and every other assignment
  violates a check at the offending incidence. As a negative control, walls are inconsistent exactly
  when the input overlap is odd. We export two instances with all labels (`explicit_layer_code_*.npz`)
  for independent inspection.
* **Distance** (Theorem SI.6, new). The explicit realization has
  `d_X ≥ (L_x+1) d_X^in / w_X` and `d_Z ≥ L_z d_Z^in / w_Z`, where `w_X`, `w_Z` are the input check weights.
  The slabs give matching upper bounds `(L_x+1) d_X^in + s m_Z n_in`, and the dual bound for `d_Z`. So
  `d = Θ(N²)` for balanced good inputs.

  The proof lifts the input distance through the complete ports. Take a nontrivial X logical `O` and a
  column `x0`. The parities of `O` on the pieces of the column-`x0` Z slab define a vector `v(x0)`. Because
  `O` pairs with every Z slab as the input logical does, `v(x0)` lies in `x + rowspace(H_X)`. It is
  therefore a nontrivial input logical, of weight at least `d_X^in`. Each piece changes at most `w_X`
  entries of `v(x0)`, so at least `d_X^in / w_X` pieces in that plane hold a qubit of `O`. The `L_x+1`
  planes are disjoint.

  `distance_lower_bound_check.py` checks every step of the proof on seven inputs. First, each basis slab's
  support is exactly the union of the pieces the proof uses, lies in its own plane or layer, and obeys the
  upper bound: 1,010 of 1,010 slabs pass. Second, the plane step holds for random representatives at every
  column and height: 3,858 of 3,858 cases pass. It also confirms that the bound lies below every exact
  distance in Table S2. There the exact values `d_X = L_x+1` and `d_Z = L_z` sit about a factor two above
  the bound.
* We removed or corrected the old claims ("exact-cell certified", "16/16 PASS", the truncated
  `n = 66` instance), and kept the old scripts for provenance only.

## 2. Make inherited-syndrome propagation explicit

**Does the point hold?** Yes. Your example (stabilizer `X1X2`, error `Z1`) is correct, and
linearity does not imply `C = 0`.

**What changed.**

* A15b/A15c and S3/S4 now read `λ_out = Lλ_in + C s_in + E e` and `o = Qλ_in + G s_in + F e`.
  The text explains that `C` and `G` are fixed by how representatives at the two cuts differ by
  stabilizers. It includes your example. It also notes that `C = 0` is achievable for unitary
  primitives but not in general for code deformations, so we keep the terms rather than fix a convention.
* The **relative effect** is now defined as `Δλ = λ_out + Lλ_in + C s_in` and `Δo = o + Qλ_in + G s_in`
  (Eq. A15-rel). This depends only on the internal faults.
* **The gap the missing terms opened, and how it is closed.** The padded-composition theorem identifies a
  contained component's relative effect with its global effect. That identification fails if the
  component leaves nonzero syndrome at its right cut, because the downstream `C s` term could then turn a
  relatively trivial component into a logical error. The new **Lemma SV.12 ("contained components leave no cut
  syndrome")** proves `s_L = s_R = 0` for components strictly between idle-pad cuts. The proof telescopes
  along T0/T3/T4 rows. One subtlety came up while writing it: at a gauging exit, a detector-trivial
  pattern (for example a Gauss-law stabilizer applied mid-window) can leave a data deviation with nonzero
  syndrome that the recorded byproduct frame cancels. So the lemma is stated in the frame of the realized
  branch, and the T4 rows carry exactly that frame. The padded-composition proof and the rectangle-restriction
  proof now cite the lemma. The courier-spectator lemma now says explicitly that the syndrome terms of an
  inherited class stay inside `R_term`.
* **Check** (`frame_propagation_check.py`). The script covers three things:
  * Your example gives `C = [1]`.
  * On an explicit layer-code block whose cuts use slab representatives at different heights,
    the reduced law fails for 986 of 1,979 random inherited deviations with nonzero syndrome.
    The corrected law holds for all 2,000, and the corrected relative effect equals the internal
    effect in all 2,000.
  * An exhaustive idle-chain test (2^16 patterns) confirms the contained-component lemma.

## 3. Test the actual logical measurement operation

**Does the point hold?** Yes. The surface-code campaigns test neither the reported bit, nor both
sectors, nor spectators, nor entangled inputs. We confirm your counts (255 points, 4,534,000 shots,
15 zero-failure noiseless controls), and the manuscript keeps describing those campaigns within their scope.

**What changed.** We added a new experiment (`two_block_measurement.py`; supplement Sec. SXV;
Table S17).

* **Setup.** Two explicit `[[4,2,2]]` layer-code blocks (`n = 224` each, `K = 2`, `d_X = 5`, `d_Z = 4`).
  We measure `X̄1(a) X̄1(b)` by Williamson–Yoder gauging on a grid attached to the single-layer slabs;
  logical 2 of each block is a spectator.
* **Circuit.** The circuit is complete, with ancilla-based extraction of every check. It runs 4 rounds
  before, during, and after the deformation, and ends with a noisy transversal readout. Noise is uniform
  circuit-level at strength `p`.
* **Input.** Every logical qubit is maximally entangled with a noiseless reference. Two readout bases
  test all eight generators of the output Choi state:
  * the reported bit;
  * the X frames of both measured qubits and both spectators;
  * `Z̄Z̄` of the measured pair;
  * the Z frames of the spectators.
* **Noiseless check.** All detectors and observables are deterministic. So the circuit implements the
  projective measurement with the correct outcome and acts as the identity on the spectators, for an
  arbitrary entangled input.
* **Short undetectable fault patterns first**, as you suggested. We compute exact circuit fault
  distances per observable, as minimum-weight undetectable mechanism sets, with CP-SAT and a proven
  CSS sector reduction. Results:

| interface | cut | deformed d̃_X | reported bit m | spectator X frame | Z̄1Z̄1 of measured pair | spectator Z frame | failure rate at p = 0.002: m / Z̄1Z̄1 |
|---|---|---|---|---|---|---|---|
| bridge, 1 row | 1 | 1 | 4 | 4 | **1** | 4–12 | 2.7% / 5.3% |
| bridge, 2 rows | 2 | 2 | 4 | 4 | **2** | 5–12 | 3.3% / 5.3% |
| bridge, 3 rows | 3 | 3 | 4 | 4 | **3** | 5–13 | 0.0% / 0.7% |
| grid spanning both blocks | 5 | 5 | 4 | 4 | **5** | 5–12 | 3.3% / 0.0% |
| same grid, 2 ancillas per edge | 10 | 5 | 4 | 4 | **4–5** | 5–12 | 7.3% / 0.0% |

  Entries are exact minima unless a range is given; a range means the solver proved the lower bound and
  found a pattern at the upper bound. The reported bit and the spectators' X frames have fault distance 4
  for every interface. That is both the block distance d_Z and the number of deformed rounds. The measured pair's
  Z̄1Z̄1 fault distance is **exactly the bridge width w = 1, 2, 3**, so a single fault on a one-edge
  bridge breaks the measurement undetectably. This is the cut obstruction at circuit level: an X fault on a
  bridge edge changes the gauging byproduct frame by X̄1(a). A grid spanning both blocks restores it
  to **exactly 5**, the blocks' d_X. The deformed-code distance d̃_X follows the same pattern at code
  level. Data: `two_block_distance.csv`, `two_block_mc.csv`.

* **Monte Carlo.** A small BP+LSD sample (150 shots per basis at p = 0.002) shows the same ordering for Z̄1Z̄1
  (last column). The reported bit's failure rate does not improve with the interface, as its interface-independent
  fault distance predicts. These rates are indicative only. Matching is not applicable because the error model
  does not decompose into edges.

## 4. Quantify the space–time tradeoff

**Does the point hold?** Yes.

**What changed.**

* The abstract and main-text Sec. IV now state the depth `O(T f^{3/2} Ψ) ⊆ O(T f^{5.41})` next to
  the space bound. They explain that the serial schedule minimizes space, not qubit-time: with `P`
  factories per cell, space is `O(WfP + f^{3/2})` and depth `O(TKfΨ/P)`, so qubit-time does not depend
  on `P`. They also note that the exponent comes from the proven quadratic distillation recursion;
  cubic 15-to-1 suppression would give `Ψ = f^{2.47}` but is not proved for local-stochastic inputs.
* **Concrete estimate** (`resource_estimate.py`, `resource_estimate.md`; supplement Sec. SXVI). Every
  constant is a named assumption: routing, extraction depth (28 CNOT layers per round), ancilla
  multiplicity `m0 = 2`, factory slots, a 15-to-1 on ⌈16/K⌉ blocks, and the decoder/failure model. We
  compare against surface-code lattice surgery at equal target failure, in a serial regime and a
  throughput-limited regime, and report peak qubits, wall time and qubit-seconds. For `W = 100`,
  `T = 10^6`, `p = 10^{-3}`, `ε = 10^{-2}`:

| architecture | schedule | f | peak qubits | wall time | qubit-seconds |
|---|---|---|---|---|---|
| layer code | serial (theorem) | 27 | 2.4e7 | 18.5 h | 1.6e12 |
| surface code | serial | 23 | 2.2e5 | 45 h | 3.6e10 |
| surface code | throughput-limited | 23 | 1.0e8 | 18 s | 1.9e9 |
| layer code | throughput-limited | 23 | 8.6e10 | 97 s | 8.3e12 |

  The layer-code constants come from the explicit realization of a rate-0.1, relative-distance-0.1
  input with weight-8 checks. Its size is fixed by the construction. For its distance we take
  `d = (L_x+1) d_in / 2`, the ratio seen in every exact small instance. This gives
  `n = 509 d^{3/2}` and `K = 0.47 d^{1/2}`. The two ends of the proved bracket are also reported:
  the slab upper bound gives `(180, 0.33)` and the proved lower bound gives `(4076, 0.94)`.
  (An earlier draft of this estimate used `n = 190 d^{3/2}`. That value assumed `d ≈ (L_x+1) d_in`, which is
  optimistic by the factor two above.)

  With these constants, a layer-code block holds fewer than three logicals at the needed distances. It
  stores a logical more cheaply than a surface-code patch only for `d ≳ 1,100` (≈550 to ≈4,300 across the
  bracket). In qubit-time it wins only for `f ≳ 6·10^5` (`7·10^4` at the slab upper bound, beyond `10^7` at
  the proved lower bound). The cause is that distillation dominates both architectures and a 15-to-1
  run occupies whole layer-code blocks. The manuscript
  now says plainly that the asymptotic advantage is far from the practical regime with present constants.

## What would still make the result stronger

* **Tighten the distance constant.** Theorem SI.6 loses a factor of the check weight `w`. The exact
  small instances sit a factor of about two above it. Closing this gap would pin the layer-code
  storage constant, which the resource estimate brackets within a factor of about 23.
* **A practical decoder for code changes.** The decoders of Gu et al. (arXiv:2510.06659, already cited)
  are the natural starting point, as you suggest.
* **Larger two-block simulations** (s ≥ 3, good-code inputs) once a faster decoder exists. The present
  exact-distance search takes about 10 minutes per basis at `n ≈ 1,300` total qubits.

## Items for the author to confirm

* The acknowledgments do not mention the reviewer. Add a line if appropriate.
* The AI-assistance acknowledgement is exactly as in the original submission, as the journal asked.
  The build checks this with a diff against the original source.
* The explicit realization is our construction. Please check that its relation to the published WB cells
  (supplement Sec. SI B) is described as you would like, and consider contacting D. J. Williamson about it.
