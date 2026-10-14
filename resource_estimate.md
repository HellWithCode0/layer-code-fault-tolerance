# Resource estimate (explicit assumptions, explicit allocations)

Every schedule has an integer allocation of blocks (layer code) or tiles (surface code) by role, and its distance f is the least odd f with L_live x R_total x p_L(f) <= eps_mem, where L_live counts every live logical qubit (data, working, factory and courier) and R_total is the complete runtime of that schedule including factory waiting. The model 'postselected_optimistic' is NOT covered by the paper's theorem.

## Assumptions

| assumption | value | meaning |
|---|---|---|
| `W` | 100 | logical sites |
| `T` | 1000000 | logical layers; one distilled |A> per site per layer (worst case) |
| `eps` | 0.01 | total target failure; eps_mem = eps/2 (memory and gadgets), eps_res = eps/2 (final resources) |
| `p_base` | 0.001 | physical error rate, base noise point |
| `q0_base` | 0.01 | raw injected resource error, base noise point |
| `p_secondary` | 0.0001 | physical error rate, secondary noise point |
| `q0_secondary` | 0.001 | raw injected resource error, secondary noise point |
| `p_th` | 0.01 | threshold in the logical failure model (both architectures) |
| `A` | 0.1 | prefactor: failure per logical qubit per round p_L(f) = A (p/p_th)^((f+1)/2) |
| `cY_LC` | 15 | conditional S-correction slots per 15-to-1 run on layer-code blocks (each consumes one |Y>) |
| `cY_SC` | 0 | conditional S-correction slots per 15-to-1 run in the surface-code unit (auto-corrected) |
| `max_levels` | 12 | more distillation levels than this counts as not converging |
| `slot_rounds` | 3 | rounds per gadget slot, in units of f |
| `G15` | 30 | gadget slots per 15-to-1 run on layer-code blocks |
| `G7` | 14 | gadget slots per Steane 7-to-1 run on layer-code blocks |
| `G_inj` | 2 | gadget slots per raw resource (injection plus encoded courier) |
| `G_consume` | 1 | gadget slots to consume one resource into the data |
| `A_run_logicals` | 17 | logical slots held by one 15-to-1 run in the throughput schedule |
| `Y_run_logicals` | 9 | logical slots held by one 7-to-1 run in the throughput schedule |
| `courier_blocks` | 2 | live courier blocks per raw resource (throughput) or per factory (serial), full size |
| `layer_slots` | 2 | gadget slots per logical layer in the throughput-limited layer-code schedule |
| `s` | 2 | spacing of the explicit realization |
| `rate` | 0.1 | input code rate: K = max(1, floor(rate N)) |
| `check_frac` | 0.45 | m_X = m_Z = round(check_frac N) input checks (round half up) |
| `check_weight` | 8 | input check weight w (supports assumed to span the whole input) |
| `rel_dist` | 0.1 | input relative distance: d_in = ceil(rel_dist N) |
| `N_min` | 8 | smallest input length considered (a weight-w check needs N >= w) |
| `N_max` | 6000 | largest input length tabulated |
| `anc_LC` | 1.0 | syndrome-extraction ancillas per layer-code data qubit |
| `m0` | 2 | edge multiplicity of the gauging grid (full-distance measurement) |
| `grid_ratio` | 1.0 | gauging-grid edges per block qubit, before multiplicity |
| `cnot_layers_LC` | 28 | CNOT layers per layer-code extraction round |
| `tile_coeff` | 2.0 | physical qubits per surface-code tile, in units of f^2 |
| `route_SC` | 2 | tiles per logical site (data plus routing) |
| `layer_rounds_SC` | 2 | rounds per logical layer, in units of f |
| `fac_tiles_SC` | 11 | tiles per 15-to-1 unit (Litinski-style) |
| `fac_rounds_SC` | 11 | rounds per 15-to-1 run, in units of f |
| `cnot_layers_SC` | 8 | CNOT/measure layers per surface-code round |
| `t_layer_ns` | 50.0 | duration of one CNOT/measure layer in ns |
| `f_max` | 4001 | largest odd distance searched |

Derived: eps_mem = 0.005, eps_res = 0.005, required q_final <= eps_res/(W T) = 5e-11. Layer-code distance scenarios: slab_upper d = L_z d_in; small_instance d = L_z d_in / 2; proved_lower d = L_z ceil(d_in / eta), eta = 4.

## Distillation maps

- proved: q_(r+1) = C_dist q_r^2 with C_dist = binom(15 + c_Y, 2) = 435 (layer code, c_Y = 15) and 105 (surface code, c_Y = 0). The Steane constant binom(7,2) = 21 is smaller; A and Y share the family parameter q_r.
- exact_always_output, RM15 malignant counts by weight (enumerated): [0, 0, 105, 35, 1260, 168, 4725, 435, 6000, 280, 2835, 105, 420, 0, 15, 1]
- exact_always_output, Steane 7-to-1 malignant counts by weight (enumerated): [0, 0, 21, 7, 28, 0, 7, 1]
- postselected_optimistic, RM15 accepted counts [1, 0, 0, 35, 105, 168, 280, 435, 435, 280, 168, 105, 35, 0, 0, 1], accepted logical [0, 0, 0, 35, 0, 168, 0, 435, 0, 280, 0, 105, 0, 0, 0, 1]; Steane accepted [1, 0, 0, 7, 7, 0, 0, 1], accepted logical [0, 0, 0, 7, 0, 0, 0, 1].
- Layer-code A factory per-coordinate error q_c = q_A + q_Y/2 - q_A q_Y; surface code q_c = q_A.

## Largest raw error q0 at which each recursion converges

| model | layer code (c_Y = 15) | surface code (c_Y = 0) |
|---|---|---|
| proved | 2.30e-03 | 9.52e-03 |
| exact_always_output | 4.65e-03 | 0.0109 |
| postselected_optimistic | 0.0844 | 0.141 |

## Audit of the old model

- (A) old 'quadratic' law 35 q^2 versus the exact always-output leading coefficient 105 (q_out = 105 q^2 + O(q^3)); the proved bound is C_dist q^2 with C_dist = 435 (layer code) or 105 (surface code).
- (B) old surface-code serial distance f = 23 was chosen from a 2f-round layer; the serial runtime is 4.048e+11 rounds and W R_total p_L(f) = 4.048 against a budget of 0.005; the least f passing under the same model is 31.
- (C) old serial layer code at f = 27: K = 2.442, reserve leaves 1.22 logical slots per block, 83 blocks allocated in total, but each of the 82 factories needs ceil(16/K) = 7 blocks, i.e. 574 factory blocks in total, of which 573 were never allocated (the one block beyond the data cells is credited).

## Storage crossover

computation: block with extraction ancillas and gauging grid, n (1 + anc_LC + m0 grid_ratio) per K_d sites, versus route_SC surface-code tiles per site (the accounting of the space tables); storage: block with extraction ancillas only versus one tile. 'first d' is the least distance at which the layer code is cheaper; 'stable from' is the distance from which it stays cheaper.

| accounting | scenario | first d | stable from | N | K | K_d | n | qubits per logical (LC) | qubits per logical (SC) |
|---|---|---|---|---|---|---|---|---|---|
| computation | slab_upper | 609 | 609 | 81 | 8 | 7 | 2,592,072 | 1.48e+06 | 1.48e+06 |
| computation | small_instance | 1103 | 1169 | 151 | 15 | 14 | 1.70e+07 | 4.86e+06 | 4.87e+06 |
| computation | proved_lower | 2000 | 2309 | 281 | 28 | 27 | 1.08e+08 | 1.60e+07 | 1.60e+07 |
| storage | slab_upper | 609 | 609 | 81 | 8 | 7 | 2,592,072 | 7.41e+05 | 7.42e+05 |
| storage | small_instance | 1103 | 1169 | 151 | 15 | 14 | 1.70e+07 | 2.43e+06 | 2.43e+06 |
| storage | proved_lower | 2000 | 2309 | 281 | 28 | 27 | 1.08e+08 | 8.00e+06 | 8.00e+06 |

## Main table: base noise point (p = 0.001, q0 = 0.01), small_instance scenario

| model | architecture | schedule | converged | levels | q_final | f | N | K | n (block or tile) | data | working/routing | factory | courier | peak qubits | rounds | seconds | qubit-seconds | memory failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proved | layer_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | layer_code | serial_shared | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | layer_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | surface_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0105) |
| proved | surface_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0105) |
| exact_always_output | layer_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | layer_code | serial_shared | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | layer_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | surface_code | serial | True | 9 | 1.17e-16 | 47 | - | 1 | 4,418 | 100 | 100 | 11 | 0 | 9.32e+05 | 1.42e+20 | 5.68e+13 | 5.29e+19 | 3.00e-03 |
| exact_always_output | surface_code | throughput | True | 9 | 1.17e-16 | 45 | - | 1 | 4,050 | 100 | 100 | 1.66e+13 | 0 | 6.73e+16 | 9.00e+07 | 36 | 2.42e+18 | 1.50e-03 |
| postselected_optimistic | layer_code | serial | True | 3 | 1.11e-29 | 35 | 24 | 2 | 76,224 | 100 | 0 | 3,100 | 200 | 1.04e+09 | 3.78e+12 | 5.30e+06 | 5.49e+15 | 2.57e-03 |
| postselected_optimistic | layer_code | serial_shared | True | 3 | 1.11e-29 | 37 | 26 | 2 | 97,552 | 100 | 0 | 31 | 2 | 5.19e+07 | 4.00e+14 | 5.60e+08 | 2.91e+16 | 1.06e-03 |
| postselected_optimistic | layer_code | throughput | True | 3 | 1.11e-29 | 33 | 22 | 2 | 58,256 | 100 | 0 | 5,320,931 | 2,190,210 | 1.75e+12 | 2.02e+08 | 282 | 4.94e+14 | 3.03e-03 |
| postselected_optimistic | surface_code | serial | True | 2 | 1.65e-12 | 31 | - | 1 | 1,922 | 100 | 100 | 11 | 0 | 4.06e+05 | 6.29e+11 | 2.52e+05 | 1.02e+11 | 1.33e-03 |
| postselected_optimistic | surface_code | throughput | True | 2 | 1.65e-12 | 29 | - | 1 | 1,682 | 100 | 100 | 111,623 | 0 | 1.88e+08 | 5.80e+07 | 23.2 | 4.36e+09 | 6.49e-04 |

## Main table: secondary noise point (p = 0.0001, q0 = 0.001), small_instance scenario

| model | architecture | schedule | converged | levels | q_final | f | N | K | n (block or tile) | data | working/routing | factory | courier | peak qubits | rounds | seconds | qubit-seconds | memory failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proved | layer_code | serial | True | 5 | 6.21e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 10,500 | 200 | 1.79e+09 | 4.15e+14 | 5.81e+08 | 1.04e+18 | 4.52e-03 |
| proved | layer_code | serial_shared | True | 5 | 6.21e-15 | 21 | 21 | 2 | 45,600 | 100 | 0 | 53 | 2 | 2.83e+07 | 4.59e+16 | 6.42e+10 | 1.82e+18 | 1.42e-04 |
| proved | layer_code | throughput | True | 5 | 6.21e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 1.99e+09 | 4.30e+08 | 3.98e+14 | 5.29e+08 | 741 | 2.95e+17 | 1.28e-03 |
| proved | surface_code | serial | True | 4 | 2.08e-18 | 17 | - | 1 | 578 | 100 | 100 | 11 | 0 | 1.22e+05 | 6.76e+13 | 2.70e+07 | 3.30e+12 | 1.43e-03 |
| proved | surface_code | throughput | True | 4 | 2.08e-18 | 17 | - | 1 | 578 | 100 | 100 | 2.19e+07 | 0 | 1.26e+10 | 3.40e+07 | 13.6 | 1.72e+11 | 7.44e-05 |
| exact_always_output | layer_code | serial | True | 4 | 1.64e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 8,300 | 200 | 1.43e+09 | 2.71e+13 | 3.79e+07 | 5.42e+16 | 2.36e-04 |
| exact_always_output | layer_code | serial_shared | True | 4 | 1.64e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 83 | 2 | 4.69e+07 | 2.71e+15 | 3.79e+09 | 1.78e+17 | 7.72e-04 |
| exact_always_output | layer_code | throughput | True | 4 | 1.64e-15 | 17 | 17 | 1 | 29,448 | 100 | 100 | 1.30e+08 | 2.82e+07 | 1.87e+13 | 1.26e+08 | 177 | 3.30e+15 | 2.00e-03 |
| exact_always_output | surface_code | serial | True | 4 | 1.87e-18 | 17 | - | 1 | 578 | 100 | 100 | 11 | 0 | 1.22e+05 | 6.76e+13 | 2.70e+07 | 3.30e+12 | 1.43e-03 |
| exact_always_output | surface_code | throughput | True | 4 | 1.87e-18 | 17 | - | 1 | 578 | 100 | 100 | 2.19e+07 | 0 | 1.26e+10 | 3.40e+07 | 13.6 | 1.72e+11 | 7.44e-05 |
| postselected_optimistic | layer_code | serial | True | 2 | 6.34e-20 | 15 | 15 | 1 | 20,224 | 100 | 100 | 3,900 | 200 | 3.48e+08 | 8.26e+10 | 1.16e+05 | 4.02e+13 | 3.55e-03 |
| postselected_optimistic | layer_code | serial_shared | True | 2 | 6.34e-20 | 17 | 17 | 1 | 29,448 | 100 | 100 | 39 | 2 | 2.84e+07 | 9.36e+12 | 1.31e+07 | 3.72e+14 | 2.26e-04 |
| postselected_optimistic | layer_code | throughput | True | 2 | 6.34e-20 | 15 | 15 | 1 | 20,224 | 100 | 100 | 511,870 | 113,196 | 5.06e+10 | 9.01e+07 | 126 | 6.38e+12 | 5.63e-04 |
| postselected_optimistic | surface_code | serial | True | 2 | 1.51e-21 | 15 | - | 1 | 450 | 100 | 100 | 11 | 0 | 9.5e+04 | 2.68e+11 | 1.07e+05 | 1.02e+10 | 5.65e-04 |
| postselected_optimistic | surface_code | throughput | True | 2 | 1.51e-21 | 13 | - | 1 | 338 | 100 | 100 | 98,173 | 0 | 3.33e+07 | 2.60e+07 | 10.4 | 3.46e+08 | 2.56e-03 |

## All scenarios: base noise point (p = 0.001, q0 = 0.01)

Surface-code rows do not depend on the layer-code scenario and repeat across scenarios.

### slab_upper

| model | architecture | schedule | converged | levels | q_final | f | N | K | n (block or tile) | data | working/routing | factory | courier | peak qubits | rounds | seconds | qubit-seconds | memory failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proved | layer_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | layer_code | serial_shared | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | layer_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | surface_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0105) |
| proved | surface_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0105) |
| exact_always_output | layer_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | layer_code | serial_shared | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | layer_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | surface_code | serial | True | 9 | 1.17e-16 | 47 | - | 1 | 4,418 | 100 | 100 | 11 | 0 | 9.32e+05 | 1.42e+20 | 5.68e+13 | 5.29e+19 | 3.00e-03 |
| exact_always_output | surface_code | throughput | True | 9 | 1.17e-16 | 45 | - | 1 | 4,050 | 100 | 100 | 1.66e+13 | 0 | 6.73e+16 | 9.00e+07 | 36 | 2.42e+18 | 1.50e-03 |
| postselected_optimistic | layer_code | serial | True | 3 | 1.11e-29 | 35 | 17 | 1 | 29,448 | 100 | 100 | 6,100 | 200 | 7.66e+08 | 3.78e+12 | 5.30e+06 | 4.05e+15 | 2.46e-03 |
| postselected_optimistic | layer_code | serial_shared | True | 3 | 1.11e-29 | 37 | 19 | 1 | 41,120 | 100 | 100 | 61 | 2 | 4.33e+07 | 4.00e+14 | 5.60e+08 | 2.42e+16 | 1.05e-03 |
| postselected_optimistic | layer_code | throughput | True | 3 | 1.11e-29 | 33 | 17 | 1 | 29,448 | 100 | 100 | 9,935,689 | 2,190,210 | 1.43e+12 | 2.02e+08 | 282 | 4.03e+14 | 2.44e-03 |
| postselected_optimistic | surface_code | serial | True | 2 | 1.65e-12 | 31 | - | 1 | 1,922 | 100 | 100 | 11 | 0 | 4.06e+05 | 6.29e+11 | 2.52e+05 | 1.02e+11 | 1.33e-03 |
| postselected_optimistic | surface_code | throughput | True | 2 | 1.65e-12 | 29 | - | 1 | 1,682 | 100 | 100 | 111,623 | 0 | 1.88e+08 | 5.80e+07 | 23.2 | 4.36e+09 | 6.49e-04 |

### small_instance

| model | architecture | schedule | converged | levels | q_final | f | N | K | n (block or tile) | data | working/routing | factory | courier | peak qubits | rounds | seconds | qubit-seconds | memory failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proved | layer_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | layer_code | serial_shared | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | layer_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | surface_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0105) |
| proved | surface_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0105) |
| exact_always_output | layer_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | layer_code | serial_shared | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | layer_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | surface_code | serial | True | 9 | 1.17e-16 | 47 | - | 1 | 4,418 | 100 | 100 | 11 | 0 | 9.32e+05 | 1.42e+20 | 5.68e+13 | 5.29e+19 | 3.00e-03 |
| exact_always_output | surface_code | throughput | True | 9 | 1.17e-16 | 45 | - | 1 | 4,050 | 100 | 100 | 1.66e+13 | 0 | 6.73e+16 | 9.00e+07 | 36 | 2.42e+18 | 1.50e-03 |
| postselected_optimistic | layer_code | serial | True | 3 | 1.11e-29 | 35 | 24 | 2 | 76,224 | 100 | 0 | 3,100 | 200 | 1.04e+09 | 3.78e+12 | 5.30e+06 | 5.49e+15 | 2.57e-03 |
| postselected_optimistic | layer_code | serial_shared | True | 3 | 1.11e-29 | 37 | 26 | 2 | 97,552 | 100 | 0 | 31 | 2 | 5.19e+07 | 4.00e+14 | 5.60e+08 | 2.91e+16 | 1.06e-03 |
| postselected_optimistic | layer_code | throughput | True | 3 | 1.11e-29 | 33 | 22 | 2 | 58,256 | 100 | 0 | 5,320,931 | 2,190,210 | 1.75e+12 | 2.02e+08 | 282 | 4.94e+14 | 3.03e-03 |
| postselected_optimistic | surface_code | serial | True | 2 | 1.65e-12 | 31 | - | 1 | 1,922 | 100 | 100 | 11 | 0 | 4.06e+05 | 6.29e+11 | 2.52e+05 | 1.02e+11 | 1.33e-03 |
| postselected_optimistic | surface_code | throughput | True | 2 | 1.65e-12 | 29 | - | 1 | 1,682 | 100 | 100 | 111,623 | 0 | 1.88e+08 | 5.80e+07 | 23.2 | 4.36e+09 | 6.49e-04 |

### proved_lower

| model | architecture | schedule | converged | levels | q_final | f | N | K | n (block or tile) | data | working/routing | factory | courier | peak qubits | rounds | seconds | qubit-seconds | memory failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proved | layer_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | layer_code | serial_shared | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | layer_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0435) |
| proved | surface_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0105) |
| proved | surface_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0105) |
| exact_always_output | layer_code | serial | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | layer_code | serial_shared | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | layer_code | throughput | False | - | 0.01 | - | - | - | - | - | - | - | - | - | - | - | - | q_A stops decreasing at level 1 (0.01 -> 0.0193) |
| exact_always_output | surface_code | serial | True | 9 | 1.17e-16 | 47 | - | 1 | 4,418 | 100 | 100 | 11 | 0 | 9.32e+05 | 1.42e+20 | 5.68e+13 | 5.29e+19 | 3.00e-03 |
| exact_always_output | surface_code | throughput | True | 9 | 1.17e-16 | 45 | - | 1 | 4,050 | 100 | 100 | 1.66e+13 | 0 | 6.73e+16 | 9.00e+07 | 36 | 2.42e+18 | 1.50e-03 |
| postselected_optimistic | layer_code | serial | True | 3 | 1.11e-29 | 35 | 37 | 3 | 272,160 | 50 | 0 | 1,050 | 100 | 1.31e+09 | 7.57e+12 | 1.06e+07 | 1.38e+16 | 2.72e-03 |
| postselected_optimistic | layer_code | serial_shared | True | 3 | 1.11e-29 | 37 | 39 | 3 | 320,568 | 50 | 0 | 21 | 2 | 9.36e+07 | 4.00e+14 | 5.60e+08 | 5.24e+16 | 8.76e-04 |
| postselected_optimistic | layer_code | throughput | True | 3 | 1.11e-29 | 33 | 35 | 3 | 228,888 | 50 | 0 | 3,461,069 | 2,190,210 | 5.17e+12 | 2.02e+08 | 282 | 1.46e+15 | 3.42e-03 |
| postselected_optimistic | surface_code | serial | True | 2 | 1.65e-12 | 31 | - | 1 | 1,922 | 100 | 100 | 11 | 0 | 4.06e+05 | 6.29e+11 | 2.52e+05 | 1.02e+11 | 1.33e-03 |
| postselected_optimistic | surface_code | throughput | True | 2 | 1.65e-12 | 29 | - | 1 | 1,682 | 100 | 100 | 111,623 | 0 | 1.88e+08 | 5.80e+07 | 23.2 | 4.36e+09 | 6.49e-04 |

## All scenarios: secondary noise point (p = 0.0001, q0 = 0.001)

Surface-code rows do not depend on the layer-code scenario and repeat across scenarios.

### slab_upper

| model | architecture | schedule | converged | levels | q_final | f | N | K | n (block or tile) | data | working/routing | factory | courier | peak qubits | rounds | seconds | qubit-seconds | memory failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proved | layer_code | serial | True | 5 | 6.21e-15 | 19 | 11 | 1 | 7,968 | 100 | 100 | 10,500 | 200 | 3.47e+08 | 4.15e+14 | 5.81e+08 | 2.02e+17 | 4.52e-03 |
| proved | layer_code | serial_shared | True | 5 | 6.21e-15 | 21 | 11 | 1 | 7,968 | 100 | 100 | 105 | 2 | 9.78e+06 | 4.59e+16 | 6.42e+10 | 6.28e+17 | 1.41e-04 |
| proved | layer_code | throughput | True | 5 | 6.21e-15 | 19 | 11 | 1 | 7,968 | 100 | 100 | 1.99e+09 | 4.30e+08 | 7.72e+13 | 5.29e+08 | 741 | 5.72e+16 | 1.28e-03 |
| proved | surface_code | serial | True | 4 | 2.08e-18 | 17 | - | 1 | 578 | 100 | 100 | 11 | 0 | 1.22e+05 | 6.76e+13 | 2.70e+07 | 3.30e+12 | 1.43e-03 |
| proved | surface_code | throughput | True | 4 | 2.08e-18 | 17 | - | 1 | 578 | 100 | 100 | 2.19e+07 | 0 | 1.26e+10 | 3.40e+07 | 13.6 | 1.72e+11 | 7.44e-05 |
| exact_always_output | layer_code | serial | True | 4 | 1.64e-15 | 19 | 11 | 1 | 7,968 | 100 | 100 | 8,300 | 200 | 2.77e+08 | 2.71e+13 | 3.79e+07 | 1.05e+16 | 2.36e-04 |
| exact_always_output | layer_code | serial_shared | True | 4 | 1.64e-15 | 19 | 11 | 1 | 7,968 | 100 | 100 | 83 | 2 | 9.08e+06 | 2.71e+15 | 3.79e+09 | 3.44e+16 | 7.72e-04 |
| exact_always_output | layer_code | throughput | True | 4 | 1.64e-15 | 17 | 11 | 1 | 7,968 | 100 | 100 | 1.30e+08 | 2.82e+07 | 5.05e+12 | 1.26e+08 | 177 | 8.92e+14 | 2.00e-03 |
| exact_always_output | surface_code | serial | True | 4 | 1.87e-18 | 17 | - | 1 | 578 | 100 | 100 | 11 | 0 | 1.22e+05 | 6.76e+13 | 2.70e+07 | 3.30e+12 | 1.43e-03 |
| exact_always_output | surface_code | throughput | True | 4 | 1.87e-18 | 17 | - | 1 | 578 | 100 | 100 | 2.19e+07 | 0 | 1.26e+10 | 3.40e+07 | 13.6 | 1.72e+11 | 7.44e-05 |
| postselected_optimistic | layer_code | serial | True | 2 | 6.34e-20 | 15 | 11 | 1 | 7,968 | 100 | 100 | 3,900 | 200 | 1.37e+08 | 8.26e+10 | 1.16e+05 | 1.58e+13 | 3.55e-03 |
| postselected_optimistic | layer_code | serial_shared | True | 2 | 6.34e-20 | 17 | 11 | 1 | 7,968 | 100 | 100 | 39 | 2 | 7.68e+06 | 9.36e+12 | 1.31e+07 | 1.01e+14 | 2.26e-04 |
| postselected_optimistic | layer_code | throughput | True | 2 | 6.34e-20 | 15 | 11 | 1 | 7,968 | 100 | 100 | 511,870 | 113,196 | 1.99e+10 | 9.01e+07 | 126 | 2.51e+12 | 5.63e-04 |
| postselected_optimistic | surface_code | serial | True | 2 | 1.51e-21 | 15 | - | 1 | 450 | 100 | 100 | 11 | 0 | 9.5e+04 | 2.68e+11 | 1.07e+05 | 1.02e+10 | 5.65e-04 |
| postselected_optimistic | surface_code | throughput | True | 2 | 1.51e-21 | 13 | - | 1 | 338 | 100 | 100 | 98,173 | 0 | 3.33e+07 | 2.60e+07 | 10.4 | 3.46e+08 | 2.56e-03 |

### small_instance

| model | architecture | schedule | converged | levels | q_final | f | N | K | n (block or tile) | data | working/routing | factory | courier | peak qubits | rounds | seconds | qubit-seconds | memory failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proved | layer_code | serial | True | 5 | 6.21e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 10,500 | 200 | 1.79e+09 | 4.15e+14 | 5.81e+08 | 1.04e+18 | 4.52e-03 |
| proved | layer_code | serial_shared | True | 5 | 6.21e-15 | 21 | 21 | 2 | 45,600 | 100 | 0 | 53 | 2 | 2.83e+07 | 4.59e+16 | 6.42e+10 | 1.82e+18 | 1.42e-04 |
| proved | layer_code | throughput | True | 5 | 6.21e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 1.99e+09 | 4.30e+08 | 3.98e+14 | 5.29e+08 | 741 | 2.95e+17 | 1.28e-03 |
| proved | surface_code | serial | True | 4 | 2.08e-18 | 17 | - | 1 | 578 | 100 | 100 | 11 | 0 | 1.22e+05 | 6.76e+13 | 2.70e+07 | 3.30e+12 | 1.43e-03 |
| proved | surface_code | throughput | True | 4 | 2.08e-18 | 17 | - | 1 | 578 | 100 | 100 | 2.19e+07 | 0 | 1.26e+10 | 3.40e+07 | 13.6 | 1.72e+11 | 7.44e-05 |
| exact_always_output | layer_code | serial | True | 4 | 1.64e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 8,300 | 200 | 1.43e+09 | 2.71e+13 | 3.79e+07 | 5.42e+16 | 2.36e-04 |
| exact_always_output | layer_code | serial_shared | True | 4 | 1.64e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 83 | 2 | 4.69e+07 | 2.71e+15 | 3.79e+09 | 1.78e+17 | 7.72e-04 |
| exact_always_output | layer_code | throughput | True | 4 | 1.64e-15 | 17 | 17 | 1 | 29,448 | 100 | 100 | 1.30e+08 | 2.82e+07 | 1.87e+13 | 1.26e+08 | 177 | 3.30e+15 | 2.00e-03 |
| exact_always_output | surface_code | serial | True | 4 | 1.87e-18 | 17 | - | 1 | 578 | 100 | 100 | 11 | 0 | 1.22e+05 | 6.76e+13 | 2.70e+07 | 3.30e+12 | 1.43e-03 |
| exact_always_output | surface_code | throughput | True | 4 | 1.87e-18 | 17 | - | 1 | 578 | 100 | 100 | 2.19e+07 | 0 | 1.26e+10 | 3.40e+07 | 13.6 | 1.72e+11 | 7.44e-05 |
| postselected_optimistic | layer_code | serial | True | 2 | 6.34e-20 | 15 | 15 | 1 | 20,224 | 100 | 100 | 3,900 | 200 | 3.48e+08 | 8.26e+10 | 1.16e+05 | 4.02e+13 | 3.55e-03 |
| postselected_optimistic | layer_code | serial_shared | True | 2 | 6.34e-20 | 17 | 17 | 1 | 29,448 | 100 | 100 | 39 | 2 | 2.84e+07 | 9.36e+12 | 1.31e+07 | 3.72e+14 | 2.26e-04 |
| postselected_optimistic | layer_code | throughput | True | 2 | 6.34e-20 | 15 | 15 | 1 | 20,224 | 100 | 100 | 511,870 | 113,196 | 5.06e+10 | 9.01e+07 | 126 | 6.38e+12 | 5.63e-04 |
| postselected_optimistic | surface_code | serial | True | 2 | 1.51e-21 | 15 | - | 1 | 450 | 100 | 100 | 11 | 0 | 9.5e+04 | 2.68e+11 | 1.07e+05 | 1.02e+10 | 5.65e-04 |
| postselected_optimistic | surface_code | throughput | True | 2 | 1.51e-21 | 13 | - | 1 | 338 | 100 | 100 | 98,173 | 0 | 3.33e+07 | 2.60e+07 | 10.4 | 3.46e+08 | 2.56e-03 |

### proved_lower

| model | architecture | schedule | converged | levels | q_final | f | N | K | n (block or tile) | data | working/routing | factory | courier | peak qubits | rounds | seconds | qubit-seconds | memory failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proved | layer_code | serial | True | 5 | 6.21e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 10,500 | 200 | 1.79e+09 | 4.15e+14 | 5.81e+08 | 1.04e+18 | 4.52e-03 |
| proved | layer_code | serial_shared | True | 5 | 6.21e-15 | 21 | 22 | 2 | 58,256 | 100 | 0 | 53 | 2 | 3.61e+07 | 4.59e+16 | 6.42e+10 | 2.32e+18 | 1.42e-04 |
| proved | layer_code | throughput | True | 5 | 6.21e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 1.99e+09 | 4.30e+08 | 3.98e+14 | 5.29e+08 | 741 | 2.95e+17 | 1.28e-03 |
| proved | surface_code | serial | True | 4 | 2.08e-18 | 17 | - | 1 | 578 | 100 | 100 | 11 | 0 | 1.22e+05 | 6.76e+13 | 2.70e+07 | 3.30e+12 | 1.43e-03 |
| proved | surface_code | throughput | True | 4 | 2.08e-18 | 17 | - | 1 | 578 | 100 | 100 | 2.19e+07 | 0 | 1.26e+10 | 3.40e+07 | 13.6 | 1.72e+11 | 7.44e-05 |
| exact_always_output | layer_code | serial | True | 4 | 1.64e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 8,300 | 200 | 1.43e+09 | 2.71e+13 | 3.79e+07 | 5.42e+16 | 2.36e-04 |
| exact_always_output | layer_code | serial_shared | True | 4 | 1.64e-15 | 19 | 19 | 1 | 41,120 | 100 | 100 | 83 | 2 | 4.69e+07 | 2.71e+15 | 3.79e+09 | 1.78e+17 | 7.72e-04 |
| exact_always_output | layer_code | throughput | True | 4 | 1.64e-15 | 17 | 17 | 1 | 29,448 | 100 | 100 | 1.30e+08 | 2.82e+07 | 1.87e+13 | 1.26e+08 | 177 | 3.30e+15 | 2.00e-03 |
| exact_always_output | surface_code | serial | True | 4 | 1.87e-18 | 17 | - | 1 | 578 | 100 | 100 | 11 | 0 | 1.22e+05 | 6.76e+13 | 2.70e+07 | 3.30e+12 | 1.43e-03 |
| exact_always_output | surface_code | throughput | True | 4 | 1.87e-18 | 17 | - | 1 | 578 | 100 | 100 | 2.19e+07 | 0 | 1.26e+10 | 3.40e+07 | 13.6 | 1.72e+11 | 7.44e-05 |
| postselected_optimistic | layer_code | serial | True | 2 | 6.34e-20 | 15 | 15 | 1 | 20,224 | 100 | 100 | 3,900 | 200 | 3.48e+08 | 8.26e+10 | 1.16e+05 | 4.02e+13 | 3.55e-03 |
| postselected_optimistic | layer_code | serial_shared | True | 2 | 6.34e-20 | 17 | 17 | 1 | 29,448 | 100 | 100 | 39 | 2 | 2.84e+07 | 9.36e+12 | 1.31e+07 | 3.72e+14 | 2.26e-04 |
| postselected_optimistic | layer_code | throughput | True | 2 | 6.34e-20 | 15 | 15 | 1 | 20,224 | 100 | 100 | 511,870 | 113,196 | 5.06e+10 | 9.01e+07 | 126 | 6.38e+12 | 5.63e-04 |
| postselected_optimistic | surface_code | serial | True | 2 | 1.51e-21 | 15 | - | 1 | 450 | 100 | 100 | 11 | 0 | 9.5e+04 | 2.68e+11 | 1.07e+05 | 1.02e+10 | 5.65e-04 |
| postselected_optimistic | surface_code | throughput | True | 2 | 1.51e-21 | 13 | - | 1 | 338 | 100 | 100 | 98,173 | 0 | 3.33e+07 | 2.60e+07 | 10.4 | 3.46e+08 | 2.56e-03 |

## Checks

- ok: RM15 always-output counts match the stated list
- ok: RM15 accepted patterns = 2048 Hamming codewords
- ok: every converged row meets the memory budget
- ok: every converged row uses the least passing f (f - 2 fails)
- ok: every converged row meets the resource budget
- ok: all allocations are integers
- ok: serial layer-code factories hold their logical slots
- ok: audit (C): old serial layer code needs ceil(16/K) = 7 blocks per factory at f = 27
- ok: audit (B): old SC serial memory sum 4.048 at f=23, least passing f=31
- ok: audit (A): exact leading coefficient is 105
