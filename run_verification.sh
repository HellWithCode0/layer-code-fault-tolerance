#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

# Checks of the revised proofs (explicit realization, distance, boundary
# interface, defective resources, resource estimate).
for s in distance_lower_bound_check.py frame_propagation_check.py glue_example_check.py \
         boundary_composition_check.py resource_defect_check.py resource_estimate.py; do
  echo "== $s =="
  "$PY" "$s"
  echo
done

# Archived certificate of an earlier version (superseded; provenance only).
echo "== wb_exact_cell_patch_audit.py (ARCHIVED, superseded) =="
"$PY" wb_exact_cell_patch_audit.py

echo
echo "== wb_bullet_plus_geometric_audit.py (SM cross-check) =="
"$PY" wb_bullet_plus_geometric_audit.py

echo
echo "== wb_layer_code_instance.py (finite bridge instance) =="
"$PY" wb_layer_code_instance.py

echo
echo "== wb_microscopic_bridge_audit.py =="
"$PY" wb_microscopic_bridge_audit.py

echo
echo "== reproducibility_checks.py =="
"$PY" reproducibility_checks.py

echo
echo "== slab_interface_audit.py =="
"$PY" slab_interface_audit.py

echo
echo "== finite-size crossing fit (archived Monte Carlo data) =="
"$PY" numerical/continuous_transition/threshold_estimate/fit_threshold_estimate.py \
  --csv numerical/continuous_transition/threshold_estimate/all_points.csv \
  --out numerical/continuous_transition/threshold_estimate/estimate

echo
echo "== factory_map_audit.py =="
"$PY" factory_map_audit.py

echo
echo "== operational_anchor.py =="
"$PY" operational_anchor.py


echo
echo "== measurement_interface_check.py (thin vs thick interfaces) =="
"$PY" measurement_interface_check.py

echo
echo "== gauging_rectangle_mwpm.py =="
"$PY" gauging_rectangle_mwpm.py

echo
echo "VERIFICATION COMPLETE"
