#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== exact finite-cell certificate =="
"$PY" wb_exact_cell_patch_audit.py

echo
echo "== bullet/plus geometric cross-check =="
"$PY" wb_bullet_plus_geometric_audit.py

echo
echo "== finite WB incidence test instance =="
"$PY" wb_layer_code_instance.py

echo
echo "== master microscopic bridge ledger =="
"$PY" wb_microscopic_bridge_audit.py

echo
echo "== slab interface audit =="
"$PY" slab_interface_audit.py

echo
echo "BRIDGE VERIFICATION COMPLETE"
