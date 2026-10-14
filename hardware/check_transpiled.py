"""Exact fault distance of the circuits exactly as transpiled for the backend."""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw
from circuit_distance_check import transpiled_distance

be = hw.service().backend(hw.BACKEND)
circs, init = hw.transpiled(be)
out = {}
for w, qc in circs.items():
    r, c = transpiled_distance(qc, w)
    out[w] = dict(distance=r["distance"], status=r["status"])
    print(f"transpiled w={w}: circuit fault distance {r['distance']} ({r['status']})", flush=True)
(HERE / "transpiled_distance.json").write_text(json.dumps(dict(physical_qubits=init, result=out), indent=1))
