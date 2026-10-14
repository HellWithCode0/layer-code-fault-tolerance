"""SWAP-free placement of the cut-obstruction circuits on IQM Garnet and Emerald.

The two-qubit gates of build(3) form a graph (ancilla x_v joined to data v and
to every edge qubit at v).  A subgraph monomorphism of that graph into the
coupling map of the chip is a placement that needs no SWAP.  Qubit numbers are
the 0-based numbers of the Open Quantum coupling map.
"""
import json
import sys
from pathlib import Path

import rustworkx as rx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw


def interaction_graph(w=3):
    g = rx.PyGraph()
    idx = {q: g.add_node(q) for q in hw.QUBITS}
    es = hw.edges_for(w)
    for v in hw.DATA:
        for t in [v] + [e for e in es if v in hw.EDGES[e]]:
            g.add_edge(idx[f"x_{v}"], idx[t], None)
    return g


def chip(name):
    b = json.loads((HERE / f"oq_backend_iqm_{name}.json").read_text())
    cm = b["constraint_data"]["topology"]["coupling_map"]
    n = b["constraint_data"]["n_qubits"]
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    for a, c in cm:
        g.add_edge(a, c, None)
    return g


def placements(name, limit=2000):
    g, h = chip(name), interaction_graph(3)
    out = []
    for m in rx.vf2_mapping(g, h, subgraph=True, induced=False, call_limit=None):
        place = {hw.QUBITS[hq]: gq for gq, hq in m.items()}
        out.append(place)
        if len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    for name in ("garnet", "emerald"):
        ps = placements(name)
        print(f"{name}: {len(ps)} SWAP-free placements found (search capped at 2000)")
        if ps:
            print("   example:", ps[0])
