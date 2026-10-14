#!/usr/bin/env python3
"""Lightweight WB catalog helpers shared by the microscopic bridge and operational_anchor.

No decoder dependency (no pymatching). Pure Python / NumPy / stdlib only.
"""
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path


def combinatorial_incidence_certificate(catalog: Path) -> dict:
    """Audited incidence radii from wb_junction_support_catalog.csv.

    r_inc^(infty) = max L^infty endpoint distance from junction (0,0,0).
    r_inc^(G)     = max cubic-lattice vertex-graph (L1) distance from (0,0,0).
    These are distinct: e.g. endpoint (0,1,1) has L^infty=1 but graph distance 2.
    Do NOT equate graph distance with L^infty.
    """
    rows = list(csv.DictReader(catalog.open()))
    weights = [int(r["weight"]) for r in rows]
    max_weight = max(weights)
    origin = (0, 0, 0)
    max_linf = 0
    max_graph = 0
    endpoints = set()
    for r in rows:
        for part in r["support_endpoints"].split(";"):
            m = re.search(r":(\([^)]+\))->(\([^)]+\))", part)
            if not m:
                continue
            a = ast.literal_eval(m.group(1))
            b = ast.literal_eval(m.group(2))
            for pt in (a, b):
                endpoints.add(pt)
                dx = abs(pt[0] - origin[0])
                dy = abs(pt[1] - origin[1])
                dz = abs(pt[2] - origin[2])
                max_linf = max(max_linf, max(dx, dy, dz))
                max_graph = max(max_graph, dx + dy + dz)
    r_inc_infty = max_linf
    r_inc_G = max_graph
    assert max_weight == 6
    assert r_inc_infty == 1, f"expected r_inc^(infty)=1, got {r_inc_infty}"
    assert r_inc_G == 2, f"expected r_inc^(G)=2, got {r_inc_G}"
    assert any(
        abs(pt[0]) + abs(pt[1]) + abs(pt[2]) == 2 for pt in endpoints
    ), "no endpoint at graph distance 2"
    return {
        "n_checks": len(rows),
        "max_support_weight": max_weight,
        "r_inc_infty": r_inc_infty,
        "r_inc_G": r_inc_G,
        "r_inc": r_inc_infty,
        "n_distinct_endpoints": len(endpoints),
        "scope": (
            "Audited constants r_inc^(infty)=1 (max L^infty endpoint radius) and "
            "r_inc^(G)=2 (max vertex-graph distance); max support weight 6. "
            "These pin the finite incidence interface; not a metric slab width."
        ),
    }
