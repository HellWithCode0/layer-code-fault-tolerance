#!/usr/bin/env python3
"""Checks and key numbers for the routed two-block campaign (Supplemental Sec. two-block-routed).

The routed circuits were generated and solved on the author's machine; this script
does not rebuild them.  It reads the archived tables in numerical/routed_campaign/
and checks what can be checked without the circuits:

  * every noiseless certification row is PASS with zero detector flips and every
    observable deterministic and correct;
  * every archived witness (the lightest pattern found) is a set of detector-error-model
    mechanisms whose detector symptoms cancel and whose observable effect contains
    the target observable, and its size equals the reported distance;
  * the numbers quoted in the paper, collected in key_numbers.json.

Usage:  python routed_campaign_check.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / "numerical" / "routed_campaign"
STAGES = ("short_schedule_R1_R2", "rerun_R3_R4", "long_schedule_R5_R7")


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def check_certification():
    n = 0
    for st in STAGES:
        for r in rows(D / st / "certification.csv"):
            obs = r["observables"].split()
            assert r["status"] == "PASS", (st, r)
            assert int(r["detector_flips"]) == 0, (st, r)
            assert all(o.endswith(":0") for o in obs), (st, r)
            n += 1
    return n


def check_witnesses():
    n = 0
    for st in ("rerun_R3_R4", "long_schedule_R5_R7"):
        for f in sorted((D / st / "witnesses").glob("*.json")):
            w = json.loads(f.read_text())
            wit = w.get("witness") or []
            if not wit:
                assert w["status"] == "UNKNOWN", f
                continue
            det, obs = {}, {}
            for e in wit:
                for x in e["dets"]:
                    det[x] = det.get(x, 0) ^ 1
                for o in e["obs"]:
                    obs[o] = obs.get(o, 0) ^ 1
            assert not any(det.values()), f"{f.name}: symptoms do not cancel"
            assert obs.get(w["observable_index"], 0) == 1, f"{f.name}: target not flipped"
            assert len(wit) == w["distance"], f"{f.name}: size {len(wit)} != {w['distance']}"
            n += 1
    return n


def fd_table():
    out = {}
    for st in STAGES:
        for r in rows(D / st / "fault_distance.csv"):
            key = (r["input"], r["interface"], r["observable"], int(r["rounds"]))
            out[key] = dict(distance=r["distance"] or None, lower_bound=r["lower_bound"] or None,
                            status=r["status"], stage=st)
    return out


def fmt(e):
    if e["status"] == "OPTIMAL":
        return e["distance"]
    if e["distance"]:
        return f"{e['lower_bound']}-{e['distance']}"
    return f">={e['lower_bound']}"


def main():
    ncert = check_certification()
    nwit = check_witnesses()
    fd = fd_table()
    table = {f"{i}|{f}|{o}|R={R}": fmt(e) for (i, f, o, R), e in sorted(fd.items())}
    ex = rows(D / "rerun_R3_R4" / "exhaustive_iceberg.csv")
    audit = {f"{r['interface']}|{r['observable']}": dict(single_mechanisms=int(r["n_weight1"]),
             below_distance=int(r["n_below"]), uncorrected=int(r["n_below_uncorrected"]),
             undetectable=int(r["n_below_undetectable"])) for r in ex}
    params = {r["input"]: r for r in rows(D / "short_schedule_R1_R2" / "code_parameters.csv")}
    cert = rows(D / "short_schedule_R1_R2" / "certification.csv")
    qubits = {}
    for r in cert:
        qubits.setdefault(r["input"], {})[r["interface"]] = int(r["qubits"])
    mc = rows(D / "short_schedule_R1_R2" / "monte_carlo.csv")
    rates = [int(r["fails_obs0"]) / int(r["shots"]) for r in mc]
    spacelike = rows(D / "band_spacelike.csv") if (D / "band_spacelike.csv").exists() else []
    key = dict(
        certification_rows_pass=ncert,
        witnesses_checked=nwit,
        fault_distances=table,
        decoder_audit_R3=audit,
        circuit_qubits=qubits,
        params={k: {c: v[c] for c in ("n", "K", "Lx", "dX_theorem_lb", "dZ_theorem_lb", "bridge_cut",
                                       "thick_cut", "colors_X", "colors_Z", "n_route", "tour_median",
                                       "tour_max", "n_checks")} for k, v in params.items()},
        monte_carlo_failure_fraction_range=[round(min(rates), 3), round(max(rates), 3)],
        band_spacelike=spacelike,
    )
    (D / "key_numbers.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"certification rows PASS: {ncert}; witnesses checked: {nwit}")
    for k, v in table.items():
        print(f"  {k:40s} {v}")
    print("decoder audit:", json.dumps(audit))
    print("MC failure fraction range:", key["monte_carlo_failure_fraction_range"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
