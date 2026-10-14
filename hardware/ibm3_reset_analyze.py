"""Analyze the IBM reset follow-up (ibm3_reset_submit.py).

Analysis plan, fixed before submission (2026-10-14):
  Persistence: for the check ancillas ca01 and cb01 of each copy, a shot counts
  when the ancilla reads 1 in rounds 1 and 2 while the ancillas sharing its
  data qubits show no change (no data error explains it).  Pooled over the
  two ancillas, two copies and both R = 2 and R = 3 circuits.
  Primary tests (one-sided Fisher exact, level 0.05 each):
    T1  persistence lower with two resets than with one;
    T2  reported-bit error at R = 2 lower with two resets than with one;
    T3  the same at R = 3.
  Replication: R = 2 below R = 1 and R = 3 below R = 2, for each reset variant.
  Reading: T1 passing means failed resets cause part of the persistence.  If
  T1 fails while persistence stays far above the independent-error level, the
  cause is more likely leakage, which a reset cannot remove.
Postselected errors are pooled over the two copies.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ibm_rounds_hw as rh
from cut_obstruction_hw import wilson
from ibm_rounds_analyze import fisher_greater

PAIRS = {"ca01": [("sa", "star"), ("ca12", "check")], "cb01": [("sb", "star"), ("cb12", "check")]}


def persistence(b, R):
    """(events, trials, independent-error expectation) pooled over ancillas and copies."""
    ev = tr = 0
    indep = 0.0
    for k in (0, 1):
        g = lambda a, r: b[:, rh.clbit(k, r, a, R)].astype(bool)
        for a, nbs in PAIRS.items():
            clean = np.ones(b.shape[0], bool)
            for nb, kind in nbs:
                clean &= (g(nb, 1) == g(nb, 2)) if kind == "star" else (~g(nb, 1) & ~g(nb, 2))
            r1, r2 = g(a, 1) & clean, g(a, 2) & clean
            ev += int((r1 & r2).sum())
            tr += b.shape[0]
            indep += r1.mean() * r2.mean() / max(clean.mean(), 1e-12) * b.shape[0]
    return ev, tr, indep


def pooled_error(b, R):
    cs = [rh.analyze_bits(b, R, k) for k in (0, 1)]
    f = sum(c["postselected_failures"] for c in cs)
    a = sum(c["accepted"] for c in cs)
    return dict(copies=cs, failures=f, accepted=a, error=f / max(a, 1), wilson95=wilson(f, a))


def analyze(bits, plan):
    out = {"per_circuit": {}, "tests": {}}
    for label, b in bits.items():
        R = plan["circuits"][label]["R"]
        res = pooled_error(b, R)
        if R >= 2:
            ev, tr, ind = persistence(b, R)
            res["persistence"] = dict(events=ev, trials=tr, rate=ev / tr, independent_expectation=ind / tr)
        out["per_circuit"][label] = res
    pc = out["per_circuit"]
    s = [pc[l]["persistence"] for l in ("R2_single", "R3_single")]
    d = [pc[l]["persistence"] for l in ("R2_double", "R3_double")]
    out["tests"]["T1_persistence_double_below_single"] = fisher_greater(
        sum(x["events"] for x in s), sum(x["trials"] for x in s), sum(x["events"] for x in d), sum(x["trials"] for x in d))
    for R in (2, 3):
        a, b_ = pc[f"R{R}_single"], pc[f"R{R}_double"]
        out["tests"][f"T{R}_R{R}_error_double_below_single"] = fisher_greater(
            a["failures"], a["accepted"], b_["failures"], b_["accepted"])
    for v in ("single", "double"):
        r1, r2, r3 = pc["R1"], pc[f"R2_{v}"], pc[f"R3_{v}"]
        out["tests"][f"replication_{v}_R2_below_R1"] = fisher_greater(r1["failures"], r1["accepted"], r2["failures"], r2["accepted"])
        out["tests"][f"replication_{v}_R3_below_R2"] = fisher_greater(r2["failures"], r2["accepted"], r3["failures"], r3["accepted"])
    return out


def report(out):
    for label, r in out["per_circuit"].items():
        line = (f"{label:10s}: {r['failures']}/{r['accepted']} = {100 * r['error']:.4f}% "
                f"[{100 * r['wilson95'][0]:.4f}%, {100 * r['wilson95'][1]:.4f}%]  (copies "
                + ", ".join(f"{100 * c['postselected_error']:.3f}%" for c in r["copies"]) + ")")
        if "persistence" in r:
            p = r["persistence"]
            line += f" | persistence {100 * p['rate']:.3f}% (independent {100 * p['independent_expectation']:.3f}%)"
        print(line)
    for name, p in out["tests"].items():
        tag = "CLEAR" if p < 0.05 else "not clear"
        print(f"{name}: one-sided Fisher p = {p:.3g}{' (below 1e-300)' if p == 0 else ''} -> {tag}")


def load(path=HERE / "ibm3_reset_bits.npz"):
    z = np.load(path)
    return {k[2:]: np.unpackbits(z[k], axis=1, count=int(z["n_" + k[2:]]), bitorder="little")
            for k in z.files if k.startswith("b_")}


def main():
    plan = json.loads((HERE / "ibm3_reset_plan.json").read_text())
    out = analyze(load(), plan)
    report(out)
    (HERE / "ibm3_reset_results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
