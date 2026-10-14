#!/usr/bin/env python3
"""Collect every hardware result into one summary, LaTeX tables and macros.

Reads the analysis outputs written by the run-specific scripts (and, for two
quantities, the raw shots), recomputes the statistics, and writes
  hardware/hardware_summary.json   all numbers used in the paper,
  ../hardware_tables.tex           Tables of the supplementary hardware section,
  ../hardware_macros.tex           \\newcommand macros for numbers in the text.
No number in the hardware sections of the manuscript or supplement is typed by
hand; rerun this script after any change to the data.

Usage:  python hardware/hardware_summary.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw
from cut_obstruction_hw import wilson
from ibm_rounds_analyze import fisher_greater
import ibm_rounds_analyze as ira
import ibm3_reset_analyze as i3


def J(name):
    return json.loads((HERE / name).read_text())


def row(fail, kept, shots, ctrl=None, model=None):
    lo, hi = wilson(fail, kept)
    r = dict(failures=int(fail), kept=int(kept), shots=int(shots), error=fail / kept, wilson95=[lo, hi])
    if ctrl is not None:
        r["control"] = dict(failures=int(ctrl[0]), kept=int(ctrl[1]), error=ctrl[0] / ctrl[1])
    if model is not None:
        r["model"] = model
    return r


def block_control_ibm(counts, w):
    acc = fail = 0
    for s, c in counts.items():
        s = s.replace(" ", "")
        if any(sum(int(s[-1 - hw.IDX[q]]) for q in d) % 2 for d in hw.detectors(w)[:4]):
            continue
        acc += c
        fail += c * (sum(int(s[-1 - hw.IDX[q]]) for q in hw.observable(w)) % 2)
    return fail, acc


def collect():
    S = {"cut": {}, "multiplicity": {}, "rounds": {}, "tests": {}, "jobs": {}, "excluded": {}}
    # ---------------- cut obstruction, IQM Emerald
    r1, c1 = J("oq_iqm_results.json"), J("cycle_check_control.json")
    m1 = J("iqm_plan.json")["prediction"]
    r2, c2 = J("oq_iqm2_results.json"), J("cycle_check_control2.json")
    m2 = J("iqm2_plan.json")["prediction"]
    pr3 = J("pooled_w23_results.json")
    for w in (1, 2, 3):
        a = r1[f"w{w}"]
        ctrl = c1[f"w{w} block checks only"]
        S["cut"][f"IQM run 1 w{w}"] = row(a["postselected_failures"], a["accepted"], a["shots"],
                                          (ctrl["failures"], ctrl["accepted"]) if w >= 2 else None,
                                          m1[f"w={w} p2=0.005"]["A"]["postselected_error"])
    for w in (1, 2):
        a = r2[f"w{w}"]
        ctrl = c2[f"w{w} block checks only"]
        S["cut"][f"IQM run 2 w{w}"] = row(a["postselected_failures"], a["accepted"], a["shots"],
                                          (ctrl["failures"], ctrl["accepted"]) if w >= 2 else None,
                                          m2[f"w={w} p2=0.005"]["A"]["postselected_error"])
    for w in (2, 3):
        a = pr3["run3"][str(w)]
        S["cut"][f"IQM run 3 w{w}"] = row(a["failures"], a["kept"], a["shots"],
                                          (a["control_failures"], a["control_kept"]),
                                          m1[f"w={w} p2=0.005"]["A"]["postselected_error"])
    hr, hc, hp = J("hw_results.json"), J("hw_counts.json"), J("prediction.json")["result"]
    for w in (0, 1, 2, 3):
        a = hr[f"w{w}"]
        ctrl = block_control_ibm(hc[str(w)], w) if w >= 2 else None
        S["cut"][f"IBM fez w{w}"] = row(a["postselected_failures"], a["accepted"], a["shots"], ctrl,
                                        hp[str(w)]["postselected_error"])
    # decoded (secondary) for IQM width 2 against 3
    r3 = J("oq_iqm3_results.json")
    S["cut_decoded"] = {"IQM run 1": {w: [r1[f"w{w}"]["decoded_failures"], r1[f"w{w}"]["shots"]] for w in (1, 2, 3)},
                        "IQM run 3": {w: [r3[f"w{w}"]["decoded_failures"], r3[f"w{w}"]["shots"]] for w in (2, 3)}}
    # ---------------- edge multiplicity, IQM Emerald
    mm = J("oq_iqm_m2_results.json")
    a, c = mm["all_checks"], mm["without_two_edge_check"]
    S["multiplicity"]["IQM w1 m2"] = row(a["postselected_failures"], a["accepted"], a["shots"],
                                         (c["postselected_failures"], c["accepted"]))
    # ---------------- repeated rounds
    rr = J("rounds_results.json")
    rplan = J("rounds_plan.json")["prediction"]
    for R in ("1", "2"):
        cps = rr[R]["copies"]
        ctrl = None
        if R == "2":
            cc = rr["R2_control"]["copies"]
            ctrl = (sum(x["postselected_failures"] for x in cc), sum(x["accepted"] for x in cc))
        S["rounds"][f"IQM fresh R{R}"] = row(sum(x["postselected_failures"] for x in cps),
                                             sum(x["accepted"] for x in cps), sum(x["shots"] for x in cps), ctrl,
                                             rplan[f"R={R} p2=0.005"]["error"])
    ra = J("ibm2_rounds_results.json")
    aplan = J("ibm2_rounds_plan.json")["rounds"]
    bitsA = ira.load()
    for R in ("1", "2", "3"):
        res = ra["per_R"][R]
        ctrl = None
        if "control_no_star_comparisons" in res:
            cc = res["control_no_star_comparisons"]
            ctrl = (sum(x["postselected_failures"] for x in cc), sum(x["accepted"] for x in cc))
        r = row(res["pooled"]["failures"], res["pooled"]["accepted"], sum(x["shots"] for x in res["copies"]), ctrl,
                float(np.mean([p["error"] for p in aplan[R]["predicted"]])))
        if int(R) >= 2:
            ev, tr, ind = i3.persistence(bitsA[int(R)], int(R))
            r["persistence"] = dict(events=ev, trials=tr, rate=ev / tr, independent=ind / tr)
        S["rounds"][f"IBM kingston A R{R}"] = r
    rb = J("ibm3_reset_results.json")
    bplan = J("ibm3_reset_plan.json")["circuits"]
    bitsB = i3.load()
    import ibm_rounds_hw as irh
    for label, res in rb["per_circuit"].items():
        Rb = bplan[label]["R"]
        ctrl = None
        if Rb >= 2:
            cc = [irh.analyze_bits(bitsB[label], Rb, k, include_star_comparisons=False) for k in (0, 1)]
            ctrl = (sum(x["postselected_failures"] for x in cc), sum(x["accepted"] for x in cc))
        r = row(res["failures"], res["accepted"], sum(x["shots"] for x in res["copies"]), ctrl,
                float(np.mean(bplan[label]["predicted"])))
        if "persistence" in res:
            p = res["persistence"]
            r["persistence"] = dict(events=p["events"], trials=p["trials"], rate=p["rate"],
                                    independent=p["independent_expectation"])
        S["rounds"][f"IBM kingston B {label}"] = r
    # ---------------- tests (one-sided Fisher exact: first group fails more often)
    cut = S["cut"]
    pool = lambda *ks: (sum(cut[k]["failures"] for k in ks), sum(cut[k]["kept"] for k in ks))
    T = S["tests"]

    def test(name, a, b, prereg):
        T[name] = dict(p=fisher_greater(a[0], a[1], b[0], b[1]), a=list(a), b=list(b), preregistered=prereg)
    test("IQM run 1: w2 below w1", pool("IQM run 1 w1"), pool("IQM run 1 w2"), False)
    test("IQM run 2: w2 below w1", pool("IQM run 2 w1"), pool("IQM run 2 w2"), False)
    test("IQM runs 1+3: w3 below w2", pool("IQM run 1 w2", "IQM run 3 w2"), pool("IQM run 1 w3", "IQM run 3 w3"), True)
    test("IBM fez: w2 below w1", pool("IBM fez w1"), pool("IBM fez w2"), False)
    test("IBM fez: w3 below w2", pool("IBM fez w2"), pool("IBM fez w3"), False)
    for run in ("IQM run 1", "IQM run 3"):
        d = S["cut_decoded"][run]
        test(f"{run}, decoded: w3 below w2", tuple(d[2]), tuple(d[3]), False)
    m = S["multiplicity"]["IQM w1 m2"]
    test("IQM: two parallel bridge qubits below one bridge qubit (runs 1+2)",
         pool("IQM run 1 w1", "IQM run 2 w1"), (m["failures"], m["kept"]), False)
    test("IQM: two separate bridge edges below two parallel bridge qubits (runs 1+2)",
         (m["failures"], m["kept"]), pool("IQM run 1 w2", "IQM run 2 w2"), False)
    rd = S["rounds"]
    g = lambda k: (rd[k]["failures"], rd[k]["kept"])
    test("IQM rounds: R2 below R1", g("IQM fresh R1"), g("IQM fresh R2"), True)
    test("IBM rounds A: R2 below R1", g("IBM kingston A R1"), g("IBM kingston A R2"), True)
    test("IBM rounds A: R3 below R2", g("IBM kingston A R2"), g("IBM kingston A R3"), True)
    for v in ("single", "double"):
        test(f"IBM rounds B ({v} reset): R2 below R1", g("IBM kingston B R1"), g(f"IBM kingston B R2_{v}"), True)
        test(f"IBM rounds B ({v} reset): R3 below R2", g(f"IBM kingston B R2_{v}"), g(f"IBM kingston B R3_{v}"), True)
    for R in (2, 3):
        test(f"IBM rounds B: R{R} error lower with two resets", g(f"IBM kingston B R{R}_single"),
             g(f"IBM kingston B R{R}_double"), True)
    ps = [rd[f"IBM kingston B R{R}_single"]["persistence"] for R in (2, 3)]
    pd_ = [rd[f"IBM kingston B R{R}_double"]["persistence"] for R in (2, 3)]
    test("IBM rounds B: persistence lower with two resets",
         (sum(x["events"] for x in ps), sum(x["trials"] for x in ps)),
         (sum(x["events"] for x in pd_), sum(x["trials"] for x in pd_)), True)
    # ---------------- jobs, QPU time, excluded runs
    S["jobs"] = {
        "IQM run 1": [j["job_id"] for j in J("oq_iqm_jobs.json")["jobs"].values()],
        "IQM run 2": [j["job_id"] for j in J("oq_iqm2_jobs.json")["jobs"].values()],
        "IQM run 3": [j["job_id"] for j in J("oq_iqm3_jobs.json")["jobs"].values()],
        "IQM multiplicity": [J("oq_iqm_m2_job.json")["job_id"]],
        "IQM rounds": [J("oq_rounds_R1_job.json")["job_id"], J("oq_rounds_R2_job.json")["job_id"]],
        "IBM fez widths": [hr["job_id"]],
        "IBM kingston rounds A": [J("ibm2_rounds_job.json")["job_id"]],
        "IBM kingston rounds B": [J("ibm3_reset_job.json")["job_id"]],
    }
    S["qpu_seconds"] = {"IBM fez widths": hr.get("usage_seconds"),
                        "IBM kingston rounds A": J("ibm2_rounds_job.json").get("usage_seconds"),
                        "IBM kingston rounds B": J("ibm3_reset_job.json").get("usage_seconds")}
    rg1, rg2 = J("oq_results.json"), J("oq_phys_results.json")
    S["excluded"]["Rigetti Cepheus-1-108Q, compiler placement"] = {
        w: [rg1[w]["postselected_failures"], rg1[w]["accepted"], rg1[w]["shots"]] for w in ("w1", "w2", "w3")}
    S["excluded"]["Rigetti Cepheus-1-108Q, fixed placement"] = {
        w: [sum(rg2[w][c]["postselected_failures"] for c in "AB"), sum(rg2[w][c]["accepted"] for c in "AB"),
            sum(rg2[w][c]["shots"] for c in "AB")] for w in ("w1", "w2", "w3")}
    # stuck qubit in the fixed-placement run: copy B ancilla x_a0 on physical qubit 76
    stuck = 0
    for w in (1, 2, 3):
        d = J(f"oq_phys_output_w{w}.json")
        i = 19 + hw.IDX["x_a0"]
        stuck += sum(c for s, c in d.items() if s[-1 - i] == "1")
    S["excluded"]["stuck_qubit_76_reads_1"] = [stuck, 30000]
    shots = sum(v["shots"] for v in S["cut"].values()) + sum(v["shots"] for v in S["multiplicity"].values()) \
        + sum(v["shots"] for v in S["rounds"].values())
    S["total_shots_analyzed"] = shots
    return S


# ------------------------------------------------------------- LaTeX output
def pct(x, sig=2):
    """Percentage with about `sig` significant figures, no trailing noise."""
    v = 100 * x
    if v == 0:
        return "0"
    d = max(0, sig - 1 - int(math.floor(math.log10(abs(v)))))
    return f"{v:.{d}f}"


def fold(a, b):
    """Ratio a/b: whole number from 10 up, one decimal below."""
    q = a / b
    return f"{q:.0f}" if q >= 10 else f"{q:.1f}"


def pval(p):
    """\\ensuremath, so a macro works both in text and inside $p=\\macro$."""
    if p == 0 or p < 1e-300:
        return r"\ensuremath{<10^{-300}}"
    if p >= 1e-3:
        return rf"\ensuremath{{{p:.2g}}}" if p < 0.1 else rf"\ensuremath{{{p:.2f}}}"
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    return rf"\ensuremath{{{m:.1f}\times10^{{{e}}}}}"


def ci(r):
    lo, hi = r["wilson95"]
    return f"[{pct(lo)}, {pct(hi)}]"


def err_cell(r):
    if r["failures"] == 0:
        return rf"$0$ ($\le{pct(r['wilson95'][1])}$)"
    return f"{pct(r['error'])} {ci(r)}"


def tables(S):
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\caption{Cut obstruction and edge multiplicity on hardware.  Two three-qubit blocks joined by a "
             r"bridge of $w$ edges ($w=0$: no gauging); $m_0=2$ puts two parallel ancillas on a single bridge "
             r"edge.  Errors are postselected on every detector, in percent, with Wilson 95\% intervals; a zero "
             r"count shows the upper end.  Control: the same shots postselected on the block checks only, "
             r"ignoring the cycle checks that the extra bridge edges create.  Model: uniform circuit noise "
             r"(IQM: CZ error $0.5\%$, readout $2\%/4\%$) or typical published medians (IBM), fixed before "
             r"the run.  Generated by \texttt{hardware/hardware\_summary.py}.}")
    L.append(r"\label{tab:hardware-cut}")
    L.append(r"\begin{ruledtabular}")
    L.append(r"\begin{tabular}{llrrrlll}")
    L.append(r"Device, run & Bridge & Shots & Kept & Wrong & Error (\%) & Control (\%) & Model (\%)\\")
    L.append(r"\hline")
    order = [("IQM Emerald, run 1", "IQM run 1", (1, 2, 3)), ("IQM Emerald, run 2", "IQM run 2", (1, 2)),
             ("IQM Emerald, run 3", "IQM run 3", (2, 3)), ("IBM \\texttt{ibm\\_fez}", "IBM fez", (0, 1, 2, 3))]
    for title_, key, ws in order:
        for i, w in enumerate(ws):
            r = S["cut"][f"{key} w{w}"]
            ctrl = pct(r["control"]["error"]) if "control" in r else ""
            L.append(f"{title_ if i == 0 else ''} & $w={w}$ & {r['shots']:,} & {r['kept']:,} & {r['failures']:,} & "
                     f"{err_cell(r)} & {ctrl} & {pct(r['model'])}\\\\".replace(",", "{,}"))
        L.append(r"\hline")
    m = S["multiplicity"]["IQM w1 m2"]
    L.append(f"IQM Emerald & $w=1$, $m_0=2$ & {m['shots']:,} & {m['kept']:,} & {m['failures']:,} & {err_cell(m)} & "
             f"{pct(m['control']['error'])} & \\\\".replace(",", "{,}"))
    L.append(r"\end{tabular}")
    L.append(r"\end{ruledtabular}")
    L.append(r"\end{table*}")
    L.append("")
    L.append(r"\begin{table*}[t]")
    L.append(r"\caption{Repeated rounds on hardware.  Phase-flip blocks ($d_Z=3$) joined by one gauging edge; the "
             r"two vertex stars and the four block checks are measured in $R$ rounds and the reported bit is the "
             r"round-one star product.  IQM: fresh ancillas in every round, all measured at the end.  IBM "
             r"(\texttt{ibm\_kingston}): ancillas measured and reset between rounds, once or twice; runs A and B "
             r"on different days.  Errors postselected on every detector, in percent, pooled over two copies.  "
             r"Control: the round-to-round star comparisons ignored.  Persistence: a check ancilla reads 1 in "
             r"rounds 1 and 2 while the ancillas sharing its data qubits show no change; in brackets, the rate "
             r"that independent errors would give.  Model: uniform noise (IQM, CZ $0.5\%$) or the device "
             r"calibration (IBM).  Samples: shots times the two copies.  Generated by "
             r"\texttt{hardware/hardware\_summary.py}.}")
    L.append(r"\label{tab:hardware-rounds}")
    L.append(r"\begin{ruledtabular}")
    L.append(r"\begin{tabular}{llrrrllll}")
    L.append(r"Device & Rounds & Samples & Kept & Wrong & Error (\%) & Control (\%) & Persistence (\%) & Model (\%)\\")
    L.append(r"\hline")
    rows = [("IQM Emerald", "IQM fresh R1", "$R=1$"), ("", "IQM fresh R2", "$R=2$"),
            ("IBM, run A", "IBM kingston A R1", "$R=1$"), ("", "IBM kingston A R2", "$R=2$, one reset"),
            ("", "IBM kingston A R3", "$R=3$, one reset"),
            ("IBM, run B", "IBM kingston B R1", "$R=1$"), ("", "IBM kingston B R2_single", "$R=2$, one reset"),
            ("", "IBM kingston B R2_double", "$R=2$, two resets"), ("", "IBM kingston B R3_single", "$R=3$, one reset"),
            ("", "IBM kingston B R3_double", "$R=3$, two resets")]
    for i, (dev, key, lab) in enumerate(rows):
        r = S["rounds"][key]
        ctrl = pct(r["control"]["error"]) if "control" in r else ""
        per = (f"{pct(r['persistence']['rate'], 3)} ({pct(r['persistence']['independent'])})"
               if "persistence" in r else "")
        if dev and i:
            L.append(r"\hline")
        L.append(f"{dev} & {lab} & {r['shots']:,} & {r['kept']:,} & {r['failures']:,} & {err_cell(r)} & {ctrl} & "
                 f"{per} & {pct(r['model'])}\\\\".replace(",", "{,}"))
    L.append(r"\end{tabular}")
    L.append(r"\end{ruledtabular}")
    L.append(r"\end{table*}")
    L.append("")
    L.append(r"\begin{table}[t]")
    L.append(r"\caption{Significance of the hardware comparisons: one-sided Fisher exact tests that the second "
             r"group fails less often than the first.  P: test and level $0.05$ fixed in the analysis script "
             r"before the data were taken.  Generated by \texttt{hardware/hardware\_summary.py}.}")
    L.append(r"\label{tab:hardware-tests}")
    L.append(r"\begin{ruledtabular}")
    L.append(r"\begin{tabular}{p{0.66\columnwidth}lc}")
    L.append(r"Comparison & $p$ & P\\")
    L.append(r"\hline")
    import re
    for name, t in S["tests"].items():
        label = re.sub(r"\b([wR])(\d)\b", r"$\1=\2$", name)
        L.append(f"{label} & {pval(t['p'])} & {'P' if t['preregistered'] else ''}\\\\")
    L.append(r"\end{tabular}")
    L.append(r"\end{ruledtabular}")
    L.append(r"\end{table}")
    (ROOT / "hardware_tables.tex").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def macros(S):
    c, r, t = S["cut"], S["rounds"], S["tests"]
    m = S["multiplicity"]["IQM w1 m2"]
    pool = lambda *ks: sum(c[k]["failures"] for k in ks) / sum(c[k]["kept"] for k in ks)
    pB = [r[f"IBM kingston B R{R}_{v}"]["persistence"] for R in (2, 3) for v in ("single", "double")]
    ps = sum(x["events"] for x in pB[0::2]) / sum(x["trials"] for x in pB[0::2])
    pd_ = sum(x["events"] for x in pB[1::2]) / sum(x["trials"] for x in pB[1::2])
    ind = float(np.mean([x["independent"] for x in pB]))
    M = {
        "hwIQMwOne": pct(pool("IQM run 1 w1", "IQM run 2 w1")),
        "hwIQMwTwo": pct(pool("IQM run 1 w2", "IQM run 2 w2", "IQM run 3 w2")),
        "hwIQMwThreeUpper": pct(c["IQM run 1 w3"]["wilson95"][1] if False else
                                wilson(0, c["IQM run 1 w3"]["kept"] + c["IQM run 3 w3"]["kept"])[1]),
        "hwIQMwThreeKept": f"{c['IQM run 1 w3']['kept'] + c['IQM run 3 w3']['kept']:,}".replace(",", "{,}"),
        "hwIQMctrl": pct(min(c[k]["control"]["error"] for k in c if k.startswith("IQM") and "control" in c[k])) + " to "
                     + pct(max(c[k]["control"]["error"] for k in c if k.startswith("IQM") and "control" in c[k])),
        "hwIQMpOneTwo": pval(max(t["IQM run 1: w2 below w1"]["p"], t["IQM run 2: w2 below w1"]["p"])),
        "hwIQMpTwoThree": pval(t["IQM runs 1+3: w3 below w2"]["p"]),
        "hwIBMwOne": pct(c["IBM fez w1"]["error"]), "hwIBMwTwo": pct(c["IBM fez w2"]["error"]),
        "hwIBMwThree": pct(c["IBM fez w3"]["error"]),
        "hwIBMpOneTwo": pval(t["IBM fez: w2 below w1"]["p"]), "hwIBMpTwoThree": pval(t["IBM fez: w3 below w2"]["p"]),
        "hwIBMctrl": pct(min(c["IBM fez w2"]["control"]["error"], c["IBM fez w3"]["control"]["error"])) + " to "
                     + pct(max(c["IBM fez w2"]["control"]["error"], c["IBM fez w3"]["control"]["error"])),
        "hwMult": pct(m["error"]), "hwMultCtrl": pct(m["control"]["error"]),
        "hwMultP": pval(t["IQM: two parallel bridge qubits below one bridge qubit (runs 1+2)"]["p"]),
        "hwMultVsTwoP": pval(t["IQM: two separate bridge edges below two parallel bridge qubits (runs 1+2)"]["p"]),
        "hwRndIQMOne": pct(r["IQM fresh R1"]["error"]), "hwRndIQMTwo": pct(r["IQM fresh R2"]["error"]),
        "hwRndIQMModelOne": pct(r["IQM fresh R1"]["model"]), "hwRndIQMModelTwo": pct(r["IQM fresh R2"]["model"]),
        "hwRndIQMp": pval(t["IQM rounds: R2 below R1"]["p"]), "hwRndIQMctrl": pct(r["IQM fresh R2"]["control"]["error"]),
        "hwRndIBMOne": pct(r["IBM kingston A R1"]["error"]), "hwRndIBMTwo": pct(r["IBM kingston A R2"]["error"]),
        "hwRndIBMThree": pct(r["IBM kingston A R3"]["error"]),
        "hwRndIBMModelTwo": pct(r["IBM kingston A R2"]["model"]), "hwRndIBMModelThree": pct(r["IBM kingston A R3"]["model"]),
        "hwPersSingle": pct(ps, 3), "hwPersDouble": pct(pd_, 3), "hwPersIndep": pct(ind),
        "hwPersRatio": f"{round(ps / ind)}",
        "hwPersRel": f"{100 * (ps - pd_) / ps:.0f}",
        "hwPersP": pval(t["IBM rounds B: persistence lower with two resets"]["p"]),
        "hwResetTwoS": pct(r["IBM kingston B R2_single"]["error"]), "hwResetTwoD": pct(r["IBM kingston B R2_double"]["error"]),
        "hwResetThreeS": pct(r["IBM kingston B R3_single"]["error"]), "hwResetThreeD": pct(r["IBM kingston B R3_double"]["error"]),
        "hwResetTwoP": pval(t["IBM rounds B: R2 error lower with two resets"]["p"]),
        "hwResetThreeP": pval(t["IBM rounds B: R3 error lower with two resets"]["p"]),
        "hwRndIBMpOneTwo": pval(t["IBM rounds A: R2 below R1"]["p"]),
        "hwRndIBMpTwoThree": pval(t["IBM rounds A: R3 below R2"]["p"]),
        "hwRndIBMmaxP": pval(max(v["p"] for k, v in t.items()
                                 if k.startswith("IBM rounds A:") or k.startswith("IBM rounds B ("))),
        "hwRndIBMctrl": pct(min(r[f"IBM kingston A R{R}"]["control"]["error"] for R in (2, 3))) + " to "
                        + pct(max(r[f"IBM kingston A R{R}"]["control"]["error"] for R in (2, 3))),
        "hwFezCZ": ", ".join(str(J("plan.json")["circuits"][str(w)]["cz"]) for w in (1, 2)) + " and "
                   + str(J("plan.json")["circuits"]["3"]["cz"]),
        "hwQpuFez": str(S["qpu_seconds"]["IBM fez widths"]),
        "hwQpuA": str(S["qpu_seconds"]["IBM kingston rounds A"]),
        "hwQpuB": str(S["qpu_seconds"]["IBM kingston rounds B"]),
        "hwDecTwo": pct(S["cut_decoded"]["IQM run 3"][2][0] / S["cut_decoded"]["IQM run 3"][2][1]),
        "hwDecThree": pct(S["cut_decoded"]["IQM run 3"][3][0] / S["cut_decoded"]["IQM run 3"][3][1]),
        "hwDecP": pval(max(t["IQM run 1, decoded: w3 below w2"]["p"], t["IQM run 3, decoded: w3 below w2"]["p"])),
        "hwTotalShots": f"{S['total_shots_analyzed'] / 1e6:.1f}",
        "hwStuck": f"{S['excluded']['stuck_qubit_76_reads_1'][0]:,}".replace(",", "{,}"),
        # fold changes, formatted as in make_hardware_figure.py
        "hwIQMfold": fold(pool("IQM run 1 w1", "IQM run 2 w1"),
                          pool("IQM run 1 w2", "IQM run 2 w2", "IQM run 3 w2")),
        "hwIBMfoldOneTwo": fold(c["IBM fez w1"]["error"], c["IBM fez w2"]["error"]),
        "hwIBMfoldTwoThree": fold(c["IBM fez w2"]["error"], c["IBM fez w3"]["error"]),
        "hwMultFold": fold(pool("IQM run 1 w1", "IQM run 2 w1"), m["error"]),
        "hwRndIQMfold": fold(r["IQM fresh R1"]["error"], r["IQM fresh R2"]["error"]),
        "hwRndIBMModelRatio": fold(r["IBM kingston A R3"]["error"], r["IBM kingston A R3"]["model"]),
    }
    lines = ["% Generated by hardware/hardware_summary.py; do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in M.items()]
    (ROOT / "hardware_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return M


def main():
    S = collect()
    (HERE / "hardware_summary.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8", newline="\n")
    tables(S)
    M = macros(S)
    for k, v in M.items():
        print(f"{k:22s} {v}")
    for name, t in S["tests"].items():
        print(f"{'P' if t['preregistered'] else ' '} {name}: p = {t['p']:.3g}")


if __name__ == "__main__":
    main()
