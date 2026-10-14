"""Width 2 against width 3 on IQM Emerald, run 1 plus run 3 (same qubits, same files).

Analysis plan, fixed before run 3 was submitted (2026-10-13):
  Primary test: postselected failures, pooled over run 1 and run 3, one-sided
  Fisher exact test that width 3 fails less often than width 2, at level 0.05.
  Pooling check: for each width, run 1 and run 3 are compared with a two-sided
  Fisher test; a p value below 0.05 is reported as drift between the runs.
  Also reported: each run alone, Wilson intervals, and the control without the
  bridge cycle checks (which should return to the width-one level).
Run 3 uses byte-identical circuit files (the platform's input hashes for run 1
match the local files) on the same physical qubits.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw
import analyze_iqm as ai


def fisher_greater(k1, n1, k2, n2):
    """P(X >= k1) for X ~ hypergeometric: one-sided test that group 1 fails more often."""
    K, N = k1 + k2, n1 + n2
    return sum(math.comb(n1, k) * math.comb(n2, K - k) for k in range(k1, min(K, n1) + 1)) / math.comb(N, K)


def fisher_two_sided(k1, n1, k2, n2):
    K, N = k1 + k2, n1 + n2
    pmf = lambda k: math.comb(n1, k) * math.comb(n2, K - k) / math.comb(N, K)
    p0 = pmf(k1)
    return min(1.0, sum(pmf(k) for k in range(max(0, K - n2), min(K, n1) + 1) if pmf(k) <= p0 * (1 + 1e-9)))


def control(raw, w):
    """Postselect on the four block checks only (no bridge cycle checks)."""
    pos = ai.candidates(len(next(iter(raw))))["clbits right to left (Qiskit)"]
    acc = fail = 0
    for s, c in raw.items():
        s = s.replace(" ", "")
        if any(sum(int(s[pos[q]]) for q in d) % 2 for d in hw.detectors(w)[:4]):
            continue
        acc += c
        fail += c * (sum(int(s[pos[q]]) for q in hw.observable(w)) % 2)
    return fail, acc


def run(files):
    out = {}
    for w, f in files.items():
        raw = json.loads(Path(f).read_text())
        a = ai.analyze(w, raw)
        cf, ca = control(raw, w)
        out[w] = dict(failures=a["postselected_failures"], kept=a["accepted"], shots=a["shots"],
                      bit_order=a["bit_order"], control_failures=cf, control_kept=ca)
    return out


def report(r1, r3):
    res = {"run1": r1, "run3": r3}
    for w in (2, 3):
        a, b = r1[w], r3[w]
        p = fisher_two_sided(a["failures"], a["kept"], b["failures"], b["kept"])
        res[f"pooling_check_w{w}"] = p
        print(f"width {w}: run 1 {a['failures']}/{a['kept']}, run 3 {b['failures']}/{b['kept']} "
              f"(bit order {b['bit_order']}); runs agree p = {p:.3f}{'  DRIFT' if p < 0.05 else ''}")
        print(f"         control without cycle checks, run 3: {b['control_failures']}/{b['control_kept']} "
              f"= {100 * b['control_failures'] / b['control_kept']:.2f}%")
    k2 = r1[2]["failures"] + r3[2]["failures"]
    n2 = r1[2]["kept"] + r3[2]["kept"]
    k3 = r1[3]["failures"] + r3[3]["failures"]
    n3 = r1[3]["kept"] + r3[3]["kept"]
    p = fisher_greater(k2, n2, k3, n3)
    lo2, hi2 = hw.wilson(k2, n2)
    lo3, hi3 = hw.wilson(k3, n3)
    res["pooled"] = dict(w2=[k2, n2, [lo2, hi2]], w3=[k3, n3, [lo3, hi3]], p_one_sided=p)
    print(f"pooled width 2: {k2}/{n2} = {100 * k2 / n2:.3f}% [{100 * lo2:.3f}%, {100 * hi2:.3f}%]")
    print(f"pooled width 3: {k3}/{n3} = {100 * k3 / n3:.3f}% [{100 * lo3:.3f}%, {100 * hi3:.3f}%]")
    print(f"primary test, width 3 below width 2: one-sided Fisher p = {p:.4f} -> "
          f"{'CLEAR (p < 0.05)' if p < 0.05 else 'not clear at 0.05'}")
    return res


def main():
    r1 = run({2: HERE / "oq_iqm_output_w2.json", 3: HERE / "oq_iqm_output_w3.json"})
    missing = [w for w in (2, 3) if not (HERE / f"oq_iqm3_output_w{w}.json").exists()]
    if missing:
        print("run 3 output missing for widths", missing)
        return
    r3 = run({2: HERE / "oq_iqm3_output_w2.json", 3: HERE / "oq_iqm3_output_w3.json"})
    res = report(r1, r3)
    (HERE / "pooled_w23_results.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
