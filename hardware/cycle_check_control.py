"""Free control on the IQM Emerald data: drop the bridge cycle checks.

With w >= 2 the extra bridge edges create cycle checks (the bridge squares).
The cut obstruction says these are what catch a single fault on a bridge edge.
If the cycle checks are ignored, a single bridge fault is again invisible, so
the postselected error of w = 2 and w = 3 should rise back to the w = 1 level.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cut_obstruction_hw as hw
import analyze_iqm as ai

out = {}
for w in (1, 2, 3):
    f = HERE / f"oq_iqm{ai.RUN}_output_w{w}.json"
    if not f.exists():
        continue
    raw = json.loads(f.read_text())
    pos = ai.candidates(len(next(iter(raw))))["clbits right to left (Qiskit)"]
    dets_all = hw.detectors(w)
    for label, dets in (("all checks", dets_all), ("block checks only", dets_all[:4])):
        acc = fail = 0
        for s, c in raw.items():
            if any(sum(int(s[pos[q]]) for q in d) % 2 for d in dets):
                continue
            acc += c
            fail += c * (sum(int(s[pos[q]]) for q in hw.observable(w)) % 2)
        lo, hi = hw.wilson(fail, acc)
        out[f"w{w} {label}"] = dict(accepted=acc, failures=fail, error=fail / acc, wilson95=[lo, hi])
        print(f"w={w} {label:18s}: {fail:3d} wrong of {acc:4d} kept = {100 * fail / acc:.2f}% "
              f"[{100 * lo:.2f}%, {100 * hi:.2f}%]")
(HERE / f"cycle_check_control{ai.RUN}.json").write_text(json.dumps(out, indent=1))
