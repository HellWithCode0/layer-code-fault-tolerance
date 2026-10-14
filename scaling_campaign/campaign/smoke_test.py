#!/usr/bin/env python3
"""Five-minute check that the package works on this machine.  Run it first.

    python campaign/smoke_test.py

1. The k = 2, s = 2 circuits are identical to the paper's two_block_measurement.py
   circuits (frozen SHA-256 of the circuit text, 8 circuits).
2. Every block family and interface of the campaign gives a valid deformed code
   (commuting checks, 2k - 1 logical qubits) and a noiseless circuit whose
   detectors and observables are all deterministic.
3. The decoder makes no mistake without noise and a plausible number with noise.
4. The runner and the analysis work end to end (profile "smoke", results_smoke/).
Prints PASS or FAIL for each step.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

FROZEN = {
    ("thick", 0, 1, "X"): "b8a9a209b0ab435e7d0bc0160a8a90331286f8af47615e2370e78f553b3b597e",
    ("thick", 0, 1, "Z"): "6b533b373ed53411e174d0dc590888a8b394aedbd2bcf1183573fb44a3ce377a",
    ("bridge", 1, 1, "X"): "eef39cec69aa95091d480d382d09925fc1e1bc90e16c813f35130fab4aadf4d7",
    ("bridge", 1, 1, "Z"): "9fd3a61701a43f7e15ddf6949ca2b542a00fd712f208de5a02fdc224b2d88caf",
    ("thick", 0, 2, "X"): "5810f71b72bf87c864bf837e22135db6061d80f4790fce5d3ed2856066452037",
    ("thick", 0, 2, "Z"): "3f0c99cbcfa6a0bd225fe369ebd57f5dcecd04decf86cfb48beb4011582f4023",
    ("bridge", 3, 1, "X"): "7974a8501a7773eaef047e553535cde7fb282310d260c6d102fd6044435b0308",
    ("bridge", 3, 1, "Z"): "27af9f64e86a8d2c6595c3b09bfc2b2e40eca3b095f9e27842cfe19c73e90fe1",
}


def main():
    import numpy as np
    import two_block_family as F
    from decoders import BpLsd
    ok_all = True

    def report(name, ok, extra=""):
        nonlocal ok_all
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}", flush=True)

    # 1. regression against the paper's circuits
    same = 0
    for (iface, w, m, basis), h in FROZEN.items():
        tb = F.TwoBlock(k=2, s=2, interface=iface, width=max(1, w), mult=m)
        c, _ = F.build_circuit(tb, 0.002, basis, 4, 4, 4)
        same += hashlib.sha256(str(c).encode()).hexdigest() == h
    report("1. identical to the paper's circuits", same == len(FROZEN), f"({same}/{len(FROZEN)})")

    # 2. validity of every geometry used by the profiles
    good = tot = 0
    for k, s in ((2, 2), (2, 3), (4, 2)):
        for iface, w, m in (("thick", 1, 1), ("bridge", 1, 1), ("bridge", 3, 1), ("thick", 1, 2), ("idle", 1, 1)):
            tb = F.TwoBlock(k=k, s=s, interface=iface, width=w, mult=m)
            code_ok = True if iface == "idle" else tb.check_deformed_code()[0]
            R = 2 * s + 1
            for basis in "XZ":
                c, _ = F.build_circuit(tb, 0.002, basis, R, R, R, idle=(iface == "idle"))
                tot += 1
                good += code_ok and F.noiseless_check(c, 16)
    report("2. deformed codes valid and noiseless circuits deterministic", good == tot, f"({good}/{tot})")

    # 3. decoder: zero mistakes without noise, plausible rate with noise
    tb = F.TwoBlock(k=2, s=2, interface="thick")
    c, names = F.build_circuit(tb, 0.002, "X", 5, 5, 5)
    comp = BpLsd({0, 2}).compile_decoder_for_dem(
        dem=c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True))
    d0, o0 = c.without_noise().compile_detector_sampler(seed=1).sample(50, separate_observables=True, bit_packed=True)
    zero_ok = not np.any(comp.decode_shots_bit_packed(bit_packed_detection_event_data=d0) ^ o0)
    t0 = time.time()
    d, o = c.compile_detector_sampler(seed=2).sample(300, separate_observables=True, bit_packed=True)
    pred = comp.decode_shots_bit_packed(bit_packed_detection_event_data=d)
    rate = float(np.any(np.unpackbits(pred ^ o, axis=1, bitorder="little")[:, :len(names)], axis=1).mean())
    sps = (time.time() - t0) / 300
    report("3. decoder sane", zero_ok and 0.0 < rate < 0.3,
           f"(noiseless mistakes: {'none' if zero_ok else 'SOME'}; p=0.002 failure {rate:.3f}; "
           f"{sps * 1000:.0f} ms per shot)")

    # 4. runner and analysis end to end
    res = ROOT / "results_smoke"
    shutil.rmtree(res, ignore_errors=True)
    workers = str(min(4, os.cpu_count() or 1))
    r1 = subprocess.run([sys.executable, str(HERE / "run_mc.py"), "--profile", "smoke", "--workers", workers,
                         "--out", str(res / "mc_batches.csv")], capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--res", str(res)],
                        capture_output=True, text=True)
    import csv
    from profiles import smoke, task_id
    want = {task_id(t): t["max_shots"] for t in smoke()["tasks"]}
    got = {}
    rows = 0
    if (res / "mc_batches.csv").exists():
        with open(res / "mc_batches.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                rows += 1
                got[r["task_id"]] = got.get(r["task_id"], 0) + int(r["shots"])
    complete = all(got.get(tid, 0) >= n for tid, n in want.items())
    figs = sorted(p.name for p in res.glob("fig_*.png"))
    ok4 = (r1.returncode == 0 and r2.returncode == 0 and complete and len(figs) == 3
           and (res / "SUMMARY.md").exists())
    report("4. runner and analysis", ok4, f"({rows} batches, figures: {', '.join(figs) or 'none'})")
    if not ok4:
        print(r1.stdout[-3000:], r1.stderr[-3000:], r2.stdout[-2000:], r2.stderr[-2000:])

    # speed estimate for planning
    print(f"\nSpeed on this machine: about {sps * 1000:.0f} ms per shot at the smallest size.")
    print("Rule of thumb (per shot, one core, p = 0.002): s=3 ~5x, s=4 ~15x, s=5 ~25x that.")
    print("\nSMOKE TEST:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
