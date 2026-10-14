"""Why is the IBM repeated-rounds error larger than predicted?  Read-only on the saved bits.

For postselected shots, splits failures by the round-one star readings (sa, sb).
Ideal: each star is 0 or 1 with probability 1/2, independent of the other.
A star ancilla that reads 1 in every round regardless of the true value
(for example leakage that reset cannot undo) would show up as failures
concentrated where that star reads 1, and as P(star = 1) above 1/2.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ibm_rounds_hw as rh
import ibm_rounds_analyze as ra

bits = ra.load()
for R in sorted(bits):
    b = bits[R]
    for k in (0, 1):
        fired = np.zeros(b.shape[0], dtype=bool)
        for d in rh.detectors(R, k):
            fired |= (b[:, d].sum(axis=1) % 2).astype(bool)
        acc = ~fired
        sa = b[:, rh.clbit(k, 1, "sa", R)]
        sb = b[:, rh.clbit(k, 1, "sb", R)]
        fail = (sa ^ sb).astype(bool)
        line = f"R={R} copy {'AB'[k]}: P(sa=1) {sa.mean():.3f} P(sb=1) {sb.mean():.3f} (all shots); kept shots by (sa,sb): "
        for va in (0, 1):
            for vb in (0, 1):
                m = acc & (sa == va) & (sb == vb)
                line += f"{va}{vb}:{int(m.sum())} "
        print(line)
        # per-ancilla readout of 1 in each round (all shots)
        rows = []
        for a in rh.ANCS:
            rows.append(f"{a} " + "/".join(f"{b[:, rh.clbit(k, r, a, R)].mean():.3f}" for r in range(1, R + 1)))
        print("      P(read 1) per round: " + "; ".join(rows))
