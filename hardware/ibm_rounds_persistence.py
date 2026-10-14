"""Persistence of ancilla errors across rounds in the IBM repeated-rounds data.
A check ancilla that reads 1 in rounds 1 and 2 while the ancillas sharing its data
qubits show no change points to an ancilla error that survives the reset
(reset failure or leakage), not to a data error."""
import sys, numpy as np
sys.path.insert(0, r"C:\kw\r\hardware")
import ibm_rounds_hw as rh, ibm_rounds_analyze as ra
bits = ra.load()
c = rh.clbit
for R in (2, 3):
    b = bits[R]
    for k in (0, 1):
        g = lambda a, r: b[:, c(k, r, a, R)].astype(bool)
        # ca01 shares a0 with sa and a1 with ca12; ca12's a2 is not shared; cb01 shares b0 with sb, b1 with cb12
        cases = {"ca01": [("sa", "star"), ("ca12", "check")], "cb01": [("sb", "star"), ("cb12", "check")]}
        out = []
        for a, nbs in cases.items():
            r1, r2 = g(a, 1), g(a, 2)
            clean = np.ones(b.shape[0], bool)
            for nb, kind in nbs:
                clean &= (g(nb, 1) == g(nb, 2)) if kind == "star" else (~g(nb, 1) & ~g(nb, 2))
            both = (r1 & r2 & clean).mean()
            indep = (r1 & clean).mean() * (r2 & clean).mean() / clean.mean()
            out.append(f"{a}: read 1 in rounds 1 and 2 with clean neighbours {100*both:.3f}% (independent errors would give {100*indep:.3f}%)")
        # stars: change patterns over rounds
        sa = [g("sa", r) for r in range(1, R + 1)]
        ch12 = (sa[0] != sa[1]).mean()
        line = f"R={R} copy {'AB'[k]}: " + " | ".join(out) + f" | sa changes r1->r2 {100*ch12:.2f}%"
        if R == 3:
            iso = ((sa[0] != sa[1]) & (sa[1] != sa[2])).mean()   # one-round blip (isolated readout error in round 2)
            step = ((sa[0] != sa[1]) & (sa[1] == sa[2])).mean()  # lasting change from round 2
            line += f", blip at r2 {100*iso:.2f}%, lasting change {100*step:.2f}%"
        print(line)
