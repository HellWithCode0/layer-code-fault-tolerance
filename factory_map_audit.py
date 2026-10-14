#!/usr/bin/env python3
"""Audit the ideal transversal phase conventions used by the 7->1 and 15->1 factories.

This is a finite algebraic check of the codeword phases and the single-Z syndrome
columns used by the manuscript's always-output recovery argument.  It is not a
noise simulation.
"""
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "factory_map_audit.csv"


def nonzero_columns(m: int) -> list[tuple[int, ...]]:
    return [tuple((i >> b) & 1 for b in range(m)) for i in range(1, 2**m)]


def row_words(cols: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    m = len(cols[0])
    rows = [tuple(c[r] for c in cols) for r in range(m)]
    out = []
    for coeff in product([0, 1], repeat=m):
        word = tuple(sum(a * rows[r][j] for r, a in enumerate(coeff)) % 2
                     for j in range(len(cols)))
        out.append(word)
    return out


def wt(word: tuple[int, ...]) -> int:
    return sum(word)


def xor1(word: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 ^ x for x in word)


def syndrome_columns_unique(cols: list[tuple[int, ...]]) -> bool:
    return all(any(c) for c in cols) and len(set(cols)) == len(cols)


def audit(name: str, m: int, phase_mod: int, phase_sign: int,
          target_exp: int, physical_gate: str, ideal_output: str) -> dict[str, str]:
    cols = nonzero_columns(m)
    n = len(cols)
    c0 = row_words(cols)
    c1 = [xor1(x) for x in c0]
    w0 = sorted(set(map(wt, c0)))
    w1 = sorted(set(map(wt, c1)))

    # phase exponent convention: exp(i*pi*e/(phase_mod/2)) for phase_mod=4 (S)
    # and exp(i*pi*e/4) for phase_mod=8 (T).  We only compare exponents modulo phase_mod.
    e0 = {phase_sign * w % phase_mod for w in w0}
    e1 = {phase_sign * w % phase_mod for w in w1}
    assert e0 == {0}, (name, w0, e0)
    assert e1 == {target_exp % phase_mod}, (name, w1, e1)
    assert syndrome_columns_unique(cols)

    return {
        "factory": name,
        "n": str(n),
        "physical_transversal_gate": physical_gate,
        "logical0_weight_classes": ";".join(map(str, w0)),
        "logical1_weight_classes": ";".join(map(str, w1)),
        "logical0_phase_exponent_mod": "0",
        "logical1_phase_exponent_mod": str(target_exp % phase_mod),
        "phase_modulus": str(phase_mod),
        "single_Z_syndromes_unique": "True",
        "ideal_output": ideal_output,
    }


def main() -> None:
    rows = [
        # Steane: (S^dagger)^tensor7 gives logical S on |+_L>.
        audit("Steane_7_to_1_Y", 3, 4, -1, 1, "Sdagger", "|Y>"),
        # Punctured RM: (T^dagger)^tensor15 gives logical T on |+_L>.
        audit("RM_15_to_1_A", 4, 8, -1, 1, "Tdagger", "|A>"),
    ]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("PASS ideal factory-map audit")
    print("  Steane: (Sdagger)^7 maps encoded |+> to encoded |Y>; all single-Z syndromes unique")
    print("  RM15:   (Tdagger)^15 maps encoded |+> to encoded |A>; all single-Z syndromes unique")
    print(f"  wrote {OUT.name}")


if __name__ == "__main__":
    main()
