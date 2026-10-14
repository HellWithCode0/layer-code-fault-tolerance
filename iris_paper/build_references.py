"""Write references_iris.tex: the entries cited in iris_paper.tex, in citation order.

Entries come from the paper's reference database (references_db.tex of the source
package) plus the extra entries below, which only this paper cites.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "references_db.tex"

EXTRA = {
    "Repo": r"Project repository: code, data, hardware records and technical report for this paper, \url{https://github.com/HellWithCode0/layer-code-fault-tolerance} (2026).",
    "Stim": r"C. Gidney, Stim: a fast stabilizer circuit simulator, Quantum \textbf{5}, 497 (2021), \href{https://arxiv.org/abs/2103.02202}{arXiv:2103.02202}.",
    "CPSAT": r"L. Perron and F. Didier, CP-SAT solver, Google OR-Tools, \url{https://developers.google.com/optimization} (2024).",
    "LSD": r"T. Hillmann, L. Berent, A. O. Quintavalle, J. Eisert, R. Wille, and J. Roffe, Localized statistics decoding: A parallel decoding algorithm for quantum low-density parity-check codes, \href{https://arxiv.org/abs/2406.18655}{arXiv:2406.18655} (2024).",
    "PyMatching": r"O. Higgott and C. Gidney, Sparse Blossom: correcting a million errors per core second with minimum-weight matching, Quantum \textbf{9}, 1600 (2025), \href{https://arxiv.org/abs/2303.15933}{arXiv:2303.15933}.",
    "Wilson": r"E. B. Wilson, Probable inference, the law of succession, and statistical inference, J. Am. Stat. Assoc. \textbf{22}, 209 (1927).",
    "Fisher": r"R. A. Fisher, On the interpretation of $\chi^2$ from contingency tables, and the calculation of P, J. R. Stat. Soc. \textbf{85}, 87 (1922).",
    "Qiskit": r"A. Javadi-Abhari \emph{et al.}, Quantum computing with Qiskit, \href{https://arxiv.org/abs/2405.08810}{arXiv:2405.08810} (2024).",
}


def main():
    tex = (HERE / "iris_paper.tex").read_text(encoding="utf8")
    order = []
    for group in re.findall(r"\\cite\{([^}]*)\}", tex):
        for key in (k.strip() for k in group.split(",")):
            if key and key not in order:
                order.append(key)
    db = DB.read_text(encoding="utf8")
    entries = {}
    for m in re.finditer(r"\\bibitem\{([^}]*)\}\s*(.*?)(?=\n\s*\\bibitem\{|\n\s*\\end\{thebibliography\}|\s*\Z)", db, re.S):
        entries[m.group(1)] = m.group(2).strip()
    entries.update(EXTRA)
    missing = [k for k in order if k not in entries]
    if missing:
        raise SystemExit(f"missing reference entries: {missing}")
    out = ["\\begin{thebibliography}{99}"]
    for k in order:
        out.append(f"\\bibitem{{{k}}}\n{entries[k]}\n")
    out.append("\\end{thebibliography}\n")
    (HERE / "references_iris.tex").write_text("\n".join(out), encoding="utf8", newline="\n")
    print(f"{len(order)} references written")


if __name__ == "__main__":
    main()
