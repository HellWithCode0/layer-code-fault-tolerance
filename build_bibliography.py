#!/usr/bin/env python3
"""Write references_main.tex and references_supp.tex from references_db.tex.

Each output contains only the entries cited in the corresponding document,
numbered in order of first citation, as required for APS numeric style.
Run before pdflatex whenever citations change.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'references_db.tex'


def load_db():
    txt = DB.read_text(encoding='utf-8')
    parts = re.split(r'(?=\\bibitem\{)', txt)[1:]
    return {re.match(r'\\bibitem\{([^}]*)\}', p).group(1): p.strip() for p in parts}


def cite_order(tex_files):
    keys = []
    for f in tex_files:
        src = re.sub(r'(?<!\\)%.*', '', (ROOT / f).read_text(encoding='utf-8'))
        for group in re.findall(r'\\cite[a-zA-Z]*\{([^}]*)\}', src):
            for k in (x.strip() for x in group.split(',')):
                if k and k not in keys:
                    keys.append(k)
    return keys


def write(out, keys, db):
    missing = [k for k in keys if k not in db]
    if missing:
        raise SystemExit(f'{out}: missing bibliography entries {missing}')
    body = '\n\n'.join(db[k] for k in keys)
    (ROOT / out).write_text('\\begin{thebibliography}{99}\n' + body + '\n\\end{thebibliography}\n',
                            encoding='utf-8')
    return len(keys)


def main():
    db = load_db()
    n_main = write('references_main.tex', cite_order(['main_body.tex']), db)
    n_supp = write('references_supp.tex', cite_order(['supplement.tex']), db)
    unused = set(db) - set(cite_order(['main_body.tex'])) - set(cite_order(['supplement.tex']))
    print(f'references_main.tex: {n_main} entries; references_supp.tex: {n_supp} entries; '
          f'database entries cited nowhere: {sorted(unused) or "none"}')


if __name__ == '__main__':
    main()
