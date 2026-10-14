#!/usr/bin/env python3
"""Re-runnable bullet/plus geometric transcription + chart/bulk audit (same PASS as junctions)."""
from __future__ import annotations
import csv
from pathlib import Path
import reproducibility_checks as rc
import wb_microscopic_bridge_audit as aud

ROOT = Path(__file__).resolve().parent

def main(radius: int = 2) -> int:
    E, jdata = rc.wb_junction_support_data()
    named_key = {n: aud._edge_key(*ends) for n, ends in E.items()}

    cat_rows = []
    for r in jdata:
        if r["cell"] == "middle_z":
            cat_rows.append({**r, "cell": "plus",
                "role": "plus (⊕)=middle z-junction lattice of Fig.22"})
        if r["cell"] == "first_z":
            cat_rows.append({**r, "cell": "bullet",
                "role": "bullet (•) finite SX=first z-junction (Fig.18 / Fig.23)"})

    with (ROOT / "wb_bullet_plus_support_catalog.csv").open("w", newline="") as fh:
        fields = ["cell","source_item","source_figure","check","pauli",
                  "support_edges","support_endpoints","weight","role","scope"]
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in cat_rows:
            ends = ";".join(f"{e}:{E[e][0]}->{E[e][1]}" for e in r["support"])
            w.writerow(dict(
                cell=r["cell"], source_item=r["source_item"], source_figure=r["source_figure"],
                check=r["check"], pauli=r["pauli"],
                support_edges=" ".join(r["support"]), support_endpoints=ends,
                weight=len(r["support"]), role=r["role"],
                scope="WB transcription; see wb_bullet_plus_transcription_provenance.md",
            ))

    by = {}
    for r in cat_rows:
        by.setdefault(r["cell"], []).append(r)

    seeds = {
        "plus": {(0,0,0):set(), (1,1,0):{"ym","yp"}, (1,0,1):{"ym","xm"}, (0,1,1):{"yp","xm"}},
        "bullet": {(0,0,0):set(), (1,1,1):{"zm","zp","xp"}},
    }
    stated = {"plus":{"ym","yp","xm"}, "bullet":{"zm","zp","xp"}}
    kernels = {"plus":{(0,0,0),(1,1,0),(1,0,1),(0,1,1)}, "bullet":{(0,0,0),(1,1,1)}}

    chart_rows=[]; chart_pauli={}
    for cell, K in kernels.items():
        checks=by[cell]; pool=sorted({e for ch in checks for e in ch["support"]}); n=len(pool)
        for w in sorted(K):
            word="".join(map(str,w)); seed=seeds[cell][w]
            if all(b==0 for b in w):
                chart_rows.append(dict(cell=cell,catalog_cell=cell,kernel_word=word,pauli_type="X",
                    pauli_edges="(identity)",pauli_weight=0,seed_edges="",novel_syndrome="",
                    novel_syndrome_weight=0,confined_to_cell_novel_checks="yes",
                    fully_novel_neutral="yes",status="PASS",note="identity realizes vacuum kernel word"))
                chart_pauli[(cell,word)]=("X",set()); continue
            found=None
            for mask in range(1<<n):
                edges={pool[i] for i in range(n) if mask&(1<<i)}
                if not seed<=edges: continue
                syn=aud._syndrome_X_pauli(edges, checks)
                if sum(syn.values())==0:
                    cand=(len(edges), sorted(edges))
                    if found is None or cand < (found[0], sorted(found[1])): found=(len(edges), set(edges))
            assert found is not None, (cell, word)
            wt, edges = found
            chart_rows.append(dict(cell=cell,catalog_cell=cell,kernel_word=word,pauli_type="X",
                pauli_edges=" ".join(sorted(edges)),pauli_weight=wt,
                seed_edges=" ".join(sorted(seed)),novel_syndrome="",novel_syndrome_weight=0,
                confined_to_cell_novel_checks="yes",fully_novel_neutral="yes",status="PASS",
                note="WB-transcribed legs + completing factor; novel-check syndrome empty"))
            chart_pauli[(cell,word)]=("X", edges)

    with (ROOT/"wb_bullet_plus_chart_syndrome_audit.csv").open("w",newline="") as fh:
        w=csv.DictWriter(fh, fieldnames=list(chart_rows[0].keys())); w.writeheader(); w.writerows(chart_rows)

    novel_z={frozenset(named_key[e] for e in ch["support"]) for ch in cat_rows if ch["pauli"]=="Z"}
    stars, plaquettes = aud._ordinary_sc_checks(radius, novel_z)
    bulk_rows=[]; n_pass=n_fail=n_nz=0
    for cell,K in kernels.items():
        checks=by[cell]
        catalog={e for ch in checks for e in ch["support"]}
        cat_keys=frozenset(named_key[e] for e in catalog)
        occ_keys=frozenset(named_key[e] for e in stated[cell])
        for w in sorted(K):
            word="".join(map(str,w)); ptype, edges = chart_pauli[(cell,word)]
            pkeys=frozenset(named_key[e] for e in edges)
            novel_wt=sum(aud._syndrome_X_pauli(set(edges), checks).values())
            fires=[]
            for ch in plaquettes:
                if len(pkeys & ch["support"])%2==0: continue
                t_occ=bool(ch["support"]&occ_keys); t_cat=bool(ch["support"]&cat_keys)
                cls="bulk_exterior" if not t_cat else ("leg_local" if t_occ else "cell_local")
                fires.append((ch["loc"], cls, aud._loc_rinf(ch["loc"])))
            n_ext=sum(1 for f in fires if f[1]=="bulk_exterior")
            ok=(novel_wt==0) and (n_ext==0)
            status="PASS" if ok else "FAIL"
            if ok:
                n_pass+=1
                if any(w): n_nz+=1
            else: n_fail+=1
            bulk_rows.append(dict(
                cell=cell,catalog_cell=cell,kernel_word=word,pauli_type=ptype,
                pauli_edges="(identity)" if not edges else " ".join(sorted(edges)),
                pauli_weight=len(edges),
                stated_occupation_legs=" ".join(sorted(stated[cell])),
                catalog_edges=" ".join(sorted(catalog)),
                patch_radius_Linf=radius,
                n_ordinary_stars=len(stars), n_ordinary_plaquettes=len(plaquettes),
                novel_syndrome_weight=novel_wt, n_ordinary_fires=len(fires),
                n_leg_local_fires=sum(1 for f in fires if f[1]=="leg_local"),
                n_cell_local_fires=sum(1 for f in fires if f[1]=="cell_local"),
                n_bulk_exterior_fires=n_ext,
                max_fire_rinf=max((f[2] for f in fires), default=0),
                bulk_exterior_locs=" ".join(f[0] for f in fires if f[1]=="bulk_exterior"),
                ordinary_fire_detail=" ".join(f"{a}:{b}" for a,b,_ in fires),
                status=status,
                note="vacuum identity" if not edges else "WB-transcribed Pauli; ordinary bulk exterior syndrome empty",
            ))
    with (ROOT/"wb_bullet_plus_bulk_syndrome_audit.csv").open("w",newline="") as fh:
        w=csv.DictWriter(fh, fieldnames=list(bulk_rows[0].keys())); w.writeheader(); w.writerows(bulk_rows)

    print(f"bullet/plus chart PASS {sum(1 for r in chart_rows if r['status']=='PASS')}/{len(chart_rows)}")
    print(f"bullet/plus bulk  PASS {n_pass}/{len(bulk_rows)} nonzero={n_nz}/4 fail={n_fail} R={radius}")
    for r in bulk_rows:
        if any(int(c) for c in r["kernel_word"]):
            print(f"  {r['cell']} {r['kernel_word']} {r['status']} edges={r['pauli_edges']} ext={r['n_bulk_exterior_fires']}")
    return 0 if n_fail==0 else 1

if __name__ == "__main__":
    raise SystemExit(main(2) or main(3))
