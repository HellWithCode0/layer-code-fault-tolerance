#!/usr/bin/env python3
"""Finite-size threshold estimate from continuous-transition Monte Carlo."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_rows(path):
    rows=list(csv.DictReader(open(path)))
    out=[]
    for r in rows:
        out.append({
            'p': float(r['p']), 'L': int(float(r['size'])),
            'G': int(float(r['gauge_scale'])),
            'shots': int(float(r['shots'])),
            'fails': int(float(r['failures'])),
            'pL': float(r['logical_error_rate']),
            'wlo': float(r['wilson_lo']), 'whi': float(r['wilson_hi']),
            'profile': r.get('profile', r.get('noise_profile','baseline')),
        })
    return out

def crossing_linear(p, y_lo, y_hi):
    d=np.asarray(y_lo)-np.asarray(y_hi)
    for i in range(len(p)-1):
        if d[i]==0: return float(p[i])
        if d[i]*d[i+1]<0:
            t=d[i]/(d[i]-d[i+1])
            return float(p[i]+t*(p[i+1]-p[i]))
    return float('nan')

def jeffreys_crossing_samples(pgrid, klo, nlo, khi, nhi, npost=30000, rng=0):
    rng = np.random.default_rng(rng)
    crosses=[]
    for _ in range(npost):
        ylo=rng.beta(klo+0.5, np.maximum(nlo-klo,0)+0.5)
        yhi=rng.beta(khi+0.5, np.maximum(nhi-khi,0)+0.5)
        xs=[]; d=[]
        for i,p in enumerate(pgrid):
            if nlo[i]<=0 or nhi[i]<=0: continue
            xs.append(p); d.append(ylo[i]-yhi[i])
        if len(xs)<2: continue
        xs=np.array(xs); d=np.array(d)
        found=None
        for i in range(len(xs)-1):
            if d[i]*d[i+1]<0:
                t=d[i]/(d[i]-d[i+1]); found=float(xs[i]+t*(xs[i+1]-xs[i])); break
            if d[i]==0: found=float(xs[i]); break
        if found is not None and xs[0] < found < xs[-1]:
            crosses.append(found)
    return np.array(crosses)

def analyze(rows, G, outdir):
    sub=[r for r in rows if r['G']==G and r['p']>0]
    zero=[r for r in rows if r['G']==G and r['p']==0]
    sizes=sorted({r['L'] for r in sub})
    ps=sorted({r['p'] for r in sub})
    table={(r['p'],r['L']):r for r in sub}
    Lmin,Lmax=sizes[0], sizes[-1]
    mono=[]; wsep=[]
    for p in ps:
        if not all((p,L) in table for L in sizes):
            continue
        rates=[table[(p,L)]['pL'] for L in sizes]
        if all(rates[i]+1e-15>=rates[i+1] for i in range(len(rates)-1)):
            mono.append(p)
            if all(table[(p,sizes[i+1])]['whi'] < table[(p,sizes[i])]['wlo'] for i in range(len(sizes)-1)):
                wsep.append(p)
    ps2=[p for p in ps if (p,Lmin) in table and (p,Lmax) in table]
    ylo=np.array([table[(p,Lmin)]['pL'] for p in ps2])
    yhi=np.array([table[(p,Lmax)]['pL'] for p in ps2])
    x_lin=crossing_linear(ps2, ylo, yhi)
    klo=np.array([table[(p,Lmin)]['fails'] for p in ps2], dtype=float)
    nlo=np.array([table[(p,Lmin)]['shots'] for p in ps2], dtype=float)
    khi=np.array([table[(p,Lmax)]['fails'] for p in ps2], dtype=float)
    nhi=np.array([table[(p,Lmax)]['shots'] for p in ps2], dtype=float)
    samples=jeffreys_crossing_samples(np.array(ps2), klo, nlo, khi, nhi, npost=40000, rng=20260919+G)
    if len(samples):
        med=float(np.median(samples)); lo=float(np.quantile(samples,0.025)); hi=float(np.quantile(samples,0.975)); frac=len(samples)/40000
    else:
        med=lo=hi=float('nan'); frac=0.
    slopes={}
    for p in ps:
        Ls=[L for L in sizes if (p,L) in table]
        if len(Ls)<3: continue
        x=np.log(np.array(Ls,float))
        y=np.log(np.clip([table[(p,L)]['pL'] for L in Ls], 1e-12, 1))
        slopes[p]=float(np.polyfit(x,y,1)[0])

    fig,ax=plt.subplots(figsize=(6.4,4.3))
    cmap=plt.cm.viridis(np.linspace(0.1,0.9,len(sizes)))
    for color,L in zip(cmap,sizes):
        xs=[]; ys=[]; elo=[]; ehi=[]
        for p in ps:
            if (p,L) not in table: continue
            r=table[(p,L)]; xs.append(p); ys.append(r['pL']); elo.append(max(0,r['pL']-r['wlo'])); ehi.append(max(0,r['whi']-r['pL']))
        ax.errorbar(xs, ys, yerr=[elo,ehi], marker='o', ms=3.5, lw=1, capsize=2, color=color, label=f'L={L}')
    if not math.isnan(x_lin):
        ax.axvline(x_lin, color='k', ls='--', lw=1, label=f'linear cross≈{x_lin:.4g}')
    if not math.isnan(med):
        ax.axvspan(lo, hi, color='0.75', alpha=0.45, label=f'Jeffreys 95% crossing band')
        ax.axvline(med, color='crimson', ls='-', lw=1.2, label=f'Jeffreys med≈{med:.4g}')
    ax.set_yscale('log'); ax.set_xlabel(r'two-qubit noise parameter $p$'); ax.set_ylabel(r'logical error rate $p_L$')
    ax.set_title(f'Continuous SI1000-like code-switch (gauged duration = {G}d)')
    ax.legend(fontsize=7, loc='best'); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir/f'threshold_G{G}d.pdf'); fig.savefig(outdir/f'threshold_G{G}d.png', dpi=180); plt.close(fig)

    return {
        'gauge_scale': G, 'sizes': sizes, 'monotone_ps': mono, 'wilson_separated_ps': wsep,
        'linear_crossing_Lmin_Lmax': x_lin, 'jeffreys_median': med, 'jeffreys_95_lo': lo, 'jeffreys_95_hi': hi,
        'jeffreys_valid_fraction': frac, 'log_slopes': {str(k):v for k,v in slopes.items()},
        'total_shots': int(sum(r['shots'] for r in sub)+sum(r['shots'] for r in zero)),
        'noiseless_all_zero': all(r['fails']==0 for r in zero) if zero else None,
        'n_noiseless': len(zero),
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv', required=True); ap.add_argument('--out', required=True)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows=load_rows(args.csv)
    reports=[analyze(rows, G, out) for G in sorted({r['G'] for r in rows})]
    meds=[r['jeffreys_median'] for r in reports if not math.isnan(r['jeffreys_median'])]
    headline=float(np.median(meds)) if meds else float('nan')
    los=[r['jeffreys_95_lo'] for r in reports if not math.isnan(r['jeffreys_95_lo'])]
    his=[r['jeffreys_95_hi'] for r in reports if not math.isnan(r['jeffreys_95_hi'])]
    band=(float(min(los)), float(max(his))) if los and his else (float('nan'), float('nan'))
    lines=[
        '# Numerical threshold estimate',
        '',
        '## Scope',
        '',
        'Finite-size **circuit-level crossing estimate** for the continuous algebraic',
        '`base → gauged-strip → base` code-switch under an SI1000-like parameterization on the growing HGP/layer-code substrate,',
        'with exact graphlike Stim DEM + MWPM.',
        '',
        '**Not** a proof of the asymptotic committed-MW threshold in the FT theorem,',
        '**not** a CAD-complete Williamson–Baspin T3/T4 DEM, and **not** an efficient-decoder theorem.',
        '',
        'Paper wording: *finite-size crossing estimate for the algebraic code-switch model under an SI1000-like parameterization*.',
        '',
        '## Headline estimate',
        '',
        f'- Median over G=d,2d,3d Jeffreys medians: **p_th^FS ≈ {headline:.6f}**',
        f'- Envelope of per-duration Jeffreys 95% crossing bands: **[{band[0]:.6f}, {band[1]:.6f}]**',
        f'- Total Monte Carlo shots (all families): **{sum(r["total_shots"] for r in reports):,}**',
        '',
    ]
    for r in reports:
        lines += [
            f"## G={r['gauge_scale']}d", '',
            f"- sizes: {r['sizes']}",
            f"- noiseless all-zero: {r['noiseless_all_zero']} ({r['n_noiseless']} points)",
            f"- monotone p: {r['monotone_ps']}",
            f"- Wilson-separated monotone p: {r['wilson_separated_ps']}",
            f"- linear Lmin–Lmax crossing: {r['linear_crossing_Lmin_Lmax']:.6g}",
            f"- **Jeffreys median: {r['jeffreys_median']:.6g}**",
            f"- Jeffreys 95%: [{r['jeffreys_95_lo']:.6g}, {r['jeffreys_95_hi']:.6g}] (valid frac {r['jeffreys_valid_fraction']:.3f})",
            f"- shots: {r['total_shots']:,}",
            f"- log p_L vs log L slopes: {r['log_slopes']}",
            '',
        ]
    (out/'THRESHOLD_ESTIMATE.md').write_text('\n'.join(lines))
    (out/'threshold_fit.json').write_text(json.dumps({'headline_fs_pth':headline,'band':band,'by_G':reports}, indent=2))
    print(json.dumps({'headline_fs_pth':headline,'band':band}, indent=2))

if __name__=='__main__':
    main()
