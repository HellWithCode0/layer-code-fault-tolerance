import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'figures'
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'mathtext.fontset': 'dejavusans',
    'figure.dpi': 180,
    'savefig.bbox': 'tight',
})

# ---------- Architecture courier ----------
fig, ax = plt.subplots(figsize=(12.0,4.6))
ax.set_xlim(0,15); ax.set_ylim(0,6.8); ax.axis('off')

def box(x,y,w,h,text,fc,ec,fs=13,weight='normal'):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.14',facecolor=fc,edgecolor=ec,lw=1.7)
    ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,fontweight=weight)
    return p

def arrow(x1,y1,x2,y2,label=None,dy=.22):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=1.6,color='#4b5563'))
    if label: ax.text((x1+x2)/2,(y1+y2)/2+dy,label,ha='center',va='bottom',fontsize=10.5,color='#374151')

ax.text(.5,6.15,'Encoded resource path',fontsize=16,fontweight='bold')
box(.5,4.25,2.15,1.15,'raw resource\nlocal bare port','#fff1df','#b56b1f',13)
box(3.2,4.25,2.2,1.15,'first encoded\nblock $B_0$','#e9f3ff','#49637f',13)
box(6.0,4.25,2.45,1.15,'encoded courier\n$B_j\\to B_{j+1}$','#e9f3ff','#49637f',13)
box(9.05,4.25,2.0,1.15,'factory /\ndistillation','#f0eafe','#735db0',13)
box(12.0,4.25,2.35,1.15,'protected\nresource output','#e6f5e8','#4b7d52',13)
for a,b in [(2.65,3.2),(5.4,6.0),(8.45,9.05),(11.05,12.0)]: arrow(a,4.83,b,4.83)
for x in [2.92,5.7,8.75,11.52]: ax.plot([x,x],[4.0,5.7],lw=5,color='#d5dae0')

ax.text(.5,2.55,'Persistent layer-code data',fontsize=16,fontweight='bold')
box(1.2,.72,3.0,1.25,'3D nearest-neighbour\nlayer-code data','#eaf5ea','#4b7d52',13)
box(5.65,.72,3.0,1.25,'constant-width slab\nlogical port','#eaf5ea','#4b7d52',13)
box(10.3,.72,3.25,1.25,'local 3D gauging\n$data\\leftrightarrow$ resource','#eaf5ea','#4b7d52',13)
arrow(4.2,1.35,5.65,1.35,'slab theorem',.30)
arrow(8.65,1.35,10.3,1.35,'protected measurement',.30)
arrow(13.2,4.25,12.45,1.98,'consume locally',.05)
ax.text(7.5,.12,'All inter-block operations are geometrically local; separators denote padding between active boundaries.',ha='center',fontsize=10.5,color='#4b5563')
fig.subplots_adjust(left=.02,right=.99,bottom=.03,top=.98)
fig.savefig(OUT/'architecture_courier.pdf'); fig.savefig(OUT/'architecture_courier.png',dpi=320); plt.close(fig)

# ---------- Gate compilation ----------
fig, ax = plt.subplots(figsize=(5.0,3.55)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,8.2)
ax.set_title(r'Clifford+$T$ from pure-$X/Z$ measurements',fontsize=13,pad=7,fontweight='bold')
rows=[(r'$S$',r'phase teleportation + $|Y\rangle$',r'$Z^{m\oplus n}X^m$'),(r'$T$',r'phase teleportation + $|A\rangle$',r'$S^mZ^nX^m$'),(r'$H$',r'$S\to S_X\to S$  ($3|Y\rangle$)',r'Pauli frame'),(r'CNOT',r'$Z_cZ_a,\ X_aX_t,\ Z_a$',r'$Z_c^vX_t^{u\oplus w}$'),(r'CZ',r'$H_t\cdot$CNOT$\cdot H_t$',r'$\leq 6|Y\rangle$')]
cols=[1.1,5.0,8.65]
ax.add_patch(Rectangle((.15,6.7),9.7,.85,facecolor='#ededed',edgecolor='#333',lw=.8))
for x,h in zip(cols,['Gate','Measurements / resources','Correction']): ax.text(x,7.12,h,ha='center',va='center',fontsize=10.5,fontweight='bold')
for i,(g,m,c) in enumerate(rows):
    y=5.95-i*1.05
    ax.add_patch(Rectangle((.15,y-.38),9.7,.82,facecolor='#f4f7f4' if i%2==0 else 'white',edgecolor='#555',lw=.65))
    ax.text(cols[0],y,g,ha='center',va='center',fontsize=11,fontweight='bold')
    ax.text(cols[1],y,m,ha='center',va='center',fontsize=9.8)
    ax.text(cols[2],y,c,ha='center',va='center',fontsize=9.5)
ax.text(5,.35,r'Only pure-$X$ or pure-$Z$ logical measurements are required.',ha='center',fontsize=9.5,style='italic')
fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.90)
fig.savefig(OUT/'gate_compilation.pdf'); fig.savefig(OUT/'gate_compilation.png',dpi=320); plt.close(fig)

# ---------- Raw injection ----------
fig, ax = plt.subplots(figsize=(5.2,3.05)); ax.set_xlim(0,11); ax.set_ylim(0,6.4); ax.axis('off')
ax.set_title('Raw injection: only the travelling resource is cheap',fontsize=12.5,pad=6,fontweight='bold')
ax.add_patch(FancyBboxPatch((.35,2.05),1.9,2.25,boxstyle='round,pad=.05',facecolor='#ffcdd2',edgecolor='#333',lw=1.1))
ax.text(1.30,3.18,'bare $q$\n(weight 1)',ha='center',va='center',fontsize=11)
ax.add_patch(FancyBboxPatch((3.15,1.45),3.65,3.55,boxstyle='round,pad=.05',facecolor='#d9eddb',edgecolor='#333',lw=1.1))
ax.text(4.98,4.55,r'block $B_0$',ha='center',fontsize=12,fontweight='bold')
# three clean internal logical-class boxes
inner=[(3.52,2.45,0.92,0.95,'spectator\n$\Omega(d)$','#e3f2fd'),(4.60,2.45,0.92,0.95,'spectator\n$\Omega(d)$','#e3f2fd'),(5.68,2.45,0.92,0.95,'travelling\nresource','#fff9c4')]
for x,y,w,h,lab,fc in inner:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025',facecolor=fc,edgecolor='#555',lw=.75))
    ax.text(x+w/2,y+h/2,lab,ha='center',va='center',fontsize=7.4)
ax.annotate('',xy=(3.15,3.18),xytext=(2.25,3.18),arrowprops=dict(arrowstyle='-|>',lw=1.4,color='#444'))
ax.text(2.70,3.63,r'gauge $Z_qZ_{B_0}$',ha='center',fontsize=9.2)
ax.add_patch(FancyBboxPatch((7.85,2.05),2.65,2.25,boxstyle='round,pad=.05',facecolor='#e1bee7',edgecolor='#333',lw=1.1))
ax.text(9.18,3.18,'encoded courier\nthereafter',ha='center',va='center',fontsize=11)
ax.annotate('',xy=(7.85,3.18),xytext=(6.80,3.18),arrowprops=dict(arrowstyle='-|>',lw=1.4,color='#444'))
ax.text(5.45,.45,r'All spectator logicals remain $\Omega(d)$ in the quotient; only the intended resource class is weight one.',ha='center',fontsize=9.3)
fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.88)
fig.savefig(OUT/'raw_injection_intuition.pdf'); fig.savefig(OUT/'raw_injection_intuition.png',dpi=320); plt.close(fig)

# ---------- Fragmentation ----------
fig, axes = plt.subplots(1,3,figsize=(9.3,3.35))
configs=[('Filled cube',r'$B_f(A)=O(W^{2/3}/b^{2}+1)$'),(r'Thin slab, thickness $h<b$',r'$B_f(A)=\Theta(W/(hb^{2}))$'),('Isolated sites',r'$B_f(A)=W$')]  # corrected labels; see make_fragmentation_figure.py
for ax,(title,sub) in zip(axes,configs):
    ax.set_xlim(0,4); ax.set_ylim(0,4); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title+'\n'+sub,fontsize=11.2,fontweight='bold',pad=5)
for i in range(3):
    for j in range(3): axes[0].add_patch(Rectangle((.55+i*.9,.55+j*.9),.78,.78,facecolor='#79ad64',edgecolor='#333',lw=.7))
axes[0].text(2,.15,'few partial coarse cells',ha='center',fontsize=9.2)
for i in range(9): axes[1].add_patch(Rectangle((.22+i*.4,1.55),.34,.72,facecolor='#f6b26b',edgecolor='#333',lw=.6))
axes[1].text(2,.15,'same $W$, many partial cells',ha='center',fontsize=9.2)
for x,y in [(0.35,.55),(.35,2.8),(1.5,1.65),(2.85,.45),(2.85,2.85),(1.5,3.0),(.95,.45),(2.2,1.55),(3.2,1.65)]: axes[2].add_patch(Rectangle((x,y),.53,.53,facecolor='#e77a7a',edgecolor='#333',lw=.6))
axes[2].text(2,.15,'same $W$, many islands',ha='center',fontsize=9.2)
fig.suptitle(r'At fixed logical volume $W$, fragmentation $B_f(A)$ controls the boundary-resource penalty',fontsize=13,fontweight='bold',y=.99)
fig.subplots_adjust(left=.02,right=.98,bottom=.10,top=.78,wspace=.18)
fig.savefig(OUT/'fragmentation_Bf.pdf'); fig.savefig(OUT/'fragmentation_Bf.png',dpi=320); plt.close(fig)

# ---------- Analytic bound plots ----------
r=np.linspace(.05,1,300)
fig,ax=plt.subplots(figsize=(4.0,3.1))
for t in (1,2,4): ax.plot(r,r/(2*t),label=rf'$t={t}$',lw=1.8)
ax.set_xlabel(r'Aspect ratio $a_{\min}/a_{\max}$'); ax.set_ylabel(r'Lower bound on $\beta$'); ax.set_xlim(0,1); ax.set_ylim(0,.52); ax.grid(True,alpha=.2); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/'relative_expansion_bound.pdf'); fig.savefig(OUT/'relative_expansion_bound.png',dpi=320); plt.close(fig)

k=np.arange(1,21); rect=12; pad=1+np.floor(k-1).astype(int); fph=np.minimum(rect,pad)
fig,ax=plt.subplots(figsize=(4.0,3.1)); ax.step(k,fph,where='mid',label=r'composed lower bound $f_{\rm ph}$',lw=1.8); ax.plot(k,pad,'--',lw=1.4,label='padding term'); ax.axhline(rect,ls=':',lw=1.4,label=r'$\delta_g=\delta_{\rm mem}=12$'); ax.set_xlabel(r'Padding ratio $R_{\rm pad}/\tau$'); ax.set_ylabel(r'Lower bound on $f_{\rm ph}$'); ax.set_xlim(1,20); ax.set_ylim(0,14); ax.grid(True,alpha=.2); ax.legend(frameon=False,loc='lower right')
fig.tight_layout(); fig.savefig(OUT/'padded_composition_bound.pdf'); fig.savefig(OUT/'padded_composition_bound.png',dpi=320); plt.close(fig)

# ---------- CSV sanity-check plots ----------
df=pd.read_csv(ROOT/'phenomenological_mw_demo.csv')
fig,ax=plt.subplots(figsize=(4.0,3.1))
for d,g in df.groupby('distance'):
    y=g.logical_error_rate.to_numpy(); lo=g.wilson_lo.to_numpy(); hi=g.wilson_hi.to_numpy(); p=g.p.to_numpy(); yerr=np.vstack([y-lo,hi-y]); ax.errorbar(p,y,yerr=yerr,marker='o',ms=4,lw=1.5,capsize=2,label=rf'$d={d}$ toric')
ax.set_yscale('log'); ax.set_xlabel(r'Phenomenological error probability $p$'); ax.set_ylabel(r'Logical error rate $p_L$'); ax.set_title('Phenomenological MWPM sanity check',fontsize=11.5); ax.grid(True,which='both',alpha=.2); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/'phenomenological_mw_demo.pdf'); fig.savefig(OUT/'phenomenological_mw_demo.png',dpi=320); plt.close(fig)

df=pd.read_csv(ROOT/'gauging_rectangle_mwpm.csv')
fig,ax=plt.subplots(figsize=(4.0,3.1))
for d,g in df.groupby('d'):
    y=g.logical_error_rate.to_numpy(); lo=g.wilson_lo.to_numpy(); hi=g.wilson_hi.to_numpy(); p=g.p.to_numpy(); yerr=np.vstack([y-lo,hi-y]); ax.errorbar(p,y,yerr=yerr,marker='o',ms=4,lw=1.5,capsize=2,label=rf'$d={d}$')
ax.set_yscale('log'); ax.set_xlabel(r'Code-capacity error probability $p$'); ax.set_ylabel(r'Logical error rate $p_L$'); ax.set_title('Gauging-rectangle MWPM sanity check',fontsize=11.5); ax.grid(True,which='both',alpha=.2); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/'gauging_rectangle_mwpm.pdf'); fig.savefig(OUT/'gauging_rectangle_mwpm.png',dpi=320); plt.close(fig)

print('print-scale pass complete')
