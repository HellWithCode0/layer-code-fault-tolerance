import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 160,
    'savefig.bbox': 'tight',
})

# Figure 1: exact integer distance floor from Eq. (distance-floor).
p = np.logspace(-4, np.log10(0.3), 700)
q = p * (1.0 - p) / 3.0
fig, ax = plt.subplots(figsize=(3.35, 2.45))
for eps in (1e-2, 1e-4, 1e-6):
    ratio = np.log(1.0/(2.0*eps)) / np.log(1.0/q)
    dstar = np.maximum(1, 2*np.ceil(ratio).astype(int) - 1)
    ax.step(p, dstar, where='post', label=rf'$\varepsilon={eps:g}$')
ax.set_xscale('log')
ax.set_xlabel(r'Physical depolarizing probability $p$')
ax.set_ylabel(r'Distance floor $d_*$')
ax.grid(True, which='both', alpha=0.22)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/'storage_distance_floor.pdf')
fig.savefig(OUT/'storage_distance_floor.png', dpi=240)
plt.close(fig)

# Figure 2: prefix-parity repair network, exact illustrative bit pattern.
x = np.array([1, 0, 1, 0], dtype=int)
pref = np.concatenate([[0], np.cumsum(x) % 2])  # p_0 ... p_4
n = len(x)
xs = np.arange(1, n+1)
fig, ax = plt.subplots(figsize=(3.35, 2.25))
# Baseline / possible backbone
ax.plot([0.6, n+0.4], [0, 0], linestyle=':', linewidth=1.0, alpha=0.55)
# Occupied backbone segments p_i, i=1..n-1
for i in range(1, n):
    if pref[i] == 1:
        ax.plot([i, i+1], [0, 0], linewidth=4.0, solid_capstyle='butt')
    else:
        ax.plot([i, i+1], [0, 0], linestyle=':', linewidth=1.0, alpha=0.55)
# Incidence nodes and selected data strings
ax.scatter(xs, np.zeros(n), s=28, zorder=3)
for i, bit in enumerate(x, start=1):
    if bit:
        ax.plot([i, i], [0, 0.78], linewidth=3.0)
        ax.text(i, 0.88, rf'$x_{i}=1$', ha='center', va='bottom')
    else:
        ax.plot([i, i], [0, 0.48], linestyle=':', linewidth=1.2, alpha=0.65)
        ax.text(i, 0.58, rf'$x_{i}=0$', ha='center', va='bottom')
    triplet = f'{pref[i-1]}{pref[i]}{bit}'
    ax.text(i, -0.27, triplet, ha='center', va='top', fontsize=8)
# Backbone labels
for i in range(1, n):
    ax.text(i+0.5, 0.08, rf'$p_{i}={pref[i]}$', ha='center', va='bottom', fontsize=8)
ax.text(0.62, 0.08, r'$p_0=0$', ha='left', va='bottom', fontsize=8)
ax.text(n+0.38, 0.08, rf'$p_{n}=0$', ha='right', va='bottom', fontsize=8)
ax.annotate('pass-through 110', xy=(2, 0), xytext=(2.0, -0.62),
            ha='center', arrowprops={'arrowstyle':'->', 'lw':0.8})
ax.set_xlim(0.5, n+0.5)
ax.set_ylim(-0.78, 1.18)
ax.axis('off')
fig.tight_layout(pad=0.25)
fig.savefig(OUT/'prefix_parity_network.pdf')
fig.savefig(OUT/'prefix_parity_network.png', dpi=240)
plt.close(fig)

# Figure 3: exact lower bound from the cubic-box relative-expansion corollary.
r = np.linspace(0.05, 1.0, 300)  # a_min/a_max
fig, ax = plt.subplots(figsize=(3.35, 2.45))
for t in (1, 2, 4):
    beta_lb = r / (2.0*t)
    ax.plot(r, beta_lb, label=rf'$t={t}$')
ax.set_xlabel(r'Aspect ratio $a_{\min}/a_{\max}$')
ax.set_ylabel(r'Lower bound on $\beta$')
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.52)
ax.grid(True, alpha=0.22)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/'relative_expansion_bound.pdf')
fig.savefig(OUT/'relative_expansion_bound.png', dpi=240)
plt.close(fig)

# Figure 4: exact padded-composition minimum for illustrative fixed rectangle distances.
k = np.arange(1, 21)  # R_pad / tau, integer multiples
rect = 12
pad_term = 1 + np.floor(k - 1).astype(int)  # exact theorem term for tau=1, R_pad=k
fph = np.minimum(rect, pad_term)
fig, ax = plt.subplots(figsize=(3.35, 2.45))
ax.step(k, fph, where='mid', label=r'composed lower bound $f_{\rm ph}$')
ax.plot(k, pad_term, linestyle='--', linewidth=1.1, label='padding term')
ax.axhline(rect, linestyle=':', linewidth=1.1, label=r'$\delta_g=\delta_{\rm mem}=12$')
ax.set_xlabel(r'Padding ratio $R_{\rm pad}/\tau$')
ax.set_ylabel(r'Lower bound on $f_{\rm ph}$')
ax.set_xlim(1, 20)
ax.set_ylim(0, 14)
ax.grid(True, alpha=0.22)
ax.legend(frameon=False, loc='lower right')
fig.tight_layout()
fig.savefig(OUT/'padded_composition_bound.pdf')
fig.savefig(OUT/'padded_composition_bound.png', dpi=240)
plt.close(fig)

print('Generated:')
for f in sorted(OUT.glob('*.pdf')):
    print(f.name)

# Early proof-dependency diagram (later replaced by the final wide version).
fig, ax = plt.subplots(figsize=(3.35, 4.25))
ax.axis('off')
labels = [
    'Uniform local detector\ngeneration',
    'Boundary gluing with\ncomplete $(s,\\lambda)$',
    'Relative rectangle\nrestriction',
    'Relative rectangle\ndistance',
    'Padded composition',
    'Raw / encoded\ncourier lemmas',
    'Resource-boundary\nquarantine',
    'Complete scheduled-\ncircuit theorem',
]
y = np.linspace(0.93, 0.08, len(labels))
for i, (lab, yy) in enumerate(zip(labels, y)):
    ax.text(0.5, yy, lab, ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.32', fc='none', lw=0.9))
    if i < len(labels)-1:
        ax.annotate('', xy=(0.5, y[i+1]+0.045), xytext=(0.5, yy-0.045),
                    arrowprops=dict(arrowstyle='->', lw=0.8))
ax.text(0.03, 0.76, r'$\mathcal{R}_g^{\mathrm{full}}$', ha='left', va='center', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', fc='none', lw=0.8))
ax.annotate('algebraic audit only', xy=(0.36, y[1]), xytext=(0.18, 0.70),
            fontsize=7, ha='center', arrowprops=dict(arrowstyle='->', lw=0.7))
fig.tight_layout(pad=0.2)
fig.savefig(OUT/'proof_dependency.pdf')
fig.savefig(OUT/'proof_dependency.png', dpi=240)
plt.close(fig)

# Architecture overview: resource path plus persistent data lane
fig, ax = plt.subplots(figsize=(7.2, 2.75))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')

def box(x, y, w, h, text, fc='#eef3f8', ec='#415a77', lw=1.25, fs=9.0):
    r = patches.FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.025,rounding_size=0.08',
                               facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(r); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
    return r

def arrow(x1,y1,x2,y2, text=None):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.4,color='#4b5563'))
    if text: ax.text((x1+x2)/2,(y1+y2)/2+0.18,text,ha='center',va='bottom',fontsize=7.7,color='#374151')

ax.text(0.2,3.55,'Encoded resource path',fontsize=9.3,fontweight='bold',ha='left')
box(0.35,2.55,1.25,0.72,'raw resource\nlocal bare port',fc='#fff4e6',ec='#b56b1f')
box(2.05,2.55,1.35,0.72,'first encoded\nblock $B_0$',fc='#edf6ff')
box(3.95,2.55,1.55,0.72,'encoded courier\n$B_j\\to B_{j+1}$',fc='#edf6ff')
box(6.05,2.55,1.35,0.72,'factory /\ndistillation',fc='#f3efff',ec='#6d5ba8')
box(8.05,2.55,1.45,0.72,'protected\nresource output',fc='#eef8ef',ec='#4b7d52')
arrow(1.60,2.91,2.05,2.91); arrow(3.40,2.91,3.95,2.91); arrow(5.50,2.91,6.05,2.91); arrow(7.40,2.91,8.05,2.91)

ax.text(0.2,1.55,'Persistent layer-code data',fontsize=9.3,fontweight='bold',ha='left')
box(0.80,0.48,2.30,0.72,'3D nearest-neighbour\nlayer-code data',fc='#eef8ef',ec='#4b7d52')
box(4.05,0.48,2.15,0.72,'constant-width slab\nlogical port',fc='#eef8ef',ec='#4b7d52')
box(7.15,0.48,2.05,0.72,'local 3D gauging\n$data\\,\\leftrightarrow\\,resource$',fc='#eef8ef',ec='#4b7d52')
arrow(3.10,0.84,4.05,0.84); arrow(6.20,0.84,7.15,0.84)
ax.text(3.58,1.34,'slab theorem',ha='center',va='bottom',fontsize=7.7,color='#374151')
ax.text(6.68,1.34,'protected measurement',ha='center',va='bottom',fontsize=7.7,color='#374151')
arrow(8.77,2.55,8.25,1.20,'consume locally')
for x in (1.78,3.67,5.78,7.72):
    ax.plot([x,x],[2.42,3.38],lw=3.5,color='#c7cdd4',alpha=0.8)
ax.text(5.0,0.08,'All inter-block operations are geometrically local; shaded separators denote padding / quarantine between active boundaries.',
        ha='center',va='bottom',fontsize=7.6,color='#4b5563')
fig.tight_layout(pad=0.25)
fig.savefig(OUT/'architecture_courier.pdf', bbox_inches='tight')
fig.savefig(OUT/'architecture_courier.png', dpi=240, bbox_inches='tight')
plt.close(fig)


# The phenomenological MWPM demo figure is produced by operational_anchor.py
# (figures/phenomenological_mw_demo.pdf/.png), not by this analytic script.
print("Note: figures/phenomenological_mw_demo.pdf is generated by operational_anchor.py")


# ---------- Additional explanatory figures ----------
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

NEW_C = '#d4edda'
IMP_C = '#cce5ff'
AUD_C = '#fff3cd'
EDGE_C = '#333333'

def _p3_box(ax, x, y, w, h, text, facecolor, lw=1.0, fontsize=7, dashed=False):
    style = 'square,pad=0.01' if dashed else 'round,pad=0.02,rounding_size=0.04'
    p = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle=style,
                       facecolor=facecolor, edgecolor=EDGE_C, lw=lw,
                       linestyle='--' if dashed else '-',
                       transform=ax.transAxes, clip_on=False)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            transform=ax.transAxes, linespacing=1.15)

fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')

def depbox(x,y,w,h,text,fc='#eef6ee',ec='#4d7c55',fs=9.2):
    r=patches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.08',
                             facecolor=fc,edgecolor=ec,linewidth=1.3)
    ax.add_patch(r); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,fontweight='bold')

def deparrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.6,color='#4b5563'))

xs=[0.35,3.15,5.95,8.95]
labels=['Slab theorem','Local gauging','Spacetime\ncomposition','Universal\nresource theorem']
for x,l in zip(xs,labels): depbox(x,2.45,2.25,1.0,l)
for i in range(3): deparrow(xs[i]+2.25,2.95,xs[i+1],2.95)

depbox(0.65,0.65,1.75,0.72,'Published layer-code\nfinite cells',fc='#edf3fb',ec='#5879a6',fs=8.2)
depbox(6.22,0.65,1.75,0.72,'Williamson--Yoder\ncleaning theorem',fc='#edf3fb',ec='#5879a6',fs=8.2)
deparrow(1.52,1.37,1.52,2.45); deparrow(7.10,1.37,7.10,2.45)
ax.text(5.95,4.45,'Geometry-preserving route from memory to universal fault tolerance',
        ha='center',va='center',fontsize=10.6,fontweight='bold')
ax.text(5.95,0.15,'Finite audits check the two local interfaces and circuit templates; the four green boxes are the analytic proof chain.',
        ha='center',va='bottom',fontsize=8.0,color='#4b5563')
fig.tight_layout(pad=0.25)
fig.savefig(OUT/'proof_dependency.pdf', bbox_inches='tight')
fig.savefig(OUT/'proof_dependency.png', dpi=240, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
ax = axes[0]; ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect('equal'); ax.axis('off')
ax.set_title(r'Filled cube: $B_f(A)=O(W^{2/3}/b^2+1)$', fontsize=8)
for i in range(3):
    for j in range(3):
        ax.add_patch(Rectangle((0.5+i*0.9, 0.5+j*0.9), 0.8, 0.8, facecolor='#6aa84f', edgecolor='k', lw=0.6, alpha=0.85))
ax.text(2, 0.15, r'$W$ sites, few partial cells', ha='center', fontsize=7)
ax.text(2, 3.7, r'small $B_f$ $\Rightarrow$ $O(Wf+f^{3/2})$', ha='center', fontsize=7, color='#1b5e20')
ax = axes[1]; ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect('equal'); ax.axis('off')
ax.set_title(r'Thin slab ($h<b$): $B_f(A)=\Theta(W/(hb^2))$', fontsize=8)
for i in range(9):
    ax.add_patch(Rectangle((0.2+i*0.4, 1.6), 0.35, 0.7, facecolor='#f6b26b', edgecolor='k', lw=0.5, alpha=0.9))
ax.text(2, 0.15, r'same $W$, many partial cells', ha='center', fontsize=7)
ax.text(2, 3.7, r'large $B_f$ in (A26)', ha='center', fontsize=7, color='#e65100')
ax = axes[2]; ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect('equal'); ax.axis('off')
ax.set_title(r'Isolated sites: $B_f(A)=W$', fontsize=8)
for (x,y) in [(0.3,0.5),(0.3,2.8),(1.5,1.6),(2.8,0.4),(2.8,2.9),(1.5,3.0),(0.9,0.4),(2.2,1.5),(3.2,1.6)]:
    ax.add_patch(Rectangle((x, y), 0.55, 0.55, facecolor='#e06666', edgecolor='k', lw=0.5, alpha=0.9))
ax.text(2, 0.05, r'same $W$, $B_f\sim$# islands', ha='center', fontsize=7)
ax.text(2, 3.7, r'wires to $O(Wf+(B_f+1)f^{3/2})$', ha='center', fontsize=7, color='#b71c1c')
fig.suptitle(r'Equal logical volume $W$: fragmentation $B_f(A)$ drives the resource bound (A26)', fontsize=9, y=1.02)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.82, wspace=0.15)
fig.savefig(OUT/'fragmentation_Bf.pdf'); fig.savefig(OUT/'fragmentation_Bf.png', dpi=240); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.8)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title(r'Clifford+$T$ from pure-$X/Z$ + $|Y\rangle$/$|A\rangle$', fontsize=9)
rows = [(r'$S$', r'phase tel. + $|Y\rangle$', r'$Z^{m\oplus n}X^m$'),
        (r'$T$', r'phase tel. + $|A\rangle$', r'$S^m Z^n X^m$'),
        (r'$H$', r'$S\to S_X\to S$ (3$|Y\rangle$)', r'Pauli frame'),
        (r'CNOT', r'$Z_cZ_a, X_aX_t, Z_a$', r'$Z_c^v X_t^{u\oplus w}$'),
        (r'CZ', r'$H_t\cdot$CNOT$\cdot H_t$', r'$\leq 6\,|Y\rangle$')]
xs = [1.0, 4.6, 8.2]
ax.add_patch(Rectangle((0.2, 6.3), 9.6, 0.9, facecolor='#eeeeee', edgecolor='k', lw=0.6))
for x, h in zip(xs, ['Gate', 'Measurements / resources', 'Correction']):
    ax.text(x, 6.75, h, ha='center', va='center', fontsize=7, fontweight='bold')
for i, (g, m, c) in enumerate(rows):
    y = 5.5 - i*1.05
    ax.add_patch(Rectangle((0.2, y-0.4), 9.6, 0.95, facecolor='#e8f5e9' if i%2==0 else '#ffffff', edgecolor='k', lw=0.5))
    ax.text(xs[0], y, g, ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(xs[1], y, m, ha='center', va='center', fontsize=6.5)
    ax.text(xs[2], y, c, ha='center', va='center', fontsize=6.5)
ax.text(5, 0.35, r'All measurements are pure $X$ or $Z$ (one- or two-block).', ha='center', fontsize=6.5, style='italic')
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.92)
fig.savefig(OUT/'gate_compilation.pdf'); fig.savefig(OUT/'gate_compilation.png', dpi=240); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
ax.set_title(r'Raw injection: only the travelling resource is cheap', fontsize=8)
ax.add_patch(FancyBboxPatch((0.3, 2.0), 1.8, 2.2, boxstyle='round,pad=0.05', facecolor='#ffcdd2', edgecolor='k', lw=0.9))
ax.text(1.2, 3.1, 'bare $q$\n(wt 1)', ha='center', va='center', fontsize=7)
ax.add_patch(FancyBboxPatch((3.0, 1.5), 3.2, 3.2, boxstyle='round,pad=0.05', facecolor='#c8e6c9', edgecolor='k', lw=0.9))
ax.text(4.6, 4.2, r'block $B_0$', ha='center', fontsize=8, fontweight='bold')
for i, lab in enumerate(['spec.\n$\\Omega(d)$', 'spec.\n$\\Omega(d)$', 'travel\nres.']):
    ax.add_patch(Circle((3.7+i*0.9, 2.8), 0.45, facecolor='#fff9c4' if i==2 else '#e3f2fd', edgecolor='k', lw=0.7))
    ax.text(3.7+i*0.9, 2.8, lab, ha='center', va='center', fontsize=5.5)
ax.annotate('', xy=(3.0, 3.1), xytext=(2.1, 3.1), arrowprops=dict(arrowstyle='->', lw=1.0))
ax.text(2.55, 3.55, r'gauge $Z_q Z_{B_0}$', ha='center', fontsize=6)
ax.text(5, 0.45, r'Quotient: spectators stay $\Omega(d)$ (bare-port / raw-quotient lemmas).', ha='center', fontsize=6.2)
ax.add_patch(FancyBboxPatch((7.0, 2.0), 2.6, 2.2, boxstyle='round,pad=0.05', facecolor='#e1bee7', edgecolor='k', lw=0.9))
ax.text(8.3, 3.1, 'encoded\ncourier\nthereafter', ha='center', va='center', fontsize=7)
ax.annotate('', xy=(7.0, 3.1), xytext=(6.2, 3.1), arrowprops=dict(arrowstyle='->', lw=1.0))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.88)
fig.savefig(OUT/'raw_injection_intuition.pdf'); fig.savefig(OUT/'raw_injection_intuition.png', dpi=240); plt.close(fig)

print('Figures regenerated into figures/')

# Figure: repeated slabs, 2-disjoint mixed logicals, and Clifford-only shallow circuits
fig = plt.figure(figsize=(7.2, 2.95))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.1], wspace=0.22)

# Panel A: many admissible gaps give many slab ports
ax = fig.add_subplot(gs[0,0])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
frame = FancyBboxPatch((0.7,0.8), 8.6, 8.2, boxstyle='round,pad=0.02,rounding_size=0.08',
                       facecolor='#f8fafc', edgecolor='#64748b', lw=1.2)
ax.add_patch(frame)
# horizontal X slabs
Y_LEVELS=[2.0,3.35,4.7,6.05,7.4]
for y in Y_LEVELS:
    ax.add_patch(Rectangle((1.05,y-0.28), 7.9, 0.56, facecolor='#cfe8d5', edgecolor='#4c8c5a', lw=0.8, alpha=0.95))
# vertical Z slabs
X_LEVELS=[2.1,3.55,5.0,6.45,7.9]
for x in X_LEVELS:
    ax.add_patch(Rectangle((x-0.22,1.15), 0.44, 7.5, facecolor='#d9d3ef', edgecolor='#6e59a5', lw=0.8, alpha=0.95))
ax.text(5.0, 9.55, r'$\Theta(N)$ admissible $X$-gaps', ha='center', va='bottom', fontsize=8.2, color='#245c30')
ax.text(9.55, 5.0, r'$\Theta(N)$ admissible $Z$-gaps', ha='left', va='center', fontsize=8.2, color='#4b3a80', rotation=270)
ax.text(5.0, 0.15, 'same-orientation slabs are pairwise disjoint', ha='center', va='bottom', fontsize=7.3, color='#475569')
ax.text(1.0, 9.32, '(a)', fontsize=8.4, fontweight='bold')
ax.text(5.0, 8.84, 'uniform-gap slab theorem', ha='center', va='top', fontsize=8.1, fontweight='bold')
ax.text(5.0, 8.2, 'every pure logical class appears in every strip', ha='center', va='top', fontsize=7.1)

# Panel B: 2-disjoint mixed logicals
ax = fig.add_subplot(gs[0,1])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
frame = FancyBboxPatch((0.7,0.8), 8.6, 8.2, boxstyle='round,pad=0.02,rounding_size=0.08',
                       facecolor='#f8fafc', edgecolor='#64748b', lw=1.2)
ax.add_patch(frame)
for idx,(y,x) in enumerate(zip(Y_LEVELS,X_LEVELS), start=1):
    ax.add_patch(Rectangle((1.05,y-0.28), 7.9, 0.56, facecolor='#bfe2c7', edgecolor='#4c8c5a', lw=0.7, alpha=0.95))
    ax.add_patch(Rectangle((x-0.22,1.15), 0.44, 7.5, facecolor='#cbc2e9', edgecolor='#6e59a5', lw=0.7, alpha=0.95))
    ax.add_patch(Rectangle((x-0.22,y-0.28), 0.44, 0.56, facecolor='#f4b183', edgecolor='#9a4d00', lw=0.7, alpha=0.95))
    ax.text(x+0.38, y+0.38, rf'$P_{{{idx}}}$', fontsize=6.7, color='#9a4d00')
ax.text(5.0, 8.82, '(b) mixed logical representatives', ha='center', va='top', fontsize=8.2, fontweight='bold')
ax.text(5.0, 8.15, r'$P_i=X_i(u)Z_i(v)$ from the $i$th $X$- and $Z$-slabs', ha='center', va='top', fontsize=7.1)
ax.text(5.0, 0.52, r'each qubit lies in at most one $X_i$ and one $Z_i$', ha='center', va='bottom', fontsize=7.0)
ax.text(5.0, 0.16, r'$\Rightarrow$ each qubit is used by at most two $P_i$ ($2$-disjoint)', ha='center', va='bottom', fontsize=7.3, color='#7c2d12')

# Panel C: theorem chain to Clifford-only shallow circuits
ax = fig.add_subplot(gs[0,2])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.text(0.7, 9.25, '(c) structural consequence', fontsize=8.4, fontweight='bold')
items = [
    (8.05, 6.2, 1.05, r'$\Delta(Q_N)=\Theta(N)$', '#e8f5e9', '#2e7d32'),
    (5.95, 6.2, 1.05, r'$d_\uparrow=d_\downarrow=\Theta(N^2)$', '#e3f2fd', '#1565c0'),
    (3.85, 7.0, 1.12, 'for fixed gate arity $q$\nand depth $h$:\n$ d_\\uparrow q^{3h}/(d_\\downarrow\\Delta)=O(N^{-1})$', '#fff8e1', '#a16207'),
    (1.55, 7.4, 1.25, 'all sufficiently large\nstatic blocks admit only\nlogical Clifford unitaries\nat constant depth', '#fce7f3', '#ad1457'),
]
for y,w,h,txt,fc,ec in items:
    patch = FancyBboxPatch((5-w/2, y-h/2), w, h, boxstyle='round,pad=0.03,rounding_size=0.08', facecolor=fc, edgecolor=ec, lw=1.2)
    ax.add_patch(patch)
    ax.text(5, y, txt, ha='center', va='center', fontsize=7.0)
for y1,y2 in [(7.45,6.52),(5.40,4.42),(3.14,2.22)]:
    ax.annotate('', xy=(5,y2), xytext=(5,y1), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='#475569'))
ax.text(5, 0.42, 'JKY disjointness theorem (no geometric locality assumption)', ha='center', va='bottom', fontsize=6.9, color='#475569')

fig.suptitle(r'Uniform slabs $\Rightarrow$ growing disjointness $\Rightarrow$ shallow native unitaries are Clifford', fontsize=10.2, y=0.99, fontweight='bold')
fig.subplots_adjust(left=0.03, right=0.99, bottom=0.02, top=0.88, wspace=0.22)
fig.savefig(OUT/'slab_disjointness_clifford.pdf')
fig.savefig(OUT/'slab_disjointness_clifford.png', dpi=240)
plt.close(fig)
