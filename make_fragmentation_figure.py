#!/usr/bin/env python3
"""Regenerate figures/fragmentation_Bf.{pdf,png} (Supplemental Fig. S5).

Coarse boxes have side b = Theta(f^{1/6}).  The labels give the scaling of the
number B_f(A) of partially filled boxes at fixed logical volume W:
  filled bounded-aspect cube : O(W^{2/3}/b^2 + 1)
  thin slab of thickness h<b : Theta(W/(h b^2))
  isolated sites, spacing >= b: exactly W
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / 'figures'
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans',
                     'font.size': 11, 'mathtext.fontset': 'dejavusans', 'figure.dpi': 180,
                     'savefig.bbox': 'tight'})

fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.35))
configs = [('Filled cube', r'$B_f(A)=O(W^{2/3}/b^{2}+1)$'),
           (r'Thin slab, thickness $h<b$', r'$B_f(A)=\Theta(W/(hb^{2}))$'),
           ('Isolated sites', r'$B_f(A)=W$')]
for ax, (title, sub) in zip(axes, configs):
    ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title + '\n' + sub, fontsize=11.2, fontweight='bold', pad=5)
for i in range(3):
    for j in range(3):
        axes[0].add_patch(Rectangle((.55 + i * .9, .55 + j * .9), .78, .78,
                                    facecolor='#79ad64', edgecolor='#333', lw=.7))
axes[0].text(2, .15, 'few partial coarse cells', ha='center', fontsize=9.2)
for i in range(9):
    axes[1].add_patch(Rectangle((.22 + i * .4, 1.55), .34, .72, facecolor='#f6b26b',
                                edgecolor='#333', lw=.6))
axes[1].text(2, .15, r'same $W$, every cell partial', ha='center', fontsize=9.2)
for x, y in [(0.35, .55), (.35, 2.8), (1.5, 1.65), (2.85, .45), (2.85, 2.85), (1.5, 3.0),
             (.95, .45), (2.2, 1.55), (3.2, 1.65)]:
    axes[2].add_patch(Rectangle((x, y), .53, .53, facecolor='#e77a7a', edgecolor='#333', lw=.6))
axes[2].text(2, .15, r'same $W$, one partial cell per site', ha='center', fontsize=9.2)
fig.suptitle(r'At fixed logical volume $W$, fragmentation $B_f(A)$ controls the boundary-resource penalty',
             fontsize=13, fontweight='bold', y=.99)
fig.subplots_adjust(left=.02, right=.98, bottom=.10, top=.78, wspace=.18)
fig.savefig(OUT / 'fragmentation_Bf.pdf')
fig.savefig(OUT / 'fragmentation_Bf.png', dpi=320)
plt.close(fig)
print('wrote figures/fragmentation_Bf.pdf and .png')
