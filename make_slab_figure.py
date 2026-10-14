import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'mathtext.fontset':'dejavusans','pdf.fonttype':42})
fig=plt.figure(figsize=(10.6,4.25)); gs=fig.add_gridspec(1,3,width_ratios=[1.03,1.03,1.08],wspace=.24)
y_levels=[2.0,3.15,4.3,5.45,6.6]; x_levels=[1.9,3.0,4.1,5.2,6.3]
# A
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(0,8.5); ax.set_ylim(0,8.2); ax.axis('off')
ax.text(4.25,7.85,'(a) uniform-gap mobility',ha='center',fontweight='bold',fontsize=12)
ax.add_patch(FancyBboxPatch((.55,.85),7.35,6.55,boxstyle='round,pad=.02',facecolor='#f8fafc',edgecolor='#64748b',lw=1.5))
for y in y_levels: ax.add_patch(Rectangle((1.0,y-.18),6.25,.36,facecolor='#cfe8d5',edgecolor='#4c8c5a',lw=.8))
for x in x_levels: ax.add_patch(Rectangle((x-.14,1.25),.28,5.75,facecolor='#ddd6f4',edgecolor='#6e59a5',lw=.8))
ax.text(4.1,6.95,r'$\Theta(N)$ disjoint $X$-slabs',ha='center',fontsize=11,color='#245c30',fontweight='bold')
ax.text(7.75,4.1,r'$\Theta(N)$ disjoint $Z$-slabs',ha='center',va='center',rotation=270,fontsize=11,color='#4b3a80',fontweight='bold')
ax.text(4.1,.35,'every admissible gap works',ha='center',fontsize=10.0,color='#475569')
# B
ax=fig.add_subplot(gs[0,1]); ax.set_xlim(0,8.5); ax.set_ylim(0,8.2); ax.axis('off')
ax.text(4.25,7.85,'(b) mixed representatives',ha='center',fontweight='bold',fontsize=12)
ax.add_patch(FancyBboxPatch((.55,.85),7.35,6.55,boxstyle='round,pad=.02',facecolor='#f8fafc',edgecolor='#64748b',lw=1.5))
for idx,(y,x) in enumerate(zip(y_levels,x_levels),1):
    ax.add_patch(Rectangle((1.0,y-.18),6.25,.36,facecolor='#c8e2cf',edgecolor='#4c8c5a',lw=.8)); ax.add_patch(Rectangle((x-.14,1.25),.28,5.75,facecolor='#d1c8ef',edgecolor='#6e59a5',lw=.8)); ax.add_patch(Rectangle((x-.14,y-.18),.28,.36,facecolor='#f4b183',edgecolor='#9a4d00',lw=.8)); ax.text(x+.20,y+.14,rf'$P_{idx}$',fontsize=9,color='#9a4d00')
ax.text(4.1,6.95,r'$P_i=X_i(u)Z_i(v)$',ha='center',fontsize=11,fontweight='bold')
ax.text(4.1,.35,r'each qubit is used by at most two $P_i$',ha='center',fontsize=9.8,color='#9a4d00')
# C
ax=fig.add_subplot(gs[0,2]); ax.set_xlim(0,8.5); ax.set_ylim(0,8.2); ax.axis('off')
ax.text(4.25,7.85,'(c) shallow-unitary obstruction',ha='center',fontweight='bold',fontsize=12)
steps=[(6.45,5.3,.95,r'$\Delta(Q_N)=\Theta(N)$','#e8f5e9','#2e7d32',11),(4.90,5.3,.95,r'$d_\uparrow=d_\downarrow=\Theta(N^2)$','#e3f2fd','#1565c0',10.7),(3.20,6.3,1.15,r'$d_\uparrow q^{3h}/(d_\downarrow\Delta)=O(N^{-1})$','#fff8e1','#a16207',10),(1.30,6.5,1.30,'fixed $q,h$  $\Longrightarrow$  logical Clifford only','#fce7f3','#ad1457',10.4)]
for cy,w,h,t,fc,ec,fs in steps:
    ax.add_patch(FancyBboxPatch((4.25-w/2,cy-h/2),w,h,boxstyle='round,pad=.03',facecolor=fc,edgecolor=ec,lw=1.5)); ax.text(4.25,cy,t,ha='center',va='center',fontsize=fs)
for y1,y2 in [(5.95,5.40),(4.40,3.83),(2.62,1.95)]: ax.annotate('',xy=(4.25,y2),xytext=(4.25,y1),arrowprops=dict(arrowstyle='-|>',lw=1.2,color='#475569'))
ax.text(4.25,.30,'JKY theorem; no geometric-locality assumption',ha='center',fontsize=9,color='#475569')
fig.suptitle('Uniform slabs  →  growing disjointness  →  shallow native unitaries are Clifford',fontsize=15,fontweight='bold',y=.99)
fig.subplots_adjust(left=.02,right=.99,bottom=.04,top=.86,wspace=.22)
fig.savefig('figures/slab_disjointness_clifford.pdf'); fig.savefig('figures/slab_disjointness_clifford.png',dpi=320)
