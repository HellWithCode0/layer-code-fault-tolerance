import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.2,'mathtext.fontset':'dejavusans','pdf.fonttype':42})
fig=plt.figure(figsize=(11.8,4.25))
axs=[fig.add_axes([.03,.12,.29,.80]),fig.add_axes([.355,.12,.29,.80]),fig.add_axes([.68,.12,.29,.80])]
blue='#4C78A8'; red='#D65F5F'; purple='#8F63B8'; dark='#222'; gray='#666'; light='#F2F4F5'
# A
ax=axs[0]; ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis('off')
ax.text(.15,7.75,'(a)',fontsize=13,fontweight='bold',va='top'); ax.text(5.2,7.75,'Overlap-corrected cleaning',ha='center',va='top',fontsize=12,fontweight='bold')
A=FancyBboxPatch((1.2,2.7),7.0,2.2,boxstyle='round,pad=.03,rounding_size=.10',facecolor=blue,edgecolor=dark,alpha=.38,lw=1.1)
B=FancyBboxPatch((3.75,1.0),2.2,5.8,boxstyle='round,pad=.03,rounding_size=.10',facecolor=red,edgecolor=dark,alpha=.34,lw=1.1)
C=Rectangle((3.75,2.7),2.2,2.2,facecolor=purple,edgecolor=dark,alpha=.62,lw=1.0)
for p in [A,B,C]: ax.add_patch(p)
ax.text(1.7,5.25,r'$A:\ g_X(A)$',fontsize=12,fontweight='bold')
ax.text(6.15,6.20,r'$B:\ g_Z(B)$',fontsize=12,fontweight='bold')
ax.text(4.85,3.80,r'$A\cap B$',fontsize=12,ha='center',va='center',color='white',fontweight='bold')
ax.annotate(r'$r(A,B)$',xy=(4.85,3.75),xytext=(1.2,1.20),arrowprops=dict(arrowstyle='-',lw=1.0,color=gray),fontsize=11)
ax.text(5.0,.30,r'$g_X(A)+g_Z(B)\leq K+|A\cap B|$',ha='center',fontsize=16)
# B
ax=axs[1]; ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis('off')
ax.text(.15,7.75,'(b)',fontsize=13,fontweight='bold',va='top'); ax.text(5.1,7.75,'Layer-code transverse slabs',ha='center',va='top',fontsize=12,fontweight='bold')
front=np.array([[1.9,1.35],[6.8,1.35],[6.8,6.15],[1.9,6.15]]); shift=np.array([1.25,.9]); back=front+shift
for seg in [(front[0],front[1]),(front[1],front[2]),(front[2],front[3]),(front[3],front[0]),(back[0],back[1]),(back[1],back[2]),(back[2],back[3]),(back[3],back[0]),(front[0],back[0]),(front[1],back[1]),(front[2],back[2]),(front[3],back[3])]: ax.plot([seg[0][0],seg[1][0]],[seg[0][1],seg[1][1]],color=dark,lw=1.0)
hs=np.array([[1.9,3.35],[6.8,3.35],[8.05,4.25],[3.15,4.25]]); vs=np.array([[4.05,1.35],[5.45,1.35],[6.70,2.25],[5.30,2.25],[5.30,7.05],[4.05,6.15]])
ax.add_patch(Polygon(hs,closed=True,facecolor=blue,edgecolor=dark,alpha=.32,lw=1)); ax.add_patch(Polygon(vs,closed=True,facecolor=red,edgecolor=dark,alpha=.30,lw=1))
it=np.array([[4.05,3.35],[5.30,3.35],[5.30,4.65],[4.05,4.65]]); top=np.array([[4.05,4.65],[5.30,4.65],[6.55,5.55],[5.30,5.55]])
ax.add_patch(Polygon(it,closed=True,facecolor=purple,edgecolor=dark,alpha=.62,lw=1)); ax.add_patch(Polygon(top,closed=True,facecolor=purple,edgecolor=dark,alpha=.55,lw=1))
ax.text(2.3,4.65,r'$X$ slab',fontsize=11.5); ax.text(6.05,1.95,r'$Z$ slab',fontsize=11.5)
ax.text(7.55,5.70,r'$|A|,|B|=\Theta(d)$',fontsize=11.5)
ax.text(6.75,3.80,r'$|A\cap B|=\Theta(K)$',fontsize=11.5)
# C
ax=axs[2]; ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis('off')
ax.text(.15,7.75,'(c)',fontsize=13,fontweight='bold',va='top'); ax.text(5.0,7.75,'Structural consequence',ha='center',va='top',fontsize=12,fontweight='bold')
items=[(5.8,r'$\chi_{XZ}(Q_N)=\Theta(K)=\Theta(n/d)$'),(3.95,r'slab supports: $\Theta(d)$ each'),(2.10,r'intersection: $\Theta(K)$')]
for y,t in items:
    ax.add_patch(FancyBboxPatch((1.0,y),8.0,1.15,boxstyle='round,pad=.05',facecolor=light,edgecolor=dark,lw=1.1)); ax.text(5.0,y+.58,t,ha='center',va='center',fontsize=12,fontweight='bold')
for y1,y2 in [(5.78,5.18),(3.93,3.33)]: ax.annotate('',xy=(5,y2),xytext=(5,y1),arrowprops=dict(arrowstyle='->',lw=1.0,color=gray))
ax.text(5.0,.55,'same slabs: fault-tolerant ports + structural no-go witness',ha='center',fontsize=10.5)
fig.savefig('figures/symplectic_overlap_layer_ports.pdf',bbox_inches='tight'); fig.savefig('figures/symplectic_overlap_layer_ports.png',dpi=320,bbox_inches='tight')
