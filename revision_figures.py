"""Publication figures. Numerical plots use the archived counts without simulation.
Schematics and resource envelopes are explanatory, not additional research data.
"""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon, FancyBboxPatch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
 'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
 'mathtext.fontset':'dejavusans','lines.linewidth':1.25})
INK='#202a35'; BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'; PURPLE='#7759A6'; GRAY='#75818D'
COLORS=[BLUE,ORANGE,GREEN,PURPLE,'#404040']
MARKERS=['o','s','^','D','v']

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.035)
    fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)

def blank(ax):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')

def arrow(ax,a,b,color=INK,lw=1.2):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=lw,color=color))

def box(ax,x,y,w,h,title,sub='',color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.01',
      ec=color,fc=color+'0D',lw=1))
    ax.text(x+w/2,y+h*.65 if sub else y+h/2,title,ha='center',va='center',fontsize=8,color=INK)
    if sub: ax.text(x+w/2,y+h*.27,sub,ha='center',va='center',fontsize=7,color=INK)

def cube(ax,origin=(.1,.2),scale=.6):
    o=np.array(origin); u=np.array([.68,0])*scale;v=np.array([.25,.21])*scale;w=np.array([0,.62])*scale
    pts=[o,o+u,o+u+v,o+v,o+w,o+u+w,o+u+v+w,o+v+w]
    for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot(*np.array([pts[a],pts[b]]).T,color='#83909B',lw=.8,zorder=1)
    z=o+.49*w
    ax.add_patch(Polygon([z,z+u,z+u+v,z+v],closed=True,fc=BLUE+'30',ec=BLUE,lw=1.2,zorder=2))
    x=o+.55*u
    ax.add_patch(Polygon([x,x+v,x+v+w,x+w],closed=True,fc=ORANGE+'25',ec=ORANGE,lw=1.2,zorder=3))
    line=z+.55*u
    ax.plot(*np.array([line,line+v]).T,color=PURPLE,lw=3,zorder=5)
    return o,u,v,w

fig=plt.figure(figsize=(7.05,3.05))
gs=fig.add_gridspec(2,2,width_ratios=[1,1.65],height_ratios=[1,.72],wspace=.12,hspace=.32)
ax=fig.add_subplot(gs[:,0]);blank(ax)
ax.text(0,1,'(a) Local logical access',weight='bold',va='top')
cube(ax,(.17,.36),.66)
ax.text(.10,.29,r'$X$ slab',color=BLUE);ax.text(.66,.29,r'$Z$ slab',color=ORANGE)
box(ax,.10,.05,.75,.15,'Grid ancilla',r'Pauli parity $\longrightarrow$ frame update',GREEN)
arrow(ax,(.49,.30),(.49,.22),GREEN)
ax=fig.add_subplot(gs[0,1]);blank(ax)
ax.text(0,1,'(b) Encoded resource supply',weight='bold',va='top')
xs=[.012,.275,.539,.805];ws=[.17,.19,.18,.18]
titles=['Inject','Transfer','Distill','Consume'];subs=['bare port',r'$d_0<d_1<\cdots$','fixed codes','data port']
for i,(x,w,t,sub) in enumerate(zip(xs,ws,titles,subs)):
    box(ax,x,.37,w,.36,t,sub,color=ORANGE if i==0 else GREEN)
    if i<3: arrow(ax,(x+w+.006,.55),(xs[i+1]-.012,.55))
ax.text(.50,.12,'Encoded blocks after raw injection',ha='center',fontsize=7.5,color=GREEN)
ax=fig.add_subplot(gs[1,1]);blank(ax)
ax.text(0,1,'(c) Protected measurement schedule',weight='bold',va='top')
for x,w,t,col in [(.015,.26,'Gauging',BLUE),(.29,.26,'Idle padding',GRAY),(.565,.26,'Gauging',BLUE)]:
    box(ax,x,.30,w,.38,t,color=col)
arrow(ax,(.02,.17),(.97,.17));ax.text(.97,.12,'time',ha='right',va='top',fontsize=7)
ax.text(.88,.49,r'$\cdots$',ha='center',fontsize=13)
save(fig,'revision_architecture')

fig,axs=plt.subplots(1,2,figsize=(7.05,2.50),gridspec_kw={'width_ratios':[1,1.65]})
ax=axs[0];blank(ax);ax.text(0,1,'(a) Transverse slab ports',weight='bold',va='top')
o,u,v,w=cube(ax,(.12,.22),.77)
for zf in [.25,.73]:
    z=o+zf*w; ax.plot(*np.array([z,z+u,z+u+v,z+v,z]).T,color=BLUE,ls=':',lw=.8)
ax.text(.04,.11,r'$|X\mathrm{\ slab}|,|Z\mathrm{\ slab}|=O(N^2)$',fontsize=7.5)
ax.text(.04,.01,r'$|X\cap Z|=O(N)$',fontsize=7.5,color=PURPLE)
ax=axs[1];blank(ax);ax.text(0,1,'(b) Prefix parity repairs one check',weight='bold',va='top')
pos=[.12,.36,.60,.84];vals=[1,0,1,0]
ax.plot([.035,.93],[.43,.43],ls=':',c=GRAY,lw=1)
ax.plot([pos[0],pos[2]],[.43,.43],c=GREEN,lw=3)
for i,(x,val) in enumerate(zip(pos,vals)):
    ax.plot([x,x],[.43,.74],c=ORANGE if val else GRAY,lw=2 if val else 1,ls='-' if val else ':')
    ax.plot(x,.43,'o',c=INK,ms=3)
    ax.text(x,.82,rf'$x_{i+1}={val}$',ha='center',fontsize=8)
    ax.text(x,.29,['11','110','101','00'][i],ha='center',fontsize=8,color=GREEN if i==1 else INK)
for i,x in enumerate([.04,.24,.48,.72,.94]):
    ax.text(x,.49,rf'$p_{i}={[0,1,1,0,0][i]}$',ha='center',fontsize=7)
ax.add_patch(Rectangle((.30,.24),.12,.40,fc='none',ec=GREEN,lw=1,ls='--'))
ax.text(.36,.13,'pass-through',ha='center',fontsize=7.5,color=GREEN)
ax.text(.53,.01,r'$p_{i-1}+p_i+x_i=0\quad (\mathrm{mod}\ 2)$',ha='center',fontsize=9)
fig.subplots_adjust(wspace=.22)
save(fig,'revision_slab_prefix')

fig,axs=plt.subplots(1,2,figsize=(7.05,2.70),gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0];blank(ax);ax.text(0,1,'(a) From fault distance to space',weight='bold',va='top')
box(ax,.06,.72,.88,.16,r'$d=\Theta(f),\quad N=\Theta(\sqrt{f})$',color=BLUE)
arrow(ax,(.50,.70),(.50,.59))
box(ax,.06,.40,.88,.18,r'$K=\Theta(\sqrt{f}),\quad n=\Theta(f^{3/2})$',color=GREEN)
arrow(ax,(.50,.39),(.50,.28))
box(ax,.01,.055,.98,.21,r'$Q=O\!\left(Wf+(B_f+1)f^{3/2}\right)$',color=PURPLE)
ax=axs[1]
x=np.logspace(-2,2,250)
ax.axvspan(.01,1,color=BLUE,alpha=.045)
ax.loglog(x,1+x,c=INK,label=r'Envelope $1+x$')
ax.loglog(x,x,c=GREEN,ls='--',label=r'Extensive term $x$')
ax.axhline(1,c=BLUE,ls=':',lw=1.3,label='One-cell term 1')
ax.axvline(1,c=GRAY,lw=.7,ls=':')
ax.set(xlim=(.01,100),ylim=(.008,180),xlabel=r'Logical occupancy $x=W/\sqrt{f}$',ylabel=r'Space envelope / $f^{3/2}$')
ax.set_title('(b) Bounded-aspect logical box',loc='left',weight='bold',pad=10)
ax.legend(loc='upper left',frameon=False,fontsize=6.5)
fig.subplots_adjust(left=.025,right=.985,bottom=.19,top=.88,wspace=.35)
save(fig,'revision_resources')

# Replace the original disjointness schematic with legible vector artwork.
# Write the two Theta bounds separately to avoid implying exact equality of
# minimum and maximum logical weights.
fig,axs=plt.subplots(1,3,figsize=(7.05,2.15))
for ax in axs:blank(ax)
axs[0].text(0,1,'(a) Complete slab families',weight='bold',va='top')
for i in range(4):
    axs[0].add_patch(Rectangle((.08,.18+i*.15),.81,.07,ec=BLUE,fc=BLUE+'25',lw=.6))
    axs[0].add_patch(Rectangle((.16+i*.18,.12),.06,.69,ec=ORANGE,fc=ORANGE+'25',lw=.6))
axs[0].text(.48,.01,r'$r_X,r_Z=\Theta(N)$',ha='center')
axs[1].text(0,1,'(b) Mixed logical classes',weight='bold',va='top')
box(axs[1],.04,.54,.91,.25,r'$P_i=X_i(u)Z_i(v)$',color=GREEN)
axs[1].text(.50,.35,'At most two supports\nper physical qubit',ha='center',va='center',fontsize=8)
axs[1].text(.50,.08,r'$\Delta\geq\frac{1}{2}\min\{r_X,r_Z\}$',ha='center',fontsize=10)
axs[2].text(0,1,'(c) Static unitary restriction',weight='bold',va='top')
axs[2].text(.50,.76,r'$\Delta=\Theta(N)$',ha='center')
axs[2].text(.50,.59,r'$d_\uparrow,d_\downarrow=\Theta(N^2)$',ha='center')
arrow(axs[2],(.50,.50),(.50,.34))
box(axs[2],.06,.06,.88,.27,'Logical Clifford', 'fixed depth and gate arity',PURPLE)
fig.subplots_adjust(wspace=.22,left=.02,right=.98,top=.96,bottom=.06)
save(fig,'slab_disjointness_clifford')

DATA=ROOT/'numerical/continuous_transition/threshold_estimate/all_points.csv'
rows=list(csv.DictReader(DATA.open()))
stats=[]
for r in rows:
    for k in ['p','logical_error_rate','wilson_lo','wilson_hi']:r[k]=float(r[k])
    for k in ['size','gauge_scale','shots','failures']:r[k]=int(float(r[k]))
    assert abs(r['logical_error_rate']-r['failures']/r['shots'])<1e-12
assert len(rows)==255 and sum(r['shots'] for r in rows)==4534000
assert sum(r['failures'] for r in rows if r['p']==0)==0
cross=[(4.471,4.426,4.522),(4.431,4.392,4.466),(4.328,4.265,4.393)]
fig,axs=plt.subplots(1,3,figsize=(7.05,2.75),sharey=True)
for G,ax in enumerate(axs,1):
    med,lo,hi=cross[G-1];ax.axvspan(lo,hi,color=GRAY,alpha=.16,lw=0)
    ax.axvline(med,color=GRAY,ls='--',lw=.9)
    for L,c,mk in zip([3,5,7,9,11],COLORS,MARKERS):
        rr=sorted([r for r in rows if r['gauge_scale']==G and r['size']==L and r['p']>0],key=lambda r:r['p'])
        p=np.array([r['p'] for r in rr])*1000; y=np.array([r['logical_error_rate'] for r in rr])
        low=np.array([r['wilson_lo'] for r in rr]);high=np.array([r['wilson_hi'] for r in rr])
        ax.errorbar(p,y,yerr=[np.maximum(0,y-low),np.maximum(0,high-y)],color=c,marker=mk,ms=2.4,
                    capsize=1.2,elinewidth=.65,lw=.9,label=f'$L={L}$')
    ax.set(yscale='log',xlim=(1.3,6.65),ylim=(1e-5,.9),xlabel=r'Noise strength $p\ (10^{-3})$')
    ax.set_xticks([2,3,4,5,6]);ax.grid(axis='y',alpha=.13,which='major')
    ax.set_title(f'({chr(96+G)}) $G='+('d' if G==1 else str(G)+'d')+'$',loc='left',weight='bold')
axs[0].set_ylabel('Logical failure probability')
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.53,1.015),handlelength=1.3,columnspacing=1.5)
fig.subplots_adjust(left=.085,right=.99,bottom=.20,top=.82,wspace=.13)
save(fig,'revision_numerics')

# Redraw the supplementary numerical panels individually at an appropriate
# final-column font size, preserving the same source data and intervals.
for G in [1,2,3]:
    fig,ax=plt.subplots(figsize=(3.35,2.65))
    med,lo,hi=cross[G-1];ax.axvspan(lo,hi,color=GRAY,alpha=.16,lw=0);ax.axvline(med,color=GRAY,ls='--',lw=.9)
    for L,c,mk in zip([3,5,7,9,11],COLORS,MARKERS):
        rr=sorted([r for r in rows if r['gauge_scale']==G and r['size']==L and r['p']>0],key=lambda r:r['p'])
        y=np.array([r['logical_error_rate'] for r in rr]);lo2=np.array([r['wilson_lo'] for r in rr]);hi2=np.array([r['wilson_hi'] for r in rr])
        ax.errorbar([1000*r['p'] for r in rr],y,yerr=[np.maximum(0,y-lo2),np.maximum(0,hi2-y)],c=c,marker=mk,ms=2.5,capsize=1.2,lw=.9,label=f'$L={L}$')
    ax.set(xlabel=r'Noise strength $p\ (10^{-3})$',ylabel='Logical failure probability',yscale='log',ylim=(1e-5,.9),xlim=(1.3,6.65))
    ax.set_title('$G='+('d' if G==1 else str(G)+'d')+'$',loc='left')
    ax.legend(frameon=False,fontsize=6,ncol=2,loc='lower right');ax.grid(axis='y',alpha=.13)
    fig.tight_layout(pad=.4);save(fig,f'threshold_G{G}d')
report={'points':len(rows),'shots':sum(r['shots'] for r in rows),'noiseless_points':sum(r['p']==0 for r in rows),
 'noiseless_failures':sum(r['failures'] for r in rows if r['p']==0),
 'all_rates_match_counts':True,'numerical_source':str(DATA.relative_to(ROOT)),
 'crossings':'Archived rounded Jeffreys estimates; no new Monte Carlo runs.'}
(ROOT/'revision_data_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

def readcsv(path):
    return list(csv.DictReader((ROOT/path).open()))

def curve(ax,rr,color,marker,label,scale=1):
    rr=sorted([r for r in rr if float(r['p'])>0],key=lambda r:float(r['p']))
    x=np.array([float(r['p'])*scale for r in rr]);y=np.array([float(r['logical_error_rate']) for r in rr])
    lo=np.array([float(r['wilson_lo']) for r in rr]);hi=np.array([float(r['wilson_hi']) for r in rr])
    positive=y>0
    # Do not replace zero observations by arbitrary positive pseudocounts.
    ax.errorbar(x,np.where(positive,y,np.nan),yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],
                color=color,marker=marker,ms=3,lw=1,capsize=1.4,elinewidth=.7,label=label)
    if (~positive).any():ax.scatter(x[~positive],hi[~positive],marker='v',s=16,facecolors='none',edgecolors=color,zorder=5)

def finish(ax,title,xlabel):
    ax.set_title(title,loc='left',fontsize=9,pad=8)
    ax.set(xlabel=xlabel,ylabel='Logical failure probability',yscale='log')
    ax.grid(axis='y',alpha=.13)

for fn,key,title,xlab in [('phenomenological_mw_demo','distance','Toric memory control',r'Noise probability $p$'),
                         ('gauging_rectangle_mwpm','d','Gauging-strip control',r'Data-noise probability $p$')]:
    rr=readcsv(fn+'.csv');fig,ax=plt.subplots(figsize=(3.35,2.65))
    for i,d in enumerate(sorted({int(r[key]) for r in rr})):
        curve(ax,[r for r in rr if int(r[key])==d],COLORS[i],MARKERS[i],f'$d={d}$')
    finish(ax,title,xlab);ax.legend(frameon=False,loc='lower right')
    fig.tight_layout(pad=.45);save(fig,fn)

rr=readcsv('wb_layer_circuit_dem.csv');fig,ax=plt.subplots(figsize=(3.35,2.65))
for i,L in enumerate(sorted({int(r['L']) for r in rr})):
    curve(ax,[r for r in rr if int(r['L'])==L],COLORS[i],MARKERS[i],f'$L={L}$',1000)
finish(ax,'Truncated-WB circuit control',r'Noise strength $p\ (10^{-3})$');ax.legend(frameon=False,ncol=2,fontsize=6.5)
fig.tight_layout(pad=.45);save(fig,'final_circuit_level_arch_si1000_wb')

rr=readcsv('magic_factory_rm15_si1000.csv');fig,ax=plt.subplots(figsize=(3.35,3.25))
names={'Steane7_code_capacity_LUT':'Steane: capacity',
 'Steane7_project_Sdag_LUT':'Steane: circuit',
 'RM15_allones_code_capacity_LUT':'RM15: capacity',
 'RM15_project_Sdag_LUT':'RM15: Clifford probe',
 'MultiLevel_Steane_then_RM15_capacity':'Steane / RM15 stack',
 'Factory_plus_CAS_consume_d3_proxy':'Factory + consume proxy'}
for i,(key,label) in enumerate(names.items()):
    curve(ax,[r for r in rr if r['factory']==key],(COLORS+["#B57B00"])[i],(MARKERS+['P'])[i],label,1000)
finish(ax,'Finite factory controls',r'Noise strength $p\ (10^{-3})$')
ax.legend(frameon=False,fontsize=6.1,ncol=2,loc='upper center',bbox_to_anchor=(.45,-.32),columnspacing=.8,handlelength=1.2)
fig.subplots_adjust(left=.18,right=.98,top=.89,bottom=.40)
save(fig,'final_magic_factory_rm15_si1000')

rr=readcsv('numerical/stage_composed/all_points.csv')
fig,axs=plt.subplots(1,2,figsize=(7.05,2.75),sharey=True)
for R,ax in enumerate(axs,1):
    for i,L in enumerate([3,5,7,9,11]):
        curve(ax,[r for r in rr if r['experiment']=='hp_cas' and int(r['rounds_scale'])==R and int(r['size'])==L],COLORS[i],MARKERS[i],f'$L={L}$',1000)
    finish(ax,'(a) $R=d$' if R==1 else '(b) $R=2d$',r'Noise strength $p\ (10^{-3})$')
axs[1].set_ylabel('');axs[1].legend(frameon=False,fontsize=6.5,ncol=2,loc='lower right')
fig.tight_layout(pad=.5,w_pad=1);save(fig,'supp_heavy_stage_composed')

fig,axs=plt.subplots(1,2,figsize=(7.05,2.75))
memory=readcsv('wb_bposd_threshold.csv')
for i,L in enumerate([3,4,5,6]):
    curve(axs[0],[r for r in memory if int(r['L'])==L and r['noise_model']=='code_capacity_independent_XZ'],COLORS[i],MARKERS[i],f'$L={L}$')
for i,L in enumerate([3,5,7,9,11]):
    curve(axs[1],[r for r in rr if r['experiment']=='primitive' and int(r['size'])==L],COLORS[i],MARKERS[i],f'$d={L}$',1000)
finish(axs[0],'(a) WB-incidence memory',r'Data-noise probability $p$')
finish(axs[1],'(b) Circuit-level gauging',r'Noise strength $p\ (10^{-3})$')
for ax in axs:ax.legend(frameon=False,fontsize=6.5,ncol=2,loc='lower right')
fig.tight_layout(pad=.5,w_pad=1.1);save(fig,'fig10_building_blocks')
