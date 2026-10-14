#!/usr/bin/env python3
"""Generate a numbering-agnostic dependency map for the FT theorem."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
NEW = "#dcefdc"; IMPORTED = "#dce9f8"; SUPPORT = "#f8e7b8"
EDGE = "#3f4752"; ARROW = "#59616b"; TEXT = "#171b21"

def box(ax,x,y,w,h,text,fc,fs=7.0,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.018,rounding_size=0.05",
                                facecolor=fc,edgecolor=EDGE,linewidth=1.0))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,
            fontweight="bold" if bold else "normal",linespacing=1.10,color=TEXT)

def arrow(ax,a,b):
    ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.15,color=ARROW))

def main():
    fig,ax=plt.subplots(figsize=(7.3,4.25))
    ax.set_xlim(0,12); ax.set_ylim(0,7.3); ax.axis("off")
    ax.text(6,6.98,"Dependency chain for the universal fault-tolerance theorem",
            ha="center",va="center",fontsize=11,fontweight="bold")

    w,h=2.30,0.94; xs=[0.25,3.05,5.85,8.65]; yt=4.75; yb=2.58
    top=[
        (xs[0],"Source-derived boundary\nand incidence inputs"),
        (xs[1],"Constant-width\nslab theorem"),
        (xs[2],"Local 3D gauging\nand static distance"),
        (xs[3],"Pure-X/Z compilation\nand resource maps"),
    ]
    bottom=[
        (xs[3],"Local detector catalog\nand boundary gluing"),
        (xs[2],"Spacetime cleaning\nand rectangle distance"),
        (xs[1],"Padded composition\nand encoded courier"),
        (xs[0],"Universal resource bound\nand positive threshold"),
    ]
    for x,t in top: box(ax,x,yt,w,h,t,NEW,7.0,True)
    for x,t in bottom: box(ax,x,yb,w,h,t,NEW,7.0,True)
    for (x,_),(x2,_) in zip(top,top[1:]): arrow(ax,(x+w,yt+h/2),(x2,yt+h/2))
    arrow(ax,(xs[3]+w/2,yt),(xs[3]+w/2,yb+h))
    for (x,_),(x2,_) in zip(bottom,bottom[1:]): arrow(ax,(x,yb+h/2),(x2+w,yb+h/2))

    # Cited inputs aligned directly with the step they support.
    box(ax,0.34,6.05,2.12,0.50,"Williamson--Baspin local defect cells",IMPORTED,6.1)
    arrow(ax,(1.40,6.05),(xs[0]+w/2,yt+h))
    box(ax,5.95,6.05,2.08,0.50,"Grid-gauging / expansion input",IMPORTED,6.1)
    arrow(ax,(6.99,6.05),(xs[2]+w/2,yt+h))

    box(ax,5.95,1.46,2.08,0.50,"Williamson--Yoder cleaning theorem",IMPORTED,6.0)
    arrow(ax,(6.99,1.96),(xs[2]+w/2,yb))
    box(ax,0.34,1.46,2.42,0.50,"Balanced good CSS family + noise model",IMPORTED,5.95)
    arrow(ax,(1.55,1.96),(xs[0]+w/2,yb))

    box(ax,8.62,1.40,2.38,0.62,"Finite transcription / circuit checks\n(scripts + CSV ledgers)",SUPPORT,6.1)
    arrow(ax,(9.81,2.02),(xs[3]+w/2,yb))

    box(ax,3.13,0.47,1.42,0.44,"PROVED HERE",NEW,6.2,True)
    box(ax,4.75,0.47,1.42,0.44,"CITED INPUT",IMPORTED,6.2,True)
    box(ax,6.37,0.47,1.78,0.44,"SUPPORTING CHECK",SUPPORT,6.0,True)
    ax.text(6,0.12,"Finite numerical crossings test selected models; they are not inputs to the theorem.",
            ha="center",va="center",fontsize=6.3,color="#424950")

    fig.tight_layout(pad=0.12)
    for stem in ("ft_proof_dependency","ft_dependency","proof_dependency"):
        fig.savefig(OUT/f"{stem}.pdf",bbox_inches="tight")
        fig.savefig(OUT/f"{stem}.png",dpi=240,bbox_inches="tight")
    plt.close(fig)
    print("wrote dependency figure and aliases")
if __name__=="__main__": main()
