"""Boxplots from the fixed SI fold results; no modelling is rerun."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/modeling/fold_metrics'
OUT = ROOT / 'analysis_outputs/reproduction/figures/modeling'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':10,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.7,
 'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
colors=['#91B8D5','#A3C3AF','#DAB98E']
def make(frame,key,groups,labels,stem):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.2),layout='constrained')
    for i,(ax,metric,ylabel) in enumerate(zip(axes.flat,['R2','RMSE','MAE','rho'],
                       [r'$R^2$','RMSE (log$_{10}$ units)','MAE (log$_{10}$ units)',r'Spearman $\rho$'])):
        vals=[frame.loc[frame[key].eq(g),metric].to_numpy() for g in groups]
        assert all(len(v)==5 for v in vals)
        bp=ax.boxplot(vals,positions=[1,2,3],widths=.48,patch_artist=True,
            showfliers=False,medianprops={'color':'#222','linewidth':1.3},
            boxprops={'linewidth':.8},whiskerprops={'linewidth':.8},capprops={'linewidth':.8})
        for b,c in zip(bp['boxes'],colors):b.set_facecolor(c);b.set_alpha(.6)
        for j,(v,c) in enumerate(zip(vals,colors),1):
            ax.scatter(j+np.array([-.11,-.055,0,.055,.11]),v,s=25,c=c,
                       edgecolors='#333',linewidths=.45,zorder=3)
        if metric in ['R2','rho']:ax.axhline(0,color='#999',ls='--',lw=.7,zorder=0)
        ax.set_xticks([1,2,3],labels);ax.set_ylabel(ylabel);ax.set_xlim(.5,3.5)
        ax.text(-.17,1.06,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=12)
        ax.margins(y=.16)
    for ext in ['png','pdf','svg','tiff']:
        kwargs={'dpi':600} if ext in ['png','tiff'] else {}
        if ext=='tiff':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(OUT/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.08,**kwargs)
    plt.close(fig)
fixed=pd.read_csv(DATA / 'fixed_ridge_plot_points.csv')
nested=pd.read_csv(DATA / 'nested_plot_points.csv')
make(fixed,'split',['Exact sequence','80% cluster','Substrate'],
     ['Exact\nsequence','80% sequence\ncluster','Substrate'],'Figure_S2_fixed_holdout_boxplots')
make(nested,'target',['kcat','kcat/Km','Km'],
     [r'$k_{cat}$',r'$k_{cat}/K_m$',r'$K_m$'],'Figure_S3_nested_boxplots')
print('Figures S2 and S3 reproduced from unchanged five-fold source values.')
