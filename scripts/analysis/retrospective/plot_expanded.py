from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'analysis_outputs/reproduction/figures/retrospective'
OUT.mkdir(parents=True, exist_ok=True)
r = pd.read_csv(ROOT / 'data/modeling/retrospective/all_predictions.csv')
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
fig=plt.figure(figsize=(7.2,6.8),layout='constrained')
gs=fig.add_gridspec(2,2,width_ratios=[1.25,1],hspace=.12,wspace=.08)
ax=fig.add_subplot(gs[:,0]);bx=fig.add_subplot(gs[0,1]);cx=fig.add_subplot(gs[1,1])
colors=['#668FAF','#82AA92','#CCA86F'];targets=['kcat_s','Km_uM','kcat_Km_M_s']
labels=[r'$k_{cat}$',r'$K_m$',r'$k_{cat}/K_m$'];markers=['o','s','D']
names=list(dict.fromkeys(r.enzyme));status=r.groupby('enzyme').status.first().to_dict()
offset=[-.18,0,.18]
for target,color,label,marker,shift in zip(targets,colors,labels,markers,offset):
 sub=r[r.target.eq(target)]
 for isprimary in [True,False]:
  q=sub[sub.status.eq('primary').eq(isprimary)]
  good=q.fold_error.le(2)
  if isprimary:
   for close in [False,True]:
    v=q[good.eq(close)]
    ax.scatter(v.fold_error,[names.index(x)+shift for x in v.enzyme],s=31 if close else 17,marker=marker,
     facecolors=color if close else '#C9C9C9',edgecolors='#333333' if close else '#A9A9A9',linewidths=.7,zorder=4 if close else 2)
  else:
   ax.scatter(q.fold_error,[names.index(x)+shift for x in q.enzyme],s=27,marker=marker,
    facecolors='none',edgecolors=color,linewidths=1,zorder=3)
ax.set_yticks(range(len(names)),[x+(' *' if status[x]!='primary' else '') for x in names],fontsize=6.8)
ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(.8,850);ax.set_xticks([1,2,10,100,500],['1','2','10','100','500'])
ax.axvline(1,color='#444',lw=.7);ax.axvline(2,color='#999',ls=':',lw=.7)
ax.set_xlabel('Prediction error factor');ax.set_title('All 22 candidates; 50 reported parameters',loc='left',fontsize=8)
ax.grid(axis='x',color='#DDD',lw=.4);ax.text(-.43,1.02,'a',transform=ax.transAxes,fontweight='bold',fontsize=10)
ax.legend(handles=[Line2D([],[],marker=m,color=c,ls='none',markersize=4,label=l) for m,c,l in zip(markers,colors,labels)],loc='lower right',fontsize=7)

z=r[r.group.eq('ZenA_2024')&r.target.eq('kcat_s')];lo=-2.8;hi=.75
bx.plot([lo,hi],[lo,hi],color='#888',ls='--',lw=.8)
for row in z.itertuples():
 selected=row.enzyme in ['ZenA_Sl','ZenA_Scfl']
 bx.scatter(row.observed_log10,row.predicted_log10,s=23 if selected else 18,c=colors[0] if selected else '#B8B8B8',edgecolors='#444',lw=.4)
 if selected:
  dy=-14 if row.enzyme=='ZenA_Sl' else 8
  bx.annotate(row.enzyme.replace('ZenA_',''),(row.observed_log10,row.predicted_log10),xytext=(3,dy),textcoords='offset points',fontsize=7)
bx.scatter(z.observed_log10,np.log10(z.baseline_predicted),marker='^',s=23,facecolors='none',edgecolors='#9B9B9B',lw=.7)
bx.set_xlim(lo,hi);bx.set_ylim(lo,hi);bx.set_aspect('equal');bx.set_xlabel('Reported log$_{10}$ $k_{cat}$');bx.set_ylabel('Predicted log$_{10}$ $k_{cat}$')
bx.set_title('All nine ZenA homologues',loc='left',fontsize=8);bx.text(-.22,1.04,'b',transform=bx.transAxes,fontweight='bold',fontsize=10)
bx.text(.02,.96,'Blue: two closest kcat cases\nselected after evaluation',transform=bx.transAxes,va='top',fontsize=6.5)

z=r[r.group.eq('ZHDAY3_2024')]
for j,name in enumerate(['ZHDAY3_WT','ZHDAY3_N153H']):
 q=z[z.enzyme.eq(name)].set_index('target').loc[targets]
 pos=np.arange(3)+(-.1 if j==0 else .1)
 cx.scatter(pos,q.fold_error,s=27,marker=['o','s'][j],c=colors[j],edgecolors='#444',lw=.4,label=['WT','N153H'][j])
 cx.scatter(pos,q.baseline_fold_error,s=25,marker='^',facecolors='none',edgecolors=colors[j],lw=.8)
cx.set_xticks(range(3),labels);cx.set_yscale('log');cx.set_ylim(.85,55);cx.set_yticks([1,2,5,10,20,50],['1','2','5','10','20','50'])
cx.set_ylabel('Prediction error factor');cx.axhline(2,color='#999',ls=':',lw=.7);cx.grid(axis='y',color='#DDD',lw=.4)
cx.set_title('Both ZHDAY3 variants; all three targets',loc='left',fontsize=8);cx.text(-.22,1.04,'c',transform=cx.transAxes,fontweight='bold',fontsize=10)
cx.legend(loc='upper right',fontsize=7)
fig.supxlabel('a: Colored filled markers highlight all seven primary predictions within twofold; gray = remaining primary results.\nOpen triangles in b and c: training mean.  * Sequence/construct caveat; open markers in a. Highlights selected after evaluation.',fontsize=6.7)
for ext in ['png','tiff','pdf','svg']:
 kw={'dpi':600} if ext in ['png','tiff'] else {}
 if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
 fig.savefig(OUT/f'expanded_predictions.{ext}',bbox_inches='tight',pad_inches=.08,**kw)
plt.close(fig)
(OUT/'expanded_figure_caption.md').write_text('''# Figure caption

Expanded retrospective predictions reveal target-dependent agreement. (a) Error factors for all 22 tested enzyme/variant candidates and all 50 available reported parameters. Colors and shapes identify kcat, Km and kcat/Km; an asterisk and open markers indicate a reference-sequence proxy or an unresolved construct effect. (b) All nine measured ZenA homologues for kcat. The two closest cases are highlighted after evaluation; the seven other results are retained. (c) All three targets for the ZHDAY3 wild type and N153H variant acting on alpha-zearalanol. Open triangles in b and c denote the training-mean baseline. Error factor is max(predicted/reported, reported/predicted); one is exact agreement. The dotted twofold guides are descriptive and are not prespecified success criteria. The same fixed Ridge configuration and original training features were used throughout, with no candidate-driven tuning. There are no new biological replicates, inferential tests or prospective experiments. Selected favorable cases do not establish overall predictive accuracy or successful enzyme discovery. Data: all_predictions.csv.
''',encoding='utf8')
print('Figure exported in PNG TIFF PDF SVG.')
