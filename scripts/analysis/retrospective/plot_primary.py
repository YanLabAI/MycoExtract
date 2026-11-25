"""Figure S5 from the complete, unchanged retrospective predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / 'analysis_outputs/reproduction/figures/retrospective'
ART.mkdir(parents=True, exist_ok=True)
data_path = ROOT / 'data/modeling/retrospective/all_predictions.csv'
r = pd.read_csv(data_path)
primary = r[r.status.eq('primary')].copy()
selected = primary[primary.fold_error.le(2)].copy()
assert len(r) == 50 and r.enzyme.nunique() == 22
assert len(primary) == 43 and len(selected) == 7
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'legend.frameon': False})
fig = plt.figure(figsize=(7.2, 3.85), layout='constrained')
gs = fig.add_gridspec(1, 2)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
blue = '#467CA4'; green = '#4C8B74'; gray = '#BABABA'
offsets = {'ZenA_Sl': (-30, -16), 'ZenA_Scfl': (5, 8),
    'ZenA_Nb': (5, -13), 'ZHRnZ_WT': (-56, 13),
    'DONrd': (5, 9), 'ZHDAY3_WT': (8, -17), 'ZHDAY3_N153H': (-82, -31)}
for ax, target, color, label, panel in zip(axes,
        ['kcat_s', 'Km_uM'], [blue, green],
        [r'$k_{cat}$ (s$^{-1}$)', r'$K_m$ ($\mu$M)'], ['a', 'b']):
    q = primary[primary.target.eq(target)]
    v = np.r_[q.observed_log10, q.predicted_log10, np.log10(q.baseline_predicted)]
    lo = np.floor(v.min() * 2) / 2 - .25
    hi = np.ceil(v.max() * 2) / 2 + .25
    ax.fill_between([lo, hi], [lo-np.log10(2), hi-np.log10(2)],
        [lo+np.log10(2), hi+np.log10(2)], color='#EEEEEE', zorder=0)
    ax.plot([lo, hi], [lo, hi], color='#777777', ls='--', lw=.8, zorder=1)
    for row in q.itertuples():
        good = row.fold_error <= 2
        ax.scatter(row.observed_log10, row.predicted_log10,
            s=32 if good else 21, c=color if good else gray,
            edgecolors='#444444', linewidths=.45, zorder=4 if good else 2)
        if good:
            ax.annotate(row.enzyme, (row.observed_log10, row.predicted_log10),
                xytext=offsets[row.enzyme], textcoords='offset points', fontsize=7,
                arrowprops={'arrowstyle': '-', 'lw': .45, 'color': color},
                color=color, zorder=5)
    ax.scatter(q.observed_log10, np.log10(q.baseline_predicted), marker='^',
        s=22, facecolors='none', edgecolors='#969696', linewidths=.7, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel('Reported log$_{10}$ ' + label)
    ax.set_ylabel('Predicted log$_{10}$ ' + label)
    ax.set_title(f'{label}: all {len(q)} primary cases', loc='left', fontsize=9)
    ax.text(-.17, 1.05, panel, transform=ax.transAxes, fontweight='bold', fontsize=11)

for ext in ['png', 'tiff', 'pdf', 'svg']:
    kw = {'dpi': 600} if ext in ['png', 'tiff'] else {}
    if ext == 'tiff': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(ART / f'primary_predictions.{ext}',
        bbox_inches='tight', pad_inches=.08, **kw)
plt.close(fig)
print(selected[['enzyme', 'target', 'fold_error']].to_string(index=False))
