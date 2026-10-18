"""Manuscript-only layouts from frozen Stage A/B statistics; no new inference."""
from pathlib import Path
import json, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/qai-paper-figures')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'rebuilt_figures'
OUT.mkdir(exist_ok=True)
B = json.loads((ROOT/'05_statistics/results/B_statistics.json').read_text())
A = json.loads((ROOT/'05_statistics/results/A_statistics.json').read_text())
assert B['status']=='PASS_STAGE_B_COMPLETE_ALL_K_AND_METHODS' and len(B['contrasts'])==96
assert A['status']=='PASS_K3_COMPLETE_ALL_METHOD_COMPARISON' and len(A['contrasts'])==18
METHODS=('interaction_tabu','shortgpt_bi_adapted','sleb_iterative_adapted','random')
NAMES={'interaction_tabu':'Combination search','shortgpt_bi_adapted':'ShortGPT BI','sleb_iterative_adapted':'SLEB iterative','random':'Random'}
COLORS=dict(zip(METHODS,('#0072B2','#E69F00','#009E73','#CC79A7')))
MARKERS=dict(zip(METHODS,('o','s','^','D')))
MODELS=('qwen','mistral'); TASKS=('piqa','hellaswag','gsm8k')
WORKLOADS=('B1S512T128','B4S2048T128')
DARK='#263746'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9.2,
 'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'legend.fontsize':7.7,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
legend=[Line2D([0],[0],marker=MARKERS[m],linestyle='',color=COLORS[m],label=NAMES[m],markersize=6) for m in METHODS]
def save(fig,name):
 for ext,kw in (('pdf',{}),('svg',{}),('png',{'dpi':320})):
  fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',**kw)
 plt.close(fig)
def rows(model,workload=WORKLOADS[0]):
 return [r for r in B['long_table'] if r['model']==model and r['workload']==workload]
def dense(model,workload=WORKLOADS[0]):
 return next(r for r in rows(model,workload) if r['deleted_layers']==0)

# Three task rows by two model columns at near-native manuscript width.
fig,axes=plt.subplots(3,2,figsize=(6.5,7.6),sharex=True,sharey=True,layout='constrained')
for ti,task in enumerate(TASKS):
 for mi,model in enumerate(MODELS):
  ax=axes[ti,mi]; base=dense(model)[task+'_accuracy']
  ax.axhline(base,color=DARK,ls='--',lw=1)
  ax.scatter([0],[base],marker='P',s=55,color=DARK,zorder=4)
  for mj,method in enumerate(METHODS):
   for sj,seed in enumerate((11,23,37)):
    for r in rows(model):
     if r['method']==method and r['seed']==seed:
      x=r['deleted_layers']+(mj-1.5)*.12+(sj-1)*.026
      ax.scatter(x,r[task+'_accuracy'],color=COLORS[method],marker=MARKERS[method],s=22,
                 alpha=.86,edgecolor='white',linewidth=.25)
  ax.set_title(model.upper()+' · '+task.upper(),pad=4)
  ax.set_xlim(-.28,4.28);ax.set_ylim(0,1);ax.set_xticks((0,1,2,3,4))
  ax.set_yticks((0,.25,.5,.75,1));ax.grid(axis='y',color='#D9E1E8',lw=.5)
  if mi==0:ax.set_ylabel('Question accuracy')
  if ti==2:ax.set_xlabel('Whole layers deleted (k)')
fig.legend(handles=legend+[Line2D([0],[0],marker='P',linestyle='',color=DARK,label='Dense',markersize=6)],
 loc='outside lower center',ncol=3,frameon=False)
save(fig,'paper_figure_1_accuracy_2x3')

# Four panels at manuscript width.
fig,axes=plt.subplots(2,2,figsize=(6.5,6.4),layout='constrained')
for mi,model in enumerate(MODELS):
 for wi,workload in enumerate(WORKLOADS):
  ax=axes[mi,wi];baseline=dense(model,workload)['piqa_accuracy']
  for r in rows(model,workload):
   if r['deleted_layers']==0:continue
   lo,hi=r['speedup_CI_nominal95']; y=r['speedup']
   ax.errorbar((r['piqa_accuracy']-baseline)*100,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],
    fmt=MARKERS[r['method']],color=COLORS[r['method']],markersize=3.7+.45*r['deleted_layers'],
    markeredgecolor='white',elinewidth=.6,capsize=1,alpha=.8)
  ax.axhline(1,color=DARK,ls='--',lw=.8);ax.axvline(0,color=DARK,ls=':',lw=.8)
  ax.grid(color='#D9E1E8',lw=.4);ax.set_title(model.upper()+' · '+workload)
  if mi==1:ax.set_xlabel('PIQA change from Dense (pp)')
  if wi==0:ax.set_ylabel('Dense/candidate speedup')
fig.legend(handles=legend,loc='outside lower center',ncol=2,frameon=False)
save(fig,'paper_figure_2_quality_serving')

# Holm-96, one model per page; no eighth-of-page panels.
for model in MODELS:
 fig,axes=plt.subplots(2,2,figsize=(6.5,6.4),sharex=True,sharey=True,layout='constrained')
 for ki,k in enumerate((1,2,3,4)):
  ax=axes.flat[ki]
  cells=sorted((c for c in B['contrasts'] if c['model']==model and c['k']==k),
               key=lambda c:(TASKS.index(c['task']),METHODS.index(c['method'])))
  assert len(cells)==12
  for y,c in enumerate(cells):
   p=c['accuracy_difference']*100;lo,hi=(v*100 for v in c['nominal95CI']);m=c['method']
   ax.errorbar(p,y,xerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt=MARKERS[m],color=COLORS[m],
    markersize=5,markeredgecolor=DARK if c['reject_Holm96'] else 'white',elinewidth=.9,capsize=1.5)
  ax.axvline(0,color=DARK,ls='--',lw=.8)
  ax.set_yticks(range(12),[c['task'].upper()+' · '+NAMES[c['method']] for c in cells])
  ax.invert_yaxis();ax.set_title(model.upper()+f' · k={k}')
  ax.grid(axis='x',color='#D9E1E8',lw=.4)
  if ki>=2:ax.set_xlabel('Accuracy change (pp)')
 save(fig,f'paper_figure_3_holm96_{model}')

# Tail panels vertical, each full width.
fig,axes=plt.subplots(2,1,figsize=(6.5,6.4),sharex=True,layout='constrained')
for ax,model in zip(axes,MODELS):
 for r in rows(model):
  if r['deleted_layers']==0:continue
  ax.scatter(100*r['parameter_removed_ratio'],r['C4_positive_tail_CVaR90'],
   color=COLORS[r['method']],marker=MARKERS[r['method']],s=22+r['deleted_layers']*4,
   alpha=.8,edgecolor='white',linewidth=.25)
 ax.set_title(model.upper());ax.set_ylabel('C4 positive-loss CVaR90');ax.grid(color='#D9E1E8',lw=.4)
axes[-1].set_xlabel('Parameters removed (%)')
fig.legend(handles=legend,loc='outside lower center',ncol=2,frameon=False)
save(fig,'paper_figure_4_parameter_tail')

# Cost roles vertically, retaining exact lease values and historical separation.
fig,axes=plt.subplots(2,1,figsize=(6.5,6.2),height_ratios=(3.0,0.85),layout='constrained')
for ax,historical in zip(axes,(False,True)):
 leases=[x for x in B['cost_lease_lineage'] if x['shared_or_historical']==historical]
 labels=[x['role'].replace('-',' ') for x in leases]
 values=[x['GPUh'] for x in leases]
 colors=['#D55E00' if 'fail' in x['role'].lower() else '#56B4E9' if historical else '#0072B2' for x in leases]
 ax.barh(labels,values,color=colors,edgecolor=DARK,lw=.35)
 ax.invert_yaxis();ax.set_xlim(0,max(values)*1.3)
 ax.set_xlabel('Controller lease occupancy (GPU-hours)')
 ax.set_title('Shared historical prior' if historical else 'Incremental A + B runs, including failures')
 ax.grid(axis='x',color='#D9E1E8',lw=.4)
 for y,v in enumerate(values):ax.text(v,y,f' {v:.3f}',va='center',fontsize=7.5)
save(fig,'paper_figure_5_cost_lineage')

# Holm-18 earlier boundary figure: model panels stacked.
fig,axes=plt.subplots(2,1,figsize=(6.5,5.9),sharex=True,layout='constrained')
for ax,model in zip(axes,MODELS):
 cells=[c for c in A['contrasts'] if c['model']==model]
 assert len(cells)==9
 for y,c in enumerate(cells):
  method=c['comparison'].split('-minus-')[1];p=c['difference']*100
  lo,hi=(v*100 for v in c['nominal95CI'])
  ax.errorbar(p,y,xerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt=MARKERS[method],
   color=COLORS[method],markeredgecolor=DARK if c['reject_Holm18'] else 'white',
   markersize=6,elinewidth=1,capsize=2)
 ax.axvline(0,color=DARK,ls='--',lw=.9)
 ax.set_yticks(range(9),[c['task'].upper()+' · '+c['comparison'].split('-minus-')[1].replace('_adapted','') for c in cells])
 ax.invert_yaxis();ax.set_title(model.upper());ax.grid(axis='x',color='#D9E1E8',lw=.4)
axes[-1].set_xlabel('Tabu minus comparator accuracy (pp)')
save(fig,'paper_figure_C_holm18')
print('Generated seven manuscript-only figure layouts from frozen A/B statistics')
