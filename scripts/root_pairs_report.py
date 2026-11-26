"""Verify archived runs and rebuild figures/tables for the root/phrase experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/root_pairs';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
ORDER=['roots','mixed_independent','mixed_paired','mixed_rank']
NAMES=['Целые предложения','+ подфразы, обычное обучение','+ подфразы, пары в батче','+ подфразы, парная ошибка']
COLORS=['#61778a','#14876b','#d59931','#ae5979']
rows=[];hist=[];preds={};sels={};epochs=[]
for f in sorted((OUT/'runs').glob('*/metrics.json')):
 d=json.loads(f.read_text());p=pd.read_csv(f.parent/'predictions.csv');sel=json.loads((f.parent/'selection.json').read_text())
 assert len(p)==872 and d['global_step']==615 and d['split']=='test' and not d['pilot']
 assert len(sel['indices'])==len(set(sel['indices']))==6534
 acc=accuracy_score(p.label,p.prediction);f1=f1_score(p.label,p.prediction,average='macro')
 assert np.isclose(acc,d['metrics']['accuracy']) and np.isclose(f1,d['metrics']['macro_f1'])
 m,s=d['name'],d['training_seed'];preds[m,s]=p;sels[m,s]=sel
 rows.append(dict(method=m,seed=s,accuracy=acc,macro_f1=f1,correct=int((p.label==p.prediction).sum())))
 for h in d['log_history']:
  hist.append(dict(method=m,seed=s,**h))
  if 'eval_accuracy' in h:epochs.append(dict(method=m,seed=s,epoch=h['epoch'],step=h['step'],accuracy=h['eval_accuracy'],loss=h['eval_loss']))
assert len(rows)==12
runs=pd.DataFrame(rows);runs.to_csv(OUT/'runs.csv',index=False)
summary=runs.groupby('method').agg(accuracy=('accuracy','mean'),sd=('accuracy','std'),macro_f1=('macro_f1','mean')).reindex(ORDER)
summary.to_csv(OUT/'summary.csv')
comparisons=[]
for seed in sorted(runs.seed.unique()):
 for m in ORDER[1:]:assert np.array_equal(preds[m,seed].label,preds['roots',seed].label)
 assert sels['mixed_independent',seed]['indices']==sels['mixed_paired',seed]['indices']==sels['mixed_rank',seed]['indices']
 assert sels['mixed_independent',seed]['pairs']==sels['mixed_paired',seed]['pairs']==sels['mixed_rank',seed]['pairs']
 assert len(set(sels['roots',seed]['indices'])&set(sels['mixed_rank',seed]['indices']))==6423
 for m,c in [('mixed_independent','roots'),('mixed_paired','mixed_independent'),('mixed_rank','mixed_paired'),('mixed_rank','roots')]:
  a=preds[m,seed];b=preds[c,seed];ac=a.prediction==a.label;bc=b.prediction==b.label
  comparisons.append(dict(method=m,control=c,seed=seed,delta_pp=100*(ac.mean()-bc.mean()),new_correct=int((ac&~bc).sum()),new_errors=int((~ac&bc).sum())))
pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
pd.DataFrame(epochs).to_csv(OUT/'epochs.csv',index=False)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.2))
for i,m in enumerate(ORDER):
 v=runs[runs.method==m].accuracy*100
 ax.scatter(v,[i-.12,i,i+.12],color=COLORS[i],s=45,zorder=3)
 ax.scatter([v.mean()],[i],color=COLORS[i],marker='D',s=80,zorder=4)
 ax.text(95.1,i,f'{v.mean():.2f}%',va='center',fontweight='bold',color=COLORS[i])
ax.set_yticks(range(4),NAMES);ax.invert_yaxis();ax.set_xlim(93.1,95.65);ax.set_xlabel('Accuracy на GLUE validation, %');ax.grid(axis='x',alpha=.18)
ax.set_title('Точки — три запуска, ромб — среднее',loc='left',fontsize=12);fig.tight_layout();fig.savefig(FIG/'accuracy.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
c=pd.DataFrame(comparisons);v=c[(c.method=='mixed_independent')&(c.control=='roots')]
ax.bar(v.seed.astype(str),v.delta_pp,color='#14876b',width=.55)
for i,r in enumerate(v.itertuples()):ax.text(i,r.delta_pp+.025,f'+{r.delta_pp:.2f} п.п.\n+{r.new_correct-r.new_errors} ответа',ha='center',fontsize=11)
ax.set_ylim(0,1.15);ax.set_ylabel('Прирост accuracy, п.п.');ax.set_xlabel('Seed');ax.set_title('Добавление 111 подфраз: плюс в трех запусках',loc='left',fontsize=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(FIG/'paired_gains.png',dpi=180);plt.close(fig)
h=pd.DataFrame(hist);fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for m,name,color in zip(ORDER,NAMES,COLORS):
 for ax,metric,mult in [(axes[0],'loss',1),(axes[1],'eval_accuracy',100)]:
  g=h[h.method==m].dropna(subset=[metric])
  for seed,z in g.groupby('seed'):ax.plot(z.step,z[metric]*mult,color=color,alpha=.16,lw=1)
  z=g.groupby('step')[metric].mean();ax.plot(z.index,z*mult,color=color,lw=2,label=name)
axes[0].set_ylabel('Train loss');axes[1].set_ylabel('Val accuracy, %')
for ax in axes:ax.set_xlabel('Шаг обучения');ax.grid(alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=9);fig.suptitle('Обучение: средние линии и отдельные запуски',fontsize=12);fig.tight_layout(rect=(0,.15,1,.95));fig.savefig(FIG/'learning.png',dpi=180);plt.close(fig)
# Optional diagnostic slices use exactly the existing text snapshot.
data=ROOT/'experiments/06_root_pair_training/data/sst2'
if data.exists():
 import re
 from datasets import load_from_disk
 ds=load_from_disk(str(data));text=ds['test']['sentence'];labels=np.array(ds['test']['label']);sr=[]
 masks={'Все':np.ones(len(text),dtype=bool),'Маркеры отрицания':np.array([bool(re.search(r"\b(?:not|never|no|neither|nor)\b|n't",t.lower())) for t in text]),'Маркеры противопоставления':np.array([bool(re.search(r'\b(?:but|although|though|however|yet|despite)\b',t.lower())) for t in text])}
 for (m,seed),p in preds.items():
  assert np.array_equal(labels,p.label)
  for name,mask in masks.items():sr.append(dict(method=m,seed=seed,slice=name,n=int(mask.sum()),accuracy=float((p.label[mask]==p.prediction[mask]).mean())))
 pd.DataFrame(sr).to_csv(OUT/'slices.csv',index=False)
print('Verified 12 full runs, identical mixture subsets, predictions, metrics; rebuilt 3 figures')
