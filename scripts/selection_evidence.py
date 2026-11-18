"""Rebuild selection diagnostics from dataset, saved scores and actual subset IDs."""
import argparse,hashlib,json,re,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/enn-report-mpl')
import numpy as np
import pandas as pd
from datasets import load_from_disk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);a=ap.parse_args()
 ds=load_from_disk(str(a.dataset));texts=list(ds['pool']['sentence']);y=np.array(ds['pool']['label'])
 snap=json.loads((ROOT/'results/inputs/part2/snapshot.json').read_text())
 payload=list(zip(ds['pool']['orig_idx'],texts,y.tolist()))
 assert hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==snap['splits']['pool']['sha256']
 out=ROOT/'results/analysis';out.mkdir(exist_ok=True)
 p=np.load(ROOT/'results/inputs/part2/proxy_head.npz')['probabilities'];err=(p>=.5)!=y
 lengths=np.array([len(t.split()) for t in texts]);neg=np.array([bool(re.search(r"\b(?:not|never|no)\b|n't",t.lower())) for t in texts])
 rows=[];sets={};examples=[]
 for part in ['part1','part2','part3']:
  for f in sorted((ROOT/'results/selections'/part).glob('*.json')):
   d=json.loads(f.read_text());ids=np.array(d['indices']);n=d['name'];seed=d['selection_seed'];sets[part,n,seed]=ids
   row=dict(part=part,name=n,seed=seed,n=len(ids),median_words=float(np.median(lengths[ids])),mean_words=float(lengths[ids].mean()),short_share=float((lengths[ids]<=3).mean()),positive_count=int(y[ids].sum()))
   for label in [0,1]:
    ii=ids[y[ids]==label];row[f'proxy_error_label{label}']=float(err[ii].mean());row[f'negation_label{label}']=float(neg[ii].mean())
   rows.append(row)
   if part=='part1' and seed==17 and n in ['random','rare_words','lexical_diversity','heldout_forgetting']:
    for label in [0,1]:
     for i in sorted(ids[y[ids]==label])[:2]:examples.append(dict(method=n,seed=seed,pool_index=int(i),label=label,text=texts[i],words=int(lengths[i])))
 audit=pd.DataFrame(rows);audit.to_csv(out/'selection_evidence.csv',index=False);pd.DataFrame(examples).to_csv(out/'selection_examples.csv',index=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
 for n,label,color in [('random','Случайный','#72818d'),('lexical_diversity','Разнообразие слов','#328479'),('rare_words','Редкие слова','#be6756')]:
  curves=[]
  for seed in [17,42,101]:
   lens=lengths[sets['part1',n,seed]];curves.append([(lens<=v).mean()*100 for v in range(1,31)])
  ax.plot(range(1,31),np.mean(curves,axis=0),label=label,color=color,lw=2.5)
 ax.set(xlabel='Длина фразы, слов',ylabel='Доля выборки не длиннее этой фразы, %',title='Какие длины текстов попали в обучение',xlim=(1,30),ylim=(0,102));ax.grid(alpha=.15);ax.legend(frameon=False,loc='lower right');save(fig,'selection_lengths.png')
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 groups=[('part2',['skill_random','negation_gradient','contrast_gradient'],['Случайный','Градиент:\nотрицания','Градиент:\nпротивопоставления'],'Исходный отбор'),('part3',['skill_matched_random_v2','grad_raw_mix25','grad_nobias_mix25'],['Случайный\nс ограничениями','Градиент','Без bias'],'После изменения состава')]
 for ax,(part,ns,labels,title) in zip(axes,groups):
  q=audit[audit.part==part].groupby('name').mean(numeric_only=True).loc[ns]
  for c,offset,color,label in [(0,-.18,'#497ba4','Отрицательные отзывы'),(1,.18,'#c17762','Положительные отзывы')]:
   v=q[f'proxy_error_label{c}']*100;b=ax.bar(np.arange(len(ns))+offset,v,.34,label=label,color=color);ax.bar_label(b,fmt='%.1f',padding=3,fontsize=9)
  ax.set_xticks(range(len(ns)),labels);ax.set_ylim(0,114);ax.set_ylabel('Ошибки маленькой модели, %');ax.set_title(title);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
 axes[0].legend(frameon=False,fontsize=9,loc='upper left');save(fig,'selection_class_difficulty.png')
 fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained');q=audit[audit.part=='part2'].groupby('name').mean(numeric_only=True).loc[['skill_random','negation_keyword','negation_embedding']]
 for c,off,color,label in [(0,-.18,'#497ba4','Отрицательные отзывы'),(1,.18,'#c17762','Положительные отзывы')]:
  b=ax.bar(np.arange(3)+off,q[f'negation_label{c}']*100,.34,color=color,label=label);ax.bar_label(b,fmt='%.1f',padding=3)
 ax.set_xticks(range(3),['Случайный','По словам','По сходству']);ax.set_ylim(0,120);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Фразы с маркером отрицания, %');ax.set_title('Как отбор связывает слово отрицания с меткой');ax.legend(frameon=False,fontsize=9,loc='upper left');ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True);save(fig,'selection_negation_labels.png')
 hist=pd.read_csv(ROOT/'results/analysis/forgetting_audit.csv')
 fig,ax=plt.subplots(figsize=(8,3),layout='constrained');ax.barh(hist.seed.astype(str),hist.nonzero_selected,color='#328479',label='Был переход к ошибке');ax.barh(hist.seed.astype(str),hist.zero_selected,left=hist.nonzero_selected,color='#d9e1e6',label='Нулевая оценка, случайное заполнение')
 for j,r in enumerate(hist.itertuples()):ax.text(r.nonzero_selected/2,j,str(r.nonzero_selected),ha='center',va='center',color='white');ax.text(r.nonzero_selected+r.zero_selected/2,j,str(r.zero_selected),ha='center',va='center')
 ax.set(xlabel='Число строк в выбранных 6534',ylabel='Запуск',title='Нестабильные строки — только небольшая часть выборки');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2);save(fig,'selection_forgetting_mix.png')
 chosen=pd.read_csv(ROOT/'results/preparation/part2/selected_pairs_s17.csv');chosen['size']=chosen.text_a.str.len()+chosen.text_b.str.len();chosen.sort_values(['size','a']).head(3).drop(columns='size').to_csv(out/'pair_examples.csv',index=False)
 summary=audit.groupby(['part','name']).mean(numeric_only=True).reset_index();summary.to_csv(out/'selection_evidence_summary.csv',index=False)
 print(summary[summary.name.isin(['random','lexical_diversity','rare_words','negation_keyword','negation_embedding','skill_random','grad_raw_mix25'])].to_string(index=False))
 print('Verified dataset hash; rebuilt',len(audit),'subset audits and four composition figures.')
if __name__=='__main__':main()
