"""Training diagnostics from saved Trainer logs, without rerunning training."""
import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/enn-report-mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'results/analysis';runs=pd.read_csv(out/'all_runs.csv');rows=[];checks=[]
 for r in runs.itertuples():
  h=json.loads((ROOT/'results/raw'/r.part/'runs'/r.run_id/'progress.json').read_text())['log_history']
  for v in h:
   for metric in ['loss','eval_loss','eval_accuracy']:
    if metric in v:rows.append(dict(part=r.part,name=r.name,seed=r.seed,step=v['step'],epoch=v['epoch'],metric=metric,value=v[metric]))
  ev=[v for v in h if 'eval_accuracy' in v];assert len(ev)==3
  best=max(ev,key=lambda v:v['eval_accuracy']);checks.append(dict(part=r.part,name=r.name,seed=r.seed,best_epoch=best['epoch'],val_epoch1_pct=100*ev[0]['eval_accuracy'],val_epoch3_pct=100*ev[-1]['eval_accuracy'],test_accuracy_pct=100*r.accuracy))
 data=pd.DataFrame(rows).drop_duplicates(['part','name','seed','step','metric']);data.to_csv(out/'training_diagnostics.csv',index=False);pd.DataFrame(checks).to_csv(out/'training_epochs.csv',index=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 def curve(ax,part,name,metric,label,color):
  q=data[(data.part==part)&(data.name==name)&(data.metric==metric)];g=q.groupby('step').agg(value=('value','mean'),low=('value','min'),high=('value','max'),epoch=('epoch','mean'),n=('value','count'));assert (g.n==3).all()
  factor=100 if metric=='eval_accuracy' else 1
  ax.plot(g.epoch,g.value*factor,color=color,label=label,lw=2,marker='o' if metric.startswith('eval') else None);ax.fill_between(g.epoch,g.low*factor,g.high*factor,color=color,alpha=.10)
 def save(fig,name):fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
 for n,l,c in [('random','Случайный','#71818a'),('heldout_forgetting','Нестабильные примеры','#31877b'),('rare_words','Редкие слова','#bf6b58')]:curve(ax,'part1',n,'eval_accuracy',l,c)
 ax.set(xlabel='Эпоха ModernBERT',ylabel='Верных на внутренней проверке, %',title='Первая серия: как менялась точность',xticks=[1,2,3]);ax.legend(frameon=False);ax.grid(alpha=.15);save(fig,'training_part1.png')
 fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 configs=[('part1',[('random','Случайный','#71818a'),('heldout_forgetting','Нестабильные','#31877b'),('rare_words','Редкие слова','#bf6b58')],'Первая серия'),('part2',[('skill_random','Случайный','#71818a'),('negation_gradient','Градиент: отрицания','#bf6b58'),('contrast_gradient','Градиент: противопоставления','#467ea5')],'Градиентный отбор')]
 for col,(part,methods,title) in enumerate(configs):
  for row,metric in enumerate(['loss','eval_loss']):
   ax=axes[row,col]
   for n,l,c in methods:curve(ax,part,n,metric,l,c)
   ax.set(xlabel='Эпоха ModernBERT',ylabel='Ошибка (cross-entropy)',title=title+(' · обучение' if row==0 else ' · внутренняя проверка'));ax.grid(alpha=.15);ax.set_ylim(bottom=0)
   if row==1:ax.set_xticks([1,2,3])
  axes[0,col].legend(frameon=False,fontsize=9)
 save(fig,'training_losses.png')
 print(pd.DataFrame(checks).query("name in ['random','heldout_forgetting','rare_words','skill_random','negation_gradient','contrast_gradient']").to_string(index=False))
 print('Saved',len(data),'log records; all plotted steps have 3 runs.')
if __name__=='__main__':main()
