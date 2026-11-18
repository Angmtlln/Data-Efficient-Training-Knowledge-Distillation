"""Figures computed from saved metrics and per-epoch logs."""
import json, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/enn-report-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BLUE='#3B73A5'; GREEN='#268576'; RED='#C46B59'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlepad':16})
def save(fig,out,name):
 fig.savefig(out/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
def delta_chart(s,part,names,labels,base,title,out,file):
 z=s[s.part==part].set_index('name');v=z.loc[names].accuracy_pct-z.loc[base,'accuracy_pct']
 fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained');y=np.arange(len(v))
 ax.barh(y,v,color=[GREEN if x>=0 else RED for x in v],height=.55)
 ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(0,color='#687782',lw=1)
 ax.spines[['left','bottom']].set_visible(False);ax.tick_params(axis='both',length=0)
 lo=min(-.2,float(v.min())-.35);hi=max(.3,float(v.max())+.35);ax.set_xlim(lo,hi)
 for i,x in enumerate(v):ax.text(x+(.035 if x>=0 else -.035),i,f'{x:+.2f}',va='center',ha='left' if x>=0 else 'right',fontweight='bold')
 ax.set_xlabel('Разница со случайным отбором, п.п.');ax.set_title(title);ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
 save(fig,out,file)
def plot_results(summary,out):
 delta_chart(summary,'part1',['heldout_forgetting','dedup_exact','lexical_diversity','rare_words'],['Нестабильные примеры','Без повторов','Разнообразие слов','Редкие слова'],'random','Что изменилось относительно случайных 10%',out,'part1_all_accuracy.png')
 delta_chart(summary,'part3',['grad_raw_mix25','grad_nobias_mix25','grad_randomref_nobias_mix25','neg_embedding_mix25'],['Градиент','Градиент без bias','Градиент, случайный ориентир','Сходство с отрицаниями'],'skill_matched_random_v2','Исправления: удалось ли обойти контроль?',out,'part3_skills_all_accuracy.png')
 z=summary[summary.part=='part2'].set_index('name');base=z.loc['skill_random']
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for ax,prefix,col,title in zip(axes,['negation','contrast'],['negation_pct','contrast_pct'],['Отрицания','Противопоставления']):
  ids=[prefix+'_keyword',prefix+'_embedding'];allv=z.loc[ids].accuracy_pct-base.accuracy_pct;target=z.loc[ids,col]-base[col]
  for vals,offset,color,label in [(allv,-.18,BLUE,'Весь тест'),(target,.18,GREEN,'Целевые фразы')]:
   bars=ax.bar(np.arange(2)+offset,vals,width=.32,color=color,label=label)
   ax.bar_label(bars,labels=[f'{v:+.2f}' for v in vals],padding=4,fontsize=10)
  ax.set_xticks([0,1],['По словам','По сходству']);ax.axhline(0,color='#7b8794',lw=1);ax.set_title(title);ax.set_ylabel('Разница с контролем, п.п.');ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
  vals=list(allv)+list(target);ax.set_ylim(min(vals+[0])-1.1,max(vals+[0])+1.1)
 axes[0].legend(frameon=False,loc='lower right');save(fig,out,'skills_target_vs_total.png')
 ids=['skill_random','negation_gradient','negation_randomref_gradient','contrast_gradient','contrast_randomref_gradient'];labels=['Случайный\nотбор','Отрицания','Отрицания:\nслучайный\nориентир','Противо-\nпоставления','Противо-\nпоставления:\nслучайный ориентир']
 fig,ax=plt.subplots(figsize=(10,4),layout='constrained');v=z.loc[ids].accuracy_pct
 bars=ax.bar(labels,v,color=[GREEN]+[RED]*4,width=.6);ax.bar_label(bars,labels=[f'{x:.2f}%' for x in v],padding=5);ax.set_ylim(0,105);ax.set_ylabel('Верных ответов, %');ax.set_title('Градиентный отбор: все четыре варианта хуже контроля');ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True);save(fig,out,'gradient_accuracy.png')
 runs=pd.read_csv(out/'all_runs.csv')
 ids=['pair_random','pair_matched_random','pair_pairs','pair_broken_pairs'];labels=['Случайный','С похожей\nдлиной','Целые пары','Замена\nучастника']
 fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
 for j,(seed,color) in enumerate(zip([17,42,101],[BLUE,GREEN,RED])):
  vals=[100*runs[(runs.part=='part2')&(runs.name==n)&(runs.seed==seed)].accuracy.iloc[0] for n in ids]
  ax.scatter(np.arange(4)+(j-1)*.15,vals,color=color,s=65,label=f'Запуск {seed}')
 ax.set_xticks(range(4),labels);ax.set_ylabel('Верных ответов, %');ax.set_title('Пары: результат зависит от запуска');ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,ncol=3,loc='lower center');ax.margins(y=.25);save(fig,out,'pairs_seeds.png')
 records=[]
 for r in runs.itertuples():
  p=ROOT/'results/raw'/r.part/'runs'/r.run_id/'progress.json'
  for h in json.loads(p.read_text()).get('log_history',[]):
   if 'eval_accuracy' in h:records.append({'part':r.part,'name':r.name,'seed':r.seed,'epoch':h['epoch'],'accuracy_pct':100*h['eval_accuracy']})
 hist=pd.DataFrame(records).drop_duplicates(['part','name','seed','epoch'],keep='last');hist.to_csv(out/'learning_validation.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
 configs=[(['skill_random','negation_keyword','negation_embedding'],['Случайный','По словам','По сходству'],'Отрицания: слова и сходство'),(['skill_random','negation_gradient','contrast_gradient'],['Случайный','Градиент: отрицания','Градиент: противопоставления'],'Отбор по градиенту')]
 for ax,(ids,labels,title) in zip(axes,configs):
  for n,label,color in zip(ids,labels,[GREEN,RED,BLUE]):
   h=hist[(hist.part=='part2')&(hist.name==n)];g=h.groupby('epoch').accuracy_pct.agg(['mean','min','max','count']);assert len(g)==3 and (g['count']==3).all(),n
   ax.plot(g.index,g['mean'],marker='o',color=color,label=label);ax.fill_between(g.index,g['min'],g['max'],alpha=.12,color=color)
  ax.set_xticks([1,2,3]);ax.set_xlabel('Эпоха');ax.set_ylabel('Верных на внутренней проверке, %');ax.set_title(title);ax.grid(alpha=.13);ax.legend(frameon=False,fontsize=9,loc='best')
 save(fig,out,'learning_validation.png')
if __name__=='__main__':
 out=ROOT/'results/analysis';plot_results(pd.read_csv(out/'all_summary.csv'),out)
