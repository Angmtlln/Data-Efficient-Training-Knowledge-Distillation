"""Readable figures with consistent Russian labels; no metric recomputation."""
import json,os,textwrap
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/enn-report-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/enn-report-cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
def plot_results(summary,out):
 names=json.loads((ROOT/'docs/methods.json').read_text())
 controls={'random','skill_random','pair_random','pair_matched_random','forget_random','forget_matched_zero','skill_matched_random_v2'}
 configs=[('part1','Простые способы отбора',summary.part=='part1'),('part2','Отбор под языковые явления и пары',summary.part=='part2'),('part3_forgetting','Сколько нестабильных строк добавлять',(summary.part=='part3')&summary.name.str.startswith('forget_')),('part3_skills','Исправления отбора: около 25% целевых строк',(summary.part=='part3')&~summary.name.str.startswith('forget_'))]
 for key,title,mask in configs:
  z=summary[mask].sort_values('accuracy_pct');fig,ax=plt.subplots(figsize=(13,max(4.8,len(z)*.72)))
  yy=np.arange(len(z));colors=['#278472' if n in controls else '#517ba5' for n in z.name]
  ax.errorbar(z.accuracy_pct,yy,xerr=z.sd_pp,fmt='none',ecolor='#7b8794',capsize=3,lw=1.5)
  ax.scatter(z.accuracy_pct,yy,s=52,c=colors,zorder=3)
  labels=['\n'.join(textwrap.wrap(names[n]['title'],37)) for n in z.name]
  ax.set_yticks(yy,labels,fontsize=10);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
  ax.set_xlim(65 if key=='part2' else 88,98);ax.set_xlabel('Верных ответов на тесте, %',fontsize=11)
  for i,r in enumerate(z.itertuples()):ax.text(97.6,i,f'{r.accuracy_pct:.2f}%',va='center',ha='right',fontsize=10,color='#25364a')
  ax.set_title(title,pad=16,fontsize=14,fontweight='bold')
  ax.spines[['top','right']].set_visible(False)
  fig.text(.5,.015,'Точка — среднее трех запусков; отрезок — стандартное отклонение, не доверительный интервал.',ha='center',fontsize=9,color='#566577')
  fig.legend(handles=[Line2D([0],[0],marker='o',lw=0,color='#278472',label='Случайный вариант для сравнения'),Line2D([0],[0],marker='o',lw=0,color='#517ba5',label='Проверяемый способ')],loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,fontsize=8,frameon=False)
  fig.tight_layout(rect=(0,.12,1,1));fig.savefig(out/f'{key}_all_accuracy.png',dpi=170);plt.close(fig)
if __name__=='__main__':
 out=ROOT/'results/analysis';plot_results(pd.read_csv(out/'all_summary.csv'),out)
