"""Plot completed results, preserving individual seeds and paired differences."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'results/len_forgetting'
METHODS=['random','len8','len8_matched_random','len8_forgetting']
NAMES=['Random','Длинные\n≥ 8 слов','Парный\nконтроль','Нестабильные\n+ длинные']
COLORS={17:'#5969a5',42:'#ed8e53',101:'#4a9f88'}

def main():
    results={}
    for f in (ROOT/'runs').glob('*/metrics.json'):
        r=json.loads(f.read_text())
        if r['pilot'] or r['split']!='test':raise ValueError('Unexpected pilot or split')
        results[(r['name'],r['selection_seed'])]=r['metrics']['accuracy']*100
    if set(results)!={(m,s) for m in METHODS for s in COLORS}:raise ValueError('Plot requires all 12 completed runs')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,5.2),gridspec_kw={'width_ratios':[1.7,1]})
    for position,(seed,color) in enumerate(COLORS.items()):
        vals=[results[(m,seed)] for m in METHODS]
        ax.scatter(np.arange(4),vals,s=55,c=color,label=f'seed {seed}',zorder=3)
        ax.plot([2,3],vals[2:4],color=color,alpha=.75,linewidth=1.5)
        bx.scatter(position,vals[3]-vals[2],color=color,s=70,zorder=3)
    ax.set_xticks(range(4),NAMES);ax.set_ylabel('Accuracy на GLUE validation, %')
    ax.set_title('Результаты отдельных запусков');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    diffs=[results[('len8_forgetting',s)]-results[('len8_matched_random',s)] for s in COLORS]
    bx.axhline(0,color='#666666',linewidth=1);bx.axhline(np.mean(diffs),color='#c24845',linestyle='--',label=f'Среднее: {np.mean(diffs):+.2f} п.п.')
    bx.set_xticks(range(len(COLORS)),[str(s) for s in COLORS]);bx.set_xlabel('Seed выборки и обучения')
    bx.set_ylabel('Разница accuracy, п.п.');bx.set_title('Нестабильные минус контроль');bx.grid(axis='y',alpha=.2);bx.legend(frameon=False,loc='best')
    fig.suptitle('ModernBERT: отбор 6534 примеров SST-2',fontsize=15,fontweight='bold',y=.98)
    fig.text(.5,.015,'Каждая точка — один запуск; тестовая выборка: 872 примера.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.045,1,.94));fig.savefig(ROOT/'comparison.png',dpi=180);print(ROOT/'comparison.png')

if __name__=='__main__':main()
