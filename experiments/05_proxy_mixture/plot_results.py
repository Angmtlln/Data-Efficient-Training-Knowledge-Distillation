"""Plot individual paired differences in SST-2 and IMDb."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]/'results/proxy_mixture'
NAMES={'mix_hard':'Самые трудные','mix_uncertain':'Неопределённые','mix_mid':'Средняя сложность',
       'mix_easy':'Самые лёгкие','mix_rare':'Редкие слова','mix_diverse':'Кластеры TF-IDF'}
COLORS={17:'#5969a5',42:'#ed8e53',101:'#4a9f88'}

def main():
    data=json.loads((ROOT/'summary.json').read_text())
    if data['completed_new_runs']!=21 or data.get('imdb',{}).get('completed_runs')!=27:raise ValueError('Full SST-2 and IMDb results required')
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharey=True,sharex=True)
    for ax,key,title in zip(axes,['sst2','imdb'],['SST-2: GLUE validation','IMDb: первые 64 токена']):
        deltas=data[key]['paired_deltas'];ax.axvline(0,color='#777777',linewidth=1)
        for row,(name,label) in enumerate(NAMES.items()):
            values=[]
            for offset,(seed,color) in zip([-.16,0,.16],COLORS.items()):
                d=next(x for x in deltas if x['name']==name and x['seed']==seed)['accuracy_delta_pp'];values.append(d)
                ax.scatter(d,row+offset,color=color,s=42,label=f'seed {seed}' if row==0 else None,zorder=3)
            ax.scatter(np.mean(values),row,marker='D',facecolor='white',edgecolor='#222222',s=32,zorder=4,label='среднее' if row==0 else None)
        ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Разница accuracy с парным контролем, п.п.')
        ax.set_yticks(range(len(NAMES)),list(NAMES.values()));ax.invert_yaxis()
    axes[1].legend(frameon=False,loc='best',fontsize=8)
    fig.suptitle('Выбор 25% обучающей выборки: парные сравнения',fontsize=14,fontweight='bold')
    fig.text(.5,.01,'Каждая цветная точка — один сид; 75% обучающих данных общие.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(ROOT/'comparison.png',dpi=180);print(ROOT/'comparison.png')

if __name__=='__main__':main()
