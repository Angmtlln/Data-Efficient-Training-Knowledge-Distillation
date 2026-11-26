"""Recompute metrics, paired effects and training plots from saved predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
NAMES={'roots':'Целые предложения','mixed_independent':'Добавлены подфразы','mixed_paired':'Пары в одном батче','mixed_rank':'Пары + дополнительная ошибка'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output/'report';out.mkdir(parents=True,exist_ok=True)
    rows=[];history=[]
    for f in sorted((a.output/'runs').glob('*/metrics.json')):
        r=json.loads(f.read_text());pred=pd.read_csv(f.parent/'predictions.csv')
        assert r['global_step']==615 and len(pred)==872 and r['split']=='test' and not r['pilot']
        acc=accuracy_score(pred.label,pred.prediction);f1=f1_score(pred.label,pred.prediction,average='macro')
        assert np.isclose(acc,r['metrics']['accuracy']) and np.isclose(f1,r['metrics']['macro_f1'])
        rows.append(dict(method=r['name'],seed=r['training_seed'],accuracy=acc,macro_f1=f1,seconds=r['last_session_seconds']))
        for h in r['log_history']:history.append(dict(method=r['name'],seed=r['training_seed'],**h))
    if not rows:(out/'REPORT.md').write_text('# Обучение еще не завершено\n');return
    df=pd.DataFrame(rows);assert not df.duplicated(['method','seed']).any(),'Multiple runs per method/seed: specify one output series'
    df.to_csv(out/'runs.csv',index=False);hh=pd.DataFrame(history);hh.to_csv(out/'training_history.csv',index=False)
    summary=df.groupby('method').agg(n=('accuracy','size'),accuracy=('accuracy','mean'),sd=('accuracy','std'),macro_f1=('macro_f1','mean')).reindex([m for m in NAMES if m in set(df.method)])
    summary.to_csv(out/'summary.csv');pivot=df.pivot(index='seed',columns='method',values='accuracy');comparisons=[]
    for m,c in [('mixed_independent','roots'),('mixed_paired','mixed_independent'),('mixed_rank','mixed_paired'),('mixed_rank','roots')]:
        if m in pivot and c in pivot:
            for seed,d in ((pivot[m]-pivot[c]).dropna()*100).items():comparisons.append(dict(method=m,control=c,seed=seed,delta_pp=d))
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    for i,m in enumerate(summary.index):
        vals=df[df.method==m].accuracy*100;ax.scatter([i]*len(vals),vals,s=40);ax.plot([i-.15,i+.15],[vals.mean()]*2,color='black',lw=2)
    ax.set_xticks(range(len(summary)),[NAMES[m] for m in summary.index],rotation=12,ha='right');ax.set_ylabel('Accuracy, %');ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for (method,seed),g in hh.groupby(['method','seed']):
        for ax,metric,label in [(axes[0],'eval_accuracy','Val accuracy'),(axes[1],'loss','Train loss')]:
            if metric in g:
                v=g.dropna(subset=[metric]);ax.plot(v.step,v[metric],label=f'{method}, {seed}',alpha=.7);ax.set_ylabel(label);ax.set_xlabel('Шаг обучения')
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'learning.png',dpi=160);plt.close(fig)
    lines=['# Связь целого предложения и подфразы','',f'Завершено {len(df)} из 12 основных запусков.','', '| Метод | Запусков | Accuracy, % | SD, п.п. | Macro-F1, % |','|---|---:|---:|---:|---:|']
    for m,r in summary.iterrows():lines.append(f'| {NAMES[m]} | {int(r.n)} | {100*r.accuracy:.2f} | {100*r.sd:.2f} | {100*r.macro_f1:.2f} |')
    lines+=['','![Качество](accuracy.png)','','![Обучение](learning.png)','','Главное сравнение: mixed_rank против mixed_paired на одинаковых текстах и батчах. Разницы по каждому seed сохранены в comparisons.csv. Итоговая оценка использует многократно исследованный GLUE validation; небольшие различия требуют независимого подтверждения.']
    (out/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
