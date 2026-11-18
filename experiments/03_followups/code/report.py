"""Offline reports; GPU metrics only appear after real completed training."""
import argparse
import html
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/enn-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/enn-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ROOT,read_data

STYLE='''<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#172b42;background:#f7f9fc}h1,h2{color:#143f65}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dde5ee;text-align:left}th{background:#e7eef8}.note{padding:18px;background:#e8f2ef;border-left:5px solid #297d67;margin:20px 0}img{max-width:100%}</style>'''


def main(output):
    ds,_=read_data()
    parts=['<meta charset="utf-8">'+STYLE,'<h1>Data-efficient ModernBERT</h1>',
           '<div class="note">Подготовка на CPU выполнена. Таблицы состава подвыборок не являются результатами обучения ModernBERT. Финальный test — GLUE validation (872), val — 2000 из train.</div>']
    for name in ['selection_summary','pair_selection_summary','skill_selection_summary']:
        p=ROOT/'reports'/f'{name}.csv'
        if p.exists():
            parts+=['<h2>'+html.escape(name)+'</h2>',pd.read_csv(p).round(4).to_html(index=False)]
    if (ROOT/'reports/mining.json').exists():
        m=json.loads((ROOT/'reports/mining.json').read_text())
        parts+=['<h2>Проверка гипотезы о парах</h2>',
            f"<p>Найдено {m['candidate_pairs']} широких кандидатов, {m['operator_candidates']} с явным отрицанием/отсутствием. После предварительного просмотра: {m['disjoint_reviewed']} непересекающихся пар. Используем 32 пары = 64 строки из 6534 (0.98%). Это диагностический эксперимент с ограниченной статистической мощностью.</p>"]
        audit=pd.read_csv(ROOT/'reports/pair_operator_audit.csv').head(12)
        parts+=[audit[['text_a','text_b','removed','added','verdict']].fillna('').to_html(index=False)]
    results=[]
    for p in sorted(Path(output).glob('runs/*/metrics.json')):
        d=json.loads(p.read_text())
        if d.get('pilot'):continue
        pred=pd.read_csv(p.parent/'predictions.csv')
        assert len(pred)==len(ds['test'])
        row={k:d[k] for k in ['name','selection_seed','training_seed','global_step','last_session_seconds']}
        row.update(d['metrics'])
        texts=pd.Series(ds['test']['sentence'])
        for skill,pattern in [('negation',r"\b(?:not|never|no)\b|n't"),('contrast',r'\b(?:but|although|however|yet)\b')]:
            mask=texts.str.contains(pattern,case=False,regex=True)
            row[skill+'_n']=int(mask.sum())
            row[skill+'_accuracy']=float((pred.loc[mask,'label']==pred.loc[mask,'prediction']).mean())
        results.append(row)
    target=Path(output)/'report';target.mkdir(parents=True,exist_ok=True)
    # CPU figures describe selected data, never fabricate model performance.
    cpu_file=ROOT/'reports/selection_summary.csv'
    if cpu_file.exists():
        cpu=pd.read_csv(cpu_file).groupby('method').mean(numeric_only=True)
        ax=cpu.mean_words.plot.barh(figsize=(9,4),color='#287d83')
        ax.set_xlabel('Mean words per selected text (CPU data audit)')
        plt.tight_layout();plt.savefig(target/'cpu_lengths.png',dpi=150);plt.close()
        parts+=['<h2>Состав выборок: длина</h2><img src="cpu_lengths.png" alt="Mean text length">']
    skill_file=ROOT/'reports/skill_selection_summary.csv'
    if skill_file.exists():
        cpu=pd.read_csv(skill_file).groupby('method')[['negation_fraction','contrast_fraction']].mean()
        ax=cpu.plot.barh(figsize=(10,6),color=['#287d83','#b88133'])
        ax.set_xlabel('Fraction of selected rows containing lexical markers')
        plt.tight_layout();plt.savefig(target/'cpu_skill_mix.png',dpi=150);plt.close()
        parts+=['<h2>Состав выборок: лексические срезы</h2><img src="cpu_skill_mix.png" alt="Lexical slice proportions">']
    if results:
        df=pd.DataFrame(results);df.to_csv(target/'runs.csv',index=False)
        summary=df.groupby('name').agg(runs=('accuracy','count'),accuracy=('accuracy','mean'),sd=('accuracy','std'),macro_f1=('macro_f1','mean'))
        summary.to_csv(target/'summary.csv')
        parts+=['<h2>Завершенные GPU-запуски</h2>',summary.round(4).to_html(),
            '<p>SD — разброс завершенных запусков, не доверительный интервал. Показывайте число seed; лексические срезы не заменяют независимые тесты навыков. last_session_seconds не включает время предыдущих прерванных сессий.</p>']
        ax=summary.accuracy.plot.barh(xerr=summary.sd.fillna(0),figsize=(10,6))
        ax.set_xlabel('Held-out accuracy ± SD');plt.tight_layout();plt.savefig(target/'accuracy.png',dpi=150);plt.close()
        parts+=['<img src="accuracy.png" alt="Held-out accuracy">']
    else:parts+=['<h2>GPU-результатов пока нет</h2><p>Никакие значения accuracy ModernBERT здесь не подставлены.</p>']
    (target/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    print(target/'index.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results'));a=p.parse_args();main(a.output)
