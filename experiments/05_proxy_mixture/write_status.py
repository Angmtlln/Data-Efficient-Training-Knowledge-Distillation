"""Write a clearly preliminary status from validated summary.json."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]/'results/proxy_mixture'

def main():
    d=json.loads((ROOT/'summary.json').read_text());pairs=d['sst2']['paired_deltas']
    names={'mix_random':'Парный контроль','mix_hard':'Самые трудные','mix_uncertain':'Неопределённые','mix_mid':'Средняя сложность','mix_easy':'Самые лёгкие','mix_rare':'Редкие слова','mix_diverse':'Кластеры TF-IDF'}
    complete=d['completed_new_runs']==d['expected_new_runs'] and d.get('imdb',{}).get('completed_runs')==27
    stats={x['name']:x for x in d['sst2']['summary']};lines=['# Статус исследования','',f'**Готово {d["completed_new_runs"]} из {d["expected_new_runs"]} новых обучений.** '+('Внешняя оценка 27 моделей завершена; полные результаты в `REPORT.md`.' if complete else 'До завершения серии и внешней оценки выводы предварительны.'),'',
     '| Метод | Выполнено из 3 | Accuracy, среднее | Δ к парному контролю, п.п. |','|---|---:|---:|---:|']
    for name,label in names.items():
        x=stats.get(name);diff=[p['accuracy_delta_pp'] for p in pairs if p['name']==name]
        acc=f'{100*x["accuracy_mean"]:.2f}%' if x else '—';n=x['n'] if x else 0
        delta='0,00' if name=='mix_random' and x else (f'{np.mean(diff):+.2f}' if diff else '—')
        lines.append(f'| {label} | {n} | {acc} | {delta} |')
    lines+=['',('Выполнены все три сида на каждый метод.' if complete else 'Количество запусков различается. Средние по одному или двум сидам не являются подтверждёнными результатами.'),
            'GLUE validation уже многократно использовался; проверяются несколько гипотез, поправка на множественные сравнения отсутствует.',
            ('IMDb — отдельная внешняя проверка всех 27 моделей, результаты приведены в полном отчёте.' if complete else 'IMDb — отдельная внешняя проверка; до окончания всех обучений модели на нём не оцениваются.'),
            '', 'План и источники: `experiments/05_proxy_mixture/README.md` и `LITERATURE.md`.','']
    (ROOT/'STATUS.md').write_text('\n'.join(lines));print('\n'.join(lines))
if __name__=='__main__':main()
