"""Render a concise report from validated run-level metrics and summary.json."""
import argparse
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO/'results/len_forgetting'


def fmt(value):
    return '—' if value is None else f'{100*value:.2f}%'


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    summary=json.loads((ROOT/'summary.json').read_text())
    if summary['n_runs']!=summary['expected_runs'] and not args.allow_partial:
        raise ValueError('Not all runs complete; pass --allow-partial for an explicitly preliminary report')
    status='ИТОГ' if summary['n_runs']==summary['expected_runs'] else 'ПРЕДВАРИТЕЛЬНЫЙ ОТЧЁТ'
    lines=['# Длинные и нестабильные примеры для SST-2','',
        f'**{status}: {summary["n_runs"]} из {summary["expected_runs"]} обучений.**',
        '', '## Условия', '',
        'ModernBERT-base обучается три эпохи на 6534 примерах (10% пула).',
        'Условия `len8_matched_random` и `len8_forgetting` содержат 6207 общих строк длиной ≥ 8 слов;',
        'оставшиеся 327 различаются: нулевой или положительный held-out forgetting score.',
        'Заменяемые строки сопоставлены по метке и группе длины. Прокси и отбор используют метки всего пула.',
        'Скоры получены ранее TF-IDF + SGD на примерах вне обучения соответствующего proxy;',
        'используются архивные сиды 17, 42, 101. Оценка — accuracy и F1 на GLUE validation из 872 строк;',
        'лучшая эпоха выбирается по выделенным из train 2000 строкам.',
        '', '## Результаты', '',
        'Показаны средние и выборочное стандартное отклонение (SD) по запускам;',
        'SD не является доверительным интервалом.', '',
        '| Метод | Запусков | Accuracy ± SD | Macro F1 ± SD | Positive F1 |',
        '|---|---:|---:|---:|---:|']
    order=['random','len8','len8_matched_random','len8_forgetting']
    by_name={v['name']:v for v in summary['summary']}
    for name in order:
        if name not in by_name: continue
        x=by_name[name]
        lines.append(f'| `{name}` | {x["n"]} | {fmt(x["accuracy_mean"])} ± {fmt(x["accuracy_sd"])} | {fmt(x["macro_f1_mean"])} ± {fmt(x["macro_f1_sd"])} | {fmt(x["positive_f1_mean"])} |')
    if summary['n_runs']==summary['expected_runs']:
        lines+=['', '![Результаты по сидам и парные разницы](comparison.png)']
    lines+=['','### Парное сравнение с контролем','',
        'Положительное значение означает преимущество `len8_forgetting` над `len8_matched_random`.','',
        '| Seed отбора / обучения | Δ accuracy, п.п. | Δ macro F1, п.п. | Верно только контроль | Верно только эксперимент |',
        '|---|---:|---:|---:|---:|']
    for x in summary['paired_comparisons']:
        lines.append(f'| {x["selection_seed"]} / {x["training_seed"]} | {x["accuracy_delta_pp"]:+.3f} | {x["macro_f1_delta_pp"]:+.3f} | {x["control_correct_only"]} | {x["treatment_correct_only"]} |')
    if summary['paired_comparisons']:
        delta=np.asarray([p['accuracy_delta_pp'] for p in summary['paired_comparisons']])
        lines+=['',f'Средняя парная разница accuracy: **{delta.mean():+.3f} п.п.** по {len(delta)} парам.']
        if summary['n_runs']==summary['expected_runs']:
            if delta.mean() <= 0:
                lines+=['', 'В этой серии добавка нестабильных примеров не улучшила среднюю accuracy относительно парного контроля.',
                        'Три пары не позволяют исключить небольшой эффект в любую сторону.']
            else:
                lines+=['', 'Добавка нестабильных примеров дала положительную среднюю разницу относительно парного контроля.',
                        'Трёх пар недостаточно для уверенного вывода об устойчивом эффекте.']
    lines+=['', '## Ограничения и воспроизведение','',
        '- Это исследовательский результат: GLUE validation уже многократно использовался в проекте.',
        '- Три пары — малая выборка; score seed, seed поднабора и seed обучения меняются вместе, поэтому их вклад не разделён.',
        '- Выборки используют ранее рассчитанные оценки и те же сиды proxy, что и прежние опыты; это не независимое подтверждение.',
        '- Контрольные замены имеют нулевой score. Общий фон может включать другие ненулевые строки, а `len8` не ограничен ими.',
        '- Эксперименты шли в float32 на MPS. Два процесса делили GPU, поэтому время отдельных запусков не является оценкой ускорения.',
        '- Внутренний val унаследован от прежнего протокола; ранее в проекте отмечены 24 точных совпадения текстов с pool.',
        '- Результат не учитывает историческую стоимость расчёта proxy. Для сильного вывода нужна новая итоговая оценка.',
        '', 'Состав, код и команды — в `experiments/04_len_forgetting/README.md`.',
        'Индексы, аудит, предсказания и файлы метрик — в `results/len_forgetting/`.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':main()
