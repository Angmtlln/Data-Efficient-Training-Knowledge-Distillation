"""Create a reproducible research report from validated SST-2 and IMDb results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import prepare as base
from analyze import METHODS, SEEDS

NAMES = {
    'pool_random': 'Случайные из всего пула',
    'long_random': 'Случайные длинные — контроль',
    'long_mid': '25% средней сложности',
    'long_fd': '25% по частотному расстоянию',
    'long_fd_mid': '25% расстояние + средняя сложность',
    'long_gsnr': '25% динамика proxy',
    'long_roots_matched': '25% предполагаемых предложений',
    'long_roots_max': 'Максимум предполагаемых предложений',
}


def load(name):
    return json.loads((base.ROOT / name).read_text())


def main():
    sst = load('analysis.json')
    imdb = load('imdb_holdout_analysis.json')
    original = load('original_sst_analysis.json')
    by_sst = {r['method']: r for r in sst['summary']}
    by_imdb = {r['method']: r for r in imdb['summary']}
    by_original = {r['method']: r for r in original['summary']}
    if any(set(d) != set(METHODS) for d in (by_sst, by_imdb, by_original)):
        raise ValueError('Incomplete summary')
    audit = load('audit.json')['scores']
    cost = load('selection_cost_audit.json')['selections']
    root = load('inferred_roots_audit.json')
    source_audit = load('source_diversity_audit.json')
    difficulty_audit = load('difficulty_audit.json')['selections']
    original_audit = load('original_sst_audit.json')
    ext = imdb['sample']
    entries = []
    for name in METHODS:
        x, y, z = by_sst[name], by_imdb[name], by_original[name]
        delta_x = np.mean(x['delta_vs_long_random_pp'])
        delta_y = np.mean(y['delta_vs_long_random_pp'])
        tokens = np.mean([r['tokens_total_per_epoch'] for r in cost if r['name'] == name])
        entries.append(f"| {NAMES[name]} | {100*x['accuracy_mean']:.2f} ± {100*x['accuracy_sd']:.2f} | "
                       f"{100*x['macro_f1_mean']:.2f} | {delta_x:+.2f} | "
                       f"{100*y['accuracy_mean']:.2f} ± {100*y['accuracy_sd']:.2f} | "
                       f"{100*y['macro_f1_mean']:.2f} | {delta_y:+.2f} | "
                       f"{100*z['accuracy_mean']:.2f} ± {100*z['accuracy_sd']:.2f} | {100*z['macro_f1_mean']:.2f} | "
                       f"{np.mean(z['delta_vs_long_random_pp']):+.2f} | {tokens/1000:.1f} тыс. |")
    figure, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    ordered = list(METHODS)
    offsets = [-.13, 0, .13]
    for ax, data, title in zip(axes, (by_sst, by_original, by_imdb), ('SST-2 validation', 'SST-2 test: публичное зеркало', 'IMDb: новая выборка')):
        for pos, name in enumerate(ordered):
            d = np.asarray(data[name]['delta_vs_long_random_pp'])
            ax.scatter(d, np.full(3, pos) + offsets, color='#326ca8', s=27, zorder=3)
            ax.scatter([d.mean()], [pos], color='#d25b31', marker='D', s=33, zorder=4)
        ax.axvline(0, color='#777', linewidth=1)
        ax.grid(axis='x', color='#ddd')
        ax.set_title(title)
        ax.set_xlabel('Разница accuracy с контролем, п.п.')
        ax.set_yticks(range(len(ordered)), [NAMES[n] for n in ordered])
        ax.invert_yaxis()
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    figure.suptitle('Парные разницы по трём запускам (ромб — среднее)')
    figure.tight_layout()
    figure.savefig(base.ROOT / 'comparison.png', dpi=160, bbox_inches='tight')
    plt.close(figure)
    rows = '\n'.join(entries)
    source_rows = []
    for name in METHODS:
        groups = [r for r in source_audit['selections'] if r['name'] == name]
        if len(groups) != len(SEEDS):
            raise ValueError(f'Incomplete source audit: {name}')
        source_rows.append(f"| {NAMES[name]} | {100*np.mean([r['inferred_root_fraction'] for r in groups]):.1f} | "
                           f"{np.mean([r['distinct_uniquely_assigned_parents'] for r in groups]):.0f} | "
                           f"{np.mean([r['shared_with_uniquely_assigned_val_parent'] for r in groups]):.0f} |")
    source_table = '\n'.join(source_rows)
    difficulty_rows = []
    for name in METHODS:
        group = [r for r in difficulty_audit if r['name'] == name]
        if len(group) != len(SEEDS):
            raise ValueError(f'Incomplete difficulty audit: {name}')
        difficulty_rows.append(f"| {NAMES[name]} | {group[0]['scope']} | "
                               f"{np.mean([r['mean_gold_probability'] for r in group]):.3f} | "
                               f"{100*np.mean([r['proxy_disagreement_fraction'] for r in group]):.1f} |")
    difficulty_table = '\n'.join(difficulty_rows)
    def formatted(values):
        return ' / '.join(f'{v:+.2f}' for v in values)
    pair_rows = '\n'.join(
        f"| {NAMES[name]} | {formatted(by_sst[name]['delta_vs_long_random_pp'])} | "
        f"{formatted(by_original[name]['delta_vs_long_random_pp'])} | "
        f"{formatted(by_imdb[name]['delta_vs_long_random_pp'])} |"
        for name in METHODS if name != 'long_random'
    )
    clean_rows = '\n'.join(
        f"| {NAMES[name]} | {100*by_original[name]['accuracy_mean']:.2f} | "
        f"{100*by_original[name]['clean_accuracy_mean']:.2f} |"
        for name in METHODS
    )
    def method_mean(name, records, field):
        values = [r[field] for r in records if r['name'] == name]
        if not values:
            raise ValueError(f'Missing audit feature {name}: {field}')
        return float(np.mean(values))
    features = [('Средняя длина, слова', [method_mean(n, cost, 'mean_words') for n in METHODS]),
                ('Доля позитивных меток', [method_mean(n, cost, 'positive_fraction') for n in METHODS]),
                ('Доля предполагаемых целых предложений', [method_mean(n, source_audit['selections'], 'inferred_root_fraction') for n in METHODS]),
                ('Разные предполагаемые источники', [method_mean(n, source_audit['selections'], 'distinct_uniquely_assigned_parents') for n in METHODS])]
    target_accuracy = [by_original[n]['accuracy_mean'] for n in METHODS]
    corr_rows = []
    for title, values in features:
        rho = float(spearmanr(values, target_accuracy).statistic) if np.ptp(values) and np.ptp(target_accuracy) else float('nan')
        corr_rows.append(f"| {title} | {rho:+.2f} |" if np.isfinite(rho) else f"| {title} | не определено |")
    correlation_table = '\n'.join(corr_rows)
    slice_names = ('short_lt8', 'length_8_15', 'length_16_plus', 'negative', 'positive')
    slice_rows = []
    for name in METHODS:
        subset = [r for r in sst['runs'] if r['method'] == name]
        if len(subset) != len(SEEDS):
            raise ValueError(f'Incomplete slice data: {name}')
        cells = []
        for key in slice_names:
            values = [r['slices'][key]['accuracy'] for r in subset if key in r['slices']]
            cells.append(f'{100*np.mean(values):.2f}' if values else '—')
        slice_rows.append(f"| {NAMES[name]} | " + ' | '.join(cells) + ' |')
    slice_table = '\n'.join(slice_rows)
    slice_counts = {key: sst['runs'][0]['slices'].get(key, {}).get('n', 0) for key in slice_names}
    report = f'''# Отбор 10% SST-2: статьи 2025–2026 и эксперименты

## Постановка

24 обучения ModernBERT-base, 8 вариантов × 3 связанных seed (17/42/101), одинаковая модель и гиперпараметры. Каждый вариант содержит 6534 строки — около 10% исходного пула 65 349, или {100*6534/audit['eligible_count']:.2f}% пула после очистки. Взвешивания нет. Макро-F1 и accuracy рассчитаны независимо по сохранённым предсказаниям. Epoch выбирается по внутренним 2000 строкам train; SST-2 validation (872 строки) ранее многократно использовался для исследования и здесь не является слепым тестом.

До построения proxy и отбора исключены {audit['excluded_total']} строк, совпадающих по нормализованному тексту или имеющих общий восьмисловный фрагмент с внутренней val либо GLUE validation; осталось {audit['eligible_count']}. После очистки аудит обнаружил ноль таких пересечений. Более короткие или изменённые фрагменты могут остаться. Метки всего допустимого пула доступны proxy: бюджет 10% относится к финальному обучению, а не к разметке или proxy-обучению.

У `long_random`, `long_mid`, `long_fd`, `long_fd_mid`, `long_gsnr` и `long_roots_matched` внутри каждого seed одни и те же 4900 случайных длинных строк; 1634 оставшиеся имеют одинаковые квоты по метке и группам длины. `pool_random` и `long_roots_max` не имеют этого общего фона. `long_roots_max` включает все {root['eligible_long_root_rows']} допустимых предполагаемых корней длиной ≥8 слов и случайные другие длинные фразы. Корни восстановлены по лексической вложенности, а не по исходным деревьям; больше подробностей в `PROTOCOL_ROOTS.md`.

## Качество и стоимость

Среднее ± выборочное стандартное отклонение по трём запускам. Δ — средняя разница accuracy к `long_random` на тех же seed, в процентных пунктах; у методов без общего фона она не является строго парной заменой части выборки. Число токенов — сумма необёрнутых attention-padding токенов выбранных строк за эпоху после обрезки, а не реальное количество FLOPs.

| Метод | GLUE val accuracy ± SD, % | macro-F1, % | Δ val, п.п. | IMDb accuracy ± SD, % | macro-F1, % | Δ IMDb, п.п. | SST-2 test accuracy ± SD, % | macro-F1, % | Δ test, п.п. | Токенов / эпоха |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

![Разницы с контролем по сидам](comparison.png)

### Разница с контролем по каждому seed

Порядок seed: 17 / 42 / 101, процентные пункты. `pool_random` и `long_roots_max` не используют общий фон, поэтому их разница не является точечной заменой 25% примеров.

| Метод | GLUE validation | Открытый SST-2 test | IMDb |
|---|---|---|---|
{pair_rows}

### Срезы GLUE validation

Средняя accuracy в процентах по трём seed. Размеры срезов: <8 слов — {slice_counts['short_lt8']}, 8–15 — {slice_counts['length_8_15']}, 16+ — {slice_counts['length_16_plus']}, отрицательные — {slice_counts['negative']}, положительные — {slice_counts['positive']}. Все срезы исследовательские, а GLUE validation многократно использовалась при разработке проекта.

| Метод | <8 слов | 8–15 слов | 16+ слов | Отрицательные | Положительные |
|---|---:|---:|---:|---:|---:|
{slice_table}

Методы 2025/2026 — адаптации: `long_fd` вдохновлён [SCDP, COLING 2025](https://aclanthology.org/2025.coling-main.49/) и использует иные настройки TF-IDF, долю отбора и ограничения; `long_gsnr` вдохновлён [GRADFILTERING, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.2297/), но применяет ансамбль линейных моделей, а не LoRA к генеративной модели. Результаты не являются воспроизведением этих работ.

## Состав подмножеств

Показаны средние по seed характеристики, рассчитанные до оценки качества. Число источников и пересечение с внутренней val учитывают только фразы, для которых лексическая эвристика нашла ровно одного возможного родителя; это не восстановленные исходные деревья SST. Несмотря на удаление общих 8-граммов, обнаруженные совпадения предполагаемых родителей показывают риск утечки через короткие фрагменты при выборе эпохи.

| Метод | Предполагаемых целых предложений, % | Разных однозначно сопоставленных источников | Строк с предполагаемым родителем из внутренней val |
|---|---:|---:|---:|
{source_table}

Вероятность исходной метки и доля несовпадений ответа OOF proxy с меткой позволяют увидеть, какие данные реально попали в выборку. Несовпадение не является подтверждённой ошибкой разметки. Для всех методов с общим фоном измерены 1634 добавленные позиции; для `pool_random` и `long_roots_max` — всё подмножество, поэтому строки этих двух методов не вполне сопоставимы с остальными.

| Метод | Область расчёта | P(исходная метка) | Несогласие proxy, % |
|---|---|---:|---:|
{difficulty_table}

### Исследовательская связь с качеством

Ниже — корреляция Спирмена между характеристиками и средней accuracy на открытом SST-2 test на уровне восьми стратегий. Подмножества сильно пересекаются, характеристики зависят друг от друга, а сравниваемых методов мало; коэффициенты описывают лишь эту серию и не доказывают причину или перенос на другие задачи.

| Характеристика | Спирмен ρ |
|---|---:|
{correlation_table}

## Перенос и ограничения

Исходный SST-2 test: 1821 размеченное предложение из открытого [зеркала](https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/blob/master/data/stsa.binary.test), не официальная отправка на GLUE leaderboard. Из них 1814 после нормализации текстов совпадают с неразмеченным GLUE test; у семи различается запись. Открытый dev этого источника совпал по тексту и метке со всеми 872 примерами GLUE validation. Длину в 64 токена превышает {original_audit['test_over_64_tokens']} предложение теста. Варианты отбирались после изучения предыдущих экспериментов на validation; публичный test и возможное использование тех же данных на pretraining ограничивают независимость этой проверки.

Ниже — accuracy на всех 1821 примере и на 1818 примерах без точного совпадения или общего восьмисловного фрагмента с допустимым пулом. Для каждого метода показано среднее по трём seed.

| Метод | Все 1821, % | После исключения пересечений, % |
|---|---:|---:|
{clean_rows}

IMDb: новая фиксированная выборка {ext['n']} отзывов, отдельная от 2000 IMDb-примеров серии 05, по 1000 каждого класса. Точных нормализованных совпадений с SST-2: {ext['normalized_exact_sst2_any_split']}; отзывов с общим 8-граммом: {ext['containing_any_8token_ngram_from_sst2']}. Обрезаны до 64 токенов {ext['over_64_tokens']} из {ext['n']} отзывов. IMDb — иной домен, а пересечение с данными предварительного обучения неизвестно. Это проверка переноса, не новый слепой SST-2 тест.

Выборки и инициализации связаны seed; три повтора не разделяют вариативность выбора данных и обучения. Сравниваются несколько стратегий без поправки на множественные проверки. Уверенные выводы о малых разницах и превосходстве над коллегами без новых независимых повторов и оценки невозможны. Время CPU-просчёта proxy и время GPU-обучения сохранены раздельно, а часть моделей обучалась параллельно; `long_fd` seed 101 был продолжен с полного чекпоинта после первой эпохи. Wall time не является честным сравнением FLOPs или ускорения. Равное число примеров не означает равную стоимость по токенам.

Разбор статей — `experiments/07_literature_strict/LITERATURE.md`; результаты коллеги — `COLLABORATOR.md`; протоколы, манифесты, хэши, прогнозы и исходные метрики сохранены в ветке проекта.
'''
    (base.ROOT / 'REPORT.md').write_text(report)
    print('Wrote', base.ROOT / 'REPORT.md')


if __name__ == '__main__':
    main()
