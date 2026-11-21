"""Produce a descriptive report from validated results, without post-hoc selection."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'results/proxy_mixture'
METHODS=['random','len8','mix_random','mix_hard','mix_uncertain','mix_mid','mix_easy','mix_rare','mix_diverse']
LABELS={'random':'Random всего пула','len8':'Random среди длинных','mix_random':'Парный random-контроль',
        'mix_hard':'25% самых трудных','mix_uncertain':'25% неопределённых','mix_mid':'25% средней сложности',
        'mix_easy':'25% самых лёгких','mix_rare':'25% редких слов','mix_diverse':'25% по кластерам TF-IDF'}


def fmt(x):return '—' if x is None else f'{100*x:.2f}%'
def pm(r,key):return f'{fmt(r[key+"_mean"])} ± {100*r[key+"_sd"]:.2f} п.п.' if r[key+'_sd'] is not None else fmt(r[key+'_mean'])


def delta(group,name):
    if name=='mix_random':return '0,00'
    d=[x['accuracy_delta_pp'] for x in group['paired_deltas'] if x['name']==name]
    return f'{np.mean(d):+.2f}' if d else '—'


def main():
    data=json.loads((ROOT/'summary.json').read_text());imdb= data.get('imdb',{})
    if data['completed_new_runs']!=21 or imdb.get('completed_runs')!=27:raise ValueError('Full results required')
    audit=json.loads((ROOT/'selection_audit.json').read_text());meta=json.loads((ROOT/'proxy/metadata.json').read_text())
    external=json.loads((ROOT/'imdb_manifest.json').read_text());overlap=json.loads((ROOT/'rotten_overlap.json').read_text())
    groups={'sst2':data['sst2'],'imdb':data['imdb']};stats={k:{r['name']:r for r in v['summary']} for k,v in groups.items()}
    if any(stats[k][name]['n']!=3 for k in stats for name in METHODS):raise ValueError('Missing seeds')
    lines=['# Исследование отбора обучающих примеров для ModernBERT','',
        '**Завершено: 21 новое обучение и 27 оценок на IMDb.** Результаты исследовательские: по три связанных сида, несколько сравнений.',
        '', '## Постановка и дизайн','',
        'Модель — `answerdotai/ModernBERT-base`, SST-2 pool — 65 349 строк; каждое обучение использует 6534 строки (10%).',
        'Все новые варианты отбирают длинные фразы: 4900 строк — общий случайный фон, 1634 места меняются методом.',
        'Внутри одного сида для всех вариантов совпадают количество меток и распределение по группам длины.',
        'Лучший checkpoint выбирается по внутренней валидации из 2000 строк; основная оценка — 872 строки GLUE validation.',
        'Метки всего пула используются для обучения TF-IDF proxy и отбора; это не эксперимент с бюджетом разметки 10%.',
        'Seeds 17/42/101 одновременно задают proxy, выборку и обучение target. Методические детали и протокол — в `experiments/05_proxy_mixture/README.md`.',
        '', '## Результаты','', 'Среднее ± выборочное стандартное отклонение по трём запускам; Δ — средняя парная разница accuracy',
        'с `mix_random` того же сида в процентных пунктах. У двух baseline из прошлой серии общий фон не фиксирован, поэтому Δ не приводится.','',
        '| Метод | SST-2 accuracy ± SD | Macro F1 | Δ, п.п. | IMDb accuracy ± SD | Macro F1 | Δ, п.п. |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in METHODS:
        s=stats['sst2'][name];i=stats['imdb'][name]
        lines.append(f'| {LABELS[name]} | {pm(s,"accuracy")} | {fmt(s["macro_f1_mean"])} | {delta(groups["sst2"],name)} | {pm(i,"accuracy")} | {fmt(i["macro_f1_mean"])} | {delta(groups["imdb"],name)} |')
    lines+=['','![Парные разницы по сидам на двух наборах](comparison.png)','',
        '### Разница с парным контролем по сидам','',
        '| Метод | SST-2 seed 17 / 42 / 101, п.п. | IMDb seed 17 / 42 / 101, п.п. |',
        '|---|---|---|']
    for name in METHODS[3:]:
        vals=[]
        for group in groups.values():
            pairs=[next(p['accuracy_delta_pp'] for p in group['paired_deltas'] if p['name']==name and p['seed']==seed) for seed in (17,42,101)]
            vals.append(' / '.join(f'{p:+.2f}' for p in pairs))
        lines.append(f'| {LABELS[name]} | {vals[0]} | {vals[1]} |')
    # Observational description only, not proof of causal/significant benefit.
    best=max(METHODS[2:],key=lambda n:stats['sst2'][n]['accuracy_mean'])
    mid_sst=[x['accuracy_delta_pp'] for x in groups['sst2']['paired_deltas'] if x['name']=='mix_mid']
    mid_imdb=[x['accuracy_delta_pp'] for x in groups['imdb']['paired_deltas'] if x['name']=='mix_mid']
    diverse_imdb=[x['accuracy_delta_pp'] for x in groups['imdb']['paired_deltas'] if x['name']=='mix_diverse']
    long_best=stats['sst2']['len8']['accuracy_mean']>stats['sst2'][best]['accuracy_mean']
    lines+=['', '## Что видно в данных','',
        f'- Среди новых вариантов на использованной в проекте SST-2-оценке наибольшую среднюю accuracy получил `{best}` ({fmt(stats["sst2"][best]["accuracy_mean"])}); это результат поиска среди нескольких методов, а не подтверждение превосходства.',
        f'- Простой случайный отбор длинных фраз из серии 04 дал {fmt(stats["sst2"]["len8"]["accuracy_mean"])} на SST-2; это {"выше" if long_best else "не выше"} среднего лучшего нового варианта. Выборки другие, поэтому это не парная оценка эффекта конкретного критерия.',
        f'- Отбор средней по группе уверенности показал положительную разницу с контролем на {sum(x>0 for x in mid_sst)} из 3 сидов SST-2 и на {sum(x>0 for x in mid_imdb)} из 3 сидов IMDb. Этого недостаточно, чтобы утверждать перенос эффекта.',
        f'- Кластерный отбор на IMDb показал отрицательную разницу с контролем на {sum(x<0 for x in diverse_imdb)} из 3 сидов; это наблюдение на длинных обрезанных отзывах, а не универсальный вывод о разнообразии.',
        '- Главный контроль — `mix_random`: он имеет тот же фон и те же квоты класса и длины, что и специальные варианты.',
        '- В таблице исходные `random` и `len8` взяты из серии 04, обученной с теми же моделью, сплитами и окружением; это полезные, но не идентично подобранные контроли.',
        '', '### Что реально выбрали методы','',
        'В таблице — средняя длина всей выборки и характеристики добавленных 1634 целевых строк, усреднённые по трём сидам.',
        'Ошибка proxy — несовпадение его OOF-ответа с существующей меткой, а не подтверждённая ошибка разметки.','',
        '| Метод | Средняя длина, слов | Proxy не согласен с меткой | Средняя вероятность метки | Затронуто кластеров |',
        '|---|---:|---:|---:|---:|']
    for name in METHODS[2:]:
        rr=[r for r in audit if r['name']==name]
        lines.append(f'| {LABELS[name]} | {np.mean([r["mean_words"] for r in rr]):.2f} | {100*np.mean([r["target_oof_wrong_fraction"] for r in rr]):.1f}% | {np.mean([r["target_oof_gold_mean"] for r in rr]):.3f} | {np.mean([r["target_cluster_count"] for r in rr]):.1f} из 64 |')
    lines+=['',f'Наибольший TF-IDF-кластер содержит {100*meta["cluster_largest_share"]:.1f}% всех допустимых длинных строк; равномерный отбор по кластерам меняет исходное распределение.',
        '', '## Внешняя оценка и совпадения','',
        f'IMDb: {external["n"]} отзывов, по 1000 каждого класса, случайная фиксированная выборка test split;',
        f'{external["normalized_exact_sst2_any_split"]} точных совпадений со строками SST-2 после нормализации,',
        f'{external["containing_any_8token_ngram_from_sst2"]} отзывов содержат общий фрагмент из восьми токенов.',
        f'{external["over_64_tokens"]} из {external["n"]} отзывов длиннее максимума в 64 токена. IMDb оценивает перенос на иной тип длинных отзывов;',
        'эти метрики нельзя считать прямой оценкой на новых SST-2 предложениях. Совпадения с данными предварительного обучения модели не проверены.',
        f'Rotten Tomatoes исключён как независимый тест: {overlap["exact_any_sst2_split"]} из {overlap["n"]} строк совпадают с SST-2 после нормализации.',
        '', '## Ограничения','',
        '- Лишь три сида, с привязанными друг к другу seed proxy, отбора и обучения. Малые разницы ненадёжны.',
        '- Шесть новых критериев сравниваются с контролем; поправка на множественные проверки не проведена.',
        '- GLUE validation уже многократно использовался в проекте; выбор лучшего метода по нему оптимистичен.',
        '- 24 текста внутренней валидации ранее обнаружены в пуле; родство фрагментов исходных предложений проверено не полностью.',
        '- Скор ошибки proxy не доказывает ошибку разметки; требуется независимая человеческая проверка.',
        '- Время обучения не является сравнением ускорения: два процесса делили GPU; метки и затраты proxy учтены отдельно.',
        '', '## Статьи','',
        '- [Dataset Cartography](https://arxiv.org/abs/2009.10795): динамика уверенности и разновидности трудности; наш OOF TF-IDF — иной proxy.',
        '- [Selection via Proxy](https://arxiv.org/abs/1906.11829): возможность использовать недорогую модель для отбора.',
        '- [Coverage-centric Coreset Selection](https://arxiv.org/abs/2210.15809): значимость покрытия при сильном сокращении данных; кластеризация здесь не CCS.',
        '- [Beyond Neural Scaling Laws](https://arxiv.org/abs/2206.14486): зависимость пользы критериев от режима и размера набора; основные эксперименты статьи — на изображениях.',
        '- [Disentangling the Roles of Representation and Selection in Data Pruning](https://aclanthology.org/2025.acl-long.821/): роль представления и алгоритма отбора.',
        '', 'Код, индексы, хэши, аудиты, скоры и предсказания находятся в ветке `AmirBikineev`: `experiments/05_proxy_mixture/` и `results/proxy_mixture/`.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines))

if __name__=='__main__':main()
