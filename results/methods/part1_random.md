# part1: random

**Контроль.** Стратифицированный случайный отбор из полного pool.

92.43% — контроль первой части, не универсальный baseline всех групп.

Рекомендация исходного разбора: Повторять вместе с улучшенными методами на новых seed; не интерпретировать как исследовательский вклад.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 92.431% ± 0.827 п.п. SD; macro F1 92.425%; positive F1 92.636%. Контроль `random`; средняя разность +0.000 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 92.661 | 92.659 | 92.760 | 93.050 | 90.244 | 90.083 |
| 42 | 91.514 | 91.502 | 91.814 | 93.050 | 90.854 | 86.777 |
| 101 | 93.119 | 93.112 | 93.333 | 92.700 | 90.244 | 89.256 |

## Исходные результаты

- Seed 17: [метрики](../raw/part1/runs/random__ss17__ts17__02ce2ebe8d8b/metrics.json), [предсказания](../raw/part1/runs/random__ss17__ts17__02ce2ebe8d8b/predictions.csv), [identity](../raw/part1/runs/random__ss17__ts17__02ce2ebe8d8b/identity.json).
- Seed 42: [метрики](../raw/part1/runs/random__ss42__ts42__171e814d3d46/metrics.json), [предсказания](../raw/part1/runs/random__ss42__ts42__171e814d3d46/predictions.csv), [identity](../raw/part1/runs/random__ss42__ts42__171e814d3d46/identity.json).
- Seed 101: [метрики](../raw/part1/runs/random__ss101__ts101__9428c991d4b2/metrics.json), [предсказания](../raw/part1/runs/random__ss101__ts101__9428c991d4b2/predictions.csv), [identity](../raw/part1/runs/random__ss101__ts101__9428c991d4b2/identity.json).
