# part1: dedup_exact

**Оставить.** Убрать точные повторения, затем случайно выбрать строки.

В pool всего 347 повторных строк (0.53%); у каждого random-поднабора лишь 2 повторных текста. Эффект +0.04 п.п. практически нулевой.

Рекомендация исходного разбора: Оставить как техническую проверку, новых GPU-прогонов ради этой эвристики не нужно.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 92.469% ± 0.350 п.п. SD; macro F1 92.468%; positive F1 92.573%. Контроль `random`; средняя разность +0.038 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 92.775 | 92.774 | 92.881 | 93.100 | 90.854 | 91.736 |
| 42 | 92.087 | 92.087 | 92.150 | 93.250 | 90.244 | 91.736 |
| 101 | 92.546 | 92.543 | 92.688 | 93.350 | 90.244 | 92.562 |

## Исходные результаты

- Seed 17: [метрики](../raw/part1/runs/dedup_exact__ss17__ts17__a52ceb7fa61d/metrics.json), [предсказания](../raw/part1/runs/dedup_exact__ss17__ts17__a52ceb7fa61d/predictions.csv), [identity](../raw/part1/runs/dedup_exact__ss17__ts17__a52ceb7fa61d/identity.json).
- Seed 42: [метрики](../raw/part1/runs/dedup_exact__ss42__ts42__5c86496e523b/metrics.json), [предсказания](../raw/part1/runs/dedup_exact__ss42__ts42__5c86496e523b/predictions.csv), [identity](../raw/part1/runs/dedup_exact__ss42__ts42__5c86496e523b/identity.json).
- Seed 101: [метрики](../raw/part1/runs/dedup_exact__ss101__ts101__77e99bbb2cef/metrics.json), [предсказания](../raw/part1/runs/dedup_exact__ss101__ts101__77e99bbb2cef/predictions.csv), [identity](../raw/part1/runs/dedup_exact__ss101__ts101__77e99bbb2cef/identity.json).
