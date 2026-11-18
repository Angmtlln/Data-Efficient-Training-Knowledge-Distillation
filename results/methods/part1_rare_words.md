# part1: rare_words

**Оставить.** Среднее 1/document-frequency по словам, top внутри классов.

Один и тот же набор во всех seed, медиана длины 2 слова; средняя accuracy ниже random на 1.19 п.п., большой разброс обучения.

Рекомендация исходного разбора: Сохранить как отрицательный результат лексической редкости. Улучшение через длину/фильтрацию будет пересекаться с работой коллеги.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 91.246% ± 1.941 п.п. SD; macro F1 91.218%; positive F1 91.708%. Контроль `random`; средняя разность -1.185 п.п. Различных поднаборов: 1.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 92.890 | 92.880 | 93.142 | 92.200 | 89.634 | 90.083 |
| 42 | 91.743 | 91.714 | 92.208 | 91.300 | 89.634 | 88.430 |
| 101 | 89.106 | 89.059 | 89.774 | 90.450 | 84.756 | 82.645 |

## Исходные результаты

- Seed 17: [метрики](../raw/part1/runs/rare_words__ss17__ts17__fded1532014d/metrics.json), [предсказания](../raw/part1/runs/rare_words__ss17__ts17__fded1532014d/predictions.csv), [identity](../raw/part1/runs/rare_words__ss17__ts17__fded1532014d/identity.json).
- Seed 42: [метрики](../raw/part1/runs/rare_words__ss42__ts42__989aee4f5974/metrics.json), [предсказания](../raw/part1/runs/rare_words__ss42__ts42__989aee4f5974/predictions.csv), [identity](../raw/part1/runs/rare_words__ss42__ts42__989aee4f5974/identity.json).
- Seed 101: [метрики](../raw/part1/runs/rare_words__ss101__ts101__f02a8622202c/metrics.json), [предсказания](../raw/part1/runs/rare_words__ss101__ts101__f02a8622202c/predictions.csv), [identity](../raw/part1/runs/rare_words__ss101__ts101__f02a8622202c/identity.json).
