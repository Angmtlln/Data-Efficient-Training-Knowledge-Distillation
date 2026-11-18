# part2: contrast_embedding

**Отложить.** Близость к среднему reference с маркерами контраста.

Общая accuracy ниже skill_random во всех seed (-0.99 п.п. в среднем). На контрастном срезе +1.10 п.п., только 121 строка.

Рекомендация исходного разбора: Не приоритет: более слабый компромисс, чем у negation_embedding. Можно оставить отрицательным результатом глобального top-k.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 92.011% ± 0.066 п.п. SD; macro F1 92.011%; positive F1 92.002%. Контроль `skill_random`; средняя разность -0.994 п.п. Различных поднаборов: 1.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 92.087 | 92.087 | 92.042 | 91.000 | 90.244 | 92.562 |
| 42 | 91.972 | 91.972 | 91.972 | 91.100 | 89.634 | 90.909 |
| 101 | 91.972 | 91.972 | 91.991 | 91.450 | 92.073 | 92.562 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/contrast_embedding__ss17__ts17__c1cae725f81a/metrics.json), [предсказания](../raw/part2/runs/contrast_embedding__ss17__ts17__c1cae725f81a/predictions.csv), [identity](../raw/part2/runs/contrast_embedding__ss17__ts17__c1cae725f81a/identity.json).
- Seed 42: [метрики](../raw/part2/runs/contrast_embedding__ss42__ts42__312a2b2a9c3d/metrics.json), [предсказания](../raw/part2/runs/contrast_embedding__ss42__ts42__312a2b2a9c3d/predictions.csv), [identity](../raw/part2/runs/contrast_embedding__ss42__ts42__312a2b2a9c3d/identity.json).
- Seed 101: [метрики](../raw/part2/runs/contrast_embedding__ss101__ts101__649db3f1704f/metrics.json), [предсказания](../raw/part2/runs/contrast_embedding__ss101__ts101__649db3f1704f/predictions.csv), [identity](../raw/part2/runs/contrast_embedding__ss101__ts101__649db3f1704f/identity.json).
