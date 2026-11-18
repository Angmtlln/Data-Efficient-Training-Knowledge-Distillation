# part2: contrast_keyword

**Оставить как контроль.** Приоритет but/although/however/yet; остальное случайно.

Доля маркеров увеличена до 39.73%, но общей пользы нет: -0.19 п.п.; на срезе лишь +0.28 п.п.

Рекомендация исходного разбора: Самостоятельно не расширять; использовать только контроль, если вернемся к целевому контрасту.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 92.813% ± 0.175 п.п. SD; macro F1 92.811%; positive F1 92.922%. Контроль `skill_random`; средняя разность -0.191 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 92.775 | 92.774 | 92.849 | 93.450 | 90.854 | 90.083 |
| 42 | 93.005 | 93.004 | 93.076 | 93.000 | 90.244 | 93.388 |
| 101 | 92.661 | 92.656 | 92.841 | 91.750 | 87.195 | 90.083 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/contrast_keyword__ss17__ts17__9d91c98e7bcc/metrics.json), [предсказания](../raw/part2/runs/contrast_keyword__ss17__ts17__9d91c98e7bcc/predictions.csv), [identity](../raw/part2/runs/contrast_keyword__ss17__ts17__9d91c98e7bcc/identity.json).
- Seed 42: [метрики](../raw/part2/runs/contrast_keyword__ss42__ts42__f973c689ec90/metrics.json), [предсказания](../raw/part2/runs/contrast_keyword__ss42__ts42__f973c689ec90/predictions.csv), [identity](../raw/part2/runs/contrast_keyword__ss42__ts42__f973c689ec90/identity.json).
- Seed 101: [метрики](../raw/part2/runs/contrast_keyword__ss101__ts101__1381ffca30a8/metrics.json), [предсказания](../raw/part2/runs/contrast_keyword__ss101__ts101__1381ffca30a8/predictions.csv), [identity](../raw/part2/runs/contrast_keyword__ss101__ts101__1381ffca30a8/identity.json).
