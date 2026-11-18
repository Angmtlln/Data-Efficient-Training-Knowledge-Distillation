# part2: contrast_randomref_gradient

**Сохранить как абляцию.** Случайный reference, сопоставленный contrast-reference по классу и бинам длины.

75.65% accuracy и SD 5.29 п.п.; -17.35 п.п. к skill_random. Reference-специфичность не объясняет провал.

Рекомендация исходного разбора: Старый метод больше не обучать. Повторять только как контроль новой формулы.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 75.650% ± 5.286 п.п. SD; macro F1 74.299%; positive F1 68.511%. Контроль `skill_random`; средняя разность -17.355 п.п. Различных поднаборов: 1.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 76.032 | 74.877 | 69.489 | 76.850 | 71.341 | 71.901 |
| 42 | 70.183 | 67.827 | 59.119 | 74.350 | 65.854 | 64.463 |
| 101 | 80.734 | 80.194 | 76.923 | 78.150 | 75.610 | 75.207 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/contrast_randomref_gradient__ss17__ts17__31c434010a27/metrics.json), [предсказания](../raw/part2/runs/contrast_randomref_gradient__ss17__ts17__31c434010a27/predictions.csv), [identity](../raw/part2/runs/contrast_randomref_gradient__ss17__ts17__31c434010a27/identity.json).
- Seed 42: [метрики](../raw/part2/runs/contrast_randomref_gradient__ss42__ts42__6580976fb709/metrics.json), [предсказания](../raw/part2/runs/contrast_randomref_gradient__ss42__ts42__6580976fb709/predictions.csv), [identity](../raw/part2/runs/contrast_randomref_gradient__ss42__ts42__6580976fb709/identity.json).
- Seed 101: [метрики](../raw/part2/runs/contrast_randomref_gradient__ss101__ts101__f55ebaf885ea/metrics.json), [предсказания](../raw/part2/runs/contrast_randomref_gradient__ss101__ts101__f55ebaf885ea/predictions.csv), [identity](../raw/part2/runs/contrast_randomref_gradient__ss101__ts101__f55ebaf885ea/identity.json).
