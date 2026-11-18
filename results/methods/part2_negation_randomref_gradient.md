# part2: negation_randomref_gradient

**Сохранить как абляцию.** Тот же proxy и формула, reference случайный, matched по классу и длине.

Провал сохраняется без skill-reference: -12.61 п.п. Проекция положительна везде, та же асимметрия трудности.

Рекомендация исходного разбора: Обязательный random-reference контроль исправленного градиентного метода; старые результаты не выбрасывать.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 80.390% ± 3.290 п.п. SD; macro F1 79.825%; positive F1 76.511%. Контроль `skill_random`; средняя разность -12.615 п.п. Различных поднаборов: 1.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 78.211 | 77.412 | 73.164 | 75.200 | 73.780 | 71.901 |
| 42 | 78.784 | 78.092 | 74.198 | 77.300 | 73.171 | 73.554 |
| 101 | 84.174 | 83.972 | 82.171 | 78.550 | 81.098 | 80.992 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/negation_randomref_gradient__ss17__ts17__c03ec31af20e/metrics.json), [предсказания](../raw/part2/runs/negation_randomref_gradient__ss17__ts17__c03ec31af20e/predictions.csv), [identity](../raw/part2/runs/negation_randomref_gradient__ss17__ts17__c03ec31af20e/identity.json).
- Seed 42: [метрики](../raw/part2/runs/negation_randomref_gradient__ss42__ts42__a9c036479802/metrics.json), [предсказания](../raw/part2/runs/negation_randomref_gradient__ss42__ts42__a9c036479802/predictions.csv), [identity](../raw/part2/runs/negation_randomref_gradient__ss42__ts42__a9c036479802/identity.json).
- Seed 101: [метрики](../raw/part2/runs/negation_randomref_gradient__ss101__ts101__9dbf96145f3e/metrics.json), [предсказания](../raw/part2/runs/negation_randomref_gradient__ss101__ts101__9dbf96145f3e/predictions.csv), [identity](../raw/part2/runs/negation_randomref_gradient__ss101__ts101__9dbf96145f3e/identity.json).
