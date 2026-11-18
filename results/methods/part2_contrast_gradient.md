# part2: contrast_gradient

**Переработать в общем семействе.** Та же формула, reference контраста.

Проекция положительна везде; выбраны 100% proxy-ошибок среди label=0 и 0% среди label=1. -15.86 п.п. общей accuracy.

Рекомендация исходного разбора: Не отдельный большой sweep. Использовать как один диагностический контраст при исправлении общей формулы.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 77.141% ± 1.289 п.п. SD; macro F1 76.163%; positive F1 71.340%. Контроль `skill_random`; средняя разность -15.864 п.п. Различных поднаборов: 1.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 76.032 | 74.902 | 69.578 | 76.500 | 70.732 | 71.901 |
| 42 | 76.835 | 75.828 | 70.893 | 75.150 | 70.732 | 69.421 |
| 101 | 78.555 | 77.759 | 73.550 | 78.850 | 73.171 | 75.207 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/contrast_gradient__ss17__ts17__7cac58212ed1/metrics.json), [предсказания](../raw/part2/runs/contrast_gradient__ss17__ts17__7cac58212ed1/predictions.csv), [identity](../raw/part2/runs/contrast_gradient__ss17__ts17__7cac58212ed1/identity.json).
- Seed 42: [метрики](../raw/part2/runs/contrast_gradient__ss42__ts42__9b1b3bfde462/metrics.json), [предсказания](../raw/part2/runs/contrast_gradient__ss42__ts42__9b1b3bfde462/predictions.csv), [identity](../raw/part2/runs/contrast_gradient__ss42__ts42__9b1b3bfde462/identity.json).
- Seed 101: [метрики](../raw/part2/runs/contrast_gradient__ss101__ts101__7b5ff741f6ec/metrics.json), [предсказания](../raw/part2/runs/contrast_gradient__ss101__ts101__7b5ff741f6ec/predictions.csv), [identity](../raw/part2/runs/contrast_gradient__ss101__ts101__7b5ff741f6ec/identity.json).
