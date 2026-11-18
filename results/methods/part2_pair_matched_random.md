# part2: pair_matched_random

**Контроль; не объявлять победителем.** Весь набор сопоставлен paired-набору по class×word-length-bin и исключает его строки.

93.35% — наивысшее среднее в этой серии, но выигрыш +0.38 п.п. неустойчив: отрицателен в двух seed. Это другой случайный набор, а не доказательство улучшения пары.

Рекомендация исходного разбора: Оставить как контроль. Отдельная программа улучшения длины дублировала бы коллегу.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 93.349% ± 0.526 п.п. SD; macro F1 93.346%; positive F1 93.475%. Контроль `pair_random`; средняя разность +0.382 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 93.463 | 93.462 | 93.559 | 92.350 | 91.463 | 90.909 |
| 42 | 92.775 | 92.770 | 92.961 | 93.300 | 89.634 | 90.909 |
| 101 | 93.807 | 93.806 | 93.905 | 92.750 | 92.073 | 93.388 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/pair_matched_random__ss17__ts17__e18ab15a9c7c/metrics.json), [предсказания](../raw/part2/runs/pair_matched_random__ss17__ts17__e18ab15a9c7c/predictions.csv), [identity](../raw/part2/runs/pair_matched_random__ss17__ts17__e18ab15a9c7c/identity.json).
- Seed 42: [метрики](../raw/part2/runs/pair_matched_random__ss42__ts42__dd7b24133470/metrics.json), [предсказания](../raw/part2/runs/pair_matched_random__ss42__ts42__dd7b24133470/predictions.csv), [identity](../raw/part2/runs/pair_matched_random__ss42__ts42__dd7b24133470/identity.json).
- Seed 101: [метрики](../raw/part2/runs/pair_matched_random__ss101__ts101__a587cb50ce60/metrics.json), [предсказания](../raw/part2/runs/pair_matched_random__ss101__ts101__a587cb50ce60/predictions.csv), [identity](../raw/part2/runs/pair_matched_random__ss101__ts101__a587cb50ce60/identity.json).
