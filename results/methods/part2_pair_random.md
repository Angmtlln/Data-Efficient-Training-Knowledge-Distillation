# part2: pair_random

**Контроль.** Случайный набор с тем же числом меток каждого класса, что и paired treatment.

92.97%, базовый контроль группы пар. Квоты между seed немного меняются вместе с paired-набором.

Рекомендация исходного разбора: Сохранить; не подменять random из других групп.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 92.966% ± 0.632 п.п. SD; macro F1 92.959%; positive F1 93.163%. Контроль `pair_random`; средняя разность +0.000 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 93.578 | 93.574 | 93.736 | 93.150 | 90.854 | 91.736 |
| 42 | 93.005 | 93.003 | 93.092 | 93.100 | 90.244 | 90.909 |
| 101 | 92.317 | 92.299 | 92.662 | 93.650 | 92.073 | 89.256 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/pair_random__ss17__ts17__fe3a62732c4d/metrics.json), [предсказания](../raw/part2/runs/pair_random__ss17__ts17__fe3a62732c4d/predictions.csv), [identity](../raw/part2/runs/pair_random__ss17__ts17__fe3a62732c4d/identity.json).
- Seed 42: [метрики](../raw/part2/runs/pair_random__ss42__ts42__66cc49f3161b/metrics.json), [предсказания](../raw/part2/runs/pair_random__ss42__ts42__66cc49f3161b/predictions.csv), [identity](../raw/part2/runs/pair_random__ss42__ts42__66cc49f3161b/identity.json).
- Seed 101: [метрики](../raw/part2/runs/pair_random__ss101__ts101__a5efa350ba9d/metrics.json), [предсказания](../raw/part2/runs/pair_random__ss101__ts101__a5efa350ba9d/predictions.csv), [identity](../raw/part2/runs/pair_random__ss101__ts101__a5efa350ba9d/identity.json).
