# part2: skill_random

**Контроль.** Random из eligible pool с исключением 4096 proxy-train и 512 reference строк.

93.00% — контроль только семейства навыков. Исключенные 4608 строк и другой генератор поднабора меняют baseline.

Рекомендация исходного разбора: Обязателен при каждом повторе skill-методов на том же eligible pool.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 93.005% ± 0.459 п.п. SD; macro F1 93.000%; positive F1 93.168%. Контроль `skill_random`; средняя разность +0.000 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 93.463 | 93.457 | 93.660 | 93.250 | 93.293 | 91.736 |
| 42 | 93.005 | 93.004 | 93.076 | 93.250 | 90.854 | 90.083 |
| 101 | 92.546 | 92.539 | 92.770 | 93.400 | 89.024 | 90.909 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/skill_random__ss17__ts17__26eab501c1ed/metrics.json), [предсказания](../raw/part2/runs/skill_random__ss17__ts17__26eab501c1ed/predictions.csv), [identity](../raw/part2/runs/skill_random__ss17__ts17__26eab501c1ed/identity.json).
- Seed 42: [метрики](../raw/part2/runs/skill_random__ss42__ts42__d3fa164a70be/metrics.json), [предсказания](../raw/part2/runs/skill_random__ss42__ts42__d3fa164a70be/predictions.csv), [identity](../raw/part2/runs/skill_random__ss42__ts42__d3fa164a70be/identity.json).
- Seed 101: [метрики](../raw/part2/runs/skill_random__ss101__ts101__4e4809e586ca/metrics.json), [предсказания](../raw/part2/runs/skill_random__ss101__ts101__4e4809e586ca/predictions.csv), [identity](../raw/part2/runs/skill_random__ss101__ts101__4e4809e586ca/identity.json).
