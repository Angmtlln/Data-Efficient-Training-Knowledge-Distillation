# part2: negation_keyword

**Переработать как контроль.** Приоритет всем строкам с not/never/no/нетипографским n't внутри классовых квот.

68.52% набора содержат маркер. Для label=0 доля 100%, для label=1 — 43.57%. Общая accuracy падает на 6.54 п.п.; даже срез отрицаний хуже random.

Рекомендация исходного разбора: Не повторять hard top-k. Для мягкого обогащения ограничить долю маркеров отдельно в каждом классе и сохранить немаркерный фон; это контроль для embedding-метода.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 86.468% ± 2.645 п.п. SD; macro F1 86.243%; positive F1 87.960%. Контроль `skill_random`; средняя разность -6.537 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 84.404 | 84.092 | 86.318 | 71.400 | 89.634 | 85.124 |
| 42 | 89.450 | 89.385 | 90.213 | 79.950 | 90.854 | 91.736 |
| 101 | 85.550 | 85.252 | 87.349 | 75.550 | 89.634 | 87.603 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/negation_keyword__ss17__ts17__dd08fdfa140b/metrics.json), [предсказания](../raw/part2/runs/negation_keyword__ss17__ts17__dd08fdfa140b/predictions.csv), [identity](../raw/part2/runs/negation_keyword__ss17__ts17__dd08fdfa140b/identity.json).
- Seed 42: [метрики](../raw/part2/runs/negation_keyword__ss42__ts42__704bc4a802ff/metrics.json), [предсказания](../raw/part2/runs/negation_keyword__ss42__ts42__704bc4a802ff/predictions.csv), [identity](../raw/part2/runs/negation_keyword__ss42__ts42__704bc4a802ff/identity.json).
- Seed 101: [метрики](../raw/part2/runs/negation_keyword__ss101__ts101__5e2c44ae6367/metrics.json), [предсказания](../raw/part2/runs/negation_keyword__ss101__ts101__5e2c44ae6367/predictions.csv), [identity](../raw/part2/runs/negation_keyword__ss101__ts101__5e2c44ae6367/identity.json).
