# part2: negation_embedding

**Повторить: приоритет 3.** Близость MiniLM-эмбеддинга к среднему reference с маркерами отрицания.

Общая accuracy -0.31 п.п., но срез отрицаний 92.68% против 91.06% у skill_random (+1.63 п.п.); всего 164 тестовых строки. Средняя длина 14.46 слова — возможное смешение с длиной.

Рекомендация исходного разбора: Проверить смесь 25% целевого отбора + 75% random, quotas по class×length; обязательно random с тем же профилем длины и keyword-контроль. Положительный срез — повод для проверки, не доказанный навык.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 92.699% ± 0.403 п.п. SD; macro F1 92.696%; positive F1 92.815%. Контроль `skill_random`; средняя разность -0.306 п.п. Различных поднаборов: 1.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 93.119 | 93.117 | 93.228 | 92.400 | 93.293 | 90.909 |
| 42 | 92.317 | 92.316 | 92.360 | 91.650 | 93.293 | 89.256 |
| 101 | 92.661 | 92.655 | 92.857 | 91.250 | 91.463 | 88.430 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/negation_embedding__ss17__ts17__7e90aa6fbb58/metrics.json), [предсказания](../raw/part2/runs/negation_embedding__ss17__ts17__7e90aa6fbb58/predictions.csv), [identity](../raw/part2/runs/negation_embedding__ss17__ts17__7e90aa6fbb58/identity.json).
- Seed 42: [метрики](../raw/part2/runs/negation_embedding__ss42__ts42__f29981ce71c0/metrics.json), [предсказания](../raw/part2/runs/negation_embedding__ss42__ts42__f29981ce71c0/predictions.csv), [identity](../raw/part2/runs/negation_embedding__ss42__ts42__f29981ce71c0/identity.json).
- Seed 101: [метрики](../raw/part2/runs/negation_embedding__ss101__ts101__aab197f4ac7c/metrics.json), [предсказания](../raw/part2/runs/negation_embedding__ss101__ts101__aab197f4ac7c/predictions.csv), [identity](../raw/part2/runs/negation_embedding__ss101__ts101__aab197f4ac7c/identity.json).
