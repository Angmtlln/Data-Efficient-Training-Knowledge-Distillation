# part1: lexical_diversity

**Оставить.** Доля уникальных слов внутри одного текста; это не разнообразие набора.

Средняя длина 6.41 слова против 9.28 у random. Выигрыш +0.23 п.п. создается главным образом seed 42; в двух других повторах хуже контроля.

Рекомендация исходного разбора: Не развивать как отдельное направление: слабый сигнал и смешение с длиной, уже исследованной коллегой.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 92.661% ± 0.303 п.п. SD; macro F1 92.654%; positive F1 92.864%. Контроль `random`; средняя разность +0.229 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 92.431 | 92.422 | 92.683 | 91.450 | 90.854 | 90.909 |
| 42 | 93.005 | 93.003 | 93.123 | 92.650 | 91.463 | 91.736 |
| 101 | 92.546 | 92.538 | 92.786 | 93.200 | 90.244 | 90.083 |

## Исходные результаты

- Seed 17: [метрики](../raw/part1/runs/lexical_diversity__ss17__ts17__d96db02e132e/metrics.json), [предсказания](../raw/part1/runs/lexical_diversity__ss17__ts17__d96db02e132e/predictions.csv), [identity](../raw/part1/runs/lexical_diversity__ss17__ts17__d96db02e132e/identity.json).
- Seed 42: [метрики](../raw/part1/runs/lexical_diversity__ss42__ts42__6ece66b0670a/metrics.json), [предсказания](../raw/part1/runs/lexical_diversity__ss42__ts42__6ece66b0670a/predictions.csv), [identity](../raw/part1/runs/lexical_diversity__ss42__ts42__6ece66b0670a/identity.json).
- Seed 101: [метрики](../raw/part1/runs/lexical_diversity__ss101__ts101__b646ba4847a2/metrics.json), [предсказания](../raw/part1/runs/lexical_diversity__ss101__ts101__b646ba4847a2/predictions.csv), [identity](../raw/part1/runs/lexical_diversity__ss101__ts101__b646ba4847a2/identity.json).
