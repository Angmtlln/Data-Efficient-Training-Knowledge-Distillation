# part3: neg_embedding_mix25

**Кандидат, слабый сигнал.** Мягкое negation-embedding нацеливание с тем же ~75% фоном и квотами. Небольшой плюс overall, практически нулевой плюс на лексическом negation-срезе.

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 93.005% ± 0.526 п.п. SD; macro F1 92.998%; positive F1 93.178%. Контроль `skill_matched_random_v2`; средняя разность +0.115 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 92.890 | 92.878 | 93.172 | 92.800 | 91.463 | 91.736 |
| 303 | 92.546 | 92.546 | 92.571 | 93.800 | 90.244 | 88.430 |
| 404 | 93.578 | 93.570 | 93.792 | 92.750 | 93.293 | 90.909 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/neg_embedding_mix25__ss202__ts202__bfec27e8eb8d/metrics.json), [предсказания](../raw/part3/runs/neg_embedding_mix25__ss202__ts202__bfec27e8eb8d/predictions.csv), [identity](../raw/part3/runs/neg_embedding_mix25__ss202__ts202__bfec27e8eb8d/identity.json).
- Seed 303: [метрики](../raw/part3/runs/neg_embedding_mix25__ss303__ts303__e8fea0092fbc/metrics.json), [предсказания](../raw/part3/runs/neg_embedding_mix25__ss303__ts303__e8fea0092fbc/predictions.csv), [identity](../raw/part3/runs/neg_embedding_mix25__ss303__ts303__e8fea0092fbc/identity.json).
- Seed 404: [метрики](../raw/part3/runs/neg_embedding_mix25__ss404__ts404__e4d08c5b5310/metrics.json), [предсказания](../raw/part3/runs/neg_embedding_mix25__ss404__ts404__e4d08c5b5310/predictions.csv), [identity](../raw/part3/runs/neg_embedding_mix25__ss404__ts404__e4d08c5b5310/identity.json).
