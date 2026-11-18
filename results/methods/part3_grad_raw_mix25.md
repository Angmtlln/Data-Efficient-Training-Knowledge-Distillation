# part3: grad_raw_mix25

**Исправляющая абляция.** Исходный градиент с bias; ~75% общего случайного фона, ~25% targeted в фиксированных стратах.

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 92.737% ± 0.962 п.п. SD; macro F1 92.730%; positive F1 92.930%. Контроль `skill_matched_random_v2`; средняя разность -0.153 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 93.349 | 93.347 | 93.439 | 92.800 | 94.512 | 94.215 |
| 303 | 91.628 | 91.612 | 91.987 | 92.550 | 87.195 | 85.950 |
| 404 | 93.234 | 93.231 | 93.363 | 93.650 | 90.854 | 92.562 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/grad_raw_mix25__ss202__ts202__339f1c241dec/metrics.json), [предсказания](../raw/part3/runs/grad_raw_mix25__ss202__ts202__339f1c241dec/predictions.csv), [identity](../raw/part3/runs/grad_raw_mix25__ss202__ts202__339f1c241dec/identity.json).
- Seed 303: [метрики](../raw/part3/runs/grad_raw_mix25__ss303__ts303__814456c3685f/metrics.json), [предсказания](../raw/part3/runs/grad_raw_mix25__ss303__ts303__814456c3685f/predictions.csv), [identity](../raw/part3/runs/grad_raw_mix25__ss303__ts303__814456c3685f/identity.json).
- Seed 404: [метрики](../raw/part3/runs/grad_raw_mix25__ss404__ts404__28fae8c3dd02/metrics.json), [предсказания](../raw/part3/runs/grad_raw_mix25__ss404__ts404__28fae8c3dd02/predictions.csv), [identity](../raw/part3/runs/grad_raw_mix25__ss404__ts404__28fae8c3dd02/identity.json).
