# part3: forget_random

**Контроль.** Стратифицированный random из полного pool, новые seeds 202/303/404.

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 92.737% ± 0.175 п.п. SD; macro F1 92.731%; positive F1 92.941%. Контроль `forget_matched_zero`; средняя разность +0.879 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 92.546 | 92.541 | 92.737 | 92.950 | 89.024 | 90.909 |
| 303 | 92.775 | 92.771 | 92.945 | 93.350 | 92.073 | 85.124 |
| 404 | 92.890 | 92.880 | 93.142 | 92.900 | 91.463 | 90.909 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/forget_random__ss202__ts202__9c2d9b277b90/metrics.json), [предсказания](../raw/part3/runs/forget_random__ss202__ts202__9c2d9b277b90/predictions.csv), [identity](../raw/part3/runs/forget_random__ss202__ts202__9c2d9b277b90/identity.json).
- Seed 303: [метрики](../raw/part3/runs/forget_random__ss303__ts303__5f8cbc5c7e1f/metrics.json), [предсказания](../raw/part3/runs/forget_random__ss303__ts303__5f8cbc5c7e1f/predictions.csv), [identity](../raw/part3/runs/forget_random__ss303__ts303__5f8cbc5c7e1f/identity.json).
- Seed 404: [метрики](../raw/part3/runs/forget_random__ss404__ts404__6ca8eb7f57e6/metrics.json), [предсказания](../raw/part3/runs/forget_random__ss404__ts404__6ca8eb7f57e6/predictions.csv), [identity](../raw/part3/runs/forget_random__ss404__ts404__6ca8eb7f57e6/identity.json).
