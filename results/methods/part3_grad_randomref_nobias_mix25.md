# part3: grad_randomref_nobias_mix25

**Контроль reference.** Та же смесь/квоты/градиент без bias, но matched-random reference вместо negation-reference.

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 91.896% ± 0.781 п.п. SD; macro F1 91.883%; positive F1 92.180%. Контроль `skill_matched_random_v2`; средняя разность -0.994 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 91.628 | 91.602 | 92.074 | 92.200 | 89.634 | 88.430 |
| 303 | 91.284 | 91.277 | 91.537 | 92.350 | 87.805 | 89.256 |
| 404 | 92.775 | 92.772 | 92.929 | 92.150 | 90.244 | 89.256 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/grad_randomref_nobias_mix25__ss202__ts202__43a30e870c9e/metrics.json), [предсказания](../raw/part3/runs/grad_randomref_nobias_mix25__ss202__ts202__43a30e870c9e/predictions.csv), [identity](../raw/part3/runs/grad_randomref_nobias_mix25__ss202__ts202__43a30e870c9e/identity.json).
- Seed 303: [метрики](../raw/part3/runs/grad_randomref_nobias_mix25__ss303__ts303__1ea79e54a88d/metrics.json), [предсказания](../raw/part3/runs/grad_randomref_nobias_mix25__ss303__ts303__1ea79e54a88d/predictions.csv), [identity](../raw/part3/runs/grad_randomref_nobias_mix25__ss303__ts303__1ea79e54a88d/identity.json).
- Seed 404: [метрики](../raw/part3/runs/grad_randomref_nobias_mix25__ss404__ts404__1573de8fcd7c/metrics.json), [предсказания](../raw/part3/runs/grad_randomref_nobias_mix25__ss404__ts404__1573de8fcd7c/predictions.csv), [identity](../raw/part3/runs/grad_randomref_nobias_mix25__ss404__ts404__1573de8fcd7c/identity.json).
