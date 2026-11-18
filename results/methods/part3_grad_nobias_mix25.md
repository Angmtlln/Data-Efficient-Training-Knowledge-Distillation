# part3: grad_nobias_mix25

**Исправляющая абляция.** Та же смесь/квоты, но без производной bias. Не лучше raw-mix по среднему; не доказана бесполезность bias во всех условиях.

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 92.699% ± 0.403 п.п. SD; macro F1 92.692%; positive F1 92.893%. Контроль `skill_matched_random_v2`; средняя разность -0.191 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 92.317 | 92.301 | 92.645 | 92.400 | 89.024 | 88.430 |
| 303 | 93.119 | 93.115 | 93.289 | 92.450 | 91.463 | 90.083 |
| 404 | 92.661 | 92.660 | 92.744 | 92.500 | 92.073 | 90.909 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/grad_nobias_mix25__ss202__ts202__3063c865b9bb/metrics.json), [предсказания](../raw/part3/runs/grad_nobias_mix25__ss202__ts202__3063c865b9bb/predictions.csv), [identity](../raw/part3/runs/grad_nobias_mix25__ss202__ts202__3063c865b9bb/identity.json).
- Seed 303: [метрики](../raw/part3/runs/grad_nobias_mix25__ss303__ts303__69cefe301765/metrics.json), [предсказания](../raw/part3/runs/grad_nobias_mix25__ss303__ts303__69cefe301765/predictions.csv), [identity](../raw/part3/runs/grad_nobias_mix25__ss303__ts303__69cefe301765/identity.json).
- Seed 404: [метрики](../raw/part3/runs/grad_nobias_mix25__ss404__ts404__2d3300c3dda2/metrics.json), [предсказания](../raw/part3/runs/grad_nobias_mix25__ss404__ts404__2d3300c3dda2/predictions.csv), [identity](../raw/part3/runs/grad_nobias_mix25__ss404__ts404__2d3300c3dda2/identity.json).
