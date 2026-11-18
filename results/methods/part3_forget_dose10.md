# part3: forget_dose10

**Кандидат для дальнейшей проверки.** 653 matched-замены (~10% набора). Лучший наблюдаемый средний результат дозировки; оптимальность не доказана.

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 92.890% ± 0.413 п.п. SD; macro F1 92.884%; positive F1 93.083%. Контроль `forget_matched_zero`; средняя разность +1.032 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 92.431 | 92.418 | 92.731 | 92.500 | 90.244 | 90.909 |
| 303 | 93.005 | 93.001 | 93.154 | 92.750 | 90.854 | 90.083 |
| 404 | 93.234 | 93.231 | 93.363 | 94.000 | 89.634 | 90.083 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/forget_dose10__ss202__ts202__48f76ac58a9e/metrics.json), [предсказания](../raw/part3/runs/forget_dose10__ss202__ts202__48f76ac58a9e/predictions.csv), [identity](../raw/part3/runs/forget_dose10__ss202__ts202__48f76ac58a9e/identity.json).
- Seed 303: [метрики](../raw/part3/runs/forget_dose10__ss303__ts303__0321e5ff2667/metrics.json), [предсказания](../raw/part3/runs/forget_dose10__ss303__ts303__0321e5ff2667/predictions.csv), [identity](../raw/part3/runs/forget_dose10__ss303__ts303__0321e5ff2667/identity.json).
- Seed 404: [метрики](../raw/part3/runs/forget_dose10__ss404__ts404__5e8d361f6af1/metrics.json), [предсказания](../raw/part3/runs/forget_dose10__ss404__ts404__5e8d361f6af1/predictions.csv), [identity](../raw/part3/runs/forget_dose10__ss404__ts404__5e8d361f6af1/identity.json).
