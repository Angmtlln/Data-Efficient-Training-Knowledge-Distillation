# part3: forget_all

**Абляция дозы.** Все 932 нестабильные строки scoring seed17 (~14.3% набора). Больше не означает лучше в этих повторах.

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 92.546% ± 0.607 п.п. SD; macro F1 92.539%; positive F1 92.756%. Контроль `forget_matched_zero`; средняя разность +0.688 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 93.005 | 92.998 | 93.215 | 92.450 | 89.024 | 91.736 |
| 303 | 91.858 | 91.846 | 92.155 | 92.100 | 88.415 | 89.256 |
| 404 | 92.775 | 92.773 | 92.897 | 93.400 | 87.805 | 90.083 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/forget_all__ss202__ts202__9ae1e0167c02/metrics.json), [предсказания](../raw/part3/runs/forget_all__ss202__ts202__9ae1e0167c02/predictions.csv), [identity](../raw/part3/runs/forget_all__ss202__ts202__9ae1e0167c02/identity.json).
- Seed 303: [метрики](../raw/part3/runs/forget_all__ss303__ts303__15a5af623933/metrics.json), [предсказания](../raw/part3/runs/forget_all__ss303__ts303__15a5af623933/predictions.csv), [identity](../raw/part3/runs/forget_all__ss303__ts303__15a5af623933/identity.json).
- Seed 404: [метрики](../raw/part3/runs/forget_all__ss404__ts404__ad79d52c1aa0/metrics.json), [предсказания](../raw/part3/runs/forget_all__ss404__ts404__ad79d52c1aa0/predictions.csv), [identity](../raw/part3/runs/forget_all__ss404__ts404__ad79d52c1aa0/identity.json).
