# part3: forget_matched_zero

**Контроль.** Общий фон плюс 932 замены, сопоставленные с nonzero-forgetting по классу и бину длины. Все nonzero строки исключены; это не обычный random.

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 91.858% ± 0.413 п.п. SD; macro F1 91.846%; positive F1 92.149%. Контроль `forget_matched_zero`; средняя разность +0.000 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 92.317 | 92.305 | 92.597 | 92.700 | 89.634 | 90.909 |
| 303 | 91.514 | 91.498 | 91.868 | 92.650 | 89.024 | 84.298 |
| 404 | 91.743 | 91.736 | 91.982 | 93.700 | 88.415 | 87.603 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/forget_matched_zero__ss202__ts202__196b12efa185/metrics.json), [предсказания](../raw/part3/runs/forget_matched_zero__ss202__ts202__196b12efa185/predictions.csv), [identity](../raw/part3/runs/forget_matched_zero__ss202__ts202__196b12efa185/identity.json).
- Seed 303: [метрики](../raw/part3/runs/forget_matched_zero__ss303__ts303__a774f5bc9641/metrics.json), [предсказания](../raw/part3/runs/forget_matched_zero__ss303__ts303__a774f5bc9641/predictions.csv), [identity](../raw/part3/runs/forget_matched_zero__ss303__ts303__a774f5bc9641/identity.json).
- Seed 404: [метрики](../raw/part3/runs/forget_matched_zero__ss404__ts404__c8142eb4a255/metrics.json), [предсказания](../raw/part3/runs/forget_matched_zero__ss404__ts404__c8142eb4a255/predictions.csv), [identity](../raw/part3/runs/forget_matched_zero__ss404__ts404__c8142eb4a255/identity.json).
