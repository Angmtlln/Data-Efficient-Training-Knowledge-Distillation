# part3: forget_dose05

**Абляция дозы.** На общем фоне 327 из 932 контролей заменены нестабильными строками (~5% набора).

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 92.240% ± 0.814 п.п. SD; macro F1 92.234%; positive F1 92.419%. Контроль `forget_matched_zero`; средняя разность +0.382 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 92.087 | 92.080 | 92.325 | 93.000 | 89.634 | 90.909 |
| 303 | 93.119 | 93.119 | 93.119 | 92.750 | 90.244 | 89.256 |
| 404 | 91.514 | 91.502 | 91.814 | 93.500 | 87.805 | 88.430 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/forget_dose05__ss202__ts202__a4239093d63e/metrics.json), [предсказания](../raw/part3/runs/forget_dose05__ss202__ts202__a4239093d63e/predictions.csv), [identity](../raw/part3/runs/forget_dose05__ss202__ts202__a4239093d63e/identity.json).
- Seed 303: [метрики](../raw/part3/runs/forget_dose05__ss303__ts303__24b28d25f9ed/metrics.json), [предсказания](../raw/part3/runs/forget_dose05__ss303__ts303__24b28d25f9ed/predictions.csv), [identity](../raw/part3/runs/forget_dose05__ss303__ts303__24b28d25f9ed/identity.json).
- Seed 404: [метрики](../raw/part3/runs/forget_dose05__ss404__ts404__0eb857e94cb1/metrics.json), [предсказания](../raw/part3/runs/forget_dose05__ss404__ts404__0eb857e94cb1/predictions.csv), [identity](../raw/part3/runs/forget_dose05__ss404__ts404__0eb857e94cb1/identity.json).
