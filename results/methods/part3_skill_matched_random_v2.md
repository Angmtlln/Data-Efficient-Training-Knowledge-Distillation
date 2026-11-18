# part3: skill_matched_random_v2

**Контроль.** Random с теми же квотами label × length bin × proxy correctness × loss quintile, что и четыре targeted-условия.

См. общий REPORT.md: выводы предварительные, по трем связанным selection/training seed.

Accuracy 92.890% ± 1.000 п.п. SD; macro F1 92.884%; positive F1 93.075%. Контроль `skill_matched_random_v2`; средняя разность +0.000 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 202 | 93.578 | 93.575 | 93.708 | 92.800 | 92.073 | 91.736 |
| 303 | 93.349 | 93.346 | 93.483 | 93.350 | 90.854 | 91.736 |
| 404 | 91.743 | 91.732 | 92.035 | 92.850 | 91.463 | 87.603 |

## Исходные результаты

- Seed 202: [метрики](../raw/part3/runs/skill_matched_random_v2__ss202__ts202__a2c8f4b69b9b/metrics.json), [предсказания](../raw/part3/runs/skill_matched_random_v2__ss202__ts202__a2c8f4b69b9b/predictions.csv), [identity](../raw/part3/runs/skill_matched_random_v2__ss202__ts202__a2c8f4b69b9b/identity.json).
- Seed 303: [метрики](../raw/part3/runs/skill_matched_random_v2__ss303__ts303__8351e9748364/metrics.json), [предсказания](../raw/part3/runs/skill_matched_random_v2__ss303__ts303__8351e9748364/predictions.csv), [identity](../raw/part3/runs/skill_matched_random_v2__ss303__ts303__8351e9748364/identity.json).
- Seed 404: [метрики](../raw/part3/runs/skill_matched_random_v2__ss404__ts404__aef97b2d961d/metrics.json), [предсказания](../raw/part3/runs/skill_matched_random_v2__ss404__ts404__aef97b2d961d/predictions.csv), [identity](../raw/part3/runs/skill_matched_random_v2__ss404__ts404__aef97b2d961d/identity.json).
