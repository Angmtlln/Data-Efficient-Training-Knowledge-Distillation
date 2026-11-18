# part1: heldout_forgetting

**Повторить: приоритет 1.** 3-fold TF-IDF + SGD, число correct→incorrect на исключенных из обучения строках за 8 проходов.

Лучше random во всех трех seed: +0.57, +1.03, +0.57 п.п. Ненулевой счетчик лишь у 878–951 строк; 5583–5656 выбранных строк имеют нулевой score и добираются случайно.

Рекомендация исходного разбора: Проверить дозировку сигнала (0, 5%, 10%, все доступные forgetting-строки) при 6534 строках; контролировать класс и длину, фиксировать общий фон, использовать новые seed. Это гипотеза о полезной небольшой добавке нестабильных примеров, а не о полностью умном top-10%.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 93.157% ± 0.577 п.п. SD; macro F1 93.156%; positive F1 93.252%. Контроль `random`; средняя разность +0.726 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 93.234 | 93.234 | 93.273 | 92.400 | 89.634 | 92.562 |
| 42 | 92.546 | 92.544 | 92.655 | 92.550 | 90.244 | 90.909 |
| 101 | 93.693 | 93.690 | 93.827 | 93.600 | 92.683 | 93.388 |

## Исходные результаты

- Seed 17: [метрики](../raw/part1/runs/heldout_forgetting__ss17__ts17__75dd3176d841/metrics.json), [предсказания](../raw/part1/runs/heldout_forgetting__ss17__ts17__75dd3176d841/predictions.csv), [identity](../raw/part1/runs/heldout_forgetting__ss17__ts17__75dd3176d841/identity.json).
- Seed 42: [метрики](../raw/part1/runs/heldout_forgetting__ss42__ts42__64278513ccbf/metrics.json), [предсказания](../raw/part1/runs/heldout_forgetting__ss42__ts42__64278513ccbf/predictions.csv), [identity](../raw/part1/runs/heldout_forgetting__ss42__ts42__64278513ccbf/identity.json).
- Seed 101: [метрики](../raw/part1/runs/heldout_forgetting__ss101__ts101__443653b05c7c/metrics.json), [предсказания](../raw/part1/runs/heldout_forgetting__ss101__ts101__443653b05c7c/predictions.csv), [identity](../raw/part1/runs/heldout_forgetting__ss101__ts101__443653b05c7c/identity.json).
