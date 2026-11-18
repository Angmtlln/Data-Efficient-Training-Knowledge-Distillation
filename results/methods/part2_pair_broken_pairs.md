# part2: pair_broken_pairs

**Оставить как ключевую абляцию.** Один член каждой пары заменен строкой того же класса и бина длины; общий фон сохранен.

6502 общих строки с paired treatment, меняются только 32. Это самый близкий контроль взаимодополняемости; эффекта текущей дозы не обнаружено.

Рекомендация исходного разбора: Сохранить; использовать в будущем парном дизайне с фиксированным фоном и более мощной парной оценкой.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 92.890% ± 0.344 п.п. SD; macro F1 92.888%; positive F1 92.983%. Контроль `pair_random`; средняя разность -0.076 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 92.546 | 92.542 | 92.705 | 93.300 | 87.805 | 92.562 |
| 42 | 92.890 | 92.889 | 92.971 | 92.750 | 90.854 | 91.736 |
| 101 | 93.234 | 93.234 | 93.273 | 93.250 | 91.463 | 90.083 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/pair_broken_pairs__ss17__ts17__af62caaad39e/metrics.json), [предсказания](../raw/part2/runs/pair_broken_pairs__ss17__ts17__af62caaad39e/predictions.csv), [identity](../raw/part2/runs/pair_broken_pairs__ss17__ts17__af62caaad39e/identity.json).
- Seed 42: [метрики](../raw/part2/runs/pair_broken_pairs__ss42__ts42__339f053bf8fb/metrics.json), [предсказания](../raw/part2/runs/pair_broken_pairs__ss42__ts42__339f053bf8fb/predictions.csv), [identity](../raw/part2/runs/pair_broken_pairs__ss42__ts42__339f053bf8fb/identity.json).
- Seed 101: [метрики](../raw/part2/runs/pair_broken_pairs__ss101__ts101__c79c02b960ec/metrics.json), [предсказания](../raw/part2/runs/pair_broken_pairs__ss101__ts101__c79c02b960ec/predictions.csv), [identity](../raw/part2/runs/pair_broken_pairs__ss101__ts101__c79c02b960ec/identity.json).
