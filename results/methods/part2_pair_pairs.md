# part2: pair_pairs

**Оставить пилот; вернуться при новых данных.** 32 непересекающиеся пары с исходными противоположными SST-2 метками и однотокенным отрицанием/отсутствием; остальное random.

64/6534 = 0.98% строк парные. В среднем -0.11 п.п. к pair_random и -0.04 п.п. к broken_pairs; знаки по seed разные. В обучении обычный CE: модель не получает явного pair-loss.

Рекомендация исходного разбора: Сначала расширить независимо проверенный банк реальных пар, выделить независимый парный evaluation, зафиксировать filler и сравнить 0/16/32 пар. Без этого не тратить GPU на больше повторов того же слабого воздействия. Pair-loss — отдельное изменение обучения, не чистый эксперимент отбора.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 92.852% ± 1.154 п.п. SD; macro F1 92.847%; positive F1 93.025%. Контроль `pair_random`; средняя разность -0.115 п.п. Различных поднаборов: 3.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 91.628 | 91.622 | 91.862 | 93.050 | 89.634 | 90.083 |
| 42 | 93.005 | 93.003 | 93.107 | 92.900 | 90.244 | 91.736 |
| 101 | 93.922 | 93.916 | 94.105 | 92.750 | 92.683 | 91.736 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/pair_pairs__ss17__ts17__e72bfe4628c1/metrics.json), [предсказания](../raw/part2/runs/pair_pairs__ss17__ts17__e72bfe4628c1/predictions.csv), [identity](../raw/part2/runs/pair_pairs__ss17__ts17__e72bfe4628c1/identity.json).
- Seed 42: [метрики](../raw/part2/runs/pair_pairs__ss42__ts42__26c240406472/metrics.json), [предсказания](../raw/part2/runs/pair_pairs__ss42__ts42__26c240406472/predictions.csv), [identity](../raw/part2/runs/pair_pairs__ss42__ts42__26c240406472/identity.json).
- Seed 101: [метрики](../raw/part2/runs/pair_pairs__ss101__ts101__4ff9284f8eb7/metrics.json), [предсказания](../raw/part2/runs/pair_pairs__ss101__ts101__4ff9284f8eb7/predictions.csv), [identity](../raw/part2/runs/pair_pairs__ss101__ts101__4ff9284f8eb7/identity.json).
