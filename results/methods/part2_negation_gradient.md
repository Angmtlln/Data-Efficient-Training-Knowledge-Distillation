# part2: negation_gradient

**Переработать: приоритет 2.** Скалярное произведение градиента линейной головы MiniLM с усредненным градиентом reference отрицаний, включая bias.

Проекция h·g_ref,w + g_ref,b отрицательна на всем eligible pool. Выбраны только правильно распознанные proxy отрицательные примеры и только ошибочно распознанные положительные. Это классово-асимметричный difficulty sampling.

Рекомендация исходного разбора: Проверить диагностику калибровки и bias; отдельно аблировать bias и центрирование градиентов по классам. GPU запускать только после pool-only проверки: score не должен сводиться к ±loss и разрушать распределение трудности. Старую формулу не повторять без изменений.

Это рекомендация после первых 54 запусков; итоги последующих абляций приведены в общем REPORT.md.

Accuracy 78.823% ± 3.498 п.п. SD; macro F1 77.676%; positive F1 82.676%. Контроль `skill_random`; средняя разность -14.182 п.п. Различных поднаборов: 1.

## Каждый seed

Метрики в процентах. Лексические срезы: negation n=164, contrast n=121. SD не является доверительным интервалом.

| seed | accuracy | macro_f1 | positive_f1 | best_val_accuracy | negation_accuracy | contrast_accuracy |
|---|---|---|---|---|---|---|
| 17 | 78.784 | 77.692 | 82.629 | 83.250 | 70.732 | 69.421 |
| 42 | 82.339 | 81.786 | 84.961 | 82.650 | 79.268 | 74.380 |
| 101 | 75.344 | 73.552 | 80.437 | 79.350 | 67.683 | 63.636 |

## Исходные результаты

- Seed 17: [метрики](../raw/part2/runs/negation_gradient__ss17__ts17__039ef55ddf4b/metrics.json), [предсказания](../raw/part2/runs/negation_gradient__ss17__ts17__039ef55ddf4b/predictions.csv), [identity](../raw/part2/runs/negation_gradient__ss17__ts17__039ef55ddf4b/identity.json).
- Seed 42: [метрики](../raw/part2/runs/negation_gradient__ss42__ts42__ff6a3e410585/metrics.json), [предсказания](../raw/part2/runs/negation_gradient__ss42__ts42__ff6a3e410585/predictions.csv), [identity](../raw/part2/runs/negation_gradient__ss42__ts42__ff6a3e410585/identity.json).
- Seed 101: [метрики](../raw/part2/runs/negation_gradient__ss101__ts101__c270197b352d/metrics.json), [предсказания](../raw/part2/runs/negation_gradient__ss101__ts101__c270197b352d/predictions.csv), [identity](../raw/part2/runs/negation_gradient__ss101__ts101__c270197b352d/identity.json).
