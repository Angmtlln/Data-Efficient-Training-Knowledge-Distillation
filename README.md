# Отбор 10% данных для обучения ModernBERT

Я выбирала часть отзывов SST-2 и проверяла, помогает ли осмысленный отбор обучить модель лучше случайного. Учителя и дистилляции нет.

**Сохранены 96 обучений: 32 условия, по три запуска на каждое.** Лучший результат — **94.72% на 10% SST-2**, +0.57 п.п. к контролю из целых предложений. [Новая серия: пары и обучение](results/root_pairs/REPORT.md).

В первых трех сериях сохранены 84 обучения; новая серия добавляет 12. Подготовленные серии 04/05 в это число не входят.

## С чего начать

- [Отчет](REPORT.md) — короткие выводы, таблицы, графики и ссылки на каждый метод.
- [Статьи](docs/SOURCES.md) — откуда взяты идеи, что получили авторы и чем мои опыты отличаются.
- [Технические пояснения](docs/TECHNICAL_NOTES.md) — настройки, формулы и ограничения.
- [Все результаты по запускам](results/analysis/all_runs.csv) — точные числа без округления.

## Что лежит в проекте

| Папка | Что внутри |
|---|---|
| `experiments/01_cpu_selection/` | Пять простых способов отбора, 15 обучений |
| `experiments/02_hypotheses/` | Отбор под языковые явления и пары, 39 обучений |
| `experiments/03_followups/` | Проверка дозы нестабильных строк и исправлений, 30 обучений |
| `results/raw/` | Метрики, ответы моделей, параметры и версии библиотек |
| `results/selections/` | Индексы выбранных строк и описание отбора |
| `results/analysis/` | Общие таблицы, графики и проверки |
| `results/methods/` | По одной понятной карточке на каждое условие |
| `results/inputs/`, `results/preparation/` | Оценки маленькой модели и проверки состава данных |
| `results/executed_code/` | Код, который использовался в завершенных обучениях |
| `results/pairs_v2_candidates/` | Новые кандидаты в пары: еще не проверены и не использовались |
| `scripts/` | Скрипты запуска, анализа и сборки отчета |
| `docs/` | Текст отчета, описания методов и источники |

Большие кеши весов, числовые представления всех фраз и подготовленные копии датасета не входят в Git. Команды ниже скачивают или создают их заново. Веса первых трех серий удалялись после успешного обучения. В новой серии 06 веса всех 12 моделей сохранены локально в `HW-project/vm_runs/enn_root_pairs_20261126/outputs/runs/`. **Предсказания и метрики всех запусков сохранены.**

## Проверить готовые результаты

Читать отчет можно без установки библиотек. Для проверки, что файлы не изменились, достаточно Python:

```bash
python scripts/artifacts.py verify
```

Для пересчета метрик нужны зависимости и исходные тексты. Установите Python 3.11–3.12, подходящий PyTorch по https://pytorch.org/get-started/locally/, затем:

```bash
python -m pip install -r requirements.txt
python experiments/02_hypotheses/code/download_data.py
python scripts/analyze_all.py --dataset experiments/02_hypotheses/data/sst2
python scripts/selection_evidence.py --dataset experiments/02_hypotheses/data/sst2
python scripts/training_diagnostics.py
python scripts/build_all_report.py
```

Первый скрипт скачивает закрепленную версию данных. Второй проверяет метки, состав выборок и заново считает accuracy и F1 по ответам моделей. `selection_evidence.py` сверяет снимок данных и проверяет состав выбранных строк. `training_diagnostics.py` читает сохраненную историю обучения. Последний скрипт обновляет отчет и карточки. Новое обучение не запускается.

Графики можно обновить отдельно, без пересчета метрик и нового обучения:

```bash
python scripts/plot_results.py
python scripts/check_report.py
```

Кривые обучения строятся из `progress.json`; их значения сохраняются в `results/analysis/learning_validation.csv`.

После намеренного обновления таблиц или текста пересоздайте список контрольных сумм:

```bash
python scripts/artifacts.py write
```

## Запустить обучение

Готовые индексы подвыборок уже входят в проект. Для повторения обучения не надо заново выбирать данные.

Пример для третьей серии:

```bash
python experiments/03_followups/code/download_data.py
python experiments/03_followups/code/tokenize_data.py
python scripts/run_queue.py --part 03_followups --output outputs/part3 --local local_checkpoints/part3 --dry-run
python scripts/run_queue.py --part 03_followups --methods forget_random --seeds 202 --output outputs/part3 --local local_checkpoints/part3 --pilot
python scripts/run_queue.py --part 03_followups --output outputs/part3 --local local_checkpoints/part3
```

- `--dry-run` только показывает список опытов.
- `--pilot` делает короткий пробный запуск для проверки работы кода.
- `--methods` и `--seeds` позволяют запустить только нужные варианты.
- Без этих ограничений очередь выполняет всю выбранную серию последовательно.

Для первых двух серий замените номер папки. Их номера случайных запусков по умолчанию — 17, 42, 101; для третьей — 202, 303, 404.

При повторе команды готовые результаты пропускаются, незавершенное обучение продолжается с сохраненного состояния. При ошибке очередь останавливается; подробности лежат в `outputs/<часть>/logs/`.

На ВМ запускайте очередь через менеджер процессов, например Supervisor. Тогда разрыв SSH не остановит обучение. На диске нужно не менее 15 ГБ свободного места на одну активную серию.

**Если нужны веса после обучения**, заранее добавьте `keep_completed_checkpoints: true` в `config.json` и выделите больше места. Без этого тяжелые состояния удаляются после успешного завершения. В процессе обучения сохраняются и проверяются последние состояния, чтобы можно было продолжить работу после обрыва.

Я обучала на RTX 5090 с torch 2.14.1+cu132 и transformers 4.57.1. Для такой GPU нужна подходящая версия PyTorch; старую CUDA-сборку устанавливать не стоит. Полные версии для каждого запуска записаны в `environment.json`.

## Заново выбрать данные на CPU

Первая серия:

```bash
python experiments/01_cpu_selection/code/download_data.py
python experiments/01_cpu_selection/code/prepare.py
```

Вторая серия:

```bash
python experiments/02_hypotheses/code/download_data.py
```

До расчета представлений скопируйте `results/inputs/part2/embedding_provenance.json` в `experiments/02_hypotheses/data/embedding_provenance.json`. Этот файл закрепляет версию MiniLM. Затем:

```bash
python experiments/02_hypotheses/code/prepare_skills.py
python experiments/02_hypotheses/code/prepare_pairs.py
```

Третья серия использует результаты подготовки первых двух:

```bash
python experiments/03_followups/probe_gradient_variants.py
python experiments/03_followups/prepare_followups.py
```

Если файл версии MiniLM не скопирован, `prepare_skills.py` возьмет текущую версию модели. Это может дать новые оценки. Во время обучения не меняйте файлы выбранных строк.

Расширенный поиск пар запускается отдельно и не меняет обучающие выборки:

```bash
python scripts/mine_pairs_v2.py --dataset experiments/02_hypotheses/data/sst2 --output outputs/pairs_v2
```

## Изменить текст отчета

Текст не зашит в таблицы с метриками:

| Что изменить | Где редактировать |
|---|---|
| Основной текст | `docs/report_template.md` |
| Короткие описания и выводы методов | `docs/methods.json` |
| Связь семейства методов со статьей | `docs/families.json` |
| Названия, ссылки и результаты статей | `docs/sources.json` |
| Подробные настройки и формулы | `docs/TECHNICAL_NOTES.md` |

После правок выполните `python scripts/build_all_report.py`. Это обновит `REPORT.md`, карточки и список источников. Числа берутся из проверенных таблиц. В тексте используется единое написание буквы «е».

Для проверки языка, ссылок и полноты карточек: `python scripts/check_report.py`.

## Проверки кода

```bash
PYTHONPATH=experiments/01_cpu_selection/code python -m unittest discover -s experiments/01_cpu_selection/tests -v
PYTHONPATH=experiments/02_hypotheses/code python -m unittest discover -s experiments/02_hypotheses/tests -v
```

Тесты проверяют восстановление обучения после обрыва, поврежденные сохранения, работу ModernBERT, формулу градиента и парные сравнения. Они не являются новыми результатами основного обучения.

Старые скрипты `analyze_results.py` и `build_report.py` относятся только к первым 54 обучениям. Для текущего отчета используйте `analyze_all.py` и `build_all_report.py`.

Метрики старых запусков не переписаны. Код воспроизведения содержит исправления сохранения и проверки версий; новые идентификаторы запусков могут отличаться. Между разными устройствами и версиями PyTorch полное совпадение чисел не гарантируется.

## Новая завершенная серия: целые предложения и подфразы

[Отчет](results/root_pairs/REPORT.md), [код и запуск](experiments/06_root_pair_training/README.md). Для пересчета новых таблиц и графиков: `python scripts/root_pairs_report.py`.
