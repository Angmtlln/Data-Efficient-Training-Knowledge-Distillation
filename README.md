# Data-efficient training: ModernBERT на SST-2

Рабочая ветка `sphsv_experiments`. Исследуем отбор 10% обучающего pool, без teacher и дистилляции. Сохранены **84 завершенных обучения: 28 условий × 3 seed**, включая отрицательные результаты и исправляющие абляции.

Начать с [REPORT.md](REPORT.md): общие выводы, таблицы, четыре графика, ограничения и ссылки на карточки каждого условия. Это первая версия отчета, а не заявление о завершении всех исследовательских вопросов.

## Структура

| Путь | Содержимое |
|---|---|
| `experiments/01_cpu_selection/` | Random, lexical-ratio, rare words, exact dedup, heldout forgetting; 15 запусков |
| `experiments/02_hypotheses/` | Skill/reference отбор и парный пилот; 39 запусков |
| `experiments/03_followups/` | Дозировка forgetting и контролируемое исправление gradient/embedding отбора; 30 запусков |
| `results/raw/part*/runs/` | Метрики, predictions, identity/config и environment каждого запуска |
| `results/selections/`, `experiments/*/selections/` | Индексы реальных SST-2 строк, хеши и описание отбора |
| `results/analysis/` | Проверенные общие таблицы, графики и диагностика |
| `results/inputs/`, `results/preparation/` | Сохраненные CPU-оценки, proxy-head, references, аудит состава |
| `results/executed_code/` | Неизмененные снимки исполнявшегося trainer-кода; исторические различия описаны в отчете |
| `results/pairs_v2_candidates/` | Расширенный поиск пар: НЕ проверены, НЕ использованы для обучения |
| `scripts/` | Очередь обучения, анализ, формирование отчета и поиск пар |

В Git не включены pretrained/model caches, 100-МБ эмбеддинги, локальные tokenized datasets, виртуальные окружения и checkpoint. Они скачиваются/создаются командами ниже. Итоговые веса обученных моделей исходный trainer удалял; они отсутствуют и в старом локальном архиве. Предсказания и все84 метрики сохранены.

## Окружение

Python3.11–3.12. Установить подходящий к GPU PyTorch по https://pytorch.org/get-started/locally/, затем:

```bash
python -m pip install -r requirements.txt
```

Реальное GPU-обучение выполнено на RTX5090, torch2.14.1+cu132, transformers4.57.1. CPU-подготовка — macOS, torch2.8.0. Точные версии каждого запуска — в environment.json. Для RTX50xx нужна CUDA-совместимая сборка PyTorch; старый Colab pin torch2.8 не навязывается всем GPU.

## Прочитать и проверить готовые результаты без GPU

Для чтения REPORT.md ничего запускать не нужно. Проверка контрольных сумм без сторонних библиотек:

```bash
python scripts/artifacts.py verify
```

Для независимого пересчета всех метрик нужен датасет в зафиксированной revision:

```bash
python experiments/02_hypotheses/code/download_data.py
python scripts/analyze_all.py --dataset experiments/02_hypotheses/data/sst2
python scripts/build_all_report.py
```

Проверяется соответствие меток test, пересчитываются accuracy и два F1, сверяются SHA256 сплитов/подвыборок, размеры, seeds и training config. Пилоты и прерванные серии не входят в основной отчет. `analyze_results.py` и `build_report.py` сохранены для исторической первой сетки54; для общего отчета использовать команды выше.

После намеренной регенерации артефактов обновить их манифест:

```bash
python scripts/artifacts.py write
```

## Воспроизвести обучение по готовым подвыборкам

Для каждой нужной части скачать исходный split и создать токенизацию. Пример третьей серии:

```bash
python experiments/03_followups/code/download_data.py
python experiments/03_followups/code/tokenize_data.py
python scripts/run_queue.py --part 03_followups --output outputs/part3 --local local_checkpoints/part3 --dry-run
python scripts/run_queue.py --part 03_followups --methods forget_random --seeds 202 --output outputs/part3 --local local_checkpoints/part3 --pilot
python scripts/run_queue.py --part 03_followups --output outputs/part3 --local local_checkpoints/part3
```

Для первой/второй заменить номер части. Seeds по умолчанию:17/42/101 для первых двух,202/303/404 для третьей. `--methods` и `--seeds` ограничивают серию. `--dry-run` не обучает. Штатная очередь последовательна; запуск двух очередей одной части блокируется lock-файлом. Готовые selection manifests уже есть, CPU-подготовку повторять не нужно.

Повтор той же команды пропускает завершенные run IDs и восстанавливает незавершенные из проверенного checkpoint. При ошибке очередь останавливается, подробности в `outputs/<part>/logs/`. На ВМ использовать менеджер процессов (например Supervisor), чтобы обучение не зависело от SSH.

Checkpoint включает optimizer/scheduler/RNG; SHA256-верификация защищает от частично записанных копий. В новой версии хранятся два последних persistent checkpoint плюс best, чтобы был fallback при повреждении. Поддерживать не менее15ГБ свободного места на одну активную серию. После успешного завершения checkpoint удаляются. Для сохранения весов заранее задать `keep_completed_checkpoints: true` в config.json и предусмотреть существенно больше места.

## Повторить CPU-подготовку

Подготовка1 требует скачанного dataset1; подготовка2 — dataset2. Первую сетку можно пересчитать так:

```bash
python experiments/01_cpu_selection/code/download_data.py
python experiments/01_cpu_selection/code/prepare.py
python experiments/02_hypotheses/code/download_data.py
```

Для закрепления версии MiniLM до расчета embeddings скопировать `results/inputs/part2/embedding_provenance.json` в `experiments/02_hypotheses/data/embedding_provenance.json`, затем:

```bash
python experiments/02_hypotheses/code/prepare_skills.py
python experiments/02_hypotheses/code/prepare_pairs.py
python experiments/03_followups/probe_gradient_variants.py
python experiments/03_followups/prepare_followups.py
```

Третья подготовка читает fixed OOF score seed17 из части1 и proxy/references/embeddings части2. При отсутствии закрепленного MiniLM provenance `prepare_skills.py` разрешает текущую revision; это будет новый CPU-пересчет, а не точное воспроизведение. Не менять selection manifests во время обучения.

Расширенные пары (не меняет training selections):

```bash
python scripts/mine_pairs_v2.py --dataset experiments/02_hypotheses/data/sst2 --output outputs/pairs_v2
```

Для воспроизведения старых CPU-оценок без пересчета можно использовать компактные `results/inputs/`; большие embeddings намеренно не включены. Полные подготовленные datasets и кеши остаются в локальном рабочем архиве вне Git.

## Проверки кода

```bash
PYTHONPATH=experiments/01_cpu_selection/code python -m unittest discover -s experiments/01_cpu_selection/tests -v
PYTHONPATH=experiments/02_hypotheses/code python -m unittest discover -s experiments/02_hypotheses/tests -v
```

Проверяются реальное resume маленькой модели, fallback поврежденного checkpoint, ModernBERT SDPA forward/backward, формула градиента через autograd, корректность пар и matched-контролей. Это проверки реализации, не новые результаты основного обучения.

## Историческая совместимость

Переносимый код закрепляет датасет/токенизатор revision, не перезаписывает config при токенизации и хеширует реально исполняемую папку. Исполненные старые версии сохранены отдельно; identity/метрики84 запусков не подменены новыми. Из-за исправления хеша воспроизводимые новые запуски могут иметь другие run IDs. Численное побитовое совпадение между разными версиями PyTorch/устройствами не гарантируется.

Приоритеты следующих редакций отчета перечислены в конце REPORT.md. Чужие стратегии намеренно не включены как наши методы; random оставлен как обязательный контроль.
