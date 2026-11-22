# Серия 07: отбор данных по мотивам статей

Мы сравнили восемь способов выбора 6534 примеров SST-2 для ModernBERT, каждый на трёх связанных seed. Перед отбором исключали точные совпадения и общие восьмисловные фрагменты с проверочными наборами. Это уменьшает пересечения, но не гарантирует удаления всех фрагментов одного предложения. Результаты и ограничения — в [отчёте](../../results/literature_strict/REPORT.md); идеи статей — в [LITERATURE.md](LITERATURE.md). GLUE validation уже многократно использовался командой, а публичный SST test и IMDb не гарантированно отсутствовали в предобучении модели.

## Какие файлы за что отвечают

| Файл | Назначение |
|---|---|
| `PROTOCOL*.md` | Правила отбора, обучения и оценки; отдельные протоколы для динамики градиентов, предполагаемых целых предложений и открытого SST test. |
| `COLLABORATOR.md` | Связь с экспериментами коллеги и отличия условий. |
| `prepare.py` | Очистка пула, расчёт оценок proxy и базовые выборки: случайная, средняя сложность и частотное расстояние. |
| `prepare_gsnr.py` | Оценка динамики градиентов маленькой proxy-модели и соответствующие выборки. Это адаптация идеи, а не точное воспроизведение статьи. |
| `audit_roots.py`, `prepare_roots.py` | Поиск предполагаемых целых предложений по вложенности фраз и построение двух выборок. |
| `audit_fragments.py`, `audit_selections.py` | Проверки пересечений, состава, числа токенов и общих элементов выборок. |
| `audit_difficulty.py`, `audit_source_diversity.py` | Анализ сложности примеров и разнообразия предполагаемых источников для итогового отчёта. |
| `prepare_imdb_holdout.py`, `audit_original_sst.py` | Фиксация внешней выборки IMDb и проверка открытого SST test. Их готовые манифесты уже сохранены в результатах. |
| `run.sh`, `run_gsnr.sh`, `run_roots.sh`, `run_all.sh` | Очереди обучения; общая использует [train.py](../04_len_forgetting/train.py) и запускает все 24 обучения. |
| `eval_imdb_holdout.py`, `eval_original_sst.py` | Предсказания сохранённых моделей на IMDb и открытом SST test. |
| `analyze*.py`, `report.py` | Проверка предсказаний и сборка таблиц и графика. |
| `hash_checkpoints.py` | Контрольные суммы весов, чтобы не перепутать модели при внешней оценке. |
| `test_*.py` | Проверки отбора, контрольных сумм, отчёта и работы со временными файлами. |
| `check_progress.py`, `run_backfill.sh`, `power_guard.sh`, `prune_completed.py` | Служебные инструменты: состояние очереди, дополнительный worker, проверка питания на Mac и удаление ненужных промежуточных checkpoint. |

В `results/literature_strict/` лежат индексы выбранных строк (`selections/`), предсказания и метрики (`runs/`, `imdb_holdout_runs/`, `original_sst_runs/`), аудиты и итоговый `REPORT.md`. Тяжёлые данные, логи и веса в Git не входят; исторические абсолютные пути к весам в метриках описывают исходную машину и сами веса не содержат.

## Воспроизведение

Для чтения отчёта зависимости не нужны. Для нового обучения установите Python 3.11–3.12, PyTorch для вашей системы и зависимости из корневого `requirements.txt`. Для 24 обучений нужен ускоритель CUDA или Apple MPS и достаточно места под веса; точное совпадение на другом оборудовании не гарантируется. На MPS установите `PYTORCH_ENABLE_MPS_FALLBACK=1`. Работа ведётся из корня репозитория.

1. Скачайте закреплённый снимок SST-2 и подготовьте токены:

```bash
python experiments/01_cpu_selection/code/download_data.py
python experiments/01_cpu_selection/code/tokenize_data.py
```

2. Индексы выборок уже сохранены в Git. Чтобы пересоздать их и аудиты, выполните команды ниже. Они перезапишут соответствующие файлы результатов; длительность расчётов зависит от компьютера.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python experiments/07_literature_strict/prepare.py
python experiments/07_literature_strict/prepare_gsnr.py
python experiments/07_literature_strict/audit_roots.py
python experiments/07_literature_strict/prepare_roots.py
python experiments/07_literature_strict/audit_fragments.py
python experiments/07_literature_strict/audit_selections.py
python experiments/07_literature_strict/audit_difficulty.py
python experiments/07_literature_strict/audit_source_diversity.py
python -m unittest discover -s experiments/07_literature_strict -p 'test_*.py' -v
```

3. Для оценки IMDb сохраните test split датасета `stanfordnlp/imdb` с ревизией `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` через `datasets.load_dataset(..., split='test').save_to_disk(...)`. Задайте путь к нему в `IMDB_SOURCE`. Сохранённый манифест проверяет содержимое выборки по хэшу.

При новом обучении в отдельной копии проекта сначала перенесите сохранённые результаты в резервную папку: иначе обучение пропустит уже завершённые запуски, веса которых в Git не включены. Это также предотвращает повторное использование старых внешних предсказаний с новыми весами:

```bash
mkdir -p outputs/original_results
mv results/literature_strict/runs results/literature_strict/imdb_holdout_runs results/literature_strict/original_sst_runs results/literature_strict/checkpoint_integrity.json outputs/original_results/
export EXPERIMENT_PYTHON=python
export IMDB_SOURCE=/путь/к/сохранённому/imdb-test
bash experiments/07_literature_strict/run_all.sh
```

Команда обучает все модели, проверяет веса, пересчитывает результаты GLUE и оценивает IMDb. На Mac перед каждым заданием проверяется внешнее питание; `EXPERIMENT_ALLOW_BATTERY=1` разрешает запуск на батарее с зарядом не ниже 40%. На других системах проверка питания не выполняется. Очереди можно также запускать отдельно, последовательно; `run_backfill.sh` — необязательный второй worker, который конкурирует за ресурсы GPU.

4. Для финальной оценки понадобится [открытый SST binary test](https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/blob/master/data/stsa.binary.test) с зафиксированным в `PROTOCOL_SST_TEST.md` хэшем. Укажите путь к файлу и выполните:

```bash
export ORIGINAL_SST_TEST=/путь/к/stsa.binary.test
bash experiments/07_literature_strict/finalize_original_sst.sh
```

Скрипт проверит файл и контрольные суммы моделей, выполнит оценку и обновит отчёт. Предыдущие файлы внешних предсказаний не содержат хэш весов: перед повторной оценкой их нужно сохранить отдельно, как в шаге 3. Новые файлы включают хэш и не будут молча использованы с другой моделью. Переобучение даёт новое исследование и может изменить результаты; исходные результаты останутся в резервной копии.
