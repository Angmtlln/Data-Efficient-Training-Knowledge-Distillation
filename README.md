# Data-Efficient Training: отбор 10% SST-2 для ModernBERT

Исследуем, как стратегия отбора подмножества (качество / разнообразие / важность) влияет
на качество ModernBERT-base, обученного на 10% GLUE SST-2. Условие — в [Conditions.md](Conditions.md),
результаты и выводы — в **[REPORT.md](REPORT.md)**.

## Окружение

Используется [uv](https://docs.astral.sh/uv/), Python 3.11, torch с CUDA 12.8.
Проект лежит в OneDrive, поэтому venv вынесен наружу:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\sst2-selection"
uv sync
uv run python scripts/download_models.py   # ModernBERT-base, MiniLM (эмбеддинги), GPT-2 (perplexity)
uv run python scripts/prepare_data.py      # data/sst2: pool / val / test
```

## Сплиты

У SST-2 test без меток, поэтому:

| сплит | источник | размер |
|---|---|---|
| `pool` | GLUE train минус val — из него отбираем 10% | 65 349 |
| `val`  | стратифицированный кусок GLUE train (early stopping / выбор эпохи) | 2 000 |
| `test` | GLUE validation — итоговая метрика | 872 |

Бюджет: 10% пула = 6 534 примера. Гиперпараметры фиксированы для всех стратегий (`src/config.py`).

## Запуск

```powershell
uv run python -m src.selection.random_baseline --seeds 0 1 2 3 4   # -> results/subsets/*.npy
uv run python -m src.train --subset random_s0 --seed 0
uv run python -m src.train --subset full --seed 0 --log-dynamics    # + динамика обучения для cartography/EL2N
```

Каждый запуск дописывает строку в `results/results.csv` (метрики test на эпохе с лучшим val).

## Структура

```
src/config.py           пути, модели, гиперпараметры
src/data.py             загрузка сплитов, сохранение/чтение подмножеств
src/train.py            дообучение на подмножестве
src/selection/          стратегии отбора (каждая сохраняет индексы pool_idx)
src/analysis/           характеристики подмножеств, корреляции, графики
scripts/                подготовка данных и моделей
results/                results.csv, subsets/, runs/
notebooks/              отчёт
```
