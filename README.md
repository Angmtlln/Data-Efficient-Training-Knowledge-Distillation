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


## SVP и градиентные эксперименты

Результаты и ограничения четырёх серий описаны в [SVP_RESULTS.md](SVP_RESULTS.md).
Код находится в `src/analysis/` и `src/selection/`. Блокноты Colab с встроенными
архивами остаются локальными и не включаются в Git; команды ниже запускаются из
корня проекта после подготовки сплитов и установки зависимостей, описанных выше.
Для градиентного отбора также требуется `faiss-cpu` (в Colab: `pip install faiss-cpu`).
Обучение proxy и ModernBERT требует CUDA; T4 использует FP16 с gradient scaling.

### SVP

```bash
python -m src.analysis.svp_proxy --seed 0 --eval-batch-size 32
python -m src.selection.svp --method uncertainty --seeds 0 1 2
OPENBLAS_NUM_THREADS=1 python -m src.selection.svp --method kcenter --seeds 0 1 2
```

Один proxy ModernBERT обучается эпоху на `random_s0`, затем выдаёт вероятности и
CLS-эмбеддинги. При наличии `data/svp_proxy_scores.parquet` uncertainty использует
именно их, иначе — сохранённый `gold_prob_cheap` из `data/scores.parquet`. В бинарной
задаче энтропия gold probability не зависит от метки. Одно и то же подмножество
используется для всех target-сидов. Это one-shot coreset, не многораундовый active learning.

Дополнительные представления и forgetting-события:

```bash
python -m src.analysis.svp_small_model embed --batch-size 128
OPENBLAS_NUM_THREADS=1 python -m src.selection.svp_additional --method embeddings --input data/svp_minilm_l3.npy --name svp_kcenter_l3 --normalize --seeds 0 1 2
python -m src.analysis.svp_small_model forgetting --batch-size 64 --epochs 3 --seed 0
python -m src.selection.svp_additional --method forget_epoch --input data/svp_forgetting_proxy.npz --name svp_forget_l3 --seeds 0 1 2
```

MiniLM-L3 для эмбеддингов не дообучается на SST-2; для forgetting тот же энкодер
обучается на всём размеченном pool. Веса моделей и большие эмбеддинги не версионируются.

### Градиентные методы

```bash
python -m src.analysis.gradient_proxy --seed 0
python -m src.selection.gradient_coreset --method glister --seeds 0 1 2
OPENBLAS_NUM_THREADS=1 python -m src.selection.gradient_coreset --method craig --seeds 0 1 2
OPENBLAS_NUM_THREADS=1 python -m src.selection.gradient_round4 --seeds 0 1 2
python -m src.train --subset r4_craig_len8_s0 --seed 0 --val-indices data/gradient_epoch_val_idx.npy --subset-weights results/subsets/r4_craig_len8_s0_weights.npy
```

Это приближения, не полное воспроизведение GLISTER или CRAIG. Proxy MiniLM-L3
обучается на `random_s0`; его градиенты головы используются для ранжирования и
покрытия. Исходный val делится на 1 000 примеров для отбора и 1 000 для выбора
эпохи. Последняя команда обучает target на одном подмножестве с весами; для
невзвешенного варианта используются те же индексы без аргумента `--subset-weights`.
Фильтр `len8` использует `n_words` из `data/scores.parquet`, поэтому необходимы
совпадающие сплиты и порядок `pool_idx`. Бюджет 10% относится к target: метки и
вычисления для обучения proxy и отбора учитываются дополнительно.

### Данные результатов

В `results/svp_results/`, `results/svp_round2_results/` и
`results/gradient_round3_results/` сохранены компактные результаты отдельных
запусков, индексы и необходимые скоры/метаданные. Анализы можно пересчитать:

```bash
python scripts/analyze_svp.py
python scripts/analyze_svp_round2.py
python scripts/analyze_gradient_round3.py
```

В `results/gradient_round4_analysis/summary_from_notebook.csv` хранится только
агрегированная таблица четвёртого раунда, извлечённая из локального блокнота.
Архив с отдельными результатами, индексами и весами этого раунда пока не получен;
поэтому его результаты нельзя проверить по отдельным запускам или точно воспроизвести
по сохранённым подмножествам. Входные веса моделей, подготовленные датасеты,
эмбеддинги и временные Colab-артефакты исключены из Git.
