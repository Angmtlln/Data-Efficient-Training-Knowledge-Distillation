# Серия 07: воспроизведение и состояние

Протоколы: `PROTOCOL.md`, `PROTOCOL_GSNR.md`, `PROTOCOL_ROOTS.md`; обзор статей `LITERATURE.md`, коллегина ветка `COLLABORATOR.md`. Подготовлены 8 условий по три сида: всего 24 обучения. Итоговые выборки, исходные показатели отбора и контрольные суммы находятся в `results/literature_strict/`; большой model cache, логи и checkpoints — в игнорируемой папке `outputs/literature_strict/`.

Серия 06 сохранена как журнал остановленного первого варианта. В серии 07 перед построением всех прокси и отборов исключаются точные совпадения и общие 8-граммы с внутренней validation и GLUE validation. Это не гарантирует удаления всех примеров из одного исходного предложения. При сравнении с сериями 04–06 и результатами коллег учитывать другую очистку, вычислительное окружение и многократное использование GLUE validation.

Для запуска необходимо подготовить локальные dataset/tokenizer/config и ModernBERT по инструкциям серий 01 и 04, а также скачать IMDb test в локальный datasets-format каталог `IMDB_SOURCE`, соответствующий revision в сохранённом манифесте. Модель и dataset ревизии фиксированы. Для полного воспроизведения proxy и выборок (один поток BLAS):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python experiments/07_literature_strict/prepare.py
python experiments/07_literature_strict/prepare_gsnr.py
python experiments/07_literature_strict/audit_roots.py
python experiments/07_literature_strict/prepare_roots.py
python experiments/07_literature_strict/audit_fragments.py
python experiments/07_literature_strict/audit_selections.py
python -m unittest discover -s experiments/07_literature_strict -p 'test_*.py' -v
```

`run.sh` обучает первые 15 моделей, `run_gsnr.sh` — три модели динамики proxy, `run_roots.sh` — шесть root-моделей. Процессы можно выполнять последовательно; `run_all.sh` выполняет первые два, затем проверяет наличие всех 24 результатов, повторно рассчитывает SST-2 и оценивает все 24 на IMDb. Поэтому перед `run_all.sh` необходимо закончить `run_roots.sh` либо запустить этот script отдельно параллельно и дождаться его завершения до конца `run_gsnr.sh`. В скриптах нужен `EXPERIMENT_PYTHON`, в `run_all.sh` также `IMDB_SOURCE`. Повторный запуск пропускает результаты с валидными хэшами и пытается продолжить прерванные обучения с последнего checkpoint. Скрипты для Mac проверяют сетевое питание перед новым обучением.

После успешного `run_all.sh` выполнить `python experiments/07_literature_strict/report.py`; он требует все результаты SST-2 и IMDb и строит таблицу и график. При использовании MPS нужен доступ процесса к ускорителю и переменная `PYTORCH_ENABLE_MPS_FALLBACK=1`. Интерпретировать результат как поисковый эксперимент, а IMDb — как внешнюю проверку переноса с усечением до 64 токенов.
