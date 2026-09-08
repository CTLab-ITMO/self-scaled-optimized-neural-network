# Что запускать сейчас

Этот файл описывает **актуальные runnable-серии** после очистки проваленных
веток. История отрицательных результатов не удалена: она хранится в
`FINDINGS.md` и `PREREGISTRATION.md`. Здесь перечислено только то, что имеет
смысл запускать повторно.

Оценки времени зависят от машины. Все основные скрипты сохраняют результаты
по мере работы; повторный запуск продолжает уже начатую серию там, где это
поддерживается конкретным runner'ом.

---

## Шаг 0. Окружение

```bash
./setup_env.sh
```

Для GPU-машины можно явно запросить CUDA-сборку:

```bash
./setup_env.sh --cuda
```

Быстрая проверка:

```bash
.venv/bin/python src/model.py --dev
```

---

## Шаг 1. Основная сохранённая серия на MNIST

Если `results/results.csv` уже содержит 110 полных прогонов, повторно считать
их не требуется. Анализ существующей таблицы:

```bash
.venv/bin/python src/analyze.py
```

Ключевые проверки этой серии описаны в `FINDINGS.md`: изоморфный
`grid4_misaligned`, density-matched контроли, oversmoothing полного графа и
сравнение data-driven графов.

---

## Шаг 2. Восстановление структуры на синтетике

```bash
./run_structure.sh
```

Сводка:

```bash
.venv/bin/python src/eval_structure.py --summary
```

В актуальном коде оставлены:

* `corr_static` — baseline;
* `partial_static` — подтверждённый метод с известным бюджетом рёбер;
* `partial_free` — подтверждённый вариант без oracle-знания истинной
  плотности.

Standalone `aracne_static` и `perm_free` больше не запускаются: их
отрицательные результаты сохранены в `FINDINGS.md`.

---

## Шаг 3. Refine-loop

Быстрый прогон:

```bash
./run_refine.sh --quick
```

Полный прогон:

```bash
./run_refine.sh
```

Статус:

```bash
./run_refine.sh --status
```

Сводка:

```bash
.venv/bin/python src/refine.py --summary
```

Текущие правила:

* `keep` — отрицательный контроль;
* `random_rewire` — отрицательный контроль;
* `corr_static` — baseline;
* `partial_static` — подтверждённый метод;
* `partial_free` — подтверждённый метод без oracle-плотности.

Удалённые из runner'а `corr_greedy`, `grad_greedy`, `grad_corr` и standalone
`aracne_static` уже были проверены и дали отрицательный результат; числа
остаются в `FINDINGS.md`.

---

## Шаг 4. Повтор подтверждённой первой серии

Сначала посмотреть, какие из нужных запусков уже есть:

```bash
./run_series.sh --status
```

Если чего-то не хватает, досчитать:

```bash
./run_series.sh
```

Сейчас этот runner фиксирован на GCN, `L = 3`, пяти сидах и оставленных
сравнениях (`grid4` / `grid4_misaligned`, density-matched random controls,
`knn_corr_k8`, `empty`, `complete`, MLP). Старую сетку GAT/GIN и глубин он
больше не запускает.

Отдельно можно пересчитать структурный precheck по глубине:

```bash
./run_series.sh --precheck
```

Результат: `results/depth_precheck.csv`.

---

## SLURM

`job.sbatch` умеет запускать structure и refine вместе либо по отдельности.
Например:

```bash
sbatch job.sbatch
sbatch job.sbatch --structure
sbatch job.sbatch --refine
```

Перед отправкой задачи окружение на машине должно быть собрано:

```bash
./setup_env.sh --cuda
```

---

## Что намеренно больше не запускается

Из runnable-пайплайна удалены ветки, для которых уже получен отрицательный
ответ: `emb_learned`, `grad_sim`, `perm_free`, standalone `aracne_static`,
`corr_greedy`, `grad_greedy`, `grad_corr` и неудачная noise-ось синтетики.
Подробности и числа остаются в `FINDINGS.md`.
