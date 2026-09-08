# GNN graph-structure experiments

Репозиторий содержит только экспериментальные ветки, которые либо дали
положительный результат, либо нужны как честные контроли к этим результатам.

## Что осталось

### 1. Влияние структуры графа на качество GNN

`results/results.csv` — основная первая серия на MNIST. Ключевой контроль:
`grid4` и `grid4_misaligned` изоморфны и совпадают по структурным инвариантам,
но дают существенно разное качество. Это показывает, что качество зависит не
только от абстрактной топологии, но и от согласования вершин графа с данными.

Контроли `empty`, `complete`, `er_*`, `config_knn8`, `regular4` и семейство
`ws_*` сохранены намеренно: они нужны для интерпретации главного результата.

### 2. Восстановление известного графа на синтетике

`src/synth.py` генерирует задачу с известным истинным графом. Без графовой
структуры класс не определяется по гистограмме признаков.

`src/structure.py` и `src/eval_structure.py` содержат три оставленных метода:

- `corr_static` — корреляционный baseline при известном числе рёбер;
- `partial_static` — частная корреляция при известном числе рёбер;
- `partial_free` — тип оценки, регуляризация и плотность выбираются по train
  без доступа к истинному графу.

В `results/structure.csv` оставлены только эти методы.

### 3. Refine-loop

`src/refine.py` сравнивает три содержательных правила и два обязательных
контроля:

- `keep` — граф не меняется;
- `random_rewire` — случайное перепроведение;
- `corr_static` — одношаговый correlation baseline;
- `partial_static`;
- `partial_free`.

В `results/refine.csv` оставлены только эти пять правил.

## Основные команды

```bash
./setup_env.sh --cuda
python src/analyze.py
python src/eval_structure.py --summary
python src/refine.py --summary
```

Новые прогоны восстановления структуры:

```bash
./run_structure.sh
```

Новые refine-прогоны:

```bash
./run_refine.sh
```

Повтор подтверждённой первой серии:

```bash
python src/run_all.py --graphs grid4 grid4_misaligned --seeds 0 1 2 3 4
./run_series.sh --status
```

## Результаты

- `results/results.csv` — первая серия GNN;
- `results/summary_full.csv` — агрегированная первая серия;
- `results/report_full.txt` — текстовый отчёт по оставленным проверкам;
- `results/structure.csv` — восстановление истинной структуры;
- `results/refine.csv` — refine-loop;
- `results/depth_precheck.csv` — структурный предрасчёт для оси глубины.

## Структура кода

- `src/graphs.py` — оставшиеся кандидаты графов для MNIST;
- `src/train.py`, `src/model.py` — обучение GNN;
- `src/run_all.py` — полный перебор кандидатов/сидов/глубин;
- `src/analyze.py` — анализ первой серии;
- `src/graph_stats.py` — структурные дескрипторы;
- `src/synth.py` — синтетическая задача с известным графом;
- `src/structure.py` — восстановление структуры;
- `src/eval_structure.py` — сравнение оставленных методов восстановления;
- `src/refine.py` — refine-loop;
- `src/depth_precheck.py` — структурный предрасчёт по глубине.

Сырые MNIST-файлы и локальные состояния прогонов не коммитятся.
