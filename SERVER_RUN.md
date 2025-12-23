### Запуск на «обычном» арендованном сервере (без Slurm)

Ниже — минимальные шаги, чтобы запустить эксперименты без Jupyter и получить результаты в `runs/*/metrics.csv`.

## 0) Предпосылки

- На сервере есть Python 3.9+.
- Желательно GPU + драйвер NVIDIA. Проверка:

```bash
nvidia-smi
```

## 1) Забрать код

```bash
git clone <URL_репозитория>
cd self-scaled-optimized-neural-network
```

## 2) Окружение (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Проверка torch/cuda:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"
```

## 3) Данные

### MNIST / FashionMNIST

Скачаются автоматически в `--data-root`. Рекомендация:

```bash
mkdir -p ./data
```

### Cats vs Dogs (PetImages)

Ожидаемая структура:

```
<PETIMAGES_ROOT>/
  Cat/*.jpg
  Dog/*.jpg
```

## 4) Запуски (то, что ты просил)

Результаты каждого запуска будут в `runs/<run_name>/metrics.csv` и `runs/<run_name>/hyperparams.json` (для sparse).

### MNIST (sparse, с расширением)

```bash
python -u train_sparse_vit.py --dataset mnist --model sparse --preset mnist_small --data-root ./data --epochs 10 --max-new-edges 50
```

### MNIST (без расширения)

Вариант A: sparse без growth/prune:

```bash
python -u train_sparse_vit.py --dataset mnist --model sparse --preset mnist_small --data-root ./data --epochs 10 --no-expansion --max-new-edges 0
```

Вариант B: dense baseline (обычный ViT):

```bash
python -u train_sparse_vit.py --dataset mnist --model dense --preset mnist_small --data-root ./data --epochs 10
```

### FashionMNIST

С расширением (sparse):

```bash
python -u train_sparse_vit.py --dataset fashion_mnist --model sparse --preset fashion_small --data-root ./data --epochs 10 --max-new-edges 50
```

Без расширения (dense baseline):

```bash
python -u train_sparse_vit.py --dataset fashion_mnist --model dense --preset fashion_small --data-root ./data --epochs 10
```

### Cats vs Dogs (с расширением и без)

С расширением (sparse):

```bash
python -u train_sparse_vit.py --dataset catsdogs --model sparse --preset catsdogs --petimages-root <PETIMAGES_ROOT> --epochs 50 --max-new-edges 100
```

Без расширения (dense baseline):

```bash
python -u train_sparse_vit.py --dataset catsdogs --model dense --preset catsdogs --petimages-root <PETIMAGES_ROOT> --epochs 50
```

## 4.1) Запустить всё разом (batch)

В репозитории есть два скрипта:

- `run_all_experiments_sequential.sh` — **последовательно** (рекомендую для 1 GPU)
- `run_all_experiments_parallel.sh` — **параллельно** (только если 2+ GPU)

Пример (последовательно):

```bash
PETIMAGES_ROOT=<PETIMAGES_ROOT> bash run_all_experiments_sequential.sh
```

По умолчанию epochs такие:
- MNIST: 200
- FashionMNIST: 200
- CatsDogs: 100

Если хочешь поменять:

```bash
MNIST_EPOCHS=200 FASHION_EPOCHS=200 CATSDOGS_EPOCHS=100 PETIMAGES_ROOT=<PETIMAGES_ROOT> bash run_all_experiments_sequential.sh
```

Пример (параллельно на GPU 0 и 1):

```bash
GPU_LIST="0 1" PETIMAGES_ROOT=<PETIMAGES_ROOT> bash run_all_experiments_parallel.sh
```

Логи будут в `./logs/*.log`, результаты — в `runs/batch_<timestamp>/`.

## 5) Быстрые графики (опционально)

### Вариант A (рекомендую): из `metrics.csv`

После каждого запуска смотри `runs/<run_name>/metrics.csv`. Графики можно получить так:

```bash
python plot_logs.py --input runs/<run_name>/metrics.csv --outdir runs/<run_name> --prefix curves
```

Это создаст:
- `runs/<run_name>/curves_loss.png`
- `runs/<run_name>/curves_accuracy.png`
- `runs/<run_name>/curves_params.png`

### Вариант B: из текстового лога (stdout)

Если у тебя есть текстовый лог запуска (stdout), можно построить графики так:

```bash
python plot_logs.py --input <path_to_log.txt> --outdir . --prefix run
```


