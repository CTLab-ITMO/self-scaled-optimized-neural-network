# Окружение

Работа идёт на двух машинах:

| | ноутбук | машина с RTX 3060 |
|---|---|---|
| роль | разработка, отладка, подготовка данных | боевые прогоны |
| GPU | нет | RTX 3060 12 GB |
| torch | CPU-сборка | CUDA-сборка |
| режим запуска | `--dev` (d = 196, 5k train, 5 эпох) | боевой (d = 784, 55k train, 30 эпох) |

Код device-agnostic: сам берёт CUDA, если она есть, иначе CPU. Один и тот же
код работает на обеих машинах, отличается только флаг `--dev`.

---

## Одной командой на любой машине

```bash
./setup_env.sh
```

Скрипт сам смотрит, есть ли видеокарта (через `nvidia-smi`), ставит подходящую
сборку torch, доустанавливает остальное, проверяет, что всё импортируется, и
скачивает MNIST, если его ещё нет. Всё кладётся **внутрь `.venv` в папке
проекта** — системный python не трогается, удалить окружение можно просто
стерев `.venv`.

Если определение железа ошиблось, можно форсировать: `./setup_env.sh --cuda`
или `./setup_env.sh --cpu`.

---

## Вручную, если нужно

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
# на машине с картой вместо cpu: --index-url .../whl/cu124
.venv/bin/pip install -r requirements.txt
```

Скачать MNIST один раз — **это обязательно сделать здесь**, чтобы на GPU-машине
не требовался интернет:

```bash
.venv/bin/python src/prepare_data.py
```

Появится `artifacts/mnist.npz` (~16 MB). Он едет на флешке вместе с кодом.

---

## Машина с RTX 3060 (Linux, интернет есть)

Проверено, что такие сборки реально выложены: под Linux + Python 3.12/3.13
свежий torch 2.13 лежит в индексе `cu129`.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu129
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Проверка должна напечатать `True NVIDIA GeForce RTX 3060`.
RTX 3060 — это Ampere (sm_86), поддерживается всеми свежими сборками.

Версия torch на двух машинах может отличаться (тут CPU-сборка, там CUDA) — на
код это не влияет: используются только `nn.Linear`, матмулы и Adam, API
стабильное.

`torch_geometric` не нужен — ни здесь, ни там. Топология графа одна на всю
выборку, поэтому слой GNN считается как обычный батчевый матмул с
предпосчитанной `A_hat`. См. PLAN.md, п. 1.1.

---

## Что кладём на флешку

```
gnn/
  configs/          конфиг эксперимента
  src/              код
  artifacts/
    mnist.npz       данные (готовятся на ноутбуке)
    pi_*.npy        перестановка признаков
  requirements.txt
  PLAN.md  SETUP.md
```

`.venv/` **не копируем** — окружение на второй машине ставится своё.

Обратно с GPU-машины забираем `results/` — csv с метриками и матрицы ошибок.
Анализ и графики строятся уже на ноутбуке, GPU для этого не нужен.
