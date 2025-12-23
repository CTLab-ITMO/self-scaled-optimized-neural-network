#!/usr/bin/env bash
set -euo pipefail

# Запуск всех экспериментов ПОСЛЕДОВАТЕЛЬНО (лучше для сервера с 1 GPU).
#
# Ожидается запуск из корня репозитория:
#   bash run_all_experiments_sequential.sh
#
# Для CatsDogs нужно указать PETIMAGES_ROOT (папка с Cat/ и Dog/):
#   PETIMAGES_ROOT=/path/to/PetImages bash run_all_experiments_sequential.sh
#
# Можно переопределить DATA_ROOT и RUNS_ROOT:
#   DATA_ROOT=./data RUNS_ROOT=./runs bash run_all_experiments_sequential.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-./data}"
RUNS_ROOT="${RUNS_ROOT:-./runs}"
LOGS_DIR="${LOGS_DIR:-./logs}"
PETIMAGES_ROOT="${PETIMAGES_ROOT:-}"

# Epoch defaults (can be overridden):
#   MNIST_EPOCHS=200 FASHION_EPOCHS=200 CATSDOGS_EPOCHS=100 bash run_all_experiments_sequential.sh
MNIST_EPOCHS="${MNIST_EPOCHS:-200}"
FASHION_EPOCHS="${FASHION_EPOCHS:-200}"
CATSDOGS_EPOCHS="${CATSDOGS_EPOCHS:-100}"

mkdir -p "$DATA_ROOT" "$RUNS_ROOT" "$LOGS_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR_BASE="$RUNS_ROOT/batch_${STAMP}"
mkdir -p "$RUN_DIR_BASE"

echo "ROOT_DIR:     $ROOT_DIR"
echo "DATA_ROOT:    $DATA_ROOT"
echo "RUN_DIR_BASE: $RUN_DIR_BASE"
echo "LOGS_DIR:     $LOGS_DIR"
echo "PETIMAGES_ROOT: ${PETIMAGES_ROOT:-<not set>}"
echo "MNIST_EPOCHS:    $MNIST_EPOCHS"
echo "FASHION_EPOCHS:  $FASHION_EPOCHS"
echo "CATSDOGS_EPOCHS: $CATSDOGS_EPOCHS"
echo

run_one () {
  local name="$1"
  shift
  echo "===== RUN: $name ====="
  echo "CMD: python -u train_sparse_vit.py $*"
  echo "LOG: $LOGS_DIR/${name}.log"
  echo
  python -u train_sparse_vit.py "$@" 2>&1 | tee "$LOGS_DIR/${name}.log"
}

# 1) MNIST sparse + expansion
run_one "mnist_sparse_expand" \
  --dataset mnist --model sparse --preset mnist_small \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$MNIST_EPOCHS" \
  --max-new-edges 50

# 2) MNIST dense baseline (без расширения)
run_one "mnist_dense" \
  --dataset mnist --model dense --preset mnist_small \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$MNIST_EPOCHS"

# 3) FashionMNIST sparse + expansion
run_one "fashion_sparse_expand" \
  --dataset fashion_mnist --model sparse --preset fashion_small \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$FASHION_EPOCHS" \
  --max-new-edges 50

# 4) FashionMNIST dense baseline (без расширения)
run_one "fashion_dense" \
  --dataset fashion_mnist --model dense --preset fashion_small \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$FASHION_EPOCHS"

# 5) CatsDogs sparse + expansion
if [[ -z "$PETIMAGES_ROOT" ]]; then
  echo "SKIP catsdogs: PETIMAGES_ROOT не задан."
  echo "Чтобы запустить catsdogs, сделай:"
  echo "  PETIMAGES_ROOT=/path/to/PetImages bash run_all_experiments_sequential.sh"
  exit 0
fi

run_one "catsdogs_sparse_expand" \
  --dataset catsdogs --model sparse --preset catsdogs \
  --petimages-root "$PETIMAGES_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$CATSDOGS_EPOCHS" \
  --max-new-edges 100

# 6) CatsDogs dense baseline (без расширения)
run_one "catsdogs_dense" \
  --dataset catsdogs --model dense --preset catsdogs \
  --petimages-root "$PETIMAGES_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$CATSDOGS_EPOCHS"

echo
echo "DONE. Все результаты лежат в: $RUN_DIR_BASE"


