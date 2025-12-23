#!/usr/bin/env bash
set -euo pipefail

# Запуск всех экспериментов ПАРАЛЛЕЛЬНО (имеет смысл только если есть 2+ GPU).
#
# Требования:
# - nvidia-smi доступен
# - ты понимаешь, что параллельные запуски будут конкурировать за CPU/RAM/диск.
#
# Пример:
#   PETIMAGES_ROOT=/path/to/PetImages bash run_all_experiments_parallel.sh
#
# По умолчанию использует GPU 0 и 1 (если доступны). Можно переопределить:
#   GPU_LIST="0 1 2 3" PETIMAGES_ROOT=... bash run_all_experiments_parallel.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-./data}"
RUNS_ROOT="${RUNS_ROOT:-./runs}"
LOGS_DIR="${LOGS_DIR:-./logs}"
PETIMAGES_ROOT="${PETIMAGES_ROOT:-}"
GPU_LIST="${GPU_LIST:-0 1}"

# Epoch defaults (can be overridden):
#   MNIST_EPOCHS=200 FASHION_EPOCHS=200 CATSDOGS_EPOCHS=100 GPU_LIST="0 1" bash run_all_experiments_parallel.sh
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
echo "GPU_LIST:     $GPU_LIST"
echo "PETIMAGES_ROOT: ${PETIMAGES_ROOT:-<not set>}"
echo "MNIST_EPOCHS:    $MNIST_EPOCHS"
echo "FASHION_EPOCHS:  $FASHION_EPOCHS"
echo "CATSDOGS_EPOCHS: $CATSDOGS_EPOCHS"
echo

GPU_ARR=($GPU_LIST)
if [[ ${#GPU_ARR[@]} -lt 2 ]]; then
  echo "GPU_LIST содержит <2 GPU. Для одной GPU используй run_all_experiments_sequential.sh"
  exit 1
fi

run_bg () {
  local name="$1"
  local gpu="$2"
  shift 2
  echo "===== RUN(bg): $name on GPU $gpu ====="
  echo "CMD: CUDA_VISIBLE_DEVICES=$gpu python -u train_sparse_vit.py $*"
  echo "LOG: $LOGS_DIR/${name}.log"
  echo
  (CUDA_VISIBLE_DEVICES="$gpu" python -u train_sparse_vit.py "$@" 2>&1 | tee "$LOGS_DIR/${name}.log") &
}

# Раскидываем задачи по GPU в round-robin.
idx=0
pick_gpu () {
  local g="${GPU_ARR[$idx]}"
  idx=$(( (idx + 1) % ${#GPU_ARR[@]} ))
  echo "$g"
}

# MNIST dense + sparse
run_bg "mnist_sparse_expand" "$(pick_gpu)" \
  --dataset mnist --model sparse --preset mnist_small \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$MNIST_EPOCHS" --max-new-edges 50

run_bg "mnist_dense" "$(pick_gpu)" \
  --dataset mnist --model dense --preset mnist_small \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$MNIST_EPOCHS"

# Fashion dense + sparse
run_bg "fashion_sparse_expand" "$(pick_gpu)" \
  --dataset fashion_mnist --model sparse --preset fashion_small \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$FASHION_EPOCHS" --max-new-edges 50

run_bg "fashion_dense" "$(pick_gpu)" \
  --dataset fashion_mnist --model dense --preset fashion_small \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR_BASE" \
  --epochs "$FASHION_EPOCHS"

# CatsDogs (если есть датасет)
if [[ -n "$PETIMAGES_ROOT" ]]; then
  run_bg "catsdogs_sparse_expand" "$(pick_gpu)" \
    --dataset catsdogs --model sparse --preset catsdogs \
    --petimages-root "$PETIMAGES_ROOT" \
    --run-dir "$RUN_DIR_BASE" \
    --epochs "$CATSDOGS_EPOCHS" --max-new-edges 100

  run_bg "catsdogs_dense" "$(pick_gpu)" \
    --dataset catsdogs --model dense --preset catsdogs \
    --petimages-root "$PETIMAGES_ROOT" \
    --run-dir "$RUN_DIR_BASE" \
    --epochs "$CATSDOGS_EPOCHS"
else
  echo "SKIP catsdogs: PETIMAGES_ROOT не задан."
fi

echo
echo "Ждём завершения всех задач..."
wait

echo "DONE. Все результаты лежат в: $RUN_DIR_BASE"


