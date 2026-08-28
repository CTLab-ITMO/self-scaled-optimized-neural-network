#!/usr/bin/env bash
# Создаёт окружение проекта в .venv и ставит всё нужное.
#
#   ./setup_env.sh          определить железо и поставить подходящий torch
#   ./setup_env.sh --cpu    принудительно CPU-сборка
#   ./setup_env.sh --cuda   принудительно CUDA-сборка
#
# Всё ставится ВНУТРЬ .venv в папке проекта. Системный python не трогается,
# так что удалить окружение можно просто стерев .venv.

set -eu
cd "$(dirname "$0")"

BUILD=auto
[ "${1:-}" = "--cpu" ] && BUILD=cpu
[ "${1:-}" = "--cuda" ] && BUILD=cuda

C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_OFF=$'\033[0m'
[ -t 1 ] || { C_OK=""; C_WARN=""; C_OFF=""; }

# --- какая сборка torch нужна -----------------------------------------------
if [ "$BUILD" = auto ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    BUILD=cuda
    echo "${C_OK}Найдена видеокарта:${C_OFF}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /'
  else
    BUILD=cpu
    echo "${C_WARN}Видеокарта не найдена - ставлю CPU-сборку torch.${C_OFF}"
    echo "Если карта есть, но nvidia-smi не работает, поставь драйвер и"
    echo "запусти заново, либо форсируй: ./setup_env.sh --cuda"
  fi
fi

# --- окружение ---------------------------------------------------------------
if [ ! -d .venv ]; then
  echo ""
  echo "Создаю .venv ..."
  python3 -m venv .venv
else
  echo ""
  echo ".venv уже есть, доустанавливаю в него."
fi

PIP=.venv/bin/pip
$PIP install --quiet --upgrade pip

echo "Ставлю torch ($BUILD) ..."
if [ "$BUILD" = cuda ]; then
  # cu124 подходит для RTX 3060 (Ampere) и вообще для всего от 20xx и новее
  $PIP install --quiet torch --index-url https://download.pytorch.org/whl/cu124
else
  $PIP install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "Ставлю остальное ..."
$PIP install --quiet -r requirements.txt

# --- проверка ----------------------------------------------------------------
echo ""
echo "Проверяю:"
.venv/bin/python - <<'EOF'
import importlib, sys
import torch
print(f"  torch {torch.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  CUDA: ДА, {p.name}, {p.total_memory / 2**30:.1f} ГБ")
else:
    print("  CUDA: нет (прогоны пойдут на процессоре, это рабочий вариант)")
missing = [m for m in ("numpy", "scipy", "sklearn", "networkx",
                       "pandas", "matplotlib", "yaml")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"  НЕ ХВАТАЕТ: {missing}")
    sys.exit(1)
print("  все пакеты на месте")
EOF

# --- данные ------------------------------------------------------------------
if [ ! -f artifacts/mnist.npz ]; then
  echo ""
  echo "Скачиваю MNIST (один раз, нужен интернет) ..."
  .venv/bin/python src/prepare_data.py
else
  echo "  MNIST уже скачан: artifacts/mnist.npz"
fi

echo ""
echo "${C_OK}================================================================${C_OFF}"
echo "${C_OK}  ОКРУЖЕНИЕ ГОТОВО${C_OFF}"
echo "${C_OK}================================================================${C_OFF}"
echo ""
echo "Дальше все команды запускать через .venv/bin/python, например:"
echo "    .venv/bin/python src/synth.py"
echo ""
echo "Либо активировать окружение один раз на сессию:"
echo "    source .venv/bin/activate"
echo "и тогда хватает просто 'python'."
echo ""
echo "Быстрая проверка, что всё живо (~1 мин):"
echo "    .venv/bin/python src/model.py --dev"
