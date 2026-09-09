#!/usr/bin/env bash

#SBATCH --job-name=gnn-refine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-%j.log
#SBATCH --error=results/slurm-%j.log

# запуск structure/refine через SLURM

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "узел:      $(hostname)"
echo "задача:    ${SLURM_JOB_ID:-нет}"
echo "начало:    $(date '+%F %T')"
echo ""

if [ ! -x .venv/bin/python ]; then
  echo "Нет .venv. Собери окружение на головном узле, до отправки задачи:"
  echo "    ./setup_env.sh --cuda"
  exit 1
fi

.venv/bin/python - <<'EOF'
import torch
print("torch:", torch.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"карта: {p.name}, {p.total_memory / 2**30:.1f} ГБ")
else:
    print("КАРТА НЕ ВИДНА - проверь --gres=gpu:1")
EOF
echo ""

DO_STRUCT=1; DO_REFINE=1; QUICK=""
STRUCT_ARGS=${STRUCT_ARGS:-}
for a in "$@"; do
  case "$a" in
    --structure) DO_REFINE=0 ;;
    --refine)    DO_STRUCT=0 ;;
    --quick)     QUICK="--quick" ;;
    *) echo "неизвестный аргумент: $a"; exit 1 ;;
  esac
done

for f in src/structure.py src/eval_structure.py run_structure.sh; do
  [ -f "$f" ] || { echo "НЕТ ФАЙЛА $f - перенос неполный, останавливаюсь"; exit 1; }
done
NR=$(grep -c 'partial_free' run_refine.sh || true)
if [ "$NR" = 0 ]; then
  echo "run_refine.sh на сервере СТАРЫЙ: в нём нет правила partial_free."
  echo "Скопируй его заново, иначе три новых правила не посчитаются."
  exit 1
fi

fail=0

if [ "$DO_STRUCT" = 1 ]; then
  echo "=== ВОССТАНОВЛЕНИЕ СТРУКТУРЫ ==="
  PY=.venv/bin/python bash ./run_structure.sh $QUICK $STRUCT_ARGS || fail=1
  echo ""
fi

if [ "$DO_REFINE" = 1 ]; then
  echo "=== СРАВНЕНИЕ ПРАВИЛ ==="
  PY=.venv/bin/python bash ./run_refine.sh $QUICK || fail=1
  echo ""
  echo "=== сводка правил ==="
  .venv/bin/python src/refine.py --summary
fi

[ "$fail" = 0 ] || echo "ВНИМАНИЕ: часть серии завершилась с ошибкой, см. выше."
echo ""
echo "конец: $(date '+%F %T')"
