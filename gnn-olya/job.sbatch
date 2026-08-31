#!/usr/bin/env bash
#SBATCH --job-name=gnn-refine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-%j.log
#SBATCH --error=results/slurm-%j.log
#
# Запуск на кластере с SLURM:
#
#   sbatch job.sbatch              полный перебор (~2 ч)
#   sbatch job.sbatch --quick      быстрый (~40 мин)
#
#   squeue -u $USER                посмотреть очередь
#   tail -f results/slurm-*.log    смотреть лог
#   scancel <номер задачи>         отменить
#
# Задача живёт в очереди сама: обрыв SSH, закрытый ноутбук и выход из системы
# ей не мешают. Именно поэтому здесь sbatch, а не nohup - на кластере
# отвязанные от очереди процессы висят на головном узле и мешают всем.
#
# Прервать и продолжить можно: состояние пишется после каждого раунда, так что
# повторный sbatch допишет недостающее, а не начнёт заново. Если задача
# упрётся в --time, просто отправь её ещё раз.

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

PY=.venv/bin/python ./run_refine.sh "$@"

echo ""
echo "=== сводка ==="
.venv/bin/python src/refine.py --summary
echo ""
echo "конец: $(date '+%F %T')"