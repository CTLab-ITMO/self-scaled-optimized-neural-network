#!/usr/bin/env bash
# эксперимент «может ли граф сам улучшаться в процессе обучения»
# итеративный цикл refinement: обучаем gnn немного, смотрим на граф, меняем, снова обучаем...
# 5 правил того как меняем граф, 4 вида графа, 5 сидов


set -u
cd "$(dirname "$0")" || exit 1
if [ -n "${PY:-}" ]; then
  :
elif [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

if ! $PY -c "import torch" >/dev/null 2>&1; then
  echo "В окружении нет torch. Сначала создай его:"
  echo "    ./setup_env.sh"
  exit 1
fi


RULES="keep random_rewire corr_static partial_static partial_free"

GRAPHS=${GRAPHS:-"grid community geometric ring"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
START=${START:-random}     # с какого графа стартуем: random | knn_corr_k8
ROUNDS=${ROUNDS:-20}
EPOCHS=${EPOCHS:-20}
BUDGET=${BUDGET:-60}
SIDE=${SIDE:-12}
N=${N:-6000}
CLASSES=${CLASSES:-4}
M=${M:-24}

QUICK=0; STATUS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --quick)  QUICK=1 ;;
    --status) STATUS=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "неизвестный аргумент: $1"; exit 1 ;;
  esac
  shift
done

if [ "$QUICK" = 1 ]; then
  GRAPHS="grid"; SEEDS="0"; ROUNDS=4; EPOCHS=10; SIDE=10; N=3000; M=12
  [ "$STATUS" = 0 ] && echo "быстрый режим: только решётка, один сид"
fi

STATE=results/.refine
mkdir -p "$STATE"

C_OK=$'\033[32m'; C_RUN=$'\033[36m'; C_WARN=$'\033[33m'; C_OFF=$'\033[0m'
[ -t 1 ] || { C_OK=""; C_RUN=""; C_WARN=""; C_OFF=""; }

CSV_DONE_KEYS=""
if [ "$START" = random ] && [ "$SIDE" = 12 ] && [ "$N" = 6000 ] && \
   [ "$CLASSES" = 4 ] && [ "$M" = 24 ] && [ -f results/refine.csv ]; then
  CSV_DONE_KEYS=$($PY - results/refine.csv "$START" "$ROUNDS" <<'EOF' 2>/dev/null
import csv, sys
path, start, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen = {}
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("start") != start:
            continue
        try:
            key = (row["rule"], row["graph"], int(row["seed"]))
            seen.setdefault(key, set()).add(int(row["round"]))
        except (KeyError, TypeError, ValueError):
            pass
for (rule, graph, seed), rs in seen.items():
    if all(r in rs for r in range(rounds)):
        print(f"{rule}__{graph}__s{seed}")
EOF
  )
fi

is_done () {
  local key="$1" rule="$2" graph="$3" seed="$4"
  local f="$STATE/$key.npz"
  if [ -f "$f" ]; then
    $PY - "$f" "$ROUNDS" <<'EOF' 2>/dev/null
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
sys.exit(0 if int(z["round"]) >= int(sys.argv[2]) - 1 else 1)
EOF
    return $?
  fi

  # В Git локальные .npz не хранятся, поэтому полный уже посчитанный прогон
  # дополнительно узнаём по refine.csv. CSV относится именно к этим параметрам.
  case "
$CSV_DONE_KEYS
" in
    *"
${rule}__${graph}__s${seed}
"*) return 0 ;;
  esac
  return 1
}

total=0; done_n=0
for r in $RULES; do for g in $GRAPHS; do for s in $SEEDS; do
  total=$((total + 1))
  is_done "${r}__${g}__${START}__s${s}__d$((SIDE*SIDE))_n${N}_c${CLASSES}_m${M}" "$r" "$g" "$s" && done_n=$((done_n + 1))
done; done; done

if [ "$STATUS" = 1 ]; then
  echo ""
  echo "  правил: $(echo $RULES | wc -w)   графов: $(echo $GRAPHS | wc -w)   сидов: $(echo $SEEDS | wc -w)"
  echo "  старт: $START   раундов: $ROUNDS   эпох на раунд: $EPOCHS"
  echo ""
  printf "  %-16s" ""
  for g in $GRAPHS; do printf "%-13s" "$g"; done; echo
  for r in $RULES; do
    printf "  %-16s" "$r"
    for g in $GRAPHS; do
      k=0
      for s in $SEEDS; do is_done "${r}__${g}__${START}__s${s}__d$((SIDE*SIDE))_n${N}_c${CLASSES}_m${M}" "$r" "$g" "$s" && k=$((k + 1)); done
      printf "%-13s" "$k/$(echo $SEEDS | wc -w)"
    done
    echo
  done
  echo ""
  echo "  готово $done_n из $total прогонов"
  [ -f results/refine.csv ] && echo "  строк в results/refine.csv: $(($(wc -l < results/refine.csv) - 1))"
  exit 0
fi

RUNLOG=$(mktemp)
trap 'rm -f "$RUNLOG"' EXIT
trap 'echo ""; echo "${C_WARN}Прервано. Потерян максимум один раунд.${C_OFF}";
      echo "Продолжить: ./run_refine.sh"; exit 130' INT TERM

# Печатаем то, что реально вернул torch, а не только да/нет: иначе при
# расхождении невозможно понять, врёт баннер или расчёт правда на процессоре.
CUDA_INFO=$($PY -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())" 2>&1)
case "$CUDA_INFO" in
  "True "*) echo "${C_OK}CUDA доступна ($CUDA_INFO) - будет быстро.${C_OFF}" ;;
  *)        echo "${C_WARN}CUDA нет: torch вернул [$CUDA_INFO]${C_OFF}"
            echo "${C_WARN}Считаем на процессоре.${C_OFF}" ;;
esac
echo "  всего прогонов: $total, уже готово: $done_n"
echo ""

i=0; failed=0
for r in $RULES; do
  for g in $GRAPHS; do
    for s in $SEEDS; do
      i=$((i + 1))
      key="${r}__${g}__${START}__s${s}__d$((SIDE*SIDE))_n${N}_c${CLASSES}_m${M}"
      if is_done "$key" "$r" "$g" "$s"; then
        echo "${C_OK}[$i/$total сделан]${C_OFF} $r / $g / сид $s"
        continue
      fi
      echo ""
      echo "${C_RUN}[$i/$total] правило $r, граф $g, сид $s${C_OFF}"
      # tee + PIPESTATUS обязательны. Раньше статус брался у grep: прогон,
      # успевший напечатать хоть строку и упавший следом, считался успешным, а
      # трассировка уходила в grep и терялась - таблица молча оказывалась
      # неполной.
      $PY src/refine.py --rule "$r" --graph "$g" --seed "$s" \
          --start "$START" --rounds "$ROUNDS" --epochs "$EPOCHS" \
          --budget "$BUDGET" --side "$SIDE" --n "$N" \
          --classes "$CLASSES" --m "$M" 2>&1 \
          | tee "$RUNLOG" | grep -E "устройство|раунд|продолжаю|уже посчитано"
      rc=${PIPESTATUS[0]}
      if [ "$rc" != 0 ]; then
        failed=$((failed + 1))
        echo "${C_WARN}  ПРОГОН УПАЛ, код $rc. Последние строки:${C_OFF}"
        tail -n 15 "$RUNLOG" | sed 's/^/    /'
        echo "${C_WARN}  Состояние сохранено, повторный запуск продолжит.${C_OFF}"
      fi
    done
  done
done

echo ""
echo "${C_OK}================================================================${C_OFF}"
echo "${C_OK}  ГОТОВО${C_OFF}"
echo "${C_OK}================================================================${C_OFF}"
echo ""
if [ "$failed" != 0 ]; then
  echo "${C_WARN}УПАЛО ПРОГОНОВ: $failed из $total - таблица НЕПОЛНАЯ.${C_OFF}"
  echo "${C_WARN}Что именно упало, видно выше. Запусти скрипт ещё раз.${C_OFF}"
  echo ""
fi
echo "Результаты: results/refine.csv"
echo ""
echo "Главный вопрос к таблице: обгоняет ли хоть одно правило corr_static."
echo "corr_static строит граф по корреляциям сразу, без петли и без сети."
echo "Сильные методы должны обгонять контроли и сравниваются с corr_static."
echo ""
echo "Свести таблицу:  python src/refine.py --summary"
