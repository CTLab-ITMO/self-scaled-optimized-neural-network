#!/usr/bin/env bash
# Пункт 4 задания: сравнение правил обновления графа.
#
#   ./run_refine.sh              прогнать всё; продолжает с места обрыва
#   ./run_refine.sh --status     что посчитано, что осталось
#   ./run_refine.sh --quick      быстрая версия: только решётка, 1 сид
#
# ПРЕРЫВАТЬ МОЖНО В ЛЮБОЙ МОМЕНТ - Ctrl-C, сон ноутбука, отключение питания.
# Состояние пишется на диск ПОСЛЕ КАЖДОГО РАУНДА, поэтому теряется максимум
# один раунд (минуты), а не весь прогон. Повторный запуск той же командой
# продолжит с последнего сохранённого раунда.
#
# Видеокарта не обязательна: задача мелкая, процессор тянет. Если CUDA есть,
# refine.py возьмёт её сам и будет заметно быстрее.

set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}

# ---------------------------------------------------------------------------
#  Что перебираем
# ---------------------------------------------------------------------------

# Порядок правил не случаен: сначала контроли, потом базовое правило на данных,
# потом те, что используют обученную сеть. Если corr_greedy не обгоняет
# контроли, остальное можно не запускать - сигнала нет вообще.
RULES="keep random_rewire corr_greedy grad_greedy grad_corr"

# Четыре типа истинного графа: вывод не должен оказаться свойством решётки.
GRAPHS="grid community geometric ring"

SEEDS="0 1 2"
START=${START:-random}     # с какого графа стартуем: empty | random | knn_corr_k8
ROUNDS=${ROUNDS:-6}
EPOCHS=${EPOCHS:-20}
BUDGET=${BUDGET:-60}
SIDE=${SIDE:-12}
N=${N:-6000}

# Флаги разбираются в цикле: раньше оба проверялись как $1, и
# "--quick --status" молча запускал полный прогон вместо показа статуса.
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
  GRAPHS="grid"; SEEDS="0"; ROUNDS=4; EPOCHS=10; SIDE=10; N=3000
  [ "$STATUS" = 0 ] && echo "быстрый режим: только решётка, один сид"
fi

STATE=results/.refine
mkdir -p "$STATE"

C_OK=$'\033[32m'; C_RUN=$'\033[36m'; C_WARN=$'\033[33m'; C_OFF=$'\033[0m'
[ -t 1 ] || { C_OK=""; C_RUN=""; C_WARN=""; C_OFF=""; }

# Прогон считается готовым, если в его состоянии записан последний раунд.
is_done () {
  local f="$STATE/$1.npz"
  [ -f "$f" ] || return 1
  $PY - "$f" "$ROUNDS" <<'EOF' 2>/dev/null
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
sys.exit(0 if int(z["round"]) >= int(sys.argv[2]) - 1 else 1)
EOF
}

total=0; done_n=0
for r in $RULES; do for g in $GRAPHS; do for s in $SEEDS; do
  total=$((total + 1))
  is_done "${r}__${g}__${START}__s${s}__d$((SIDE*SIDE))_n${N}_c4_m24" && done_n=$((done_n + 1))
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
      for s in $SEEDS; do is_done "${r}__${g}__${START}__s${s}__d$((SIDE*SIDE))_n${N}_c4_m24" && k=$((k + 1)); done
      printf "%-13s" "$k/$(echo $SEEDS | wc -w)"
    done
    echo
  done
  echo ""
  echo "  готово $done_n из $total прогонов"
  [ -f results/refine.csv ] && echo "  строк в results/refine.csv: $(($(wc -l < results/refine.csv) - 1))"
  exit 0
fi

trap 'echo ""; echo "${C_WARN}Прервано. Потерян максимум один раунд.${C_OFF}";
      echo "Продолжить: ./run_refine.sh"; exit 130' INT TERM

if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "${C_OK}CUDA доступна - будет быстро.${C_OFF}"
else
  echo "${C_WARN}CUDA нет, считаем на процессоре. Это нормально: задача мелкая,"
  echo "но заложи несколько часов на полный перебор.${C_OFF}"
fi
echo "  всего прогонов: $total, уже готово: $done_n"
echo ""

i=0
for r in $RULES; do
  for g in $GRAPHS; do
    for s in $SEEDS; do
      i=$((i + 1))
      key="${r}__${g}__${START}__s${s}__d$((SIDE*SIDE))_n${N}_c4_m24"
      if is_done "$key"; then
        echo "${C_OK}[$i/$total сделан]${C_OFF} $r / $g / сид $s"
        continue
      fi
      echo ""
      echo "${C_RUN}[$i/$total] правило $r, граф $g, сид $s${C_OFF}"
      $PY src/refine.py --rule "$r" --graph "$g" --seed "$s" \
          --start "$START" --rounds "$ROUNDS" --epochs "$EPOCHS" \
          --budget "$BUDGET" --side "$SIDE" --n "$N" 2>&1 \
          | grep -E "раунд|продолжаю|уже посчитано" || {
            echo "${C_WARN}  прогон упал - см. вывод выше. Состояние сохранено,"
            echo "  повторный запуск продолжит с последнего раунда.${C_OFF}"; }
    done
  done
done

echo ""
echo "${C_OK}================================================================${C_OFF}"
echo "${C_OK}  ГОТОВО${C_OFF}"
echo "${C_OK}================================================================${C_OFF}"
echo ""
echo "Результаты: results/refine.csv"
echo ""
echo "Главный вопрос к таблице: обгоняет ли хоть одно правило corr_greedy."
echo "corr_greedy обновляет граф ТОЛЬКО по корреляциям, без обученной сети."
echo "Если умные правила его не бьют - статистики модели ничего не добавляют,"
echo "и это законченный отрицательный результат, а не неудача."
echo ""
echo "Свести таблицу:  python src/refine.py --summary"
