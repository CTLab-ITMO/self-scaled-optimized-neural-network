#!/usr/bin/env bash

set -u
cd "$(dirname "$0")" || exit 1

PY=${PY:-python}
STATE=results/.series
mkdir -p "$STATE"


USE_TF32=${USE_TF32:-1}

CAND="grid4 grid8 thr_corr knn_corr_k8 er_grid4 er_knn8"


SEEDS3="0 1 2"
SEEDS5="0 1 2 3 4"
DEPTHS="1 2 3 4 5 6 8"
CALIB_BUDGET=400          # верхняя граница поиска плато

TF32_FLAG=""
[ "$USE_TF32" = "1" ] && TF32_FLAG="--tf32"

SMOKE=${SMOKE:-0}
EXTRA=""        # для run_all.py: умеет --out
TRAIN_EXTRA=""  # для train.py: --out не знает, ему нужен только --dev
POOL2="--pool 2"
if [ "$SMOKE" = "1" ]; then
  CAND="grid4 er_grid4"
  SEEDS3="0"; SEEDS5="0"
  DEPTHS="1 2"
  CALIB_BUDGET=8
  TF32_FLAG=""            # на CPU бессмысленно, а на карте меняло бы ключи
  EXTRA="--dev --out results/smoke.csv"
  TRAIN_EXTRA="--dev"
  POOL2=""                # --pool и --dev несовместимы по построению
  STATE=results/.series_smoke
  mkdir -p "$STATE"
fi


C_OK=$'\033[32m'; C_RUN=$'\033[36m'; C_WARN=$'\033[33m'; C_OFF=$'\033[0m'
[ -t 1 ] || { C_OK=""; C_RUN=""; C_WARN=""; C_OFF=""; }

STAGE_IDS=(1 1b 2 3 4 5 6 7 8)
STAGE_NAMES=(
  "1 замеры: сколько дают TF32 и сколько стоит GAT"
  "1b H1-strong: разрыв по ВЫБОРКЕ перестановок (главный результат!)"
  "2 калибровка бюджета эпох при L=1 и L=2"
  "3 H4b: смена порядка кандидатов на малой глубине"
  "4 H8: внимание против богатства признаков (d=196)"
  "5 калибровка бюджета для остальных глубин"
  "6 H4/H5/H6: основная сетка глубин"
  "7 H7: полная кривая на изоморфной паре"
  "8 необязательное: L=3 в TF32, H7 до 10 сидов"
)

done_marker () { echo "$STATE/stage$1.done"; }
is_done ()     { [ -f "$(done_marker "$1")" ]; }
mark_done ()   { date +"%Y-%m-%d %H:%M:%S" > "$(done_marker "$1")"; }

banner () {
  echo ""
  echo "${C_RUN}================================================================${C_OFF}"
  echo "${C_RUN}  ЭТАП $*${C_OFF}"
  echo "${C_RUN}================================================================${C_OFF}"
}

run_log () {
  local log="$STATE/stage$1.log"; shift
  echo "  \$ $*" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

on_int () {
  echo ""
  echo "${C_WARN}Прервано. Ничего не потеряно - продолжить тем же:${C_OFF}"
  echo "    ./run_series.sh"
  exit 130
}
trap on_int INT TERM

# Достаёт рекомендованный бюджет эпох для глубины $1, при надобности калибруя.
# Результат кешируется: повторный запуск скрипта не пересчитывает.
calib_epochs () {
  local L=$1 stage=$2 cache="$STATE/epochs_L$L.txt"
  if [ -s "$cache" ]; then cat "$cache"; return 0; fi

  local devflag=""; [ "$SMOKE" = "1" ] && devflag="--dev"
  run_log "$stage" $PY src/calibrate.py --layers "$L" \
      --epochs "$CALIB_BUDGET" $TF32_FLAG $devflag >/dev/null

  # L=3 у gcn в боевом режиме исторически пишется в calibration.txt
  local mode=full; [ "$SMOKE" = "1" ] && mode=dev
  local f="results/calibration_${mode}_gcn_L$L.txt"
  [ -f "$f" ] || f="results/calibration.txt"
  local e
  e=$(sed -n 's/.*train\.epochs = \([0-9]\+\).*/\1/p' "$f" | head -1)

  if [ -z "$e" ]; then
    echo "${C_WARN}  [!] не удалось прочитать бюджет из $f${C_OFF}" >&2
    echo "${C_WARN}      беру потолок поиска $CALIB_BUDGET - это НИЖНЯЯ оценка,${C_OFF}" >&2
    echo "${C_WARN}      в отчёте оговорить, что кандидаты могли не доучиться${C_OFF}" >&2
    e=$CALIB_BUDGET
  fi
  echo "$e" > "$cache"
  echo "$e"
}

# ---------------------------------------------------------------------------
#  Этапы
# ---------------------------------------------------------------------------

stage1 () {
  banner "${STAGE_NAMES[0]}"
  echo "Ничего не пишет в results.csv - только измеряет."
  echo ""
  local log="$STATE/stage1.log"
  {
    echo "=== замеры $(date +%F\ %T) ==="
    echo "--- GCN, 10 эпох, FP32 ---"; time $PY src/train.py --graph grid4 --seed 0 --epochs 10 $TRAIN_EXTRA
    echo "--- GCN, 10 эпох, TF32 ---"; time $PY src/train.py --graph grid4 --seed 0 --epochs 10 --tf32 $TRAIN_EXTRA
    echo "--- GAT, 10 эпох, d=784 ---"; time $PY src/train.py --graph grid4 --seed 0 --epochs 10 --arch gat $TF32_FLAG $TRAIN_EXTRA
    echo "--- GAT, 10 эпох, d=196 (как в этапе 4) ---"; time $PY src/train.py --graph grid4 --seed 0 --epochs 10 --arch gat $POOL2 $TF32_FLAG $TRAIN_EXTRA
  } 2>&1 | tee -a "$log"

  echo ""
  echo "${C_WARN}ПРОЧТИ ЦИФРЫ ВЫШЕ ПЕРЕД ТЕМ, КАК ИДТИ ДАЛЬШЕ.${C_OFF}"
  echo "  * во сколько раз TF32 быстрее - на столько же сократится вся серия;"
  echo "  * во сколько раз GAT дороже GCN - от этого зависит бюджет этапа 4;"
  echo "  * если GAT при d=784 дороже больше чем в 10 раз, полное разрешение"
  echo "    для H8 не рассматривать вообще."
  echo ""
  echo "  замеры сохранены: $log"
  mark_done 1
}

stage1b () {
  banner "${STAGE_NAMES[1]}"
  echo "Главный вывод работы сейчас опирается на ОДНУ перестановку sigma:"
  echo "пять сидов меняли только веса, граф был один и тот же. Здесь берём"
  echo "десять изоморфных решётке графов и проверяем, что разрыв держится"
  echo "по выборке перестановок, а не по удаче с одной."
  echo ""
  local MIS="grid4 grid4_misaligned"
  for k in 2 3 4 5 6 7 8 9 10; do MIS="$MIS grid4_misaligned_s$k"; done
  run_log 1b $PY src/run_all.py --graphs $MIS --seeds $SEEDS3 \
      $TF32_FLAG $EXTRA || return 1
  mark_done 1b
}

stage2 () {
  banner "${STAGE_NAMES[1]}"
  echo "150 эпох подобраны только при L=3. Протокол П1 требует калибровать"
  echo "каждую глубину отдельно, иначе меряется скорость сходимости."
  echo ""
  local e1 e2
  e1=$(calib_epochs 1 2); e2=$(calib_epochs 2 2)
  echo "  L=1 -> $e1 эпох"
  echo "  L=2 -> $e2 эпох"
  mark_done 2
}

stage3 () {
  banner "${STAGE_NAMES[2]}"
  echo "Самая дешёвая гипотеза серии. Решётка выигрывает уже при L=3, значит"
  echo "пересечение обязано быть при L<3 - или его нет вовсе. Ставка"
  echo "зарегистрирована как РИСКОВАННАЯ: скорее всего не подтвердится."
  echo ""
  local e
  for L in 1 2; do
    e=$(calib_epochs "$L" 2)
    echo "  --- L=$L, бюджет $e эпох ---"
    run_log 3 $PY src/run_all.py --graphs $CAND --layers "$L" \
        --seeds $SEEDS3 --epochs "$e" $TF32_FLAG $EXTRA || return 1
  done
  mark_done 3
}

stage4 () {
  banner "${STAGE_NAMES[3]}"
  echo "H8 при d=196 (--pool 2): полная выборка, полный бюджет, но GAT в 16"
  echo "раз дешевле. Лемма про два параметра внимания от разрешения не"
  echo "зависит, проверяется только перенос на качество."
  echo "${C_WARN}В отчёте обязательно указать, что H8 мерена при d=196.${C_OFF}"
  echo ""
  local e
  e=$(calib_epochs 3 5)
  run_log 4 $PY src/run_all.py --graphs grid4 grid4_misaligned \
      --arch gcn gat --in-channels 1 8 --seeds $SEEDS5 \
      $POOL2 --epochs "$e" $TF32_FLAG $EXTRA || return 1
  echo "  --- GIN третьей точкой ---"
  run_log 4 $PY src/run_all.py --graphs grid4 grid4_misaligned \
      --arch gin --in-channels 1 8 --seeds $SEEDS3 \
      $POOL2 --epochs "$e" $TF32_FLAG $EXTRA || return 1
  mark_done 4
}

stage5 () {
  banner "${STAGE_NAMES[4]}"
  local e
  for L in $DEPTHS; do
    e=$(calib_epochs "$L" 5)
    echo "  L=$L -> $e эпох"
  done
  mark_done 5
}

stage6 () {
  banner "${STAGE_NAMES[5]}"
  echo "Самый дорогой этап. Три гипотезы с одних прогонов: H4 (разброс"
  echo "достаточной глубины), H5 (оптимум при равной зоне охвата),"
  echo "H6 (энергия Дирихле работает тем лучше, чем глубже)."
  echo ""
  local e
  for L in $DEPTHS; do
    e=$(calib_epochs "$L" 5)
    echo "  --- L=$L, бюджет $e эпох ---"
    run_log 6 $PY src/run_all.py --graphs $CAND --layers "$L" \
        --seeds $SEEDS3 --epochs "$e" $TF32_FLAG $EXTRA || return 1
  done
  mark_done 6
}

stage7 () {
  banner "${STAGE_NAMES[6]}"
  echo "Изоморфная пара: зона охвата у неё тождественна на всех L, значит"
  echo "любая разница - чистая согласованность с данными. Нужна ВСЯ кривая,"
  echo "максимума мало: гипотеза про немонотонность."
  echo ""
  echo "Здесь же считается контроль конфаунда к H4: если L_opt у пары"
  echo "совпал, H4 про структуру; если разошёлся - H4 смешана с"
  echo "согласованностью и в такой формулировке непригодна."
  echo ""
  local e
  for L in $DEPTHS; do
    e=$(calib_epochs "$L" 5)
    echo "  --- L=$L, бюджет $e эпох ---"
    run_log 7 $PY src/run_all.py --graphs grid4 grid4_misaligned \
        --layers "$L" --seeds $SEEDS5 --epochs "$e" $TF32_FLAG $EXTRA || return 1
  done
  mark_done 7
}

stage8 () {
  banner "${STAGE_NAMES[7]}"
  echo "Необязательное. Всё существенное уже посчитано - это добор точности."
  echo ""
  if [ "$USE_TF32" = "1" ]; then
    echo "  --- L=3 в TF32: однородность серии со старыми 110 прогонами ---"
    local e; e=$(calib_epochs 3 5)
    run_log 8 $PY src/run_all.py --graphs $CAND --layers 3 \
        --seeds $SEEDS3 --epochs "$e" --tf32 $EXTRA || return 1
  fi
  echo "  --- H7 до 10 сидов ---"
  local e
  for L in $DEPTHS; do
    e=$(calib_epochs "$L" 5)
    run_log 8 $PY src/run_all.py --graphs grid4 grid4_misaligned \
        --layers "$L" --seeds 5 6 7 8 9 --epochs "$e" $TF32_FLAG $EXTRA || return 1
  done
  mark_done 8
}

# ---------------------------------------------------------------------------
#  Разбор аргументов и главный цикл
# ---------------------------------------------------------------------------

show_status () {
  echo ""
  echo "  режим TF32: $([ "$USE_TF32" = 1 ] && echo включён || echo выключен)"
  echo "  кандидаты:  $CAND"
  echo ""
  local i=0 id
  for id in "${STAGE_IDS[@]}"; do
    if is_done "$id"; then
      echo "  ${C_OK}[сделан]${C_OFF}  ${STAGE_NAMES[$i]}   ($(cat "$(done_marker "$id")"))"
    else
      echo "  [ ждёт ]  ${STAGE_NAMES[$i]}"
    fi
    i=$((i + 1))
  done
  echo ""
  if [ -f results/results.csv ]; then
    echo "  строк в results.csv: $(($(wc -l < results/results.csv) - 1))"
  fi
}

ONLY=""; REDO=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)  echo "прогонка скрипта: dev-режим, результаты в results/smoke.csv"; shift ;;
    --status) show_status; exit 0 ;;
    --only)   ONLY="$2"; shift 2 ;;
    --redo)   REDO="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "неизвестный аргумент: $1"; exit 1 ;;
  esac
done

[ -n "$REDO" ] && { rm -f "$(done_marker "$REDO")"; echo "этап $REDO помечен несделанным"; }

if [ "$SMOKE" != "1" ] && ! $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "${C_WARN}[!] CUDA недоступна. На процессоре эта серия займёт недели.${C_OFF}"
  echo "    Проверь: $PY -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
  printf "    Всё равно продолжить? [y/N] "; read -r a
  [ "$a" = "y" ] || exit 1
fi

START=$(date +%s)
for n in "${STAGE_IDS[@]}"; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$n" ] && continue
  if is_done "$n" && [ -z "$ONLY" ]; then
    echo "${C_OK}[сделан]${C_OFF} этап $n"
    continue
  fi
  if ! "stage$n"; then
    echo ""
    echo "${C_WARN}Этап $n завершился с ошибкой. Этап НЕ отмечен сделанным.${C_OFF}"
    echo "Посчитанное сохранено. Разберись по логу и запусти скрипт снова:"
    echo "    tail -40 $STATE/stage$n.log"
    exit 1
  fi
  # После этапа 1 нужен человек: цифры замеров определяют бюджет остального.
  if [ "$n" = 1 ] && [ -z "$ONLY" ] && [ "$SMOKE" != "1" ]; then
    printf "\nПродолжить с этапа 2? [Y/n] "; read -r a
    [ "$a" = "n" ] && exit 0
  fi
done

echo ""
echo "${C_OK}================================================================${C_OFF}"
echo "${C_OK}  СЕРИЯ ЗАВЕРШЕНА за $((($(date +%s) - START) / 3600)) ч $(((($(date +%s) - START) % 3600) / 60)) мин${C_OFF}"
echo "${C_OK}================================================================${C_OFF}"
echo ""
echo "Дальше:"
echo "    $PY src/analyze.py            таблицы и графики"
echo "    ./run_series.sh --status      что сделано"
echo ""
echo "Проверять гипотезы только по критериям из docs/PREREGISTRATION.md."
