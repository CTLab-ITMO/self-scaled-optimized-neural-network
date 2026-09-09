#!/usr/bin/env bash

# пытаемся восстановить структуру 
# данные → один раз построили graph → сравнили с truth
# делает self-check structure.py, затем запускает eval_structure.py; результаты в results/structure.csv

# результат: partial correlation лучше всего восстанавливает скрытый граф; partial_free близок без знания размера графа

set -u
cd "$(dirname "$0")" || exit 1

if [ -n "${PY:-}" ]; then
  :
elif [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi

if ! $PY -c "import numpy, scipy, pandas" >/dev/null 2>&1; then
  echo "В окружении нет numpy/scipy/pandas. Сначала: ./setup_env.sh"
  exit 1
fi

C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_OFF=$'\033[0m'
[ -t 1 ] || { C_OK=""; C_WARN=""; C_OFF=""; }

trap 'echo ""; echo "${C_WARN}Прервано. Посчитанное сохранено в"
      echo "results/structure.csv, повторный запуск продолжит.${C_OFF}"
      exit 130' INT TERM

echo "Самопроверка структурного модуля ..."
if ! $PY src/structure.py --dev > /tmp/structure_dev.$$ 2>&1; then
  echo "${C_WARN}САМОПРОВЕРКА НЕ ПРОШЛА - дальше идти нельзя:${C_OFF}"
  tail -n 25 /tmp/structure_dev.$$ | sed 's/^/    /'
  rm -f /tmp/structure_dev.$$
  exit 1
fi
rm -f /tmp/structure_dev.$$
echo "${C_OK}пройдена${C_OFF}"
echo ""

if [ "$#" -eq 0 ]; then
  # Сохранённая таблица считалась в два шага: 5 сидов на всей сетке n и
  # ещё 5 сидов только на n=6000. Повторяем именно этот протокол.
  $PY src/eval_structure.py --seeds 5 --n-grid 500,1000,2000,6000
  rc=$?
  [ "$rc" = 0 ] && $PY src/eval_structure.py --seeds 10 --n-grid 6000
  rc=$?
else
  $PY src/eval_structure.py "$@"
  rc=$?
fi

if [ "$rc" != 0 ]; then
  echo "${C_WARN}Прогон завершился с кодом $rc.${C_OFF}"
  exit "$rc"
fi

echo ""
echo "${C_OK}Готово.${C_OFF} Таблица: results/structure.csv"
echo ""
echo "Главный вопрос: обгоняет ли partial_free (плотность НЕ сообщали)"
echo "методы, которым её подарили. Если да - петля по параметрам делает"
echo "то, чего однократная формула не может в принципе."
