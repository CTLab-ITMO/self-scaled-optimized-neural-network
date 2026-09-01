#!/usr/bin/env bash
# Восстановление истинного графа без обучения сети (пункт 4, честный режим).
#
#   ./run_structure.sh           все графы, три сида      ~10 мин
#   ./run_structure.sh --quick   один граф, один сид      ~1 мин
#   ./run_structure.sh --summary свести уже посчитанное
#
# Прерывать можно: строка пишется в results/structure.csv сразу после
# каждого метода, повторный запуск пропускает посчитанное.
#
# Сеть здесь не участвует, это чистая линейная алгебра на numpy. Видеокарта
# не ускорит, но и считать нечего: минуты, а не часы.

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

# Самопроверка перед прогоном: селекторы легко сломать незаметно, а таблица
# без них выглядит правдоподобно и молча врёт.
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

$PY src/eval_structure.py "$@"
rc=$?

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