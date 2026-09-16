#!/usr/bin/env bash

# пайплайн из множества разных экспериментов:
# 1. Обучить GNN на каждом выбранном графе для нескольких seeds
# 2. Посчитать метрики
# 3. Сравнить правильный `grid4` с изоморфным, но неправильно сопоставленным `grid4_misaligned`
# 4. Сравнить `grid4` со случайным графом той же плотности, чтобы проверить, что эффект не объясняется просто числом рёбер
# 5. Сравнить data-driven `knn_corr_k8` со случайным графом той же плотности
# 6. Сравнить рабочие графы с крайними контролями `empty` и `complete` и с MLP без message passing

# результат: правильные/data-driven графы сильно лучше misaligned и random controls

set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${PY:-}" ]; then
  PYTHON="$PY"
elif [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

GRAPHS=(
  mlp
  empty
  complete
  grid4
  grid4_misaligned
  er_grid4
  knn_corr_k8
  er_knn8
)
SEEDS=(0 1 2 3 4)
OUT=results/results.csv

case "${1:-}" in
  --smoke)
    GRAPHS=(grid4 grid4_misaligned er_grid4)
    SEEDS=(0)
    OUT=results/smoke_validated.csv
    exec "$PYTHON" src/run_all.py \
      --dev \
      --graphs "${GRAPHS[@]}" \
      --seeds "${SEEDS[@]}" \
      --layers 3 \
      --epochs 8 \
      --arch gcn \
      --in-channels 1 \
      --out "$OUT"
    ;;
  --precheck)
    exec "$PYTHON" src/depth_precheck.py
    ;;
  --status)
    "$PYTHON" - <<'PY'
import csv
from pathlib import Path

graphs = {
    "mlp", "empty", "complete", "grid4", "grid4_misaligned",
    "er_grid4", "knn_corr_k8", "er_knn8",
}
seeds = set(range(5))
p = Path("results/results.csv")
if not p.exists():
    print("0/40 validated runs present")
    raise SystemExit(0)

seen = set()
with p.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            graph = r["graph"]
            seed = int(r["seed"])
            arch = r.get("arch") or "gcn"
            layers = int(r.get("layers") or 3)
            channels = int(r.get("in_channels") or 1)
            tf32 = str(r.get("tf32") or "False")
            d = int(r.get("d") or 784)
            epochs = int(r.get("epochs") or 150)
            dev = str(r.get("dev") or "False")
        except (KeyError, TypeError, ValueError):
            continue
        if (graph in graphs and seed in seeds and arch == "gcn" and
                layers == 3 and channels == 1 and tf32 == "False" and
                d == 784 and epochs == 150 and dev == "False"):
            seen.add((graph, seed))

print(f"{len(seen)}/40 validated runs present")
missing = [(g, s) for g in sorted(graphs) for s in sorted(seeds)
           if (g, s) not in seen]
if missing:
    print("missing:")
    for g, s in missing:
        print(f"  {g} seed={s}")
PY
    exit 0
    ;;
  -h|--help)
    cat <<'EOF2'
Usage:
  ./run_series.sh            rerun/complete the validated L=3 GCN benchmark
  ./run_series.sh --status   show which validated runs are already present
  ./run_series.sh --smoke    quick dev check into results/smoke_validated.csv
  ./run_series.sh --precheck recompute results/depth_precheck.csv
EOF2
    exit 0
    ;;
  "") ;;
  *)
    echo "unknown argument: $1" >&2
    exit 2
    ;;
esac

if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch is not available in $PYTHON. Run ./setup_env.sh first." >&2
  exit 1
fi

exec "$PYTHON" src/run_all.py \
  --graphs "${GRAPHS[@]}" \
  --seeds "${SEEDS[@]}" \
  --layers 3 \
  --epochs 150 \
  --arch gcn \
  --in-channels 1 \
  --out "$OUT"
