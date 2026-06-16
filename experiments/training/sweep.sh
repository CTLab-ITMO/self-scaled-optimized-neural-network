#!/bin/bash
# 3-seed × 4-lambda sweep over the convex regularizer.
export PATH=/mnt/storage00/usr/austinov/ssonn/python/bin:${PATH:-}
export PYTHONPATH=/mnt/storage00/usr/austinov/ssonn/ssonn-full-conv:${PYTHONPATH:-}
unset CUDA_VISIBLE_DEVICES

mkdir -p sweep_logs
results="sweep_results.csv"
echo "lambda,seed,test_top1,test_top5,test_loss" > "$results"

for seed in 0 1 2; do
  for lam in 0 0.005 0.01 0.05; do
    log="sweep_logs/lambda${lam}_seed${seed}.log"
    rm -f "$log"
    echo "[$(date +%H:%M:%S)] λ=$lam seed=$seed -> $log"
    SWEEP_SEED=$seed SWEEP_LAMBDA=$lam \
      torchrun --nproc_per_node=2 --standalone cifar10_basic_cnn_convex.py > "$log" 2>&1
    line=$(grep -E "^test" "$log" | head -1)
    top1=$(echo "$line" | grep -oP 'top1=\K[0-9.]+')
    top5=$(echo "$line" | grep -oP 'top5=\K[0-9.]+')
    loss=$(echo "$line" | grep -oP 'loss=\K[0-9.]+')
    echo "$lam,$seed,$top1,$top5,$loss" >> "$results"
    echo "  -> top1=$top1%  top5=$top5%  loss=$loss"
  done
done

echo
echo "=== sweep done ==="
echo
python3 - <<'PY'
import csv, statistics
rows = list(csv.DictReader(open("sweep_results.csv")))
by_lambda = {}
for r in rows:
    by_lambda.setdefault(r["lambda"], []).append(float(r["test_top1"]))
print(f"{'lambda':<8} {'n':<3} {'mean':<7} {'std':<7} {'min':<7} {'max':<7}  values")
for lam, vals in sorted(by_lambda.items(), key=lambda x: float(x[0])):
    n = len(vals)
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    print(f"{lam:<8} {n:<3} {mean:<7.2f} {std:<7.2f} {min(vals):<7.2f} {max(vals):<7.2f}  {vals}")
PY
