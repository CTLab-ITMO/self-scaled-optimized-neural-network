from __future__ import annotations

import argparse
import sys

import numpy as np

from config import load_config
from data import load_data
from train import get_device, run_one

GATE_ORDER = ["mlp", "grid4", "empty", "complete"]

DEV_GATE_EPOCHS = 40
COLLAPSE_TOL = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None,
                    help="ТОЛЬКО для калибровки бюджета обучения")
    args = ap.parse_args()

    cfg = load_config(dev=args.dev)
    if args.epochs is not None:
        cfg.train["epochs"] = args.epochs
    elif cfg.dev and cfg.train.epochs < DEV_GATE_EPOCHS:
        cfg.train["epochs"] = DEV_GATE_EPOCHS
        print(f"[dev] бюджет поднят до {DEV_GATE_EPOCHS} эпох: на пяти из конфига\n"
              f"      кандидаты не успевают обучиться и гейт падает без причины\n")
    data = load_data(cfg)
    device = get_device()

    print(f"устройство: {device}   режим: {'dev' if args.dev else 'full'}   "
          f"d = {cfg.data.d}   эпох: {cfg.train.epochs}   сид: {args.seed}")
    print(f"hash конфига: {cfg.hash}\n")

    rows: dict[str, dict] = {}
    for name in GATE_ORDER:
        print(f"обучаю {name} ...", flush=True)
        row, _ = run_one(name, args.seed, cfg, data, device)
        rows[name] = row
        print(f"  acc {row['test_accuracy']:.4f}  F1 {row['test_macro_f1']:.4f}  "
              f"лучшая эпоха {row['best_epoch']}/{row['epochs']}  "
              f"{row['train_time_s']:.0f} с")

    header = (f"\n{'кандидат':<10} {'test acc':>9} {'macro-F1':>9} {'loss':>7} "
              f"{'луч.эп':>7} {'косинус':>8}")
    print(header)
    print("-" * (len(header) - 1))
    for name in GATE_ORDER:
        r = rows[name]
        cos = "-" if np.isnan(r["node_cosine"]) else f"{r['node_cosine']:.4f}"
        print(f"{name:<10} {r['test_accuracy']:>9.4f} {r['test_macro_f1']:>9.4f} "
              f"{r['test_loss']:>7.4f} {r['best_epoch']:>4}/{r['epochs']:<2} "
              f"{cos:>8}")

    acc = {k: rows[k]["test_accuracy"] for k in GATE_ORDER}
    checks: list[tuple[str, bool, str]] = [
        ("mlp > grid4", acc["mlp"] > acc["grid4"],
         f"{acc['mlp']:.4f} > {acc['grid4']:.4f}"),
        ("grid4 > empty", acc["grid4"] > acc["empty"],
         f"{acc['grid4']:.4f} > {acc['empty']:.4f}"),
        ("empty > complete", acc["empty"] > acc["complete"],
         f"{acc['empty']:.4f} > {acc['complete']:.4f}"),
        ("complete: вершины схлопнулись (косинус = 1)",
         abs(rows["complete"]["node_cosine"] - 1.0) < COLLAPSE_TOL,
         f"косинус = {rows['complete']['node_cosine']:.6f}"),
        ("empty: вершины НЕ схлопнулись",
         rows["empty"]["node_cosine"] < 1.0 - COLLAPSE_TOL,
         f"косинус = {rows['empty']['node_cosine']:.6f}"),
        ("complete не лучше случайного угадывания более чем вдвое",
         acc["complete"] < 0.25,
         f"{acc['complete']:.4f} < 0.25"),
    ]

    print("\nпроверки:")
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'OK ' if passed else 'ПЛОХО'}] {name}   ({detail})")
        ok &= passed

    # Не проверка, а предупреждение: диапазон между полом и оракулом - это
    # весь бюджет, в котором эксперимент вообще способен что-то различить.
    span = acc["grid4"] - acc["empty"]
    print(f"\nдиапазон grid4 - empty: {span:.4f}")
    if span < 0.10:
        print("  [!] диапазон узкий: разница между кандидатами утонет в разбросе")
        print("      по сидам. Скорее всего модель недообучена - смотри, не")
        print("      упирается ли лучшая эпоха в последнюю.")

    at_last = [n for n in ("grid4", "empty", "complete")
               if rows[n]["best_epoch"] >= rows[n]["epochs"]]
    if at_last:
        print(f"  [!] лучшая эпоха = последняя у: {', '.join(at_last)}")
        print("      бюджета обучения не хватает, нужно поднимать train.epochs")

    print(f"\n{'ГЕЙТ ПРОЙДЕН' if ok else 'ГЕЙТ НЕ ПРОЙДЕН - дальше не идти'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
