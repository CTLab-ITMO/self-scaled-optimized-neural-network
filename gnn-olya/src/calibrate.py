"""Шаг 5: выбор числа эпох. Запускается НА GPU-МАШИНЕ, до основного прогона.

Число эпох - часть зафиксированного протокола, одно на всех кандидатов. Но
выбрать его надо осознанно: если бюджета не хватает, все кандидаты окажутся
недообучены, разница между ними утонет в разбросе по сидам, и эксперимент
измерит бюджет обучения вместо влияния графа.

Скрипт гоняет два опорных кандидата - grid4 (оракул, верх шкалы) и empty (пол
шкалы) - с заведомо избыточным бюджетом, смотрит, на какой эпохе val выходит
на плато, и печатает рекомендацию.

Опорные кандидаты именно эти два: подбирать гиперпараметры под интересных
кандидатов (kNN, контроли) нельзя, это была бы подгонка под ответ.

    python src/calibrate.py                 # на GPU-машине
    python src/calibrate.py --dev           # быстрая проверка, что работает
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from config import load_config, results_dir
from data import load_data
from train import get_device, run_one

# Опорная пара: медленная мешалка и пол шкалы. На них и калибруется бюджет.
PROBE_GRAPHS = ["grid4", "empty"]
SPAN_PAIR = ("grid4", "empty")  # чей разрыв accuracy считаем; должна быть подмножеством probe
PLATEAU_FRACTION = 0.995   # доля лучшего val, считающаяся выходом на плато
HEADROOM = 1.25            # запас поверх плато
MIN_USEFUL_SPAN = 0.10     # минимальный полезный разрыв grid4 - empty


def plateau_epoch(curve: list[float]) -> int:
    """Первая эпоха, на которой достигнуто PLATEAU_FRACTION от лучшего val."""
    if not curve:
        return 0
    target = PLATEAU_FRACTION * max(curve)
    for i, v in enumerate(curve, 1):
        if v >= target:
            return i
    return len(curve)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=None,
                    help="Калибровать под конкретную глубину. 150 эпох из "
                         "конфига подобраны ТОЛЬКО при L = 3; протокол П1 в "
                         "docs/PREREGISTRATION.md требует калибровать каждое L "
                         "отдельно, иначе сетка глубин измерит скорость "
                         "сходимости вместо качества графа.")
    ap.add_argument("--arch", default=None, choices=["gcn", "gat", "gin"],
                    help="Калибровать под конкретную архитектуру.")
    ap.add_argument("--graphs", nargs="*", default=None,
                    help=f"Опорные кандидаты (по умолчанию {PROBE_GRAPHS}).")
    ap.add_argument("--epochs", type=int, default=120,
                    help="заведомо избыточный бюджет для поиска плато")
    args = ap.parse_args()

    cfg = load_config(dev=args.dev)
    cfg.train["epochs"] = args.epochs
    if args.layers is not None:
        cfg.model["layers"] = args.layers
    if args.arch is not None:
        cfg.model["kind"] = args.arch
    probe = args.graphs if args.graphs else PROBE_GRAPHS
    if not set(SPAN_PAIR).issubset(probe):
        raise SystemExit(
            f"--graphs {probe} не включает опорную пару {SPAN_PAIR}: "
            f"разрыв accuracy между ними и есть то, что калибрует бюджет. "
            f"Добавь оба имени в --graphs или не используй флаг."
        )
    data = load_data(cfg)
    device = get_device()

    lines: list[str] = []

    def say(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    say(f"устройство: {device}   режим: {'dev' if cfg.dev else 'full'}   "
        f"d = {cfg.data.d}")
    say(f"hash конфига: {cfg.hash}")
    say(f"L={cfg.model.layers} hidden={cfg.model.hidden} lr={cfg.train.lr} "
        f"batch={cfg.train.batch_size}   бюджет поиска: {args.epochs} эпох")
    say(f"архитектура: {cfg.model.kind}   L = {cfg.model.layers}")
    say()

    probes = {}
    for name in probe:
        curve: list[float] = []
        t0 = time.time()
        row, _ = run_one(name, args.seed, cfg, data, device,
                         on_epoch=lambda e, l, v: curve.append(v))
        probes[name] = {"curve": curve, "accuracy": row["test_accuracy"],
                        "best_epoch": row["best_epoch"],
                        "plateau": plateau_epoch(curve)}
        say(f"  {name:<8} acc {row['test_accuracy']:.4f}   "
            f"лучшая эпоха {row['best_epoch']}/{args.epochs}   "
            f"плато с {probes[name]['plateau']}   {time.time() - t0:.0f} с")

    span = probes[SPAN_PAIR[0]]["accuracy"] - probes[SPAN_PAIR[1]]["accuracy"]
    plateau = max(p["plateau"] for p in probes.values())
    recommended = int(np.ceil(plateau * HEADROOM / 10.0) * 10)
    still_climbing = [n for n, p in probes.items()
                      if p["best_epoch"] >= args.epochs]

    say()
    say("=" * 64)
    say(f"разрыв grid4 - empty: {span:.4f}")
    say(f"плато достигается к эпохе {plateau}")
    say(f"РЕКОМЕНДУЕМОЕ train.epochs = {recommended}   "
        f"(плато {plateau} с запасом {HEADROOM:g}x)")
    say("=" * 64)
    say()

    if still_climbing:
        say(f"[!] у {', '.join(still_climbing)} лучшая эпоха упёрлась в последнюю -")
        say(f"    плато не найдено. Перезапусти: --epochs {args.epochs * 2}")
    elif span < MIN_USEFUL_SPAN:
        say(f"[!] разрыв {span:.4f} меньше {MIN_USEFUL_SPAN}: кандидаты будут")
        say("    различаться слабо, разница утонет в разбросе по сидам.")
        say("    Основной прогон НЕ запускай, привези results/calibration*")
    else:
        say(f"[ок] разрыв {span:.4f} достаточен, чтобы различать кандидатов")
        say(f"     на фоне разброса по сидам. Впиши train.epochs = {recommended}")
        say("     в configs/fixed.yaml и запускай src/sanity.py")

    # Имя файла кодирует режим, архитектуру и глубину. Без этого dev-прогон
    # затирает боевую калибровку - что и случилось 2026-08-21: репетиция на
    # ноутбуке уничтожила запись о том, откуда взялись 150 эпох. Само число
    # уцелело в fixed.yaml, но обоснование пропало.
    #
    # Историческое имя calibration.txt сохранено ровно за одним случаем -
    # боевой режим, gcn, L = 3: именно им посчитаны 110 прогонов.
    out = results_dir(cfg)
    is_legacy = (not cfg.dev and cfg.model.kind == "gcn"
                 and cfg.model.layers == 3)
    if is_legacy:
        stem = "calibration"
    else:
        mode = "dev" if cfg.dev else "full"
        stem = f"calibration_{mode}_{cfg.model.kind}_L{cfg.model.layers}"
    (out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / f"{stem}_curves.json").write_text(
        json.dumps({n: p["curve"] for n, p in probes.items()}), encoding="utf-8")
    print(f"\nсохранено: {out / (stem + '.txt')}"
          f"{' (привезти обратно)' if not cfg.dev else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
