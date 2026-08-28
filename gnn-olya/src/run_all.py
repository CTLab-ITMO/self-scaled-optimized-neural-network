"""Шаг 8: основной прогон - все кандидаты x все сиды.

Прерываемый и возобновляемый: строки дописываются в results.csv по одной и
сразу сбрасываются на диск, а уже посчитанные пары (кандидат, сид, hash
конфига, режим) пропускаются. Если прогон убьют на середине - просто запусти
эту же команду снова.

Использование:
    python src/run_all.py                       # всё, боевой режим
    python src/run_all.py --dev                 # всё, отладочный режим
    python src/run_all.py --graphs grid4 empty  # подмножество
    python src/run_all.py --groups R O          # по группам
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from config import artifacts_dir, load_config, results_dir
from data import load_data
from graphs import _BUILDERS, available
from train import get_device, run_one, set_tf32

REFERENCE = "mlp"  # не граф, а потолок шкалы


def _plan(args, cfg) -> list[str]:
    if args.graphs:
        names = list(args.graphs)
    elif args.groups:
        names = [n for n in available() if _BUILDERS[n][1] in set(args.groups)]
        if "R" in set(args.groups):
            names.append(REFERENCE)
    else:
        names = [REFERENCE] + available()
    unknown = [n for n in names if n != REFERENCE and n not in _BUILDERS]
    if unknown:
        raise SystemExit(f"неизвестные кандидаты: {unknown}")
    return names


def _acquire_lock(res_dir: Path):
    """Не дать запустить второй прогон поверх первого.

    Блокировка файловая, а не поиск по имени процесса: pgrep ловит ещё и
    оболочку-запускатель. Снимается сама при завершении, включая падение.
    """
    lock_path = res_dir / ".run_all.lock"
    try:
        import fcntl
    except ImportError:  # не Unix - просто без защиты
        return None
    fh = lock_path.open("w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            f"прогон уже запущен (см. {lock_path}).\n"
            "Два прогона писали бы в один results.csv вперемешку.\n"
            "  посмотреть лог идущего:  tail -f run.log\n"
            "  остановить его:          pkill -f run_all.py"
        )
    fh.write(str(os.getpid()))
    fh.flush()
    return fh  # держим открытым до конца процесса: закрытие снимет блокировку


def _done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # layers добавлен позже: в старых строках столбца нет, но все
            # прогоны до его появления делались при L = 3.
            # arch/layers/in_channels добавлены позже: в старых строках их
            # нет, но все прогоны до их появления делались на gcn, L = 3,
            # со скалярным признаком.
            done.add((r["graph"], int(r["seed"]), r.get("arch") or "gcn",
                      int(r.get("layers") or 3), int(r.get("in_channels") or 1),
                      str(r.get("tf32") or "False"), int(r.get("d") or 784),
                      int(r.get("epochs") or 0),
                      r["cfg_hash"], r["dev"]))
    return done


def _done_accs(path: Path, tf32: str, d: int, cfg_hash: str,
               dev: str, epochs: int) -> list[tuple]:
    """Уже посчитанные прогоны текущего режима как (группа, L, accuracy).

    Нужно поиску максимума: при возобновлении прогона состояние обязано
    учитывать то, что посчитано в прошлый раз, иначе правило обрыва начнёт
    отсчёт заново и пересчитает уже пройденный хвост.
    """
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (str(r.get("tf32") or "False") != tf32
                    or int(r.get("d") or 784) != d
                    or int(r.get("epochs") or 0) != epochs
                    or r["cfg_hash"] != cfg_hash or r["dev"] != dev):
                continue
            if r["graph"] == REFERENCE:
                continue
            out.append(((r["graph"], int(r["seed"]), r.get("arch") or "gcn",
                         int(r.get("in_channels") or 1)),
                        int(r.get("layers") or 3),
                        float(r["test_accuracy"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--graphs", nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="Сетка глубин, например --layers 1 2 3 4 5 6 8. "
                         "По умолчанию - одно значение из конфига.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Бюджет обучения. Протокол П1 требует калибровать его "
                         "на КАЖДОЙ глубине отдельно (150 из конфига подобраны "
                         "только при L = 3), поэтому флаг здесь боевой, а не "
                         "отладочный. Входит в ключ пропуска: другой бюджет - "
                         "другой эксперимент, молча переиспользовать строку "
                         "со старым бюджетом нельзя.")
    ap.add_argument("--depth-search", type=int, default=0, metavar="ТЕРПЕНИЕ",
                    help="Не гнать всю сетку глубин, а искать максимум: для "
                         "каждого кандидата глубины идут по возрастанию, и "
                         "кандидат закрывается после N глубин подряд без "
                         "улучшения. Годится для H4 и H5 (им нужен только "
                         "argmax_L). НЕ годится для H6 и H7 - им нужны точки "
                         "по всему диапазону и вся кривая соответственно. "
                         "Пропущенные глубины печатаются в конце поимённо. "
                         "ОСТОРОЖНО: правило предполагает унимодальность "
                         "кривой по глубине, а она НЕ установлена. На "
                         "репетиции 2026-08-21 (dev, 5 эпох) вышло "
                         "0.195 при L=3, 0.177 при L=5 и 0.244 при L=8 - "
                         "с терпением 1 пик на L=8 был бы пропущен. Ставить "
                         "не меньше 2, а на финальных числах лучше не "
                         "пользоваться вовсе.")
    ap.add_argument("--pool", type=int, default=None,
                    help="Пулинг без урезания выборки (см. train.py). "
                         "d пишется в results.csv и входит в ключ пропуска.")
    ap.add_argument("--tf32", action="store_true",
                    help="TF32 на матмулах (Ampere и новее). См. train.py.")
    ap.add_argument("--arch", nargs="*", default=None,
                    choices=["gcn", "gat", "gin"],
                    help="Архитектуры, например --arch gcn gat. "
                         "По умолчанию - одна из конфига (gcn).")
    ap.add_argument("--in-channels", nargs="*", type=int, default=None,
                    help="Сетка обогащения признака, например "
                         "--in-channels 1 8. По умолчанию 1 (без обогащения).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(dev=args.dev, pool=args.pool)
    if args.epochs is not None:
        cfg.train["epochs"] = args.epochs
    cfg.train["tf32"] = bool(args.tf32)
    set_tf32(bool(args.tf32))
    data = load_data(cfg)
    device = get_device()
    seeds = args.seeds if args.seeds is not None else list(cfg.seeds)
    layers = args.layers if args.layers else [cfg.model.layers]
    archs = args.arch if args.arch else [cfg.model.kind]
    chans = args.in_channels if args.in_channels else [
        int(getattr(cfg.model, "in_channels", 1) or 1)]
    names = _plan(args, cfg)

    res_dir = results_dir(cfg)
    out_path = Path(args.out) if args.out else res_dir / "results.csv"
    cm_dir = artifacts_dir(cfg) / "confusion"
    cm_dir.mkdir(parents=True, exist_ok=True)

    _lock = _acquire_lock(res_dir)  # noqa: F841 - держит блокировку

    done = _done_keys(out_path)
    jobs = [(g, s, a, L, k)
            for g in names for s in seeds for a in archs
            for L in layers for k in chans
            if (g, s, a, L, k, str(bool(args.tf32)), cfg.data.d,
                cfg.train.epochs, cfg.hash, str(cfg.dev)) not in done]
    if args.depth_search:
        # Глубины внутри каждой (граф, сид, арх, каналы) обязаны идти по
        # возрастанию, иначе правило обрыва бессмысленно.
        jobs.sort(key=lambda j: (j[0], j[1], j[2], j[4], j[3]))

    # flush обязателен: при выводе в файл Python буферизует, и первые строки
    # висели бы в буфере минутами. Со стороны это выглядит как "ничего не
    # происходит" и провоцирует запустить прогон второй раз поверх первого.
    print(f"устройство: {device}   режим: {'dev' if cfg.dev else 'full'}   "
          f"d = {cfg.data.d}   эпох: {cfg.train.epochs}"
          f"{'   TF32' if args.tf32 else ''}", flush=True)
    print(f"hash конфига: {cfg.hash}", flush=True)
    print(f"кандидатов: {len(names)}   сидов: {len(seeds)}   "
          f"глубин: {len(layers)} {layers}   архитектур: {archs}   "
          f"каналов: {chans}", flush=True)
    total = len(names) * len(seeds) * len(layers) * len(archs) * len(chans)
    print(f"задач: {len(jobs)} (уже посчитано: {total - len(jobs)})", flush=True)
    print(f"результаты: {out_path}\n", flush=True)
    if not jobs:
        print("всё уже посчитано", flush=True)
        return 0

    header = None
    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
    fh = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=header) if header else None

    t_start = time.time()
    durations: list[float] = []
    failed: list[tuple[str, int, str]] = []

    best_acc: dict[tuple, float] = {}
    misses: dict[tuple, int] = {}
    skipped: list[tuple] = []

    if args.depth_search:
        # Проигрываем уже посчитанное в порядке возрастания глубины, чтобы
        # возобновлённый прогон продолжил поиск, а не начал его заново.
        prior = _done_accs(out_path, str(bool(args.tf32)), cfg.data.d,
                           cfg.hash, str(cfg.dev), cfg.train.epochs)
        for key, L, acc in sorted(prior, key=lambda t: (t[0], t[1])):
            if acc > best_acc.get(key, -1.0):
                best_acc[key] = acc
                misses[key] = 0
            else:
                misses[key] = misses.get(key, 0) + 1
        closed = sum(1 for v in misses.values() if v >= args.depth_search)
        if prior:
            print(f"поиск максимума: учтено {len(prior)} прошлых прогонов, "
                  f"закрыто групп: {closed}", flush=True)

    for i, (graph, seed, arch, L, k) in enumerate(jobs, 1):
        key = (graph, seed, arch, k)
        if args.depth_search and misses.get(key, 0) >= args.depth_search:
            skipped.append((graph, seed, arch, k, L))
            continue
        t0 = time.time()
        cfg.model["layers"] = L
        cfg.model["kind"] = arch
        cfg.model["in_channels"] = k
        try:
            row, cm = run_one(graph, seed, cfg, data, device)
        except Exception:  # noqa: BLE001
            # Один упавший кандидат не должен убивать многочасовой прогон.
            failed.append((f"{graph} {arch} L={L} k={k}", seed,
                           traceback.format_exc(limit=3)))
            print(f"[{i}/{len(jobs)}] {graph} сид {seed} {arch} L={L} k={k}: "
                  f"УПАЛ", flush=True)
            continue

        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            writer.writeheader()
        # Дозапись идёт по ЗАГОЛОВКУ УЖЕ СУЩЕСТВУЮЩЕГО файла. Если строка
        # содержит поле, которого в заголовке нет, DictWriter молча его
        # выбросит - и, например, глубина L потеряется, а прогоны на разных
        # L склеятся в одну строку. Падаем громко.
        missing = set(row) - set(writer.fieldnames)
        if missing:
            fh.close()
            raise RuntimeError(
                f"в {out_path.name} нет столбцов {sorted(missing)}, "
                f"а строка их содержит. Допишите столбцы в файл "
                f"(или пишите в новый через --out), иначе данные потеряются.")
        writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
        fh.flush()  # чтобы прерванный прогон не потерял уже посчитанное

        tag = "dev" if cfg.dev else "full"
        # L в имени: иначе прогоны на разной глубине затрут друг друга.
        # Для L = 3 имя оставлено прежним - там уже лежат посчитанные матрицы.
        suffix = "" if (L == 3 and arch == "gcn" and k == 1) else \
            f"_{arch}_L{L}" + (f"_k{k}" if k != 1 else "")
        np.save(cm_dir / f"cm_{tag}_d{cfg.data.d}_{graph}{suffix}_seed{seed}.npy", cm)

        # Прогноз по медиане последних прогонов, а не по среднему за всё
        # время: mlp считается в 5 раз быстрее GNN, и среднее по смеси даёт
        # заниженную втрое оценку, пока считаются первые (быстрые) задачи.
        if args.depth_search:
            acc = row["test_accuracy"]
            if acc > best_acc.get(key, -1.0):
                best_acc[key] = acc
                misses[key] = 0
            else:
                misses[key] = misses.get(key, 0) + 1

        durations.append(time.time() - t0)
        recent = sorted(durations[-5:])[len(durations[-5:]) // 2]
        eta = recent * (len(jobs) - i)
        print(f"[{i}/{len(jobs)}] {graph:<17} сид {seed} {arch} L={L:<2} "
              f"k={k} acc {row['test_accuracy']:.4f}  F1 {row['test_macro_f1']:.4f}  "
              f"эп {row['best_epoch']:>3}/{row['epochs']}  "
              f"{time.time() - t0:>5.0f} с   осталось ~{eta / 60:.0f} мин",
              flush=True)

    fh.close()
    print(f"\nготово за {(time.time() - t_start) / 60:.1f} мин -> {out_path}")

    # Никаких молчаливых обрезаний: что не посчитано, то названо поимённо.
    # Иначе таблица выглядит как полная сетка, а на деле в ней дыры.
    if skipped:
        print(f"\nпоиск максимума закрыл {len(skipped)} прогонов "
              f"(терпение {args.depth_search}):")
        for graph, seed, arch, k, L in skipped:
            print(f"  {graph} сид {seed} {arch} k={k} L={L} - не считался, "
                  f"максимум уже пройден")
        print("  ВАЖНО: такая таблица годится для H4 и H5, но НЕ для H6 и H7.")

    if failed:
        print(f"\nупало задач: {len(failed)}")
        for graph, seed, tb in failed:
            print(f"  {graph} сид {seed}\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
