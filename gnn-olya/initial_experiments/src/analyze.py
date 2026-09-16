# собирает результаты экспериментов и делает итоговый анализ
# сравнивает графы между собой и считает сводные числа

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from config import artifacts_dir, load_config, results_dir  # noqa: E402
from graphs import WS_P, ws_tag  # noqa: E402

GROUP_ORDER = ["R", "O", "D", "C", "W", "P"]
GROUP_NAME = {
    "R": "референсы", "O": "оракулы", "D": "data-driven",
    "C": "контроли", "W": "кривая перепроводки", "P": "патологии",
}
CONTROL_PAIRS = [
    ("grid4", "grid4_misaligned"),
    ("grid4", "er_grid4"),
    ("knn_corr_k8", "er_knn8"),
    ("knn_corr_k8", "config_knn8"),
    ("thr_corr", "er_grid4"),
]


def load_frames(cfg, dev: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    res_path = results_dir(cfg) / "results.csv"
    if not res_path.exists():
        raise SystemExit(f"нет {res_path} - сначала запусти src/run_all.py")
    res = pd.read_csv(res_path)
    res = res[(res.cfg_hash == cfg.hash) & (res.dev.astype(str) == str(dev))]
    if res.empty:
        raise SystemExit(
            f"в results.csv нет строк с hash={cfg.hash} и dev={dev}.\n"
            "Скорее всего конфиг правили после прогона - результаты "
            "несравнимы, нужен перезапуск."
        )
    stats_path = artifacts_dir(cfg) / f"graph_stats_d{cfg.data.d}.csv"
    if not stats_path.exists():
        raise SystemExit(f"нет {stats_path} - сначала запусти src/graph_stats.py")
    return res, pd.read_csv(stats_path)


def aggregate(res: pd.DataFrame, gstats: pd.DataFrame) -> pd.DataFrame:
    """Среднее и std по сидам + дескрипторы графа."""
    metrics = ["test_accuracy", "test_macro_f1", "test_loss",
               "val_accuracy", "train_accuracy", "node_cosine", "best_epoch"]
    agg = res.groupby("graph")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = res.groupby("graph").size()
    out = agg.reset_index().merge(gstats, on="graph", how="left")
    out["group"] = pd.Categorical(out["group"].fillna("R"),
                                  categories=GROUP_ORDER, ordered=True)
    return out.sort_values(["group", "test_accuracy_mean"],
                           ascending=[True, False])


def main_table(df: pd.DataFrame) -> str:
    lines = []
    head = (f"{'кандидат':<17} {'acc':>15} {'macro-F1':>15} {'loss':>7} "
            f"{'|E|':>7} {'плотн':>8} {'косин':>6}")
    for grp in GROUP_ORDER:
        sub = df[df.group == grp]
        if sub.empty:
            continue
        lines.append(f"\n=== {grp}: {GROUP_NAME[grp]} ===")
        lines.append(head)
        lines.append("-" * len(head))
        for _, r in sub.iterrows():
            std = 0.0 if pd.isna(r.test_accuracy_std) else r.test_accuracy_std
            f1s = 0.0 if pd.isna(r.test_macro_f1_std) else r.test_macro_f1_std
            lines.append(
                f"{r.graph:<17} "
                f"{r.test_accuracy_mean:>8.4f}±{std:<6.4f} "
                f"{r.test_macro_f1_mean:>8.4f}±{f1s:<6.4f} "
                f"{r.test_loss_mean:>7.4f} "
                f"{_fmt(r.get('stat_n_edges'), '.0f', 7)} "
                f"{_fmt(r.get('stat_density'), '.5f', 8)} "
                f"{_fmt(r.get('node_cosine_mean'), '.3f', 6)}")
    return "\n".join(lines)


def _fmt(v, spec: str, width: int) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-".rjust(width)
    return f"{v:>{width}{spec}}"


def test_h1(res: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """H1: m(A) не является функцией класса изоморфизма A.

    grid4 и grid4_misaligned изоморфны, поэтому ВСЕ структурные дескрипторы
    у них совпадают. Если качество значимо различается, никакой структурный
    дескриптор предсказать его не может.
    """
    out = ["\n" + "=" * 72, "H1: m(A) не определяется классом изоморфизма A", "=" * 72]
    a = res[res.graph == "grid4"]["test_accuracy"]
    b = res[res.graph == "grid4_misaligned"]["test_accuracy"]
    if a.empty or b.empty:
        return out + ["  нет данных по grid4 / grid4_misaligned"]

    struct = [c for c in df.columns if c.startswith("stat_")]
    ra = df[df.graph == "grid4"].iloc[0]
    rb = df[df.graph == "grid4_misaligned"].iloc[0]
    diverge = [c for c in struct
               if not np.isclose(ra[c], rb[c], rtol=1e-6, atol=1e-9)]

    out.append(f"  структурные дескрипторы расходятся в: {diverge or 'ни в одном'}")
    out.append(f"  grid4            {a.mean():.4f} ± {a.std():.4f}  (n={len(a)})")
    out.append(f"  grid4_misaligned {b.mean():.4f} ± {b.std():.4f}  (n={len(b)})")
    out.append(f"  разница          {a.mean() - b.mean():+.4f}")

    if len(a) > 1 and len(b) > 1:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        dz = (a.mean() - b.mean()) / pooled if pooled > 0 else np.inf
        out.append(f"  t-тест Уэлча: t = {t:.3f}, p = {p:.2e}, Cohen d = {dz:.2f}")
        verdict = "ПОДТВЕРЖДЕНА" if p < 0.05 and a.mean() > b.mean() else "не подтверждена"
        out.append(f"  -> H1 {verdict}")
    return out


# Десять независимых перестановок sigma для главного контроля - см.
# docs/PREREGISTRATION.md, раздел H1-strong. Порядок фиксирован там же.
H1_STRONG_VARIANTS = ["grid4_misaligned"] + [f"grid4_misaligned_s{k}" for k in range(2, 11)]
H1_STRONG_MIN_MARGIN = 0.10   # критерий подтверждения из docs/PREREGISTRATION.md


def test_h1_strong(res: pd.DataFrame) -> list[str]:
    """Разрыв grid4 / изоморфной решётки по выборке из 10 перестановок sigma.

    Базовый H1 использовал одну sigma, то есть выборку размера 1. Критерий
    подтверждения — в docs/PREREGISTRATION.md.
    """
    out = ["\n" + "=" * 72,
           "H1-strong: разрыв держится по выборке из 10 перестановок", "=" * 72]
    a = res[res.graph == "grid4"]["test_accuracy"]
    if a.empty:
        return out + ["  нет данных по grid4"]
    a_mean = a.mean()

    per_variant = []
    for name in H1_STRONG_VARIANTS:
        b = res[res.graph == name]["test_accuracy"]
        if not b.empty:
            per_variant.append((name, b.mean(), len(b)))
    if not per_variant:
        return out + ["  прогонов по изоморфным перестановкам ещё нет"]

    missing = [n for n in H1_STRONG_VARIANTS
               if n not in {v[0] for v in per_variant}]
    out.append(f"  grid4  {a_mean:.4f} ± {a.std():.4f}  (n={len(a)})")
    out.append("")
    below = 0
    for name, mean_acc, n in per_variant:
        is_below = mean_acc < a_mean
        below += is_below
        out.append(f"  [{'OK' if is_below else '!!'}] {name:<22} {mean_acc:.4f}  "
                   f"разрыв {a_mean - mean_acc:+.4f}  (n={n})")
    if missing:
        out.append(f"  нет данных: {missing}")

    variant_means = np.array([v[1] for v in per_variant])
    avg_gap = a_mean - variant_means.mean()
    out.append("")
    out.append(f"  среднее по {len(per_variant)} перестановкам: "
               f"{variant_means.mean():.4f}   средний разрыв: {avg_gap:+.4f}")

    if len(per_variant) < len(H1_STRONG_VARIANTS):
        out.append(f"  [!] {len(missing)} из {len(H1_STRONG_VARIANTS)} прогонов ещё нет - "
                   f"вердикт предварительный, не для отчёта")
    elif avg_gap >= H1_STRONG_MIN_MARGIN and below == len(H1_STRONG_VARIANTS):
        out.append(f"  -> H1-strong ПОДТВЕРЖДЕНА: средний разрыв "
                   f">= {H1_STRONG_MIN_MARGIN}, все {len(H1_STRONG_VARIANTS)} "
                   f"перестановок ниже grid4")
    else:
        reason = (f"средний разрыв {avg_gap:.4f} < {H1_STRONG_MIN_MARGIN}"
                 if avg_gap < H1_STRONG_MIN_MARGIN else
                 f"{len(H1_STRONG_VARIANTS) - below} перестановок на уровне "
                 f"grid4 или выше")
        out.append(f"  -> H1-strong НЕ ПОДТВЕРЖДЕНА: {reason}")
    return out


def test_controls(res: pd.DataFrame) -> list[str]:
    out = ["\n" + "=" * 72,
           "Парные сравнения с density-matched контролями", "=" * 72]
    for orig, ctrl in CONTROL_PAIRS:
        a = res[res.graph == orig]["test_accuracy"]
        b = res[res.graph == ctrl]["test_accuracy"]
        if a.empty or b.empty:
            continue
        line = f"  {orig:<15} {a.mean():.4f}  против  {ctrl:<17} {b.mean():.4f}   " \
               f"разница {a.mean() - b.mean():+.4f}"
        if len(a) > 1 and len(b) > 1:
            _, p = stats.ttest_ind(a, b, equal_var=False)
            line += f"   p = {p:.2e}"
        out.append(line)
    return out


def check_budget(res: pd.DataFrame) -> list[str]:
    """Кандидаты, у которых лучшая эпоха близка к последней, недообучены.

    Это не придирка: весь смысл работы в том, что протокол обучения одинаков
    для всех A. Если у части кандидатов бюджета не хватило, разница между ними
    частично отражает бюджет, а не влияние графа - и на защите это первое, о
    чём спросят. Такие кандидаты надо назвать в отчёте прямо.
    """
    out = ["\n" + "=" * 72, "Достаточность бюджета обучения", "=" * 72]
    g = res.groupby("graph").agg(best=("best_epoch", "mean"),
                                 total=("epochs", "max"))
    g["доля"] = g.best / g.total
    tight = g[g["доля"] > 0.9].sort_values("доля", ascending=False)
    out.append(f"  бюджет: {int(g.total.max())} эпох")
    if tight.empty:
        out.append("  все кандидаты вышли на плато внутри бюджета - ок")
    else:
        out.append(f"  [!] упёрлись в потолок ({len(tight)} из {len(g)}):")
        for name, r in tight.iterrows():
            out.append(f"      {name:<17} лучшая эпоха {r.best:.0f}/{r.total:.0f} "
                       f"({r['доля']:.0%})")
        out.append("  Этим кандидатам бюджета не хватило. В отчёте назвать явно:")
        out.append("  их результат - нижняя оценка, при большем бюджете был бы выше.")
    return out


def make_plots(df: pd.DataFrame, cfg, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {"R": "#888888", "O": "#d62728", "D": "#2ca02c",
              "C": "#1f77b4", "W": "#ff7f0e", "P": "#9467bd"}

    # Кривая деградации по доле перепроводки.
    ws = df[df.graph.str.startswith("ws_p", na=False)].copy()
    base = df[df.graph == "grid4"]
    if not ws.empty:
        # Имена кандидатов собираются в graphs.py как f"ws_p{p:g}" без точки.
        # Восстанавливаем соответствие тем же правилом, а не разбором строки.
        ws["p"] = ws.graph.map({ws_tag(p): p for p in WS_P})
        rows = ws[["p", "test_accuracy_mean", "test_accuracy_std"]]
        if not base.empty:
            rows = pd.concat([pd.DataFrame({
                "p": [0.0],
                "test_accuracy_mean": [base.test_accuracy_mean.iloc[0]],
                "test_accuracy_std": [base.test_accuracy_std.iloc[0]]}), rows])
        rows = rows.sort_values("p")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(rows.p, rows.test_accuracy_mean,
                    yerr=rows.test_accuracy_std.fillna(0), marker="o",
                    capsize=4, color="#ff7f0e")
        ax.set_xlabel("доля перепроводки p (|E| и степени неизменны)")
        ax.set_ylabel("accuracy на тесте")
        ax.set_title("Деградация решётки при сохранении степенной последовательности")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / "w_rewiring_curve.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths



def structure_report(cfg) -> list[str]:
    """Короткая сводка оставленного эксперимента восстановления структуры."""
    path = results_dir(cfg) / "structure.csv"
    if not path.exists():
        return ["\n" + "=" * 72, "Восстановление структуры", "=" * 72,
                "  нет results/structure.csv"]
    x = pd.read_csv(path)
    if x.empty:
        return ["\n" + "=" * 72, "Восстановление структуры", "=" * 72,
                "  structure.csv пуст"]

    base = x[x.n == x.n.max()].copy()
    methods = [m for m in ("corr_static", "partial_static", "partial_free")
               if m in set(base.method)]
    graphs = [g for g in ("tree", "grid", "smallworld", "community",
                           "scalefree", "ring", "geometric")
              if g in set(base.graph)]
    table = base.pivot_table(index="method", columns="graph", values="f1")
    table = table.reindex(index=methods, columns=graphs).round(3)

    out = ["\n" + "=" * 72, "Восстановление структуры", "=" * 72,
           f"  строк: {len(x)}   сидов: {x.seed.nunique()}   "
           f"n: {sorted(map(int, x.n.unique()))}",
           "  F1 на максимальном n, среднее по сидам:",
           table.to_string()]

    free = base[base.method == "partial_free"]
    if not free.empty:
        edges = free.groupby("graph")[["n_edges", "n_true"]].mean().round(0)
        edges = edges.reindex(graphs)
        out += ["", "  partial_free: найдено рёбер / истина:", edges.to_string()]
    return out


def refine_report(cfg) -> list[str]:
    """Короткая сводка оставленного эксперимента замены графа."""
    path = results_dir(cfg) / "refine.csv"
    if not path.exists():
        return ["\n" + "=" * 72, "Замена графа по данным", "=" * 72,
                "  нет results/refine.csv"]
    x = pd.read_csv(path)
    if x.empty:
        return ["\n" + "=" * 72, "Замена графа по данным", "=" * 72,
                "  refine.csv пуст"]

    key = ["rule", "graph", "seed"]
    x = x.sort_values(key + ["round"])
    n_run = x.groupby(key)["round"].transform("count")
    from_end = x.groupby(key)["round"].transform("max") - x["round"]
    tail = np.minimum(5, np.maximum(1, n_run // 4))
    last = x[from_end < tail].groupby(key, as_index=False).mean(numeric_only=True)
    rules = [r for r in ("keep", "random_rewire", "corr_static",
                          "partial_static", "partial_free")
             if r in set(last.rule)]
    graphs = [g for g in ("community", "geometric", "grid", "ring")
              if g in set(last.graph)]

    acc = last.pivot_table(index="rule", columns="graph", values="test_acc")
    rec = last.pivot_table(index="rule", columns="graph", values="rec_f1")
    acc = acc.reindex(index=rules, columns=graphs).round(4)
    rec = rec.reindex(index=rules, columns=graphs).round(4)
    runs = x[key].drop_duplicates().shape[0]
    rounds = x.groupby(key)["round"].count()

    return ["\n" + "=" * 72, "Замена графа по данным", "=" * 72,
            f"  строк: {len(x)}   прогонов: {runs}   сидов: {x.seed.nunique()}   "
            f"раундов на прогон: {int(rounds.min())}-{int(rounds.max())}",
            "  accuracy, среднее по хвосту каждого прогона:", acc.to_string(),
            "", "  F1 восстановления графа:", rec.to_string(),
            "", "  keep и random_rewire — контроли; corr_static — одношаговый baseline."]


def depth_report(cfg) -> list[str]:
    """Сводка сохранённого предрасчёта по глубине без обучения сети."""
    path = results_dir(cfg) / "depth_precheck.csv"
    if not path.exists():
        return ["\n" + "=" * 72, "Предрасчёт глубины", "=" * 72,
                "  нет results/depth_precheck.csv"]
    x = pd.read_csv(path)
    if x.empty:
        return ["\n" + "=" * 72, "Предрасчёт глубины", "=" * 72,
                "  depth_precheck.csv пуст"]

    keep = [g for g in ("grid4", "grid4_misaligned", "er_grid4",
                         "knn_corr_k8", "er_knn8", "empty", "complete")
            if g in set(x.graph)]
    cols = [c for c in ("graph", "n_edges", "n_components", "L_global",
                         "hops_L1", "hops_L2", "hops_L3") if c in x.columns]
    table = x[x.graph.isin(keep)][cols].copy()
    table["graph"] = pd.Categorical(table.graph, categories=keep, ordered=True)
    table = table.sort_values("graph")
    for c in [c for c in table.columns if c.startswith("hops_L")]:
        table[c] = table[c].round(2)
    return ["\n" + "=" * 72, "Предрасчёт глубины", "=" * 72,
            "  Сколько вершин в среднем доступно за 1/2/3 шага без обучения:",
            table.to_string(index=False)]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()

    cfg = load_config(dev=args.dev)
    res, gstats = load_frames(cfg, args.dev)
    df = aggregate(res, gstats)

    out_dir = results_dir(cfg) / ("figures_dev" if args.dev else "figures")
    report = [
        f"режим: {'dev' if args.dev else 'full'}   d = {cfg.data.d}   "
        f"hash = {cfg.hash}",
        f"кандидатов: {df.graph.nunique()}   прогонов: {len(res)}   "
        f"сидов на кандидата: {res.groupby('graph').size().min()}"
        f"-{res.groupby('graph').size().max()}",
        main_table(df),
        *test_h1(res, df),
        *test_controls(res), *check_budget(res),
        *structure_report(cfg), *refine_report(cfg), *depth_report(cfg),
    ]
    text = "\n".join(report)
    print(text)

    table_path = results_dir(cfg) / f"summary_{'dev' if args.dev else 'full'}.csv"
    df.to_csv(table_path, index=False)
    report_path = results_dir(cfg) / f"report_{'dev' if args.dev else 'full'}.txt"
    report_path.write_text(text, encoding="utf-8")

    figs = make_plots(df, cfg, out_dir)
    print(f"\nтаблица:  {table_path}")
    print(f"отчёт:    {report_path}")
    for p in figs:
        print(f"график:   {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
