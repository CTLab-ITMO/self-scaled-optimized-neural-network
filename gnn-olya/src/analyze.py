"""Шаг 9: анализ результатов - таблица, графики, проверка гипотез H1-H3.

Считается на ноутбуке, GPU не нужен: на вход results.csv с боевой машины
и graph_stats_*.csv, посчитанный до обучения.

Использование:
    python src/analyze.py                 # боевые результаты
    python src/analyze.py --dev           # отладочные
"""

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
# Пары "оригинал -> его density-matched контроль". Ни одно утверждение
# "граф X помогает" не принимается без сравнения по этой таблице.
GOOD_ACC = 0.48   # порог «осмысленного» графа для проверки H2

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
            f"{'|E|':>7} {'плотн':>8} {'энерг':>7} {'ассорт':>7} {'косин':>6}")
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
                f"{_fmt(r.get('sig_dirichlet'), '.3f', 7)} "
                f"{_fmt(r.get('sig_assortativity'), '.3f', 7)} "
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


def test_h2(df: pd.DataFrame) -> list[str]:
    """H2: качество предсказывается гладкостью сигнала на графе.

    Корреляция считается дважды — по всем кандидатам и по осмысленным.
    Вырожденные графы одни создают сильную связь, не ранжируя нормальные.
    """
    out = ["\n" + "=" * 72,
           "H2: m(A) предсказывается сигнальными дескрипторами", "=" * 72]
    have_desc = df.dropna(subset=["sig_dirichlet"])
    good = have_desc[have_desc.test_accuracy_mean > GOOD_ACC]
    out.append(f"  «осмысленные» = accuracy > {GOOD_ACC}: {len(good)} из {len(have_desc)} графов")
    out.append("")
    out.append(f"  {'дескриптор':<28} {'все':>16} {'осмысленные':>16}")
    out.append("  " + "-" * 60)
    for col, name, kind in [
        ("sig_dirichlet", "энергия Дирихле", "данные"),
        ("sig_assortativity", "ассортативность сигнала", "данные"),
        ("stat_density", "плотность", "структура"),
        ("stat_clustering", "кластеризация", "структура"),
    ]:
        cells = []
        for sub in (df, good):
            s2 = sub[["test_accuracy_mean", col]].dropna()
            if len(s2) < 3:
                cells.append("—".rjust(16))
                continue
            rho, p = stats.spearmanr(s2[col], s2.test_accuracy_mean)
            cells.append(f"{rho:+.3f} (p={p:.2f})".rjust(16))
        out.append(f"  {name:<28}{cells[0]}{cells[1]}")

    sub_all = df[["test_accuracy_mean", "sig_dirichlet"]].dropna()
    sub_good = good[["test_accuracy_mean", "sig_dirichlet"]].dropna()
    if len(sub_good) >= 3 and len(sub_all) >= 3:
        r_all = stats.spearmanr(sub_all.sig_dirichlet, sub_all.test_accuracy_mean)[0]
        r_good = stats.spearmanr(sub_good.sig_dirichlet, sub_good.test_accuracy_mean)[0]
        out.append("")
        if r_all * r_good <= 0:
            out.append("  [!] ЗНАК ПЕРЕВОРАЧИВАЕТСЯ на осмысленном подмножестве.")
            out.append("      Значит дескриптор отличает граф от мусора, но НЕ ранжирует")
            out.append("      нормальные графы между собой. Как критерий выбора A он")
            out.append("      непригоден - вся корреляция по всему набору держится на")
            out.append("      вырожденных кандидатах.")
            out.append("      -> H2 в форме «предиктор качества» НЕ подтверждена.")
        else:
            out.append("  Знак сохраняется на обоих подмножествах - дескриптор ранжирует")
            out.append("  и осмысленные графы тоже. -> H2 подтверждена.")
    return out


def test_h3(df: pd.DataFrame) -> list[str]:
    """H3: зависимость от плотности - перевёрнутая U."""
    out = ["\n" + "=" * 72, "H3: перевёрнутая U по плотности", "=" * 72]
    sub = df[["graph", "stat_density", "test_accuracy_mean",
              "node_cosine_mean"]].dropna().sort_values("stat_density")
    if len(sub) < 5:
        return out + ["  мало точек"]
    best = sub.loc[sub.test_accuracy_mean.idxmax()]
    out.append(f"  максимум качества при плотности {best.stat_density:.5f} "
               f"({best.graph})")
    left = sub[sub.stat_density < best.stat_density]
    right = sub[sub.stat_density > best.stat_density]
    for side, part in (("слева от максимума", left), ("справа", right)):
        if len(part) >= 3:
            rho, p = stats.spearmanr(part.stat_density, part.test_accuracy_mean)
            out.append(f"  {side:<20} rho = {rho:+.3f}  p = {p:.3f}  "
                       f"(n = {len(part)})")
    if len(right) >= 3:
        rho, p = stats.spearmanr(right.stat_density, right.node_cosine_mean)
        out.append(f"  плотность vs косинус справа: rho = {rho:+.3f} p = {p:.3f}")
        out.append("  -> если положительная, правая ветвь падает из-за "
                   "oversmoothing,")
        out.append("     а не из-за неинформативности структуры")
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

    # 1-2. качество против сигнального и структурного дескриптора
    for col, xlabel, fname in [
        ("sig_dirichlet", "энергия Дирихле (гладкость сигнала на графе)", "h2_dirichlet"),
        ("sig_assortativity", "ассортативность сигнала по рёбрам", "h2_assortativity"),
        ("stat_density", "плотность графа", "h3_density"),
    ]:
        sub = df[[col, "test_accuracy_mean", "test_accuracy_std", "graph",
                  "group"]].dropna(subset=[col, "test_accuracy_mean"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for grp in GROUP_ORDER:
            s = sub[sub.group == grp]
            if s.empty:
                continue
            ax.errorbar(s[col], s.test_accuracy_mean,
                        yerr=s.test_accuracy_std.fillna(0), fmt="o",
                        color=colors[grp], label=GROUP_NAME[grp], capsize=3)
        for _, r in sub.iterrows():
            ax.annotate(r.graph, (r[col], r.test_accuracy_mean),
                        fontsize=6.5, alpha=0.75,
                        xytext=(4, 3), textcoords="offset points")
        rho, p = stats.spearmanr(sub[col], sub.test_accuracy_mean)
        if col == "stat_density":
            # symlog, а не log: у empty плотность ровно 0, и на логарифмической
            # оси эта точка молча исчезла бы - а это пол шкалы, ключевая
            # опора для H3.
            ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy на тесте")
        ax.set_title(f"{xlabel}\nСпирмен rho = {rho:+.3f}, p = {p:.2e}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{fname}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # 3. кривая деградации по доле перепроводки
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
        *test_h1(res, df), *test_h1_strong(res), *test_h2(df), *test_h3(df),
        *test_controls(res), *check_budget(res),
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
