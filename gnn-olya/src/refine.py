# проверяет, помогает ли заменить исходный граф на граф, построенный по данным
# сравнивает хорошие методы с вариантами «ничего не менять» и «менять случайно»

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, results_dir  # noqa: E402
from model import DenseGCN  # noqa: E402

_RULES: dict[str, callable] = {}
_FREE_DENSITY: set[str] = set()


def rule(name: str, free_density: bool = False):
    """Регистрирует правило обновления связей."""
    def deco(fn):
        _RULES[name] = fn
        if free_density:
            _FREE_DENSITY.add(name)
        return fn
    return deco


def available_rules() -> list[str]:
    return sorted(_RULES)



@dataclass
class Context:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    d: int
    n_classes: int
    device: torch.device
    A_true: np.ndarray | None = None   # известен только на синтетике
    rng: np.random.Generator = field(default_factory=lambda:
                                     np.random.default_rng(0))
    # Сид прогона. Нужен правилам из structure.py: без него их внутренние
    # разбиения на половины шли бы с одним и тем же сидом при любом прогоне,
    # и три «независимых» сида делили бы одну случайность.
    seed: int = 0


@dataclass
class Stats:
    val_acc: float
    test_acc: float
    n_edges: int


def _normalize_torch(A: torch.Tensor) -> torch.Tensor:
    """D^(-1/2)(A+I)D^(-1/2), дифференцируемо по A.

    Градиент течёт и через степени: добавление ребра меняет и нормировку.
    """
    A_loop = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    dinv = A_loop.sum(1).clamp_min(1e-8).pow(-0.5)
    return dinv[:, None] * A_loop * dinv[None, :]


def train_round(A: np.ndarray, ctx: Context, epochs: int, hidden: int,
                layers: int, lr: float, batch: int, seed: int) -> Stats:
    """Один раунд: обучить сеть на текущем графе и снять статистики."""
    torch.manual_seed(seed)
    At = torch.tensor(A, dtype=torch.float32, device=ctx.device)
    A_hat = _normalize_torch(At).detach()

    net = DenseGCN(d=ctx.d, n_layers=layers, hidden=hidden,
                   n_classes=ctx.n_classes).to(ctx.device)
    opt = torch.optim.Adam(net.parameters(), lr)
    lf = torch.nn.CrossEntropyLoss()
    n = len(ctx.y_train)
    for _ in range(epochs):
        perm = torch.randperm(n, device=ctx.device)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            lf(net(ctx.x_train[b], A_hat), ctx.y_train[b]).backward()
            opt.step()

    @torch.no_grad()
    def acc(x, y):
        return float((net(x, A_hat).argmax(1) == y).float().mean())

    return Stats(val_acc=acc(ctx.x_val, ctx.y_val),
                 test_acc=acc(ctx.x_test, ctx.y_test),
                 n_edges=int(A.sum() // 2))


# ---------------------------------------------------------------------------
#  Правила обновления
# ---------------------------------------------------------------------------

def _apply(A: np.ndarray, iu, drop, add) -> np.ndarray:
    """Убирает рёбра `drop`, добавляет `add`, сохраняя симметрию.

    Обе половины правятся явно: `np.maximum(A, A.T)` в конце восстановил бы
    удалённое ребро из нижнего треугольника, и удаления бы терялись.
    """
    A = A.copy()
    di, dj = iu[0][drop], iu[1][drop]
    ai, aj = iu[0][add], iu[1][add]
    A[di, dj] = 0
    A[dj, di] = 0
    A[ai, aj] = 1
    A[aj, ai] = 1
    return A


@rule("keep")
def _keep(A: np.ndarray, st: Stats, ctx: Context, **kw) -> np.ndarray:
    """КОНТРОЛЬ: не менять граф. Показывает, сколько даёт дообучение само."""
    return A


@rule("random_rewire")
def _random_rewire(A: np.ndarray, st: Stats, ctx: Context,
                   budget: int = 200, **kw) -> np.ndarray:
    """КОНТРОЛЬ: переставить `budget` рёбер случайно. Пол шкалы.

    Единственное правило, невоспроизводимое после обрыва: берёт ctx.rng, а тот
    при возобновлении стартует заново. Для контроля это неважно.
    """
    A = A.copy()
    iu = np.triu_indices(ctx.d, 1)
    present = np.flatnonzero(A[iu])
    absent = np.flatnonzero(A[iu] == 0)
    k = min(budget, len(present), len(absent))
    if k == 0:
        return A
    drop = ctx.rng.choice(present, k, replace=False)
    add = ctx.rng.choice(absent, k, replace=False)
    return _apply(A, iu, drop, add)


def _corr_matrix(ctx: Context) -> np.ndarray:
    """|corr| между признаками по train. Кешируется: от раунда не зависит."""
    cached = getattr(ctx, "_corr", None)
    if cached is not None:
        return cached
    x = ctx.x_train.cpu().numpy()
    alive = x.std(0) > 0
    S = np.zeros((ctx.d, ctx.d))
    if alive.sum() > 1:
        C = np.nan_to_num(np.corrcoef(x[:, alive], rowvar=False), nan=0.0)
        S[np.ix_(alive, alive)] = np.abs(C)
    np.fill_diagonal(S, 0.0)
    ctx._corr = S
    return S


@rule("corr_static")
def _corr_static(A: np.ndarray, st: Stats, ctx: Context, **kw) -> np.ndarray:
    """ОРИЕНТИР БЕЗ ПЕТЛИ: сразу взять |E| пар с наибольшей |корреляцией|.

    Это сильный одношаговый ориентир: одна матрица корреляций, без сети и
    дополнительных раундов. Правило, не обогнавшее его, не оправдывает итеративность.
    Побочная польза: граф после первого раунда не меняется, поэтому разброс
    точности по раундам даёт оценку шума обучения на ХОРОШЕМ графе.
    """
    # То же, что graphs.top_m_edges, но без отсева нулевой корреляции: здесь
    # число рёбер обязано совпасть с исходным, иначе сравнение поедет по
    # плотности и петля упадёт на проверке.
    S = _corr_matrix(ctx)
    iu = np.triu_indices(ctx.d, 1)
    top = np.argsort(-S[iu])[:int(A.sum() // 2)]
    out = np.zeros_like(A)
    out[iu[0][top], iu[1][top]] = 1
    return np.maximum(out, out.T)


def _fitted(ctx: Context, score: str, n_edges: int | None) -> np.ndarray:
    """Конвейер из structure.py, посчитанный один раз на прогон.

    Кеш обязателен: от раунда результат не зависит (сеть в него не входит),
    а перебор двух сеток с подвыборками стоит куда дороже одного раунда.
    """
    import structure as ST
    key = f"_fit_{score}_{n_edges}"
    cached = getattr(ctx, key, None)
    if cached is None:
        x = ctx.x_train.cpu().numpy().astype(np.float64)
        cached = ST.fit(x, score=score, n_edges=n_edges, seed=ctx.seed).A
        setattr(ctx, key, cached)
    return cached.copy()      # копия: правило не должно отдавать общий буфер


@rule("partial_static")
def _partial_static(A: np.ndarray, st: Stats, ctx: Context, **kw) -> np.ndarray:
    """Частная корреляция, регуляризация по устойчивости. Плотность задана.

    Прямое сравнение с `corr_static` при равной плотности: обе величины
    считаются по train, ни одна не смотрит ни на модель, ни на A_true.
    """
    return _fitted(ctx, "partial", int(A.sum() // 2))


@rule("partial_free", free_density=True)
def _partial_free(A: np.ndarray, st: Stats, ctx: Context, **kw) -> np.ndarray:
    """ЧЕСТНЫЙ РЕЖИМ: плотность тоже ищется, а не сообщается.

    Единственное правило, которому не подарили число рёбер. Остальным, включая
    `corr_static`, оно известно из стартового графа. Если это правило без форы
    обгоняет их с форой - петля наконец делает то, чего одношаговый метод не
    может в принципе.
    """
    return _fitted(ctx, "auto", None)


# ---------------------------------------------------------------------------
#  Петля
# ---------------------------------------------------------------------------

def run_key(rule_name: str, graph: str, start: str, seed: int,
            side: int, n: int, classes: int, m: int) -> str:
    """Ключ прогона: по нему хранится состояние и пропускается посчитанное.

    Включает все параметры задачи, иначе прогон на другом размере подхватится
    как продолжение. Число раундов не включено: его увеличение доделывает хвост.
    """
    return (f"{rule_name}__{graph}__{start}__s{seed}"
            f"__d{side * side}_n{n}_c{classes}_m{m}")


def refine(A0: np.ndarray, rule_name: str, ctx: Context, rounds: int = 5,
           epochs: int = 20, budget: int = 200, hidden: int = 64,
           layers: int = 3, lr: float = 1e-3, batch: int = 256,
           seed: int = 0, verbose: bool = True,
           state_dir: Path | None = None, key: str = ""):
    """Главная петля. Возвращает (итоговый граф, история по раундам).

    При заданном `state_dir` состояние пишется после каждого раунда, и
    повторный запуск продолжает с него: обрыв стоит одного раунда.
    """
    from synth import recovery

    fn = _RULES[rule_name]
    A = A0.copy()
    hist = []
    start_round = 0

    state = (state_dir / f"{key}.npz") if state_dir else None
    if state is not None and state.exists():
        z = np.load(state, allow_pickle=True)
        A = z["A"]
        hist = list(z["hist"])
        start_round = int(z["round"]) + 1
        if verbose and start_round < rounds:
            print(f"  продолжаю с раунда {start_round} "
                  f"(готово {start_round} из {rounds})", flush=True)
        if start_round >= rounds:
            if verbose:
                print("  уже посчитано целиком", flush=True)
            return A, hist

    A_trained = A
    for r in range(start_round, rounds):
        t = time.time()
        st = train_round(A, ctx, epochs, hidden, layers, lr, batch, seed + r)
        row = dict(round=r, rule=rule_name, n_edges=st.n_edges,
                   val_acc=st.val_acc, test_acc=st.test_acc,
                   sec=round(time.time() - t, 1))
        if ctx.A_true is not None:
            row.update({f"rec_{k}": v for k, v in
                        recovery(A, ctx.A_true).items()
                        if k in ("precision", "recall", "f1")})
        hist.append(row)
        if verbose:
            extra = (f"  f1={row['rec_f1']:.3f}" if "rec_f1" in row else "")
            print(f"  раунд {r}: рёбер {st.n_edges:>5}  "
                  f"val {st.val_acc:.4f}  test {st.test_acc:.4f}{extra}"
                  f"   [{row['sec']:.0f} c]", flush=True)
        # Правило применяется и после последнего раунда: сохранённый граф
        # обязан быть готовым к раунду r+1. Иначе увеличение --rounds на уже
        # посчитанном прогоне доучивало бы на старом графе, теряя одну правку.
        A_trained = A
        before = int(A.sum() // 2)
        A = fn(A, st, ctx, budget=budget)
        assert A.shape == (ctx.d, ctx.d), "правило сломало форму матрицы"
        assert np.array_equal(A, A.T), "правило вернуло несимметричный граф"
        assert not np.any(np.diag(A)), "правило добавило петли"
        # Число рёбер обязано сохраняться, иначе сравнение поедет по плотности.
        # Симметричность такую ошибку не ловит.
        after = int(A.sum() // 2)
        if after != before and rule_name not in _FREE_DENSITY:
            raise AssertionError(
                f"правило {rule_name} изменило число рёбер {before} -> {after}; "
                f"сравнение с контролями станет нечестным по плотности. "
                f"Все правила обязаны сохранять число рёбер.")

        if state is not None:
            # Пишем ПОСЛЕ правила: при записи до него правка раунда r терялась
            # при возобновлении, и следующий раунд считался на старом графе.
            np.savez(state, A=A, hist=np.array(hist, dtype=object), round=r)
    return A_trained, hist


def synth_context(side: int, n: int, n_classes: int, m: int, seed: int,
                  device: torch.device, graph: str = "grid") -> Context:
    from synth import make_clusters
    t = make_clusters(side=side, n=n, n_classes=n_classes, m=m, seed=seed,
                      graph=graph)
    tt = lambda a, dt: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
    return Context(
        x_train=tt(t.x_train, torch.float32), y_train=tt(t.y_train, torch.long),
        x_val=tt(t.x_val, torch.float32), y_val=tt(t.y_val, torch.long),
        x_test=tt(t.x_test, torch.float32), y_test=tt(t.y_test, torch.long),
        d=t.d, n_classes=t.n_classes, device=device, A_true=t.A_true,
        rng=np.random.default_rng(seed), seed=seed)


def summarize() -> int:
    """Сводит results/refine.csv в таблицу «правило x граф».

    Две величины обязательны обе: точность одна обманывает из-за вырождения.
    """
    import pandas as pd
    f = results_dir(load_config()) / "refine.csv"
    if not f.exists():
        print("нет results/refine.csv - сначала ./run_refine.sh")
        return 1
    df = pd.read_csv(f)

    # Среднее по хвосту, а не последний раунд: сеть переобучается заново каждый
    # раунд, поэтому последний раунд добавляет случайный шум обучения.
    # Хвост берётся у КАЖДОГО прогона свой. Общий порог по номеру раунда
    # выбрасывал целиком правила, досчитанные с меньшим --rounds: их строки
    # просто не доходили до порога, и в таблице они молча исчезали.
    key = ["rule", "graph", "seed"]
    df = df.sort_values(key + ["round"])
    n_run = df.groupby(key)["round"].transform("count")
    from_end = df.groupby(key)["round"].transform("max") - df["round"]
    tail = np.minimum(5, np.maximum(1, n_run // 4))
    last = df[from_end < tail].groupby(key, as_index=False).mean(
        numeric_only=True)

    lens = df.groupby(key)["round"].count()
    if lens.nunique() > 1:
        print(f"\n[!] РАЗНАЯ ДЛИНА ПРОГОНОВ: от {lens.min()} до {lens.max()} "
              f"раундов. Оценка шума сравнивает неравное, пересчитай короткие "
              f"с тем же ROUNDS.")
    print(f"\nусреднено по последним {int(tail.min())}-{int(tail.max())} "
          f"раундам каждого прогона")

    for col, title in (("test_acc", "ТОЧНОСТЬ"),
                       ("rec_f1", "ВОССТАНОВЛЕНИЕ структуры (f1)")):
        if col not in last:
            continue
        t = last.pivot_table(index="rule", columns="graph", values=col,
                             aggfunc="mean")
        order = [r for r in ("keep", "random_rewire", "corr_static",
                             "partial_static", "partial_free") if r in t.index]
        print(f"\n{title}\n" + "-" * 60)
        print(t.loc[order].round(4).to_string())

    # Правило, само выбирающее плотность, сравнивать по точности с остальными
    # напрямую нельзя: у них разное число рёбер. Без этой строки разницу
    # припишут правилу, а она может быть эффектом плотности.
    free = last[last.rule.isin(_FREE_DENSITY)]
    if len(free):
        print("\nПЛОТНОСТЬ НЕ СОВПАДАЕТ - эти правила искали число рёбер сами:")
        ref = last[last.rule == "corr_static"].groupby("graph")["n_edges"].mean()
        for (r, g), v in free.groupby(["rule", "graph"])["n_edges"].mean().items():
            base = ref.get(g, float("nan"))
            print(f"  {r} / {g}: {v:.0f} рёбер против {base:.0f} у остальных")

    # У `keep` и `corr_static` граф между раундами не меняется, поэтому разброс
    # их точности по раундам - чистый шум переобучения. Без этой строки разницу
    # в третьем знаке принимают за результат.
    flat = df[df.rule.isin(["keep", "corr_static"]) & (df["round"] >= 1)]
    if len(flat):
        sp = flat.groupby(["rule", "graph", "seed"])["test_acc"].agg(
            lambda v: v.max() - v.min())
        print(f"\nШУМ обучения (граф не менялся): худший разброс по раундам "
              f"{sp.max():.3f}, средний {sp.mean():.3f}")
        print("Разница в точности меньше худшего разброса ничего не значит.")

    print("\nЧитать так: сначала смотрим на восстановление, потом на точность.")
    print("Правило обязано обгонять И `keep`, И `random_rewire`, иначе оно")
    print("ничего не делает. И обязано обгонять `corr_static` - граф по")
    print("корреляциям, построенный сразу и без петли; иначе итеративность")
    print("не даёт преимущества перед одношаговым ориентиром.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="keep", choices=available_rules())
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20,
                    help="эпох на раунд (не на весь прогон)")
    ap.add_argument("--budget", type=int, default=200,
                    help="сколько рёбер правило меняет за раунд")
    ap.add_argument("--graph", default="grid",
                    choices=["grid", "community", "geometric", "ring"],
                    help="тип ИСТИННОГО графа в синтетике")
    # `empty` убран намеренно: все правила меняют рёбра ПАРАМИ (одно убрать,
    # одно добавить), чтобы плотность не поехала. На пустом графе убирать
    # нечего, k=0, и любое правило молча возвращает пустой граф навсегда.
    ap.add_argument("--start", default="random",
                    choices=["random", "knn_corr_k8"],
                    help="с какого графа начинать")
    # Пока только синтетика: на MNIST истинного графа нет, мерить нечем.
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", action="store_true",
                    help="не считать, а свести уже посчитанное в таблицу")
    args = ap.parse_args()

    if args.summary:
        return summarize()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = synth_context(args.side, args.n, args.classes, args.m, args.seed,
                        dev, graph=args.graph)

    d = ctx.d
    if args.start == "random":
        A0 = np.zeros((d, d), dtype=np.uint8)
        iu = np.triu_indices(d, 1)
        pick = ctx.rng.choice(len(iu[0]), int(ctx.A_true.sum() // 2),
                              replace=False)
        A0[iu[0][pick], iu[1][pick]] = 1
        A0 = np.maximum(A0, A0.T)
    else:
        import graphs as G
        from types import SimpleNamespace
        x_np = ctx.x_train.cpu().numpy()
        adapter = SimpleNamespace(x_train=x_np, y_train=ctx.y_train.cpu().numpy())
        cfg = SimpleNamespace(data=SimpleNamespace(d=d, side=args.side),
                              graphs=SimpleNamespace(graph_seed=1234))
        A0 = G.build_graph("knn_corr_k8", cfg, adapter).A

    print(f"устройство: {dev}   d = {d}   старт: {args.start} "
          f"({int(A0.sum() // 2)} рёбер)")
    print(f"правило: {args.rule}   раундов: {args.rounds}   "
          f"эпох на раунд: {args.epochs}   бюджет правки: {args.budget}\n")

    res = results_dir(load_config())
    state_dir = res / ".refine"
    state_dir.mkdir(parents=True, exist_ok=True)
    key = run_key(args.rule, args.graph, args.start, args.seed,
                  args.side, args.n, args.classes, args.m)

    _, hist = refine(A0, args.rule, ctx, rounds=args.rounds,
                     epochs=args.epochs, budget=args.budget, seed=args.seed,
                     state_dir=state_dir, key=key)

    import pandas as pd
    df = pd.DataFrame(list(hist))
    for k, v in (("graph", args.graph), ("start", args.start),
                 ("seed", args.seed)):
        df[k] = v

    # Дозапись; строки того же ключа заменяются, а не дублируются.
    out = res / "refine.csv"
    if out.exists():
        old = pd.read_csv(out)
        mask = ~((old.rule == args.rule) & (old.graph == args.graph)
                 & (old.start == args.start) & (old.seed == args.seed))
        df = pd.concat([old[mask], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nсохранено: {out}")
    print("\nСравнивать правило нужно с контролями `keep` и `random_rewire` "
          "на том же старте.")
    return 0


if __name__ == "__main__":
    sys.exit(main())