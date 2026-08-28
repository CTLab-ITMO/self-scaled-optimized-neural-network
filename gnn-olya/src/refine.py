"""Итеративное улучшение матрицы смежности (пункт 4 задания).

Петля: обучить GNN -> собрать статистики -> обновить связи -> повторить.
Правило обновления подключается по имени, см. реестр ниже.

В реестре вместе с правилами лежат контроли `keep` и `random_rewire`.
Правило, не обогнавшее оба, ничего не делает.

ОСТОРОЖНО при чтении чисел: на плохом графе сеть схлопывается в один класс
(замерено: acc ровно 0.2533 = доля класса при трёх разных сидах). Поэтому
`keep` даёт одинаковую точность во всех раундах, а скачок точности означает
лишь выход из вырождения, а не хороший граф. Подробности — docs/IDEAS.md, идея 34.

Запуск: python src/refine.py --rule corr_greedy --rounds 6
"""

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


def rule(name: str):
    """Регистрирует правило обновления связей."""
    def deco(fn):
        _RULES[name] = fn
        return fn
    return deco


def available_rules() -> list[str]:
    return sorted(_RULES)


# ---------------------------------------------------------------------------
#  Контекст и статистики
# ---------------------------------------------------------------------------

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


@dataclass
class Stats:
    val_acc: float
    test_acc: float
    edge_grad: np.ndarray      # [d, d], < 0 = ребро выгодно усилить
    node_emb: np.ndarray       # [d, hidden]
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

    # --- градиент потерь по весам рёбер: A снова непрерывна и требует grad
    At = torch.tensor(A, dtype=torch.float32, device=ctx.device,
                      requires_grad=True)
    sub = min(2048, n)
    loss = lf(net(ctx.x_train[:sub], _normalize_torch(At)), ctx.y_train[:sub])
    grad = torch.autograd.grad(loss, At)[0]
    grad = ((grad + grad.t()) / 2)             # граф неориентированный
    grad.fill_diagonal_(0.0)

    # Только train: статистики влияют на граф, с теста была бы утечка.
    with torch.no_grad():
        emb = net.node_embeddings(ctx.x_train[:512], A_hat).mean(0)

    return Stats(val_acc=acc(ctx.x_val, ctx.y_val),
                 test_acc=acc(ctx.x_test, ctx.y_test),
                 edge_grad=grad.detach().cpu().numpy(),
                 node_emb=emb.cpu().numpy(),
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
    """КОНТРОЛЬ: переставить `budget` рёбер случайно. Пол шкалы."""
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


@rule("corr_greedy")
def _corr_greedy(A: np.ndarray, st: Stats, ctx: Context,
                 budget: int = 200, **kw) -> np.ndarray:
    """БАЗОВОЕ ПРАВИЛО: менять рёбра по корреляции, не глядя на модель.

    Главный ориентир: `knn_corr_k4` даёт f1 = 0.918 сразу, без итеративности.
    Правило, не дотянувшее до него, делает петлю ненужной.
    """
    S = _corr_matrix(ctx)
    iu = np.triu_indices(ctx.d, 1)
    v = S[iu]
    present = A[iu] > 0
    cand_add = np.flatnonzero(~present)
    cand_del = np.flatnonzero(present)
    k = min(budget, len(cand_add), len(cand_del))
    if k == 0:
        return A
    add = cand_add[np.argsort(-v[cand_add])[:k]]     # самая сильная связь
    drop = cand_del[np.argsort(v[cand_del])[:k]]     # самая слабая
    return _apply(A, iu, drop, add)


@rule("grad_corr")
def _grad_corr(A: np.ndarray, st: Stats, ctx: Context,
               budget: int = 200, alpha: float = 0.5, **kw) -> np.ndarray:
    """Смесь градиента и корреляции, обе величины приводятся к рангам.

    Ранги обязательны: градиент ~1e-4, корреляция ~1, прямая сумма отдала бы
    всё одному сигналу. Смесь осмысленна, если обгонит обе составляющие.
    """
    iu = np.triu_indices(ctx.d, 1)
    g = -st.edge_grad[iu]                     # больше = полезнее добавить
    c = _corr_matrix(ctx)[iu]

    def to_rank(z):
        r = np.empty(len(z))
        r[np.argsort(z)] = np.arange(len(z))
        return r / max(len(z) - 1, 1)

    score = alpha * to_rank(g) + (1 - alpha) * to_rank(c)
    present = A[iu] > 0
    cand_add = np.flatnonzero(~present)
    cand_del = np.flatnonzero(present)
    k = min(budget, len(cand_add), len(cand_del))
    if k == 0:
        return A
    add = cand_add[np.argsort(-score[cand_add])[:k]]
    drop = cand_del[np.argsort(score[cand_del])[:k]]
    return _apply(A, iu, drop, add)


@rule("grad_greedy")
def _grad_greedy(A: np.ndarray, st: Stats, ctx: Context,
                 budget: int = 200, **kw) -> np.ndarray:
    """ЗАГЛУШКА: добавить рёбра с лучшим градиентом, убрать с худшим.

    Слабое место: градиент говорит о МАЛОМ изменении веса, а делается скачок
    0 -> 1, и снят он на сети, обученной под текущий граф. См. docs/IDEAS.md, 34.
    """
    A = A.copy()
    iu = np.triu_indices(ctx.d, 1)
    g = st.edge_grad[iu]
    present = A[iu] > 0

    cand_add = np.flatnonzero(~present)
    cand_del = np.flatnonzero(present)
    k = min(budget, len(cand_add), len(cand_del))
    if k == 0:
        return A
    add = cand_add[np.argsort(g[cand_add])[:k]]        # самый отрицательный
    drop = cand_del[np.argsort(-g[cand_del])[:k]]      # самый положительный
    return _apply(A, iu, drop, add)


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
        if state is not None:
            # После раунда, но до правила: обрыв продолжится с этого графа.
            np.savez(state, A=A, hist=np.array(hist, dtype=object), round=r)

        if r < rounds - 1:
            before = int(A.sum() // 2)
            A = fn(A, st, ctx, budget=budget)
            assert A.shape == (ctx.d, ctx.d), "правило сломало форму матрицы"
            assert np.array_equal(A, A.T), "правило вернуло несимметричный граф"
            assert not np.any(np.diag(A)), "правило добавило петли"
            # Число рёбер обязано сохраняться, иначе сравнение поедет по
            # плотности. Симметричность такую ошибку не ловит.
            after = int(A.sum() // 2)
            if after != before:
                raise AssertionError(
                    f"правило {rule_name} изменило число рёбер {before} -> "
                    f"{after}; сравнение с контролями станет нечестным по "
                    f"плотности. Все правила обязаны сохранять число рёбер.")
    return A, hist


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
        rng=np.random.default_rng(seed))


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
    last = df.sort_values("round").groupby(
        ["rule", "graph", "seed"], as_index=False).last()

    for col, title in (("test_acc", "ТОЧНОСТЬ в последнем раунде"),
                       ("rec_f1", "ВОССТАНОВЛЕНИЕ структуры (f1)")):
        if col not in last:
            continue
        t = last.pivot_table(index="rule", columns="graph", values=col,
                             aggfunc="mean")
        order = [r for r in ("keep", "random_rewire", "corr_greedy",
                             "grad_greedy", "grad_corr") if r in t.index]
        print(f"\n{title}\n" + "-" * 60)
        print(t.loc[order].round(4).to_string())

    print("\nЧитать так: сначала смотрим на восстановление, потом на точность.")
    print("Правило обязано обгонять И `keep`, И `random_rewire`, иначе оно")
    print("ничего не делает. И обязано обгонять `corr_greedy`, иначе не нужна")
    print("сама итеративность - хватило бы однократного графа по корреляциям.")
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
    ap.add_argument("--start", default="empty",
                    choices=["empty", "random", "knn_corr_k8"],
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
    if args.start == "empty":
        A0 = np.zeros((d, d), dtype=np.uint8)
    elif args.start == "random":
        A0 = np.zeros((d, d), dtype=np.uint8)
        iu = np.triu_indices(d, 1)
        pick = ctx.rng.choice(len(iu[0]), int(ctx.A_true.sum() // 2),
                              replace=False)
        A0[iu[0][pick], iu[1][pick]] = 1
        A0 = np.maximum(A0, A0.T)
    else:
        import graphs as G
        from eval_builders import _Adapter, _cfg_for
        cfg = _cfg_for(d, args.side)
        A0 = G.build_graph("knn_corr_k8", cfg,
                           _Adapter(ctx.x_train.cpu().numpy(),
                                    ctx.y_train.cpu().numpy())).A

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
