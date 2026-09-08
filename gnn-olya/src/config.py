from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "fixed.yaml"


class Config(dict):
    """dict с доступом через точку: cfg.model.hidden."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    @staticmethod
    def _wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Config({k: Config._wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config._wrap(v) for v in obj]
        return obj


def load_config(path: str | Path = DEFAULT_CONFIG, dev: bool = False,
                pool: int | None = None) -> Config:
    """Читает fixed.yaml. При dev=True накатывает секцию data.dev поверх.

    Хеш (cfg.hash) считается по ИСХОДНОМУ файлу, до применения dev-оверрайдов,
    поэтому он одинаков в dev и в боевом режиме. Режим пишется отдельным полем
    cfg.dev, и в results.csv тоже - смешивать dev- и боевые строки нельзя.
    """
    pool_arg = pool
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    cfg_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12]

    raw = yaml.safe_load(raw_text)
    dev_overrides = raw["data"].pop("dev")
    # Боевое n_val нужно data.py и в режиме --dev: разбиение делается сначала
    # по-боевому, и только потом из него берётся подвыборка, чтобы dev-выборка
    # была честным подмножеством боевой.
    raw["data"]["full_n_val"] = raw["data"]["n_val"]

    if dev:
        pool = dev_overrides["pool"]
        raw["data"]["side"] = raw["data"]["side"] // pool
        raw["data"]["d"] = raw["data"]["side"] ** 2
        raw["data"]["pool"] = pool
        raw["data"]["n_train"] = dev_overrides["n_train"]
        raw["data"]["n_val"] = dev_overrides["n_val"]
        raw["data"]["n_test"] = dev_overrides["n_test"]
        raw["train"]["epochs"] = dev_overrides["epochs"]
    else:
        raw["data"]["pool"] = 1

    # pool отдельно от dev: пулинг БЕЗ урезания выборки. Нужен для H8, где
    # дорога только размерность (GAT квадратичен по d), а полная выборка и
    # полный бюджет эпох нужны - иначе меряется недообучение, а не внимание.
    # d пишется в results.csv, так что режимы в таблице различимы.
    if pool_arg is not None and pool_arg > 1:
        if dev:
            raise ValueError("--pool и --dev вместе не имеют смысла: "
                             "dev уже включает свой пулинг и урезает выборку")
        base = yaml.safe_load(raw_text)["data"]["side"]
        if base % pool_arg:
            raise ValueError(f"side={base} не делится на pool={pool_arg}")
        raw["data"]["side"] = base // pool_arg
        raw["data"]["d"] = raw["data"]["side"] ** 2
        raw["data"]["pool"] = pool_arg

    cfg = Config._wrap(raw)
    cfg["hash"] = cfg_hash
    cfg["dev"] = dev
    cfg["root"] = ROOT
    return cfg


def artifacts_dir(cfg: Config) -> Path:
    p = ROOT / cfg.paths.artifacts
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(cfg: Config) -> Path:
    p = ROOT / cfg.paths.results
    p.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    for is_dev in (False, True):
        c = load_config(dev=is_dev)
        tag = "dev " if is_dev else "full"
        print(
            f"[{tag}] hash={c.hash}  side={c.data.side}  d={c.data.d}  "
            f"pool={c.data.pool}  n_train={c.data.n_train}  "
            f"n_val={c.data.n_val}  n_test={c.data.n_test}  "
            f"epochs={c.train.epochs}"
        )
