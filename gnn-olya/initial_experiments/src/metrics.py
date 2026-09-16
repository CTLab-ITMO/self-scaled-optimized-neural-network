# считает метрики

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           n_classes: int = 10, prefix: str = "") -> dict:
    labels = list(range(n_classes))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        f"{prefix}accuracy": float((y_true == y_pred).mean()),
        f"{prefix}macro_f1": float(f1),
        f"{prefix}macro_precision": float(prec),
        f"{prefix}macro_recall": float(rec),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def mean_pairwise_cosine(emb: np.ndarray) -> float:
    """Средняя попарная косинусная близость эмбеддингов — мера oversmoothing.

    1.0 = все вершины схлопнулись. Считается замкнуто через норму суммы
    нормированных векторов: O(d*F) вместо O(d^2*F).
    """
    b, d, _ = emb.shape
    if d < 2:
        return float("nan")
    norms = np.linalg.norm(emb, axis=2, keepdims=True)
    unit = emb / np.maximum(norms, 1e-12)
    total = (np.linalg.norm(unit.sum(axis=1), axis=1) ** 2) - d
    return float((total / (d * (d - 1))).mean())
