"""Bootstrap CI and paired tests for reporting."""

from __future__ import annotations

import numpy as np


def paired_t_statistic(before: list[float], after: list[float]) -> tuple[float, int]:
    """Simple paired t-statistic (requires scipy for p-value). Returns (t, df)."""
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    d = a - b
    n = len(d)
    if n < 2:
        return 0.0, 0
    mean_d = d.mean()
    std_d = d.std(ddof=1)
    if std_d == 0:
        return 0.0, n - 1
    t = mean_d / (std_d / np.sqrt(n))
    return float(t), n - 1


def bootstrap_mean_ci(values: list[float], n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return float(arr.mean()), float(lo), float(hi)

