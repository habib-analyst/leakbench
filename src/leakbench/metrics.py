"""Numeric eval metrics. Direct sklearn numbers from predict_proba.

Readout note (POPE carry-over lesson, arXiv 2610.25964): there is no text
readout anywhere in this build. Every metric is computed numerically from
model probabilities on held-out splits; no token/logit parsing choice can
move a number.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(y_true, proba):
    return float(roc_auc_score(y_true, proba))


def auprc(y_true, proba):
    return float(average_precision_score(y_true, proba))


def ece(y_true, proba, n_bins=15):
    """Expected calibration error with equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err, total = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (proba > lo) & (proba <= hi) if lo > 0 else (proba <= hi)
        if m.sum() == 0:
            continue
        err += (m.sum() / total) * abs(y_true[m].mean() - proba[m].mean())
    return float(err)


def net_benefit(y_true, proba, threshold=0.3):
    """Decision-curve net benefit at a risk threshold."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n = len(y_true)
    tp = int(((proba >= threshold) & (y_true == 1)).sum())
    fp = int(((proba >= threshold) & (y_true == 0)).sum())
    w = threshold / (1.0 - threshold)
    return float((tp - w * fp) / n)


def paired_bootstrap_diff(a, b, n_boot=2000, seed=12345, ci=0.95):
    """Paired bootstrap CI for mean(a) - mean(b). Returns (mean_diff, lo, hi)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b) and len(a) > 0
    rng = np.random.default_rng(seed)
    diffs = a - b
    boots = np.array([
        diffs[rng.integers(0, len(diffs), len(diffs))].mean()
        for _ in range(n_boot)
    ])
    alpha = 1.0 - ci
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(diffs.mean()), float(lo), float(hi)
