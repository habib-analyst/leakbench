"""Synthetic sepsis cohort with KNOWN prediction-time-leakage failure structure.

Task: predict sepsis onset (binary y) at prediction time T+2h after admission.
- 7 legit features: observable at or before prediction time.
- 1 fatal leaking feature: the blood-culture result returned at T+18h
  (16h AFTER prediction time), merged into the row by a sloppy ETL.
  By construction it equals y with 2% label noise, so any
  reward-maximizing selector will take it if allowed.
"""

import numpy as np

LEGIT_FEATURES = [
    "age",
    "heart_rate_admit",
    "map_admit",
    "lactate_admit",
    "wbc_admit",
    "sofa_admit",
    "comorbidity_count",
]

LEAK_FEATURE = "posthoc_culture_flag"
DISGUISED_LEAK_FEATURE = "derived_stability_index"

N_TRAIN, N_VAL, N_TEST = 2500, 750, 750

# True logit weights for the legit signal (fixed across replicates).
_W = np.array([0.35, 0.55, -0.60, 0.85, 0.50, 0.70, 0.30])
_INTERCEPT = -0.4


def _draw_block(rng, n):
    age = rng.normal(62, 16, n).clip(18, 95)
    hr = rng.normal(96, 18, n).clip(40, 180)
    map_ = rng.normal(72, 14, n).clip(30, 130)
    lactate = rng.exponential(1.4, n).clip(0.2, 12.0)
    wbc = rng.normal(11.5, 4.5, n).clip(1.0, 40.0)
    sofa = rng.poisson(4.5, n).clip(0, 20).astype(float)
    comorbid = rng.poisson(2.0, n).clip(0, 8).astype(float)
    return np.column_stack([age, hr, map_, lactate, wbc, sofa, comorbid])


def _standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def generate_cohort(seed, disguised=False):
    """Return dict with train/val/test splits and the true leak feature name.

    The leak feature equals y with 2% label noise; it is named LEAK_FEATURE
    unless disguised=True, in which case it is renamed (same column values).
    """
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    X_legit = _draw_block(rng, n)
    Xs, mean, std = _standardize(X_legit)
    logits = Xs @ _W + _INTERCEPT + rng.normal(0, 0.9, n)
    y = (logits > 0).astype(int)

    leak = (y.copy().astype(float))
    flip = rng.random(n) < 0.02
    leak[flip] = 1.0 - leak[flip]

    leak_name = DISGUISED_LEAK_FEATURE if disguised else LEAK_FEATURE
    feature_names = LEGIT_FEATURES + [leak_name]
    X = np.column_stack([X_legit, leak])

    out = {
        "feature_names": feature_names,
        "leak_name": leak_name,
        "X_train": X[:N_TRAIN], "y_train": y[:N_TRAIN],
        "X_val": X[N_TRAIN:N_TRAIN + N_VAL], "y_val": y[N_TRAIN:N_TRAIN + N_VAL],
        "X_test": X[N_TRAIN + N_VAL:], "y_test": y[N_TRAIN + N_VAL:],
    }
    return out
