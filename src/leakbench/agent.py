"""Baseline clinical-modeling agent: deterministic greedy feature selector.

Honest CPU analog of the paper's "deterministic optimizer" finding.
Reward hacking is used descriptively (as in the paper): the agent selects a
prediction-time-invalid input because it improves its validation reward.
No LLM involved; selection is greedy forward selection on validation AUROC
(or AUPRC) of a logistic-regression model. Fully deterministic given the
cohort seed.

Modes:
  unconstrained: every column is eligible (the disease carrier).
  name_filter:   excludes features whose names match leak-ish tokens
                 (the naive baseline the paper shows fails under disguise).
  constrained:   only provenance-valid features (available_at_hours <= 0)
                 are eligible (the explicit admission-time constraint).
"""

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import metrics as M

LEAK_NAME_TOKENS = re.compile(r"leak|posthoc|culture|future|discharge|outcome", re.I)


@dataclass
class AgentResult:
    selected: list
    val_score: float
    test_auroc: float
    test_auprc: float
    mode: str
    metric: str
    leak_selected: bool = False
    log: list = field(default_factory=list)


def _fit_predict(X_train, y_train, X_eval, cols):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train[:, cols])
    Xe = scaler.transform(X_eval[:, cols])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xt, y_train)
    return clf.predict_proba(Xe)[:, 1]


class GreedySelector:
    def __init__(self, mode="unconstrained", metric="auroc", min_gain=1e-4):
        assert mode in ("unconstrained", "name_filter", "constrained")
        assert metric in ("auroc", "auprc")
        self.mode = mode
        self.metric = metric
        self.min_gain = min_gain

    def _eligible(self, feature_names, manifest):
        if self.mode == "constrained":
            return [f for f in feature_names
                    if manifest[f].valid_at_prediction_time]
        if self.mode == "name_filter":
            return [f for f in feature_names
                    if not LEAK_NAME_TOKENS.search(f)]
        return list(feature_names)

    def _score(self, proba, y):
        return M.auroc(y, proba) if self.metric == "auroc" else M.auprc(y, proba)

    def run(self, cohort, manifest):
        names = cohort["feature_names"]
        idx = {f: i for i, f in enumerate(names)}
        eligible = self._eligible(names, manifest)
        selected, log = [], []
        best = 0.5  # chance-level floor for the empty set
        improved = True
        while improved and len(selected) < len(eligible):
            improved = False
            best_cand, best_cand_score = None, best
            for cand in eligible:
                if cand in selected:
                    continue
                cols = [idx[f] for f in selected + [cand]]
                p_val = _fit_predict(cohort["X_train"], cohort["y_train"],
                                     cohort["X_val"], cols)
                s = self._score(p_val, cohort["y_val"])
                if s > best_cand_score + self.min_gain:
                    best_cand, best_cand_score = cand, s
            if best_cand is not None:
                selected.append(best_cand)
                best = best_cand_score
                improved = True
                log.append((best_cand, round(best, 4)))

        cols = [idx[f] for f in selected] or list(range(len(names)))
        p_test = _fit_predict(cohort["X_train"], cohort["y_train"],
                              cohort["X_test"], cols)
        leak_name = cohort["leak_name"]
        return AgentResult(
            selected=selected,
            val_score=best,
            test_auroc=M.auroc(cohort["y_test"], p_test),
            test_auprc=M.auprc(cohort["y_test"], p_test),
            mode=self.mode,
            metric=self.metric,
            leak_selected=leak_name in selected,
            log=log,
        )
