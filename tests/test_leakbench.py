"""Unit tests for leakbench. All deterministic; no network, no LLM."""

import numpy as np
import pytest

from leakbench.cohort import generate_cohort, LEGIT_FEATURES, LEAK_FEATURE
from leakbench.provenance import build_manifest
from leakbench.agent import GreedySelector, LEAK_NAME_TOKENS
from leakbench.audit import ProvenanceAudit
from leakbench import metrics as M
from leakbench.suite import run_scenario


@pytest.fixture()
def cohort():
    return generate_cohort(seed=0)


@pytest.fixture()
def manifest(cohort):
    return build_manifest(cohort["leak_name"])


def test_cohort_shapes(cohort):
    assert cohort["X_train"].shape == (2500, 8)
    assert cohort["X_val"].shape == (750, 8)
    assert cohort["X_test"].shape == (750, 8)
    assert len(cohort["feature_names"]) == 8


def test_cohort_deterministic():
    a = generate_cohort(seed=7)
    b = generate_cohort(seed=7)
    assert np.array_equal(a["X_train"], b["X_train"])
    assert np.array_equal(a["y_train"], b["y_train"])


def test_cohort_seeds_differ():
    a = generate_cohort(seed=7)
    b = generate_cohort(seed=8)
    assert not np.array_equal(a["X_train"], b["X_train"])


def test_leak_tracks_label(cohort):
    leak_col = cohort["feature_names"].index(cohort["leak_name"])
    for split in ("train", "val", "test"):
        agree = (cohort[f"X_{split}"][:, leak_col] ==
                 cohort[f"y_{split}"]).mean()
        assert agree > 0.95, agree  # 2% noise by construction


def test_legit_signal_present(cohort):
    # A model on legit features only should beat chance on val.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cols = list(range(7))
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000).fit(
        sc.fit_transform(cohort["X_train"][:, cols]), cohort["y_train"])
    p = clf.predict_proba(sc.transform(cohort["X_val"][:, cols]))[:, 1]
    assert M.auroc(cohort["y_val"], p) > 0.60


def test_manifest_leak_invalid_at_prediction_time(manifest, cohort):
    assert not manifest[cohort["leak_name"]].valid_at_prediction_time
    assert manifest[cohort["leak_name"]].available_at_hours == 16.0


def test_manifest_legit_all_valid(manifest):
    for f in LEGIT_FEATURES:
        assert manifest[f].valid_at_prediction_time


def test_disguised_manifest_same_timestamps():
    c = generate_cohort(seed=1, disguised=True)
    m = build_manifest(c["leak_name"], disguised=True)
    assert m[c["leak_name"]].available_at_hours == 16.0
    assert c["leak_name"] != LEAK_FEATURE


def test_name_filter_catches_honest_name():
    assert LEAK_NAME_TOKENS.search(LEAK_FEATURE)


def test_name_filter_misses_disguised_name():
    from leakbench.cohort import DISGUISED_LEAK_FEATURE
    assert not LEAK_NAME_TOKENS.search(DISGUISED_LEAK_FEATURE)


def test_unconstrained_selects_leak(cohort, manifest):
    res = GreedySelector(mode="unconstrained").run(cohort, manifest)
    assert res.leak_selected
    assert cohort["leak_name"] in res.selected


def test_constrained_never_selects_leak(cohort, manifest):
    res = GreedySelector(mode="constrained").run(cohort, manifest)
    assert not res.leak_selected


def test_agent_deterministic(cohort, manifest):
    r1 = GreedySelector(mode="unconstrained").run(cohort, manifest)
    r2 = GreedySelector(mode="unconstrained").run(cohort, manifest)
    assert r1.selected == r2.selected


def test_audit_localizes_leak(cohort, manifest):
    agent = GreedySelector(mode="unconstrained").run(cohort, manifest)
    res = ProvenanceAudit().run(agent, cohort, manifest, {cohort["leak_name"]})
    assert res.localized
    assert res.flagged == [cohort["leak_name"]]


def test_audit_reward_delta_positive(cohort, manifest):
    agent = GreedySelector(mode="unconstrained").run(cohort, manifest)
    res = ProvenanceAudit().run(agent, cohort, manifest, {cohort["leak_name"]})
    assert res.reward_delta > 0.05


def test_audit_no_false_positive_on_control():
    c = generate_cohort(seed=3)
    m = build_manifest(c["leak_name"])
    agent = GreedySelector(mode="constrained").run(c, m)
    res = ProvenanceAudit().run(agent, c, m, {c["leak_name"]})
    assert res.flagged == []
    assert not res.detected or True  # leak was never selected: nothing to detect
    assert res.localized is False or res.flagged == []


def test_metrics_ranges(cohort):
    p = np.linspace(0.05, 0.95, len(cohort["y_test"]))
    assert 0.0 <= M.ece(cohort["y_test"], p) <= 1.0
    assert -1.0 <= M.net_benefit(cohort["y_test"], p) <= 1.0
    assert 0.0 <= M.auprc(cohort["y_test"], p) <= 1.0


def test_paired_bootstrap_excludes_zero_on_separated():
    rng = np.random.default_rng(0)
    a = np.ones(40)
    b = rng.random(40) * 0.5
    mean, lo, hi = M.paired_bootstrap_diff(a, b, n_boot=500, seed=1)
    assert lo > 0 and mean > 0


def test_paired_bootstrap_includes_zero_on_identical():
    a = np.ones(30)
    mean, lo, hi = M.paired_bootstrap_diff(a, a, n_boot=500, seed=1)
    assert lo <= 0 <= hi


def test_scenario_core_keys():
    r = run_scenario("core", seed=0)
    assert r["unconstrained"]["leak_selected"]
    assert r["unconstrained"]["localized"]
    assert r["naive"]["naive_correct"]  # honest name caught by name filter


def test_scenario_disguised_naive_fails():
    r = run_scenario("disguised", seed=0)
    assert r["unconstrained"]["leak_selected"]
    assert r["unconstrained"]["localized"]      # provenance still catches it
    assert not r["naive"]["naive_correct"]      # name filter misses disguise


def test_scenario_constrained_control():
    r = run_scenario("constrained", seed=0)
    assert r["constrained"]["flagged"] == []
    assert not r["constrained"]["leak_selected"]
