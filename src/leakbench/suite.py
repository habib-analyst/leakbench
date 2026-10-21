"""Suite orchestration: 3 variants x 20 replicates, agent modes, audits.

Variants:
  core        - leak present, honestly named (posthoc_culture_flag)
  disguised   - same leak, renamed to derived_stability_index
  constrained - same data, agent gets explicit admission-time constraint
                (negative control: no leak should be selected/flagged)

For each replicate the suite runs the unconstrained agent (disease), the
name_filter agent (naive baseline), and the provenance audit on both.
The constrained variant runs the constrained agent + audit as a control.
"""

from .cohort import generate_cohort
from .provenance import build_manifest
from .agent import GreedySelector
from .audit import ProvenanceAudit

SCENARIOS = ["core", "disguised", "constrained"]
N_REPLICATES = 20


def run_scenario(variant, seed, metric="auroc"):
    disguised = variant == "disguised"
    cohort = generate_cohort(seed, disguised=disguised)
    manifest = build_manifest(cohort["leak_name"], disguised=disguised)
    true_leak = {cohort["leak_name"]}
    audit = ProvenanceAudit()

    if variant == "constrained":
        agent = GreedySelector(mode="constrained", metric=metric).run(cohort, manifest)
        audit_res = audit.run(agent, cohort, manifest, true_leak)
        return {
            "variant": variant, "seed": seed, "metric": metric,
            "unconstrained": None,
            "constrained": _summarize(agent, audit_res),
        }

    unc = GreedySelector(mode="unconstrained", metric=metric).run(cohort, manifest)
    naive = GreedySelector(mode="name_filter", metric=metric).run(cohort, manifest)
    audit_unc = audit.run(unc, cohort, manifest, true_leak)
    audit_naive = audit.run(naive, cohort, manifest, true_leak)
    return {
        "variant": variant, "seed": seed, "metric": metric,
        "unconstrained": _summarize(unc, audit_unc),
        "naive": _summarize(naive, audit_naive),
    }


def _summarize(agent, audit_res):
    return {
        "selected": agent.selected,
        "val_score": agent.val_score,
        "test_auroc": agent.test_auroc,
        "test_auprc": agent.test_auprc,
        "leak_selected": agent.leak_selected,
        "flagged": audit_res.flagged,
        "evidence": audit_res.evidence,
        "reward_delta": audit_res.reward_delta,
        "detected": audit_res.detected,
        "localized": audit_res.localized,
        # naive-baseline "detection": did the name filter keep the leak out?
        "naive_correct": (not agent.leak_selected),
    }


def run_suite(metric="auroc", n_replicates=N_REPLICATES, seeds=None):
    seeds = list(range(n_replicates)) if seeds is None else seeds
    results = []
    for variant in SCENARIOS:
        for seed in seeds:
            results.append(run_scenario(variant, seed, metric=metric))
    return results
