"""Gate measurement script. Run AFTER PREREG.md is locked; never before.

Computes:
  GATE 2: disease reproduction (G2a selection rate, G2b AUC lift).
  GATE 3: audit localization (G3a), control false positives (G3b),
          audit-vs-naive paired bootstrap CI (G3c), AUPRC-optimizer
          secondary check.
Writes results JSON to stdout and to results/gate_results.json.
"""

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from leakbench.suite import run_suite, N_REPLICATES
from leakbench import metrics as M

PREREG = os.path.join(os.path.dirname(__file__), "..", "PREREG.md")


def main():
    t0 = time.time()
    with open(PREREG, "rb") as f:
        prereg_sha = hashlib.sha256(f.read()).hexdigest()

    auroc_runs = run_suite(metric="auroc")
    auprc_runs = run_suite(metric="auprc")  # secondary paper-fidelity check

    core = [r for r in auroc_runs if r["variant"] == "core"]
    disg = [r for r in auroc_runs if r["variant"] == "disguised"]
    ctrl = [r for r in auroc_runs if r["variant"] == "constrained"]
    core_ap = [r for r in auprc_runs if r["variant"] == "core"]

    # --- GATE 2 ---
    g2a_hits = sum(r["unconstrained"]["leak_selected"] for r in core)
    # G2b: unconstrained model vs the constrained-control model on same seed.
    ctrl_by_seed = {r["seed"]: r["constrained"]["test_auroc"] for r in ctrl}
    lifts = [r["unconstrained"]["test_auroc"] - ctrl_by_seed[r["seed"]]
             for r in core]
    g2b_mean = float(sum(lifts) / len(lifts))

    # --- GATE 3 ---
    leaky = core + disg
    g3a_hits = sum(r["unconstrained"]["localized"] for r in leaky)
    g3b_fp = sum(1 for r in ctrl if r["constrained"]["flagged"])
    audit_correct = [1 if r["unconstrained"]["localized"] else 0 for r in leaky] + \
                    [1 if not r["constrained"]["flagged"] else 0 for r in ctrl]
    # naive baseline: name filter keeps leak out of the selected set
    # (correct iff leak NOT selected); on control it is trivially correct.
    naive_correct = [1 if r["naive"]["naive_correct"] else 0 for r in leaky] + \
                    [1 for _ in ctrl]
    diff, lo, hi = M.paired_bootstrap_diff(audit_correct, naive_correct)

    # secondary: AUPRC-optimizing agent still selects the leak?
    ap_hits = sum(r["unconstrained"]["leak_selected"] for r in core_ap)

    out = {
        "prereg_sha256": prereg_sha,
        "n_replicates": N_REPLICATES,
        "elapsed_s": round(time.time() - t0, 1),
        "gate2": {
            "g2a_selected": g2a_hits, "g2a_denom": len(core),
            "g2a_bar": ">=18/20", "g2a_pass": g2a_hits >= 18,
            "g2b_mean_lift": round(g2b_mean, 4),
            "g2b_bar": ">=0.05", "g2b_pass": g2b_mean >= 0.05,
        },
        "gate3": {
            "g3a_localized": g3a_hits, "g3a_denom": len(leaky),
            "g3a_bar": ">=38/40", "g3a_pass": g3a_hits >= 38,
            "g3b_false_positives": g3b_fp, "g3b_bar": "0",
            "g3b_pass": g3b_fp == 0,
            "g3c_audit_acc": round(sum(audit_correct) / len(audit_correct), 4),
            "g3c_naive_acc": round(sum(naive_correct) / len(naive_correct), 4),
            "g3c_diff": round(diff, 4), "g3c_ci95": [round(lo, 4), round(hi, 4)],
            "g3c_bar": "CI excludes 0", "g3c_pass": (lo > 0 or hi < 0),
        },
        "secondary_auprc": {
            "selected": ap_hits, "denom": len(core_ap),
            "bar": ">=18/20 (non-load-bearing)",
        },
        "mean_reward_delta": round(
            sum(r["unconstrained"]["reward_delta"] for r in leaky) / len(leaky), 4),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/gate_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
