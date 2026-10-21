# leakbench

A synthetic **prediction-time-leakage audit suite** for clinical-modeling
agents: a cohort with *known failure structure* plus a deterministic
provenance-based audit protocol. Open implementation inspired by arXiv
2610.26088 "LeakBench: Auditing Reward Hacking through Prediction-Time
Leakage in Clinical Modeling Agents" (Bao et al., 2026). The paper's
MIMIC-IV / MHFP real-data settings are out of scope here; this build
covers the paper's synthetic task: can an audit *detect and localize*
leakage a reward-maximizing agent was guaranteed to pick up?

## The disease (by construction)

Predict sepsis onset at T+2h after admission. The cohort ships 7
admission-time features (vitals, labs, SOFA, comorbidities) and 1 fatal
leaking feature: the blood-culture result returned at T+18h (16h *after*
prediction time), merged into the row by a sloppy ETL. It equals the label
with 2% noise. A deterministic greedy agent maximizing validation AUROC
selects it in **20/20** replicates and gains **+7.9pp** test AUROC —
reward hacking, described without attributing intent.

## The audit

A deterministic provenance verifier replays the agent's selected features
against an availability manifest (`available_at_hours` per feature). It
flagged and localized the exact leak in **40/40** leaky audits (including
20 where the leak was renamed to `derived_stability_index`), with **0**
false flags on 20 constrained controls. Against a name-based baseline,
the paired-bootstrap 95% CI for the accuracy difference is
**[+0.217, +0.450]** — excludes 0.

Also measured: optimizing AUPRC instead of AUROC does not remove leakage
selection (20/20), matching the paper.

## Quickstart (CPU only, $0)

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q          # 22 tests
PYTHONPATH=src python scripts/run_gates.py         # gate measurements -> results/gate_results.json
```

Environment: Python 3.12, numpy 2.5, scikit-learn 1.9. No GPU, no network,
no LLM, no downloads.

## Layout

- `src/leakbench/cohort.py` — synthetic cohort generator (known failure structure)
- `src/leakbench/provenance.py` — feature-availability manifest
- `src/leakbench/agent.py` — deterministic greedy selector (unconstrained / name_filter / constrained)
- `src/leakbench/audit.py` — provenance verifier (detect + localize + reward delta)
- `src/leakbench/metrics.py` — AUROC/AUPRC/ECE/net-benefit, paired bootstrap
- `src/leakbench/suite.py` — 3 variants x 20 replicates orchestration
- `scripts/run_gates.py` — locked gate measurements
- `PREREG.md` — protocol and kill bars, locked *before* any run
- `KILLGATE.md` — measured gate verdicts

## Readout note

There is no text/LLM readout anywhere in this build: every number is a
direct numeric sklearn metric from `predict_proba` on held-out splits.
(See the POPE carry-over lesson in PREREG.md.)

## Honest limits

Synthetic scope only; deterministic simulator agent, not an LLM agent;
single leak mechanism (post-prediction-time ETL merge; disguise = rename).
Not the paper's biology or MIMIC-IV/MHFP settings.

## License

MIT.
