"""leakbench: synthetic prediction-time-leakage audit suite for clinical-modeling agents.

Open implementation inspired by arXiv 2610.26088 (LeakBench). Synthetic scope
only: a cohort with KNOWN failure structure plus a deterministic
provenance-based audit protocol. No LLM, CPU only, $0.
"""

__version__ = "0.1.0"

from .cohort import generate_cohort, LEGIT_FEATURES, LEAK_FEATURE
from .provenance import build_manifest, FeatureProvenance
from .agent import GreedySelector, AgentResult, LEAK_NAME_TOKENS
from .audit import ProvenanceAudit, AuditResult
from .metrics import auroc, auprc, ece, net_benefit, paired_bootstrap_diff
from .suite import run_suite, SCENARIOS

__all__ = [
    "generate_cohort", "LEGIT_FEATURES", "LEAK_FEATURE",
    "build_manifest", "FeatureProvenance",
    "GreedySelector", "AgentResult", "LEAK_NAME_TOKENS",
    "ProvenanceAudit", "AuditResult",
    "auroc", "auprc", "ece", "net_benefit", "paired_bootstrap_diff",
    "run_suite", "SCENARIOS",
]
