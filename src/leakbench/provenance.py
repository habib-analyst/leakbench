"""Feature-availability provenance manifest.

Every feature carries `available_at_hours` relative to prediction time
(T+2h after admission). A feature is prediction-time-valid iff
available_at_hours <= 0. The leak is only available at +16h (T+18h).
"""

from dataclasses import dataclass

from .cohort import LEGIT_FEATURES


@dataclass(frozen=True)
class FeatureProvenance:
    name: str
    available_at_hours: float
    source: str
    description: str

    @property
    def valid_at_prediction_time(self):
        return self.available_at_hours <= 0


# Admission-time observations: all recorded at or before prediction time.
_LEGIT_PROVENANCE = {
    "age": (0.0, "registration desk", "demographics at admission"),
    "heart_rate_admit": (0.0, "triage monitor", "triage vitals at admission"),
    "map_admit": (0.0, "triage monitor", "triage vitals at admission"),
    "lactate_admit": (-1.0, "lab interface", "admission blood gas panel"),
    "wbc_admit": (-1.0, "lab interface", "admission CBC panel"),
    "sofa_admit": (0.0, "EHR flowsheet", "SOFA computed from admission data"),
    "comorbidity_count": (0.0, "problem list", "coded comorbidities at admission"),
}


def build_manifest(leak_name, disguised=False):
    """Build the provenance manifest for a cohort.

    `disguised` only affects the human-readable description, never the
    availability timestamps: provenance-based detection must work on
    timestamps alone (the paper's renaming result).
    """
    manifest = {}
    for feat in LEGIT_FEATURES:
        at, src, desc = _LEGIT_PROVENANCE[feat]
        manifest[feat] = FeatureProvenance(feat, at, src, desc)
    leak_desc = (
        "composite stability score (vendor feed)"
        if disguised
        else "blood culture result (returned T+18h, merged by ETL)"
    )
    manifest[leak_name] = FeatureProvenance(
        name=leak_name,
        available_at_hours=16.0,  # result returned 16h AFTER prediction time
        source="microbiology lab interface (late feed)",
        description=leak_desc,
    )
    return manifest
