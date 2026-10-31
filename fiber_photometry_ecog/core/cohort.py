"""Cohort classification by genotype (DS, WT, FCD) and seizure outcome."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_models import Session


COHORT_KEYS = ("DS_sz", "DS_no_sz", "WT_sz", "WT_no_sz",
               "FCD_sz", "FCD_no_sz")

COHORT_ORDER = ("DS_sz", "DS_no_sz", "WT_sz", "WT_no_sz",
                "FCD_sz", "FCD_no_sz")


def classify_from_metadata(genotype: str, n_seizures: int) -> str:
    """Classify a session from its raw metadata fields.

    `genotype` is the same string used in Session.genotype (e.g. "Scn1a", "WT",
    "FCD"). Scn1a and FCD map to distinct groups; all other genotype strings
    map to WT. Each group is further separated by seizure outcome.
    """
    g = (genotype or "").strip().lower()
    seized = (n_seizures or 0) > 0
    if g.startswith("scn1a"):
        return "DS_sz" if seized else "DS_no_sz"
    if g.startswith("fcd"):
        return "FCD_sz" if seized else "FCD_no_sz"
    return "WT_sz" if seized else "WT_no_sz"


def classify_session(session: "Session") -> str:
    """Classify a Session into a cohort key.

    Honors `session.cohort_override` when set (used by opto, where the
    cohort is the stim subfolder name rather than a DS/WT × seized bucket).
    Otherwise falls back to the standard genotype-by-seizure classification.
    """
    if session.cohort_override:
        return session.cohort_override
    return classify_from_metadata(session.genotype, session.n_seizures)
