"""
Cohort classification (genotype × seizure outcome).

Each session falls into one of four cohorts:
    DS_sz     - Scn1a (Dravet) genotype, animal had a seizure
    DS_no_sz  - Scn1a genotype, no seizure observed
    WT_sz     - wild-type genotype, animal had a seizure
                (e.g. PTZ-induced seizures in WT mice)
    WT_no_sz  - wild-type, no seizure

Previously the pipeline collapsed by seizure outcome alone, which dropped
WT_sz into the same cohort as DS_sz - wrong for PTZ and any future
paradigm where multiple genotypes can produce seizures. All routing goes
through this module so the partition is consistent everywhere.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_models import Session


COHORT_KEYS = ("DS_sz", "DS_no_sz", "WT_sz", "WT_no_sz")

# Reporting order: DS first, then WT; seized first within each genotype.
COHORT_ORDER = ("DS_sz", "DS_no_sz", "WT_sz", "WT_no_sz")


def classify_from_metadata(genotype: str, n_seizures: int) -> str:
    """Classify a session from its raw metadata fields.

    `genotype` is the same string used in Session.genotype (e.g. "Scn1a", "WT").
    Anything that isn't recognized as Scn1a is treated as WT - keeps the
    pipeline from inventing a fourth bucket when a new genotype string slips
    in unannounced.
    """
    is_scn1a = (genotype or "").strip().lower().startswith("scn1a")
    seized = (n_seizures or 0) > 0
    if is_scn1a:
        return "DS_sz" if seized else "DS_no_sz"
    return "WT_sz" if seized else "WT_no_sz"


def classify_session(session: "Session") -> str:
    """Classify a Session into one of the four cohort keys."""
    return classify_from_metadata(session.genotype, session.n_seizures)
