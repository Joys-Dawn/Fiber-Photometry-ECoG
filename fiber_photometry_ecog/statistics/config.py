"""Configuration for cohort-level statistical comparisons."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..visualization.colors import COHORT_DISPLAY_LABELS


# Default cohort labels shared with plots.
DEFAULT_DISPLAY_NAMES: Dict[str, str] = dict(COHORT_DISPLAY_LABELS)


@dataclass
class StatsConfig:
    """Configuration for cohort comparisons.

    Attributes
    ----------
    cohorts_to_compare : list of cohort keys (in `cohort_groups`) to include
        in the comparison. With 2 cohorts the runner does an unpaired test;
        with >2, a one-way test plus pairwise post-hocs. Default starts with
        the primary DS_sz vs WT_no_sz 2-way comparison (DS_no_sz omitted).
    display_names : optional mapping from cohort key -> display label used in
        plots and CSV column headers. Falls back to DEFAULT_DISPLAY_NAMES.
    alpha : significance threshold for annotation purposes.

    P-values are not adjusted for multiple comparisons.
    """

    cohorts_to_compare: List[str] = field(
        default_factory=lambda: ["DS_sz", "WT_no_sz"]
    )
    display_names: Optional[Dict[str, str]] = None
    alpha: float = 0.05

    def label(self, cohort_key: str) -> str:
        """Return the display label for a cohort key."""
        names = self.display_names or DEFAULT_DISPLAY_NAMES
        return names.get(cohort_key, cohort_key)


@dataclass
class PairedStatsConfig:
    """Configuration for within-subject (LMM-based) comparisons.

    The model is ``value ~ cohort + (1|mouse_id) + (1|pair_id)``, fit by
    REML with one observation per session and condition. If R is unavailable
    or the model fit fails, is singular, or does not converge, a warning is
    issued and a paired t-test or Wilcoxon signed-rank test is used on matched
    session pairs. The fallback does not model within-mouse correlation
    across sessions. Observations are not averaged per mouse.

    Attributes
    ----------
    alpha : significance threshold for annotation purposes.
    """

    alpha: float = 0.05
