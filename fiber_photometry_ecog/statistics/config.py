"""Configuration for cohort-level statistical comparisons."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..visualization.colors import COHORT_DISPLAY_LABELS


# Default mapping from internal cohort keys to publication display names.
# Single source of truth is `visualization.colors.COHORT_DISPLAY_LABELS` -
# re-exported here so stats users don't have to import from visualization.
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

    NOTE: multiple-comparison correction is intentionally omitted.
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
