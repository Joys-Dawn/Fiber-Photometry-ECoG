"""
Phase 4: Analysis modules.

Each module computes a specific set of metrics per the Excel spec.
All analyzers receive Session objects and an AnalysisConfig and return
dataclass result containers.
"""

from .cohort_characteristics import compute_cohort_characteristics as compute_cohort_characteristics
from .baseline_transients import compute_baseline_transients as compute_baseline_transients
from .preictal_mean import compute_preictal_mean as compute_preictal_mean
from .preictal_transients import compute_preictal_transients as compute_preictal_transients
from .ictal_mean import compute_ictal_mean as compute_ictal_mean
from .ictal_transients import compute_ictal_transients as compute_ictal_transients
from .postictal import compute_postictal as compute_postictal
from .spike_triggered import compute_spike_triggered_average as compute_spike_triggered_average
