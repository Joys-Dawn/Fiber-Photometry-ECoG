from typing import Protocol, runtime_checkable

import numpy as np

from ...core.config import PhotometryConfig
from ...core.data_models import PhotometryResult
from .strategy_a_chandni import (
    ChandniStrategy as ChandniStrategy,
    preprocess_chandni as preprocess_chandni,
)
from .strategy_b_meiling import (
    MeilingStrategy as MeilingStrategy,
    preprocess_meiling as preprocess_meiling,
)
from .strategy_c_irls import (
    IRLSStrategy as IRLSStrategy,
    preprocess_irls as preprocess_irls,
)
from .strategy_d_no_isosbestic import (
    NoIsosbesticStrategy as NoIsosbesticStrategy,
    preprocess_no_isosbestic as preprocess_no_isosbestic,
)
from .common import (
    z_score_baseline as z_score_baseline,
    highpass_filter as highpass_filter,
    detrend_moving_average as detrend_moving_average,
)


# Strategy letter -> short human-readable name. Used to label output folders
# (e.g. "preprocessing_IRLS") and the multi-select UI.
STRATEGY_NAMES = {
    "A": "Chandni",
    "B": "Meiling",
    "C": "IRLS",
    "D": "no_isosbestic",
}


def strategy_folder_name(strategy: str) -> str:
    """Return the output-folder name for a strategy letter (e.g. 'C' -> 'preprocessing_IRLS')."""
    return f"preprocessing_{STRATEGY_NAMES.get(strategy.upper(), strategy)}"


@runtime_checkable
class PhotometryStrategy(Protocol):
    """Protocol that all photometry strategies must implement."""

    def preprocess(
        self,
        signal_470: np.ndarray,
        signal_405: np.ndarray,
        fs: float,
        config: PhotometryConfig | None = None,
    ) -> PhotometryResult: ...
