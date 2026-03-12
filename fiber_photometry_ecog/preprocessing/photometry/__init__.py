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
from .common import (
    z_score_baseline as z_score_baseline,
    highpass_filter as highpass_filter,
)


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
