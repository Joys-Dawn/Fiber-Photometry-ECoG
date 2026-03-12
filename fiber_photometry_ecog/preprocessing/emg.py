"""
EMG preprocessing.

EMG is already aligned to the ECoG timebase after sync (same continuous.dat
stream, same sample clock). This module is a placeholder for future EMG
preprocessing steps as defined by the analysis spec.

Current scope: store aligned EMG in session data. Additional filtering
and analysis TBD per Excel spec.
"""

import numpy as np


def align_emg(
    emg: np.ndarray | None,
    time: np.ndarray,
    fs: float,
) -> np.ndarray | None:
    """Pass through EMG signal (already on ECoG timebase after sync).

    Parameters
    ----------
    emg : raw EMG signal from sync output, or None
    time : common time vector
    fs : sampling rate (Hz)

    Returns
    -------
    EMG signal (unchanged), or None if not recorded.
    """
    if emg is None:
        return None
    return emg
