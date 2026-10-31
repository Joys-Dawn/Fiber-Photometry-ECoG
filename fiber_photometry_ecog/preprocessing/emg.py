"""EMG passthrough after synchronization to the ECoG timebase."""

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
