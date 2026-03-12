"""
pyPhotometry .ppd binary file reader.

Binary format (Thomas Akam, pyPhotometry):
- 2 bytes: header size (little-endian uint16)
- header_size bytes: JSON header with metadata
- remaining bytes: interleaved uint16 samples
  - top 15 bits = analog value
  - LSB = digital value
  - samples alternate between channels (2 or 3 analog, 1 or 2 digital)

Mirrors the logic in Chandni's snr-analysis pipeline (load_ppd).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import json
import numpy as np


@dataclass
class PPDData:
    signal_470: np.ndarray    # analog_1: GCaMP / signal channel (volts)
    signal_405: np.ndarray    # analog_2: isosbestic / control channel (volts)
    digital_1: np.ndarray     # sync TTL (binary 0/1)
    pulse_inds: np.ndarray    # rising-edge indices in digital_1
    pulse_times: np.ndarray   # rising-edge times in seconds
    fs: float                 # sampling rate (Hz)
    time: np.ndarray          # time vector (seconds, relative to recording start)
    metadata: Dict[str, Any]  # full header dict from pyPhotometry


def read_ppd(file_path: str | Path) -> PPDData:
    """Read a pyPhotometry .ppd binary file.

    Parameters
    ----------
    file_path : path to the .ppd file

    Returns
    -------
    PPDData with extracted signals, sync edges, and metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PPD file not found: {file_path}")
    if file_path.suffix.lower() != ".ppd":
        raise ValueError(f"Expected .ppd file, got: {file_path.suffix}")

    with open(file_path, "rb") as f:
        header_size = int.from_bytes(f.read(2), "little")
        header_bytes = f.read(header_size)
        raw = np.frombuffer(f.read(), dtype=np.dtype("<u2"))

    header = json.loads(header_bytes)
    volts_per_division = header["volts_per_division"]
    sampling_rate = header["sampling_rate"]

    # Analog = top 15 bits; digital = LSB
    analog = raw >> 1
    digital = ((raw & 1) == 1).astype(np.int8)

    n_analog = header.get("n_analog_signals", 2)

    # De-interleave channels
    signal_470 = analog[::n_analog] * volts_per_division[0]
    signal_405 = analog[1::n_analog] * volts_per_division[1]
    digital_1 = digital[::n_analog]

    n_samples = len(signal_470)
    time = np.arange(n_samples) / sampling_rate  # seconds

    # Rising edges
    pulse_inds = 1 + np.where(np.diff(digital_1) == 1)[0]
    pulse_times = pulse_inds / sampling_rate  # seconds

    return PPDData(
        signal_470=signal_470,
        signal_405=signal_405,
        digital_1=digital_1,
        pulse_inds=pulse_inds,
        pulse_times=pulse_times,
        fs=sampling_rate,
        time=time,
        metadata=header,
    )
