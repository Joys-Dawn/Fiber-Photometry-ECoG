"""
Open Ephys binary format reader.

Directory layout expected (per Chandni's snr-analysis):
  <session_dir>/
    Record Node <id>/
      experiment1/
        recording<N>/
          structure.oebin          # JSON metadata
          continuous/
            <stream_folder>/
              continuous.dat       # int16 interleaved samples
              sample_numbers.npy   # uint64 sample indices
              timestamps.npy       # float64 timestamps
          events/
            <stream_folder>/TTL/
              sample_numbers.npy   # uint64 event sample indices
              states.npy           # int16 TTL states (+1 = rise, -1 = fall)
              timestamps.npy       # float64 event timestamps

Chandni's setup has two streams (OEPstream=1 for ECoG, NIstream=2 for NI-DAQ).
Some setups (e.g. Meiling's) have one stream with ADC channels mixed in.
The reader handles both: user specifies which stream and channel indices to use.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import json
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OEPData:
    ecog: np.ndarray               # selected ECoG channel (uV)
    emg: Optional[np.ndarray]      # selected EMG channel (uV), None if not requested
    temperature_raw: np.ndarray    # raw temperature ADC (digital units, needs bit_volts conversion)
    temp_bit_volts: float          # bit-to-volts factor for temperature channel
    fs: float                      # sampling rate (Hz)
    sample_numbers: np.ndarray     # sample number array
    timestamps: np.ndarray         # timestamp array (seconds)
    ttl_sample_numbers: np.ndarray # TTL event sample numbers
    ttl_states: np.ndarray         # TTL states (+1 rise, -1 fall)
    ttl_timestamps: np.ndarray     # TTL event timestamps
    metadata: Dict[str, Any]       # full oebin dict


def _find_recording_dir(session_dir: Path, recording_num: int = 1) -> Path:
    """Find the recording directory within an Open Ephys session.

    Searches for Record Node directories and returns the path to the
    specified recording.
    """
    record_nodes = sorted(
        [d for d in session_dir.iterdir()
         if d.is_dir() and d.name.startswith("Record Node")],
        key=lambda p: p.name,
    )
    if not record_nodes:
        raise FileNotFoundError(
            f"No 'Record Node' directories found in {session_dir}"
        )

    # Use the first record node (typically Record Node 101 or 104)
    rec_node = record_nodes[0]

    # Find experiment and recording directories (handle experiment1, experiment2, etc.)
    experiments = sorted(
        [d for d in rec_node.iterdir() if d.is_dir() and d.name.startswith("experiment")],
        key=lambda p: p.name,
    )
    if not experiments:
        raise FileNotFoundError(f"No experiment directories found in {rec_node}")

    # Search for the requested recording across all experiments
    for exp_dir in experiments:
        rec_dir = exp_dir / f"recording{recording_num}"
        if rec_dir.exists():
            return rec_dir

    # Fallback: find any recording in any experiment
    for exp_dir in experiments:
        recordings = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("recording")],
            key=lambda p: p.name,
        )
        if recordings:
            rec_dir = recordings[0]
            logger.info(f"recording{recording_num} not found, using {exp_dir.name}/{rec_dir.name}")
            return rec_dir

    raise FileNotFoundError(f"No recording directories found in {rec_node}")



def _load_oebin(rec_dir: Path) -> Dict[str, Any]:
    """Load and parse the structure.oebin metadata file."""
    oebin_path = rec_dir / "structure.oebin"
    if not oebin_path.exists():
        raise FileNotFoundError(f"structure.oebin not found in {rec_dir}")
    with open(oebin_path, "r") as f:
        return json.load(f)


def _load_continuous_stream(
    rec_dir: Path,
    stream_info: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw int16 data, sample_numbers, and timestamps for a continuous stream."""
    folder = stream_info["folder_name"].rstrip("/")
    stream_dir = rec_dir / "continuous" / folder

    dat_path = stream_dir / "continuous.dat"
    if not dat_path.exists():
        raise FileNotFoundError(f"continuous.dat not found in {stream_dir}")

    n_channels = stream_info["num_channels"]
    raw = np.memmap(dat_path, dtype="<i2", mode="r")
    n_samples = len(raw) // n_channels
    data = raw.reshape((n_samples, n_channels)).T  # shape: (n_channels, n_samples)

    sample_numbers = np.load(stream_dir / "sample_numbers.npy")
    timestamps = np.load(stream_dir / "timestamps.npy")

    return data, sample_numbers, timestamps


def _load_ttl_events(rec_dir: Path, oebin: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load TTL event data (sample_numbers, states, timestamps)."""
    # Find the TTL event entry in oebin
    ttl_events = [e for e in oebin.get("events", [])
                  if "TTL" in e.get("folder_name", "")]
    if not ttl_events:
        return np.array([]), np.array([]), np.array([])

    folder = ttl_events[0]["folder_name"].rstrip("/")
    ttl_dir = rec_dir / "events" / folder

    if not ttl_dir.exists():
        return np.array([]), np.array([]), np.array([])

    sample_numbers = np.load(ttl_dir / "sample_numbers.npy")
    states = np.load(ttl_dir / "states.npy")
    timestamps = np.load(ttl_dir / "timestamps.npy")

    return sample_numbers, states, timestamps


def read_oep(
    session_dir: str | Path,
    ecog_channel: int = 2,
    emg_channel: Optional[int] = 3,
    ecog_stream_idx: int = 0,
    recording_num: int = 1,
) -> OEPData:
    """Read an Open Ephys binary recording session.

    Parameters
    ----------
    session_dir : path to the session directory (contains Record Node folders)
    ecog_channel : 1-indexed ECoG channel number (default 2)
    emg_channel : 1-indexed EMG channel number (default 3), or None to skip
    ecog_stream_idx : 0-indexed stream for ECoG data (default 0)
    recording_num : which recording to load (default 1)

    Returns
    -------
    OEPData with ECoG, EMG (optional), temperature, TTL events, and metadata.

    Notes
    -----
    Temperature is always loaded from the NI-DAQ stream (channel 1 / AI0),
    matching Chandni's MATLAB config (cfg.NIstream, cfg.tempADC=1).  When
    only one stream exists the NI-DAQ *is* that stream, so channel 1 is used.

    Channel indexing follows Open Ephys convention (1-indexed in the GUI,
    converted to 0-indexed internally).
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    rec_dir = _find_recording_dir(session_dir, recording_num)
    oebin = _load_oebin(rec_dir)

    continuous_streams = oebin.get("continuous", [])
    if not continuous_streams:
        raise ValueError(f"No continuous streams found in {rec_dir}")
    if ecog_stream_idx >= len(continuous_streams):
        raise ValueError(
            f"ECoG stream index {ecog_stream_idx} out of range "
            f"(found {len(continuous_streams)} streams)"
        )

    # --- Load ECoG stream ---
    ecog_stream_info = continuous_streams[ecog_stream_idx]
    ecog_data, sample_numbers, timestamps = _load_continuous_stream(rec_dir, ecog_stream_info)
    fs = ecog_stream_info["sample_rate"]

    # Extract ECoG channel (convert 1-indexed to 0-indexed)
    ecog_ch_idx = ecog_channel - 1
    if ecog_ch_idx < 0 or ecog_ch_idx >= ecog_data.shape[0]:
        raise ValueError(
            f"ECoG channel {ecog_channel} out of range "
            f"(stream has {ecog_data.shape[0]} channels)"
        )
    ecog_signal = ecog_data[ecog_ch_idx, :].astype(np.float64)

    # Convert to uV using bit_volts
    ecog_bit_volts = ecog_stream_info["channels"][ecog_ch_idx]["bit_volts"]
    ecog_signal *= ecog_bit_volts

    # --- Load EMG (optional) ---
    emg_signal = None
    if emg_channel is not None:
        emg_ch_idx = emg_channel - 1
        if emg_ch_idx < 0 or emg_ch_idx >= ecog_data.shape[0]:
            raise ValueError(
                f"EMG channel {emg_channel} out of range "
                f"(stream has {ecog_data.shape[0]} channels)"
            )
        emg_signal = ecog_data[emg_ch_idx, :].astype(np.float64)
        emg_bit_volts = ecog_stream_info["channels"][emg_ch_idx]["bit_volts"]
        emg_signal *= emg_bit_volts

    # --- Load temperature ---
    # Per Chandni's MATLAB code (cfg.NIstream, cfg.tempADC=1), temperature
    # is always channel 1 (AI0) on the NI-DAQ stream.
    if len(continuous_streams) > 1:
        # Two-stream setup: find the non-ECoG stream (NI-DAQ)
        ni_idx = None
        for si, s in enumerate(continuous_streams):
            if si != ecog_stream_idx and "NI" in s.get("source_processor_name", ""):
                ni_idx = si
                break
        if ni_idx is None:
            ni_idx = 1 if ecog_stream_idx == 0 else 0

        temp_stream_info = continuous_streams[ni_idx]
        temp_data, _, _ = _load_continuous_stream(rec_dir, temp_stream_info)
        temp_ch_idx = 0  # channel 1 (AI0)
        temp_bit_volts = temp_stream_info["channels"][temp_ch_idx]["bit_volts"]
        logger.info(
            "Temperature: stream %d (%s) channel 1",
            ni_idx, temp_stream_info["source_processor_name"],
        )
    else:
        # Single-stream setup: NI-DAQ is the only stream, channel 1
        temp_ch_idx = 0
        temp_data = ecog_data
        temp_bit_volts = ecog_stream_info["channels"][temp_ch_idx]["bit_volts"]

    temperature_raw = temp_data[temp_ch_idx, :].astype(np.float64)

    # --- Load TTL events ---
    ttl_samples, ttl_states, ttl_timestamps = _load_ttl_events(rec_dir, oebin)

    return OEPData(
        ecog=ecog_signal,
        emg=emg_signal,
        temperature_raw=temperature_raw,
        temp_bit_volts=temp_bit_volts,
        fs=fs,
        sample_numbers=sample_numbers,
        timestamps=timestamps,
        ttl_sample_numbers=ttl_samples,
        ttl_states=ttl_states,
        ttl_timestamps=ttl_timestamps,
        metadata=oebin,
    )
