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
from typing import Optional, Dict, Any, List

import json
import numpy as np

from ..core.config import TemperatureConfig, lookup_meiling_calibration

logger = logging.getLogger(__name__)


@dataclass
class OEPData:
    ecog: np.ndarray               # selected ECoG channel (uV)
    emg: Optional[np.ndarray]      # selected EMG channel (uV), None if not requested
    temperature_raw: np.ndarray    # raw temperature ADC (digital units, needs bit_volts conversion)
    temp_bit_volts: float          # bit-to-volts factor for temperature channel
    temp_slope: float              # per-session T(C) = slope * V(mV) + intercept
    temp_intercept: float          # per-session intercept (C)
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


def _extract_session_date(session_dir: Path) -> Optional[str]:
    """Extract YYYY-MM-DD from an Open Ephys session folder name."""
    name = session_dir.name
    if len(name) >= 10 and name[4] == "-" and name[7] == "-":
        return name[:10]
    return None


def _find_nidaq_stream(
    continuous_streams: List[Dict[str, Any]],
) -> Optional[int]:
    """Return the index of a NI-DAQ stream if one is present, else None.

    Chandni's rig records the thermistor on a standalone NI-DAQ USB-6009
    board that surfaces as its own continuous stream. Meiling's rigs don't
    have one, so the thermistor lives on an ADC input of the Intan
    Acquisition Board stream instead.
    """
    for si, s in enumerate(continuous_streams):
        name = (s.get("folder_name", "") + " " + s.get("source_processor_name", "")).upper()
        if "NI-DAQ" in name or "NIDAQ" in name:
            return si
    return None


def read_oep(
    session_dir: str | Path,
    ecog_channel: int = 2,
    emg_channel: Optional[int] = 3,
    temperature_channel: Optional[int] = None,
    temperature_stream_idx: Optional[int] = None,
    ecog_stream_idx: int = 0,
    recording_num: int = 1,
) -> OEPData:
    """Read an Open Ephys binary recording session.

    Temperature is resolved as follows. If a NI-DAQ stream is present
    (Chandni's rig), the thermistor is assumed to live on AI0 and the
    default calibration T(C) = 0.0981 * V(mV) + 8.81 is used. Otherwise
    (Meiling's rigs) the thermistor is assumed to live on ADC1 of the
    Intan Acquisition Board stream, and the calibration is looked up by
    recording date in `MEILING_RIG_CALIBRATIONS`. If the caller specifies
    `temperature_channel` explicitly, that channel overrides the default
    pick, but the same calibration rule (NI-DAQ default vs date-keyed
    Meiling) still applies based on which stream contains the channel.

    Parameters
    ----------
    session_dir : path to the session directory (contains Record Node folders)
    ecog_channel : 1-indexed ECoG channel number (default 2)
    emg_channel : 1-indexed EMG channel number (default 3), or None to skip
    temperature_channel : 1-indexed temperature channel override. If None,
        uses AI0 (NI-DAQ rigs) or ADC1 (Intan-only rigs).
    temperature_stream_idx : 0-indexed stream containing the override
        channel. Ignored unless `temperature_channel` is given.
    ecog_stream_idx : 0-indexed stream for ECoG data (default 0)
    recording_num : which recording to load (default 1)

    Returns
    -------
    OEPData with ECoG, EMG (optional), temperature, calibration, TTL events,
    and metadata. Channel indexing follows Open Ephys convention (1-indexed
    in the GUI, converted to 0-indexed internally).
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
    temp_cfg = TemperatureConfig()
    nidaq_idx = _find_nidaq_stream(continuous_streams)

    if temperature_stream_idx is not None:
        ts_idx = temperature_stream_idx
    elif nidaq_idx is not None:
        ts_idx = nidaq_idx
    else:
        ts_idx = ecog_stream_idx

    # Default channel: AI0 for NI-DAQ rigs, ADC1 for Intan-only rigs.
    # ADC1 on the 24-channel Intan stream is at 0-indexed channel 16
    # (channels 0..15 are CH1..CH16 neural, 16..23 are ADC1..ADC8).
    if temperature_channel is not None:
        temp_ch_idx = temperature_channel - 1
    elif ts_idx == nidaq_idx:
        temp_ch_idx = 0   # AI0
    else:
        temp_ch_idx = 16  # ADC1 on the Intan stream

    temp_stream_info = continuous_streams[ts_idx]
    if ts_idx == ecog_stream_idx:
        temp_data = ecog_data
    else:
        temp_data, _, _ = _load_continuous_stream(rec_dir, temp_stream_info)

    if temp_ch_idx < 0 or temp_ch_idx >= temp_data.shape[0]:
        raise ValueError(
            f"Temperature channel index {temp_ch_idx} out of range "
            f"(stream {ts_idx} has {temp_data.shape[0]} channels)"
        )

    temp_bit_volts = temp_stream_info["channels"][temp_ch_idx]["bit_volts"]
    temperature_raw = temp_data[temp_ch_idx, :].astype(np.float64)

    # Calibration: NI-DAQ rig uses the Physitemp formula in TemperatureConfig;
    # Intan-only (Meiling) rigs use the date-range mapping Meiling provided.
    if ts_idx == nidaq_idx:
        temp_slope = temp_cfg.slope
        temp_intercept = temp_cfg.intercept
        logger.info(
            "Temperature: NI-DAQ stream %d channel %d (AI0), "
            "calibration T=%.4f*mV+%.2f",
            ts_idx, temp_ch_idx + 1, temp_slope, temp_intercept,
        )
    else:
        rec_date = _extract_session_date(session_dir)
        if rec_date is None:
            raise ValueError(
                f"Cannot determine recording date from session folder "
                f"{session_dir.name!r}; required to select the Meiling rig "
                f"calibration."
            )
        temp_slope, temp_intercept = lookup_meiling_calibration(rec_date)
        logger.info(
            "Temperature: Intan stream %d channel %d (ADC%d), date=%s, "
            "calibration T=%.5f*mV+%.2f",
            ts_idx, temp_ch_idx + 1, temp_ch_idx - 15, rec_date,
            temp_slope, temp_intercept,
        )

    # --- Load TTL events ---
    ttl_samples, ttl_states, ttl_timestamps = _load_ttl_events(rec_dir, oebin)

    return OEPData(
        ecog=ecog_signal,
        emg=emg_signal,
        temperature_raw=temperature_raw,
        temp_bit_volts=temp_bit_volts,
        temp_slope=temp_slope,
        temp_intercept=temp_intercept,
        fs=fs,
        sample_numbers=sample_numbers,
        timestamps=timestamps,
        ttl_sample_numbers=ttl_samples,
        ttl_states=ttl_states,
        ttl_timestamps=ttl_timestamps,
        metadata=oebin,
    )
