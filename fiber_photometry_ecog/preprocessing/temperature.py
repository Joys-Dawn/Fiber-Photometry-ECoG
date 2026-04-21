"""
Temperature processing: voltage-to-Celsius conversion, smoothing, and landmark extraction.

Calibration: T(C) = slope * V(mV) + intercept
Default calibration constants from the lab's thermistor setup.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.config import TemperatureConfig


@dataclass
class TemperatureResult:
    """Processed temperature data and extracted landmarks."""
    temperature_c: np.ndarray        # converted to Celsius
    temperature_smooth: np.ndarray   # smoothed (moving average)
    baseline_temp: float             # mean of first N seconds
    max_temp: float                  # peak temperature
    max_temp_time: float             # time of peak temperature (seconds)
    terminal_temp: float             # temperature at end of recording


def process_temperature(
    temperature_raw: np.ndarray,
    temp_bit_volts: float,
    fs: float,
    config: TemperatureConfig | None = None,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
) -> TemperatureResult:
    """Convert raw ADC values to Celsius and extract landmarks.

    Parameters
    ----------
    temperature_raw : raw ADC values (digital units from Open Ephys)
    temp_bit_volts : bit-to-volts conversion factor from oebin metadata
    fs : sampling rate (Hz)
    config : temperature parameters (smoothing window, baseline duration).
        Uses defaults if None.
    slope, intercept : per-session calibration, T(C) = slope * V(mV) + intercept.
        When None, falls back to `config.slope` / `config.intercept` (which are
        Chandni's Physitemp defaults). Meiling's rigs pass rig-date-specific
        values here; see `MEILING_RIG_CALIBRATIONS` in core/config.py.

    Returns
    -------
    TemperatureResult with converted signal and landmarks.
    """
    if config is None:
        config = TemperatureConfig()
    if slope is None:
        slope = config.slope
    if intercept is None:
        intercept = config.intercept

    # Convert to millivolts: raw * bit_volts gives volts, * 1000 for mV
    voltage_mv = temperature_raw * temp_bit_volts * 1000.0

    # Linear calibration (per-session slope/intercept)
    temperature_c = slope * voltage_mv + intercept

    # Moving average smoothing
    if config.smoothing_window > 1 and len(temperature_c) >= config.smoothing_window:
        kernel = np.ones(config.smoothing_window) / config.smoothing_window
        temperature_smooth = np.convolve(temperature_c, kernel, mode="same")
    else:
        temperature_smooth = temperature_c.copy()

    # Mask raw-signal dropouts: thermistor disconnects produce brief rail
    # swings that survive smoothing as excursions far outside the mouse's
    # physiological range. NaN them so plots show gaps and downstream code
    # can skip them.
    PHYS_LO, PHYS_HI = 20.0, 44.0
    dropout_mask = (temperature_smooth < PHYS_LO) | (temperature_smooth > PHYS_HI)
    if dropout_mask.any():
        temperature_smooth = temperature_smooth.copy()
        temperature_smooth[dropout_mask] = np.nan

    # Landmarks (nan-aware so dropouts in the baseline window don't contaminate)
    baseline_n = int(config.baseline_duration_s * fs)
    baseline_n = min(baseline_n, len(temperature_smooth))
    baseline_temp = float(np.nanmean(temperature_smooth[:baseline_n]))

    max_idx = int(np.nanargmax(temperature_smooth))
    max_temp = float(temperature_smooth[max_idx])
    max_temp_time = max_idx / fs

    # Terminal temp: fall back to the last non-NaN sample if the trace
    # ends in a dropout.
    if np.isnan(temperature_smooth[-1]):
        valid = np.where(~np.isnan(temperature_smooth))[0]
        terminal_temp = float(temperature_smooth[valid[-1]]) if len(valid) else float("nan")
    else:
        terminal_temp = float(temperature_smooth[-1])

    return TemperatureResult(
        temperature_c=temperature_c,
        temperature_smooth=temperature_smooth,
        baseline_temp=baseline_temp,
        max_temp=max_temp,
        max_temp_time=max_temp_time,
        terminal_temp=terminal_temp,
    )


def detect_heating_start(
    temperature_smooth: np.ndarray,
    fs: float,
    derivative_threshold: float = 0.01,
    sustained_s: float = 5.0,
) -> Optional[float]:
    """Auto-detect heating start from sustained temperature rise.

    Finds the first time point where the temperature derivative
    exceeds the threshold for a sustained period.

    Parameters
    ----------
    temperature_smooth : smoothed temperature trace (Celsius)
    fs : sampling rate (Hz)
    derivative_threshold : minimum dT/dt (degrees C per second)
    sustained_s : how long the derivative must stay above threshold (seconds)

    Returns
    -------
    Time in seconds of heating onset, or None if not detected.
    """
    # Compute derivative (degrees C per second)
    dt = np.gradient(temperature_smooth, 1.0 / fs)

    sustained_n = int(sustained_s * fs)
    if sustained_n < 1:
        sustained_n = 1

    above = dt > derivative_threshold

    # Find first run of `sustained_n` consecutive True values
    count = 0
    for i in range(len(above)):
        if above[i]:
            count += 1
            if count >= sustained_n:
                # Onset is the start of this sustained run
                onset_idx = i - sustained_n + 1
                return onset_idx / fs
        else:
            count = 0

    return None


def temp_at_time(
    temperature_smooth: np.ndarray,
    fs: float,
    t: float,
) -> Optional[float]:
    """Look up temperature at a given time.

    Parameters
    ----------
    temperature_smooth : smoothed temperature trace (Celsius)
    fs : sampling rate (Hz)
    t : time in seconds

    Returns
    -------
    Temperature in Celsius, or None if t is out of range.
    """
    idx = int(t * fs)
    if idx < 0 or idx >= len(temperature_smooth):
        return None
    return float(temperature_smooth[idx])
