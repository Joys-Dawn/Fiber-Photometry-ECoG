"""Cohort and landmark display constants."""

# Per-session trace colors
ECOG_COLOR = "dimgray"
TEMP_COLOR = "black"

COHORT_COLORS = {
    "DS_sz": "red",
    "DS_no_sz": "purple",
    "WT_sz": "orange",
    "WT_no_sz": "blue",
    "FCD_sz": "red",
    "FCD_no_sz": "blue",
    # Dataset-specific pooled hyperthermia group keys. Individual points in
    # "No seizure" are overridden by source genotype in run_hyperthermia.py;
    # this fallback is used only if source metadata is unavailable.
    "Seizure": "red",
    "No seizure": "#555555",
    # LC baseline-transient genotype-only groups.
    "DS": "red",
    "WT": "blue",
    # Pooled PTZ cohort, combining DS and WT.
    "PTZ_pooled": "red",
    # Black is used for single-burst optogenetic traces and cohort means.
    "10HZ_DG": "black",
    "10HZ_DG_CPP": "black",
    "1HZ_DG": "black",
    "20HZ_DG": "black",
    "20HZ_DG_Atomoxetine": "black",
    "20HZ_DG_mcherry": "black",
    "20HZ_LC": "black",
}

# Fallback palette for unconfigured cohort keys.
_FALLBACK_PALETTE = (
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
)


def cohort_color(cohort: str) -> str:
    """Resolve a cohort key to a plot color.

    Use `COHORT_COLORS` for configured keys and a deterministic palette
    selection for other keys.
    """
    if cohort in COHORT_COLORS:
        return COHORT_COLORS[cohort]
    if not _FALLBACK_PALETTE:
        return "gray"
    # `hash()` is salted per Python process; use a stable fold instead so
    # the assignment doesn't drift across runs.
    h = 0
    for ch in cohort:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return _FALLBACK_PALETTE[h % len(_FALLBACK_PALETTE)]

COHORT_DISPLAY_LABELS = {
    "DS_sz": "DS sz",
    "DS_no_sz": "DS no sz",
    "WT_sz": "WT sz",
    "WT_no_sz": "WT",
    "FCD_sz": "FCD sz",
    "FCD_no_sz": "FCD no sz",
    "Seizure": "Seizure",
    "No seizure": "No seizure",
    "DS": "DS",
    "WT": "WT",
    "PTZ_pooled": "PTZ pooled",
# Optogenetic cohort display labels.
    "10HZ_DG": "10 Hz DG",
    "10HZ_DG_CPP": "10 Hz DG\n+ CPP",
    "1HZ_DG": "1 Hz DG",
    "20HZ_DG": "20 Hz DG",
    "20HZ_DG_Atomoxetine": "20 Hz DG\n+ Atomoxetine",
    "20HZ_DG_mcherry": "20 Hz DG\nmCherry",
    "20HZ_LC": "20 Hz LC",
}

# Landmark colors (consistent across per-session and group plots).
LANDMARK_COLORS = {
    "heating_start": "green",
    "eec": "orange",
    "ueo": "red",
    "behavioral_onset": "purple",
    "off": "blue",
}

# Display labels for landmarks on axes and legends.
LANDMARK_DISPLAY_LABELS = {
    "heating_start": "heating start",
    "eec": "earliest\nelectrographic change",
    "ueo": "seizure onset",
    "behavioral_onset": "behavioral onset",
    "off": "seizure offset",
    "max_temp": "max temp",
}


def landmark_label(key: str) -> str:
    """Map a landmark key (any case, with or without _time suffix) to its display label.

    Accepts forms like "UEO", "ueo", "ueo_time", "behavioral_onset", "Behavioral onset".
    Falls back to the original string if no mapping is found.
    """
    if not key:
        return key
    norm = key.strip().lower().replace(" ", "_")
    if norm.endswith("_time"):
        norm = norm[:-5]
    return LANDMARK_DISPLAY_LABELS.get(norm, key)


DFF_LABEL = "z_ΔF/F"
