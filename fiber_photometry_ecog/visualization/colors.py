"""
Cohort and landmark display constants.

Internal keys (cohort names, landmark names) are unchanged across modules; only
the strings that appear on plots are defined here so the user-facing wording
can be tuned in one place.
"""

# Per-session trace colors
ECOG_COLOR = "dimgray"
TEMP_COLOR = "black"

# Cohort colors for group plots
COHORT_COLORS = {
    "seizure": "red",
    "failed_seizure": "purple",
    "wt": "blue",
}

# Display labels (what shows up on plots) — internal keys above are unchanged.
COHORT_DISPLAY_LABELS = {
    "seizure": "DS sz",
    "failed_seizure": "DS no sz",
    "wt": "WT",
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
    "eec": "earliest electrographic change",
    "ueo": "seizure onset",
    "behavioral_onset": "behavioral onset",
    "off": "seizure offset",
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


# Z-scored dF/F display string — single source of truth so wording changes
# propagate to every plot.
DFF_LABEL = "z_ΔF/F"
