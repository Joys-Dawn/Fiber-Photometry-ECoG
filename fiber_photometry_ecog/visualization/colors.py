"""
Cohort and landmark display constants.

Internal keys (cohort names, landmark names) are unchanged across modules; only
the strings that appear on plots are defined here so the user-facing wording
can be tuned in one place.
"""

# Per-session trace colors
ECOG_COLOR = "dimgray"
TEMP_COLOR = "black"

# Cohort colors for group plots. Canonical genotype/seizure cohort_keys come
# from core.cohort.classify_session (DS_sz / DS_no_sz / WT_sz / WT_no_sz);
# opto experiment cohorts come from subfolder names in run_opto.py and are
# pinned explicitly here so paired baseline/treatment plots use distinct
# colors rather than both falling back to gray.
COHORT_COLORS = {
    "DS_sz": "red",
    "DS_no_sz": "purple",
    "WT_sz": "orange",      # NEW: WT with seizure (e.g. PTZ-induced)
    "WT_no_sz": "blue",
    "FCD_sz": "red",        # FCD seizure: red to match the canonical
                            # "seized" color used for DS_sz.
    "FCD_no_sz": "blue",
    # Dataset-specific pooled hyperthermia group keys. Individual points in
    # "No seizure" are overridden by source genotype in run_hyperthermia.py;
    # this fallback is used only if source metadata is unavailable.
    "Seizure": "red",
    "No seizure": "#555555",
    # LC baseline-transient genotype-only groups.
    "DS": "red",
    "WT": "blue",
    # PTZ pooled cohort (DS + WT combined) — the canonical view for PTZ
    # since the chemoconvulsant evokes seizures uniformly across genotypes.
    # Plotted in red because the only relevant sessions for this view are
    # seized ones (PTZ universally induces seizures).
    "PTZ_pooled": "red",
    # Opto cohorts render in black per spec — each panel shows a single
    # cohort so there's no overlap to distinguish. Affects both per-session
    # trace plots and the stim-aligned per-cohort mean plots.
    "10HZ_DG": "black",
    "10HZ_DG_CPP": "black",
    "1HZ_DG": "black",
    "20HZ_DG": "black",
    "20HZ_DG_Atomoxetine": "black",
    "20HZ_DG_mcherry": "black",
    "20HZ_LC": "black",
}

# Fallback palette for unknown cohort keys (any subfolder name we haven't
# explicitly pinned above). Hash-indexed in _color() so the same cohort
# string always lands on the same color across runs and plots.
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

    Pinned keys in `COHORT_COLORS` win. Anything else gets a stable
    palette color (deterministic per cohort key) instead of plain gray —
    so paired opto cohorts (e.g. ``10HZ_DG`` vs ``10HZ_DG_CPP``) never
    end up sharing the gray fallback.
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

# Display labels (what shows up on plots) - internal keys above are unchanged.
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
    # Opto cohorts: spaces in for readability, "10 Hz DG + CPP" line-broken
    # so it fits FIG_SCATTER at 40pt tick size.
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
    # 2-line for "earliest electrographic change" so the 48pt title fits
    # inside its column instead of overflowing into the next subplot.
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


# Z-scored dF/F display string - single source of truth so wording changes
# propagate to every plot.
DFF_LABEL = "z_ΔF/F"
