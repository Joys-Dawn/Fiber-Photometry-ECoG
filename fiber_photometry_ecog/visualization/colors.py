"""
Cohort color scheme constants per the Excel spec.
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
