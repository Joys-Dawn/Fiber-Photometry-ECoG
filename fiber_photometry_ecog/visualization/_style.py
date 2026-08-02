"""Shared plot style + canonical figure sizes for saved analysis plots.

The font sizes here are applied scoped (via matplotlib.rc_context) only to
the plot functions decorated with @styled - typically the public plot_*
entry points in group_plots.py and the matplotlib writers in trace_plots.py.
That keeps matplotlib figures embedded elsewhere (e.g. the Tk GUI) at
matplotlib defaults; only the saved PNG/HTML output gets the large fonts.

History: copied verbatim from the parent ephys project (47/40/32 for
label/tick/legend), halved to 24/24/20/16 because they overran the smaller
figsizes we render at, then doubled to 48/48/40/32 per user request once
figures were scaled up to match.

Canonical figure sizes group plots by shape so that everything in the same
"shape family" shares one aspect ratio:

  PANEL_W x PANEL_H  - per-panel size for 1-row multi-panel composites.
                       A 1xN figure is sized (PANEL_W*N, PANEL_H). 1.67:1.
  FIG_WIDE           - single-axes "wide rectangle" for time-series and
                       signal-vs-temperature plots (cooling curves, STA
                       traces, vs-temp, UEO standalones). 2:1.
  FIG_SCATTER        - single-axes cohort scatter (mean +/- SEM). Vertical
                       rectangle (tall + narrow) to mirror the slice-physiology
                       style; per-cohort width is added on top of this base by
                       `_save_standalone_scatter`.
  FIG_2X2            - 2x2 grid composite. 1.5:1.
"""

import functools


LABEL_SIZE = 48
TITLE_SIZE = 48
TICK_SIZE = 40
LEGEND_SIZE = 32

# Axis spine thickness (matplotlib default is 0.8). 3.0 reads cleanly at
# the 48pt/40pt font scale used for these plots.
AXIS_LINEWIDTH = 3.0

# Linewidth for event/landmark vertical lines (stim onset, UEO, etc.).
# Same scale as the axes for visual prominence — these mark the
# experimental anchor on every aligned-trace plot, so they should stand
# out, not blend into a grid.
EVENT_LINEWIDTH = 3.5

# Zero-reference lines are deliberately darker and slightly thinner than
# event lines. They should be easy to find without competing with the
# experimental landmark at x=0.
ZERO_LINE_COLOR = "#202020"
ZERO_LINEWIDTH = 2.5
ZERO_LINESTYLE = "--"

# Distance (points) between top of axes and the title. matplotlib default
# is 6, which sits the title right on top of the plot. Bumped to 30 so
# the title clears tick labels and stays readable when constrained_layout
# packs subplots tightly.
TITLE_PAD = 30

_PLOT_RC = {
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.titlepad": TITLE_PAD,
    "axes.linewidth": AXIS_LINEWIDTH,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    # Tick marks scaled to match the thicker spines.
    "xtick.major.width": AXIS_LINEWIDTH,
    "ytick.major.width": AXIS_LINEWIDTH,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": LABEL_SIZE,
    # Constrained layout dynamically pushes subplots apart so the oversized
    # 48pt labels can't overlap each other - tight_layout's heuristics give
    # up at these font sizes. Combined with savefig(bbox_inches="tight") this
    # handles both inter-subplot collisions and outer-edge clipping.
    #
    # Pads/spacings are tuned for 48pt fonts. h_pad/w_pad are in inches:
    # a 48pt line is ~0.67" tall, so 0.4" minimum keeps a row's xlabel from
    # touching the next row's title. h/wspace are fractions of subplot size.
    "figure.constrained_layout.use": True,
    # Pads bumped to handle 48pt labels + 40pt rotated tick labels +
    # thicker (3pt) axis spines. Smaller values let multi-line ylabels
    # collide with tick labels and let rotated x-tick labels run off the
    # figure even with bbox_inches="tight".
    "figure.constrained_layout.h_pad": 0.80,
    "figure.constrained_layout.w_pad": 0.80,
    "figure.constrained_layout.hspace": 0.15,
    "figure.constrained_layout.wspace": 0.15,
    # Outer padding for bbox_inches="tight". 0.5 wasn't enough — rotated
    # x-tick labels, multi-line ylabels, and the new lifted titles were
    # still getting clipped on the standalone scatter and trace plots.
    # 1.0 inch keeps everything inside the saved PNG.
    "savefig.pad_inches": 1.0,
}


def styled(fn):
    """Apply this package's plot style while `fn` runs.

    Scoped via matplotlib's rc_context so matplotlib figures created outside
    these decorated functions (e.g. embedded GUI panels) keep matplotlib's
    default rcParams instead of inheriting the oversized plot fonts.
    """
    import matplotlib.pyplot as plt

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_PLOT_RC):
            return fn(*args, **kwargs)
    return wrapper


def add_zero_y(ax, ensure_visible: bool = False) -> None:
    """Draw a dark dashed y=0 reference line.

    When ensure_visible is true, expand an all-positive or all-negative
    y-range just past zero. The small margin keeps the reference line inside
    the plotting area instead of hiding it under the bottom/top axis spine.
    """
    if ensure_visible:
        lo, hi = ax.get_ylim()
        if lo > 0:
            span = max(hi - lo, abs(hi), 1.0)
            ax.set_ylim(-0.04 * span, hi)
        elif hi < 0:
            span = max(hi - lo, abs(lo), 1.0)
            ax.set_ylim(lo, 0.04 * span)
    ax.axhline(
        0,
        color=ZERO_LINE_COLOR,
        linestyle=ZERO_LINESTYLE,
        linewidth=ZERO_LINEWIDTH,
        alpha=0.9,
        zorder=1,
    )


def add_zero_x(ax) -> None:
    """Draw a dark dashed x=0 reference line."""
    ax.axvline(
        0,
        color=ZERO_LINE_COLOR,
        linestyle=ZERO_LINESTYLE,
        linewidth=EVENT_LINEWIDTH,
        alpha=0.9,
        zorder=4,
    )


# ---- Canonical figure sizes (aspect-ratio groups) ----

PANEL_W = 12.0
PANEL_H = 9.0

FIG_WIDE = (18.0, 10.0)
FIG_SCATTER = (6.0, 12.0)
# Four panels need enough per-axis width for the publication-scale labels.
# At 22 x 16, the repeated time-axis labels could run into the neighboring
# panel or the saved-image edge even under constrained layout.
FIG_2X2 = (30.0, 20.0)


def panel_figsize(n: int) -> tuple:
    """Per-panel canonical size for 1-row multi-panel composites.

    Returns (PANEL_W * max(n, 1), PANEL_H) so every 1xN summary plot in the
    package shares the same per-panel aspect ratio.
    """
    return (PANEL_W * max(n, 1), PANEL_H)
