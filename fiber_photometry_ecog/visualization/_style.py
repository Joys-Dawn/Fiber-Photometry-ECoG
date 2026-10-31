"""Shared styles and dimensions for exported figures."""

import functools

from matplotlib.ticker import FixedLocator, MaxNLocator


LABEL_SIZE = 60
TITLE_SIZE = 60
TICK_SIZE = 52
LEGEND_SIZE = 32
MAX_NUMERIC_TICK_INTERVALS = 5

AXIS_LINEWIDTH = 3.0

# Vertical event and landmark lines.
EVENT_LINE_COLOR = "black"
EVENT_LINEWIDTH = 3.5

# Horizontal zero-reference lines.
ZERO_LINE_COLOR = "#202020"
ZERO_LINEWIDTH = 2.5
ZERO_LINESTYLE = "--"

# Title padding for quality-check and parameter-detection plots.
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
    # Constrained layout reserves space for axis decorations.
    # h_pad/w_pad are inches; hspace/wspace are fractions of subplot size.
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.80,
    "figure.constrained_layout.w_pad": 0.80,
    "figure.constrained_layout.hspace": 0.15,
    "figure.constrained_layout.wspace": 0.15,
    # Outer padding, in inches, for bbox_inches="tight".
    "savefig.pad_inches": 1.0,
}


def styled(fn):
    """Apply this package's plot style while `fn` runs.

    rc_context restores the caller's rcParams afterward.
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
        color=EVENT_LINE_COLOR,
        linestyle=ZERO_LINESTYLE,
        linewidth=EVENT_LINEWIDTH,
        alpha=1.0,
        zorder=4,
    )


def sparsify_numeric_ticks(ax) -> None:
    """Use fewer ticks on continuous axes while preserving fixed labels."""
    for axis in (ax.xaxis, ax.yaxis):
        if isinstance(axis.get_major_locator(), FixedLocator):
            continue
        axis.set_major_locator(
            MaxNLocator(nbins=MAX_NUMERIC_TICK_INTERVALS, min_n_ticks=3)
        )


def finalize_publication_figure(fig) -> None:
    """Apply the title-free, legend-free publication output policy."""
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        sparsify_numeric_ticks(ax)


def fix_axes_size(ax, size_inches, *, colorbar=None) -> None:
    """Fix a standalone data rectangle; tight export accommodates its labels."""
    fig = ax.figure
    fig.set_layout_engine("none")
    width, height = size_inches
    fig_width, fig_height = width + 5.0, height + 4.0
    fig.set_size_inches(fig_width, fig_height)
    ax.set_position((2.5 / fig_width, 2.5 / fig_height,
                     width / fig_width, height / fig_height))
    if colorbar is not None:
        colorbar.ax.set_axes_locator(None)
        colorbar.ax.set_box_aspect(None)
        colorbar.ax.set_position(((2.5 + width + 0.3) / fig_width,
                                 2.5 / fig_height, 0.3 / fig_width,
                                 height / fig_height))


# ---- Canonical figure sizes (aspect-ratio groups) ----

PANEL_W = 12.0
PANEL_H = 9.0

FIG_WIDE = (18.0, 10.0)
FIG_SCATTER = (8.0, 12.0)
FIG_2X2 = (30.0, 20.0)


def panel_figsize(n: int) -> tuple:
    """Per-panel canonical size for 1-row multi-panel composites.

    Returns (PANEL_W * max(n, 1), PANEL_H) so every 1xN summary plot in the
    package shares the same per-panel aspect ratio.
    """
    return (PANEL_W * max(n, 1), PANEL_H)
