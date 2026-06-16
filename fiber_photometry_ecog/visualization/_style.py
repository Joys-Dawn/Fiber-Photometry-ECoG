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
  FIG_SCATTER        - single-axes cohort scatter (mean +/- SEM). 1.33:1.
  FIG_2X2            - 2x2 grid composite. 1.5:1.
"""

import functools


LABEL_SIZE = 48
TITLE_SIZE = 48
TICK_SIZE = 40
LEGEND_SIZE = 32

_PLOT_RC = {
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
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
    "figure.constrained_layout.h_pad": 0.40,
    "figure.constrained_layout.w_pad": 0.40,
    "figure.constrained_layout.hspace": 0.10,
    "figure.constrained_layout.wspace": 0.10,
    # Outer padding for bbox_inches="tight" - default 0.1 leaves no room
    # for overhanging rotated labels when constrained_layout pushes them
    # outside the figure. 0.5 inch keeps the full label intact.
    "savefig.pad_inches": 0.5,
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


# ---- Canonical figure sizes (aspect-ratio groups) ----

PANEL_W = 12.0
PANEL_H = 9.0

FIG_WIDE = (18.0, 10.0)
FIG_SCATTER = (10.0, 10.0)
FIG_2X2 = (22.0, 16.0)


def panel_figsize(n: int) -> tuple:
    """Per-panel canonical size for 1-row multi-panel composites.

    Returns (PANEL_W * max(n, 1), PANEL_H) so every 1xN summary plot in the
    package shares the same per-panel aspect ratio.
    """
    return (PANEL_W * max(n, 1), PANEL_H)
