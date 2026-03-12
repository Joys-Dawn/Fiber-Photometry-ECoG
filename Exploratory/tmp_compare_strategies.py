"""Temp script: compare all 3 photometry strategies overlaid on the same axis.

Run from repo root:
    python tmp_compare_strategies.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fiber_photometry_ecog.data_loading.ppd_reader import read_ppd
from fiber_photometry_ecog.preprocessing.photometry.strategy_a_chandni import ChandniStrategy
from fiber_photometry_ecog.preprocessing.photometry.strategy_b_meiling import MeilingStrategy
from fiber_photometry_ecog.preprocessing.photometry.strategy_c_irls import IRLSStrategy

# --- GRABne files only ---
DATA_ROOT = Path("test_data/Meiling_FiberPhotometry/GRABne")
ppd_files = sorted(DATA_ROOT.rglob("*.ppd"))

MAX_FILES = 12
ppd_files = ppd_files[:MAX_FILES]

if not ppd_files:
    raise FileNotFoundError(f"No .ppd files found under {DATA_ROOT}")

print(f"Found {len(ppd_files)} files to process")

strategies = {
    "A (Chandni)": ChandniStrategy(),
    "B (Meiling)": MeilingStrategy(),
    "C (IRLS/Keevers)": IRLSStrategy(),
}
colors = {
    "A (Chandni)": "#1f77b4",
    "B (Meiling)": "#ff7f0e",
    "C (IRLS/Keevers)": "#2ca02c",
}

for ppd_path in ppd_files:
    print(f"\n{'='*60}")
    print(f"File: {ppd_path.relative_to(DATA_ROOT)}")
    print(f"{'='*60}")

    data = read_ppd(ppd_path)
    print(f"  fs={data.fs} Hz, duration={data.time[-1]:.1f}s, n_samples={len(data.signal_470)}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})

    # Top: raw signals
    axes[0].plot(data.time, data.signal_470, linewidth=0.3, alpha=0.8, label="470 (signal)")
    axes[0].plot(data.time, data.signal_405, linewidth=0.3, alpha=0.8, label="405 (iso)")
    axes[0].set_ylabel("Volts")
    axes[0].set_title(f"Raw — {ppd_path.name}")
    axes[0].legend(loc="upper right", fontsize=8)

    # Bottom: all 4 strategies overlaid
    for name, strat in strategies.items():
        try:
            result = strat.preprocess(data.signal_470, data.signal_405, data.fs)
            dff = result.dff
            axes[1].plot(data.time, dff, linewidth=0.5, alpha=0.8,
                         color=colors[name], label=name)
            print(f"  {name}: mean={np.mean(dff):.4f}, std={np.std(dff):.4f}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    axes[1].set_ylabel("dF/F")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("All strategies overlaid")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"tmp_strategy_compare_{ppd_path.stem}.png", dpi=150)
    print(f"  Saved: tmp_strategy_compare_{ppd_path.stem}.png")

plt.close("all")
print("\nDone.")
