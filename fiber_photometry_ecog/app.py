"""
Fiber Photometry & ECoG Analysis Tool — Tkinter GUI.

Three-tab interface following the EphysAutomatedAnalysis pattern:
  Tab 1 — Data Loading & Cohort Management
  Tab 2 — Preprocessing
  Tab 3 — Extraction
"""

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import numpy as np

from .core.config import (
    AnalysisConfig,
    ECoGConfig,
    PhotometryConfig,
    PreprocessingConfig,
    SpikeDetectionConfig,
    TransientConfig,
)
from .core.data_models import (
    ProcessedData,
    RawData,
    Session,
    SessionLandmarks,
)
from .data_loading import read_ppd, read_oep, synchronize
from .preprocessing import filter_ecog, process_temperature, detect_transients, detect_spikes
from .preprocessing.photometry import (
    ChandniStrategy,
    MeilingStrategy,
    IRLSStrategy,
    z_score_baseline,
    highpass_filter,
)
from .pairing.engine import assign_all_controls
from .analysis import (
    compute_cohort_characteristics,
    compute_baseline_transients,
    compute_preictal_mean,
    compute_preictal_transients,
    compute_ictal_mean,
    compute_ictal_transients,
    compute_postictal,
    compute_spike_triggered_average,
)
from .visualization import (
    plot_sanity_check,
    plot_zoomed,
    plot_cohort_characteristics,
    plot_baseline_transients,
    plot_preictal_mean,
    plot_preictal_transients,
    plot_ictal_mean,
    plot_ictal_transients,
    plot_postictal,
    plot_spike_triggered,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "A": ChandniStrategy,
    "B": MeilingStrategy,
    "C": IRLSStrategy,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class GUIOutputCapture:
    """Captures stdout/stderr and redirects to a GUI text widget."""

    def __init__(self, text_widget: tk.Text, root: tk.Tk):
        self.text_widget = text_widget
        self.root = root

    def write(self, text: str) -> int:
        if text:
            self.root.after(0, lambda t=text: self._add_text(t))
        return len(text)

    def flush(self) -> None:
        pass

    def _add_text(self, text: str) -> None:
        self.text_widget.config(state="normal")
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.config(state="disabled")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class FiberPhotometryApp:
    """Main Tkinter application for Fiber Photometry & ECoG analysis."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Fiber Photometry & ECoG Analysis")
        self.root.geometry("1100x750")

        # Pipeline state
        self.sessions: List[Session] = []
        self.preprocessing_config = PreprocessingConfig()
        self.analysis_config = AnalysisConfig()
        self.output_dir = Path("Results")
        self.running = False

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        notebook = ttk.Notebook(self.root)

        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Data Loading & Cohort Management")
        self._setup_loading_tab(tab1)

        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Preprocessing")
        self._setup_preprocessing_tab(tab2)

        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text="Extraction")
        self._setup_extraction_tab(tab3)

        notebook.pack(expand=True, fill="both", padx=10, pady=10)

    # ==================================================================
    # TAB 1 — Data Loading & Cohort Management
    # ==================================================================

    def _setup_loading_tab(self, parent: ttk.Frame) -> None:
        # --- Add session section ---
        add_frame = ttk.LabelFrame(parent, text="Add Session", padding=10)
        add_frame.pack(fill="x", padx=5, pady=5)

        # Row 1: PPD file
        row1 = ttk.Frame(add_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="PPD file:").pack(side="left")
        self.ppd_path_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.ppd_path_var, width=70).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row1, text="Browse...", command=self._browse_ppd).pack(side="right")

        # Row 2: OEP folder
        row2 = ttk.Frame(add_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="OEP folder:").pack(side="left")
        self.oep_path_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.oep_path_var, width=70).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row2, text="Browse...", command=self._browse_oep).pack(side="right")

        # Row 3: Metadata
        meta_frame = ttk.Frame(add_frame)
        meta_frame.pack(fill="x", pady=5)

        ttk.Label(meta_frame, text="Mouse ID:").pack(side="left")
        self.mouse_id_var = tk.StringVar()
        ttk.Entry(meta_frame, textvariable=self.mouse_id_var, width=12).pack(side="left", padx=(2, 10))

        ttk.Label(meta_frame, text="Genotype:").pack(side="left")
        self.genotype_var = tk.StringVar(value="Scn1a")
        ttk.Combobox(meta_frame, textvariable=self.genotype_var, values=["Scn1a", "WT"],
                      state="readonly", width=8).pack(side="left", padx=(2, 10))

        ttk.Label(meta_frame, text="Cohort:").pack(side="left")
        self.cohort_var = tk.StringVar(value="seizure")
        ttk.Combobox(meta_frame, textvariable=self.cohort_var,
                      values=["seizure", "failed_seizure", "wt"],
                      state="readonly", width=14).pack(side="left", padx=(2, 10))

        ttk.Label(meta_frame, text="Heating #:").pack(side="left")
        self.heating_session_var = tk.StringVar(value="1")
        ttk.Entry(meta_frame, textvariable=self.heating_session_var, width=4).pack(side="left", padx=(2, 10))

        ttk.Label(meta_frame, text="# Seizures:").pack(side="left")
        self.n_seizures_var = tk.StringVar(value="0")
        ttk.Entry(meta_frame, textvariable=self.n_seizures_var, width=4).pack(side="left", padx=(2, 10))

        # Row 4: Landmark times (user-entered)
        lm_frame = ttk.LabelFrame(add_frame, text="Landmark Times (seconds from recording start)", padding=5)
        lm_frame.pack(fill="x", pady=5)

        self.heating_start_var = tk.StringVar()
        self.eec_var = tk.StringVar()
        self.ueo_var = tk.StringVar()
        self.behav_var = tk.StringVar()
        self.off_var = tk.StringVar()

        for label, var in [
            ("Heating start:", self.heating_start_var),
            ("EEC:", self.eec_var),
            ("UEO:", self.ueo_var),
            ("Behavioral onset:", self.behav_var),
            ("OFF:", self.off_var),
        ]:
            ttk.Label(lm_frame, text=label).pack(side="left")
            ttk.Entry(lm_frame, textvariable=var, width=8).pack(side="left", padx=(2, 10))

        # Add button
        ttk.Button(add_frame, text="Load & Add Session", command=self._load_session).pack(pady=8)

        # --- Session list ---
        list_frame = ttk.LabelFrame(parent, text="Loaded Sessions", padding=10)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        cols = ("mouse_id", "genotype", "cohort", "n_seizures", "status")
        self.session_tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=8)
        for col, w in zip(cols, [100, 80, 120, 80, 120]):
            self.session_tree.heading(col, text=col.replace("_", " ").title())
            self.session_tree.column(col, width=w, anchor="center")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.session_tree.yview)
        self.session_tree.configure(yscrollcommand=scrollbar.set)
        self.session_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Buttons below session list
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_session).pack(side="left", padx=5)

        # Output dir
        out_frame = ttk.Frame(parent)
        out_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(out_frame, text="Output directory:").pack(side="left")
        self.output_dir_var = tk.StringVar(value="Results")
        ttk.Entry(out_frame, textvariable=self.output_dir_var, width=40).pack(side="left", padx=5)
        ttk.Button(out_frame, text="Browse...", command=self._browse_output_dir).pack(side="left")

        # Log area
        log_frame = ttk.LabelFrame(parent, text="Log", padding=5)
        log_frame.pack(fill="x", padx=5, pady=5)
        self.loading_log = tk.Text(log_frame, height=4, state="disabled", font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.loading_log.yview)
        self.loading_log.configure(yscrollcommand=log_scroll.set)
        self.loading_log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    # ==================================================================
    # TAB 2 — Preprocessing
    # ==================================================================

    def _setup_preprocessing_tab(self, parent: ttk.Frame) -> None:
        # --- Photometry strategy + params ---
        phot_frame = ttk.LabelFrame(parent, text="Photometry Parameters", padding=10)
        phot_frame.pack(fill="x", padx=5, pady=5)

        row0 = ttk.Frame(phot_frame)
        row0.pack(fill="x", pady=2)
        ttk.Label(row0, text="Strategy:").pack(side="left")
        self.strategy_var = tk.StringVar(value="A")
        ttk.Combobox(row0, textvariable=self.strategy_var,
                      values=["A", "B", "C"], state="readonly", width=5).pack(side="left", padx=5)
        ttk.Label(row0, text="(A=Chandni, B=Meiling, C=IRLS/Keevers)").pack(side="left")

        # Strategy-specific params
        param_row = ttk.Frame(phot_frame)
        param_row.pack(fill="x", pady=2)

        ttk.Label(param_row, text="Gaussian sigma (A):").pack(side="left")
        self.gaussian_sigma_var = tk.StringVar(value="75")
        ttk.Entry(param_row, textvariable=self.gaussian_sigma_var, width=6).pack(side="left", padx=(2, 10))

        ttk.Label(param_row, text="LP cutoff B (Hz):").pack(side="left")
        self.lp_cutoff_b_var = tk.StringVar(value="10.0")
        ttk.Entry(param_row, textvariable=self.lp_cutoff_b_var, width=6).pack(side="left", padx=(2, 10))

        ttk.Label(param_row, text="LP cutoff C (Hz):").pack(side="left")
        self.lp_cutoff_c_var = tk.StringVar(value="3.0")
        ttk.Entry(param_row, textvariable=self.lp_cutoff_c_var, width=6).pack(side="left", padx=(2, 10))

        ttk.Label(param_row, text="IRLS c:").pack(side="left")
        self.irls_c_var = tk.StringVar(value="1.4")
        ttk.Entry(param_row, textvariable=self.irls_c_var, width=6).pack(side="left", padx=(2, 10))

        # Post-processing
        post_row = ttk.Frame(phot_frame)
        post_row.pack(fill="x", pady=2)
        ttk.Label(post_row, text="HPF cutoff (Hz):").pack(side="left")
        self.hpf_cutoff_var = tk.StringVar(value="0.01")
        ttk.Entry(post_row, textvariable=self.hpf_cutoff_var, width=6).pack(side="left", padx=(2, 10))

        # --- ECoG params ---
        ecog_frame = ttk.LabelFrame(parent, text="ECoG Filter Parameters", padding=10)
        ecog_frame.pack(fill="x", padx=5, pady=5)

        ecog_row = ttk.Frame(ecog_frame)
        ecog_row.pack(fill="x")
        ttk.Label(ecog_row, text="Bandpass (Hz):").pack(side="left")
        self.bp_low_var = tk.StringVar(value="1.0")
        ttk.Entry(ecog_row, textvariable=self.bp_low_var, width=6).pack(side="left", padx=2)
        ttk.Label(ecog_row, text="–").pack(side="left")
        self.bp_high_var = tk.StringVar(value="70.0")
        ttk.Entry(ecog_row, textvariable=self.bp_high_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(ecog_row, text="Notch (Hz):").pack(side="left")
        self.notch_var = tk.StringVar(value="60.0")
        ttk.Entry(ecog_row, textvariable=self.notch_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(ecog_row, text="Notch Q:").pack(side="left")
        self.notch_q_var = tk.StringVar(value="30.0")
        ttk.Entry(ecog_row, textvariable=self.notch_q_var, width=6).pack(side="left", padx=2)

        # --- Transient detection params ---
        trans_frame = ttk.LabelFrame(parent, text="Transient Detection Parameters", padding=10)
        trans_frame.pack(fill="x", padx=5, pady=5)

        trans_row = ttk.Frame(trans_frame)
        trans_row.pack(fill="x")
        ttk.Label(trans_row, text="Method:").pack(side="left")
        self.transient_method_var = tk.StringVar(value="prominence")
        ttk.Combobox(trans_row, textvariable=self.transient_method_var,
                      values=["prominence", "mad"], state="readonly", width=10).pack(side="left", padx=(2, 10))
        ttk.Label(trans_row, text="Min prominence:").pack(side="left")
        self.min_prominence_var = tk.StringVar(value="1.0")
        ttk.Entry(trans_row, textvariable=self.min_prominence_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(trans_row, text="Max width (s):").pack(side="left")
        self.max_width_var = tk.StringVar(value="8.0")
        ttk.Entry(trans_row, textvariable=self.max_width_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(trans_row, text="MAD k:").pack(side="left")
        self.mad_k_var = tk.StringVar(value="3.0")
        ttk.Entry(trans_row, textvariable=self.mad_k_var, width=6).pack(side="left", padx=2)

        # --- Spike detection params ---
        spike_frame = ttk.LabelFrame(parent, text="Spike Detection Parameters", padding=10)
        spike_frame.pack(fill="x", padx=5, pady=5)

        spike_row = ttk.Frame(spike_frame)
        spike_row.pack(fill="x")
        ttk.Label(spike_row, text="Threshold mul:").pack(side="left")
        self.tmul_var = tk.StringVar(value="3.0")
        ttk.Entry(spike_row, textvariable=self.tmul_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(spike_row, text="Abs threshold:").pack(side="left")
        self.abs_thresh_var = tk.StringVar(value="0.4")
        ttk.Entry(spike_row, textvariable=self.abs_thresh_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(spike_row, text="Duration (ms):").pack(side="left")
        self.spk_min_var = tk.StringVar(value="70")
        ttk.Entry(spike_row, textvariable=self.spk_min_var, width=5).pack(side="left", padx=2)
        ttk.Label(spike_row, text="–").pack(side="left")
        self.spk_max_var = tk.StringVar(value="200")
        ttk.Entry(spike_row, textvariable=self.spk_max_var, width=5).pack(side="left", padx=(2, 10))

        # --- Run button + progress ---
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=5, pady=10)
        ttk.Button(btn_frame, text="Run Preprocessing", command=self._run_preprocessing, width=25).pack(anchor="center")

        self.preproc_progress = ttk.Progressbar(parent, mode="determinate", length=500)
        self.preproc_progress.pack(pady=5)
        self.preproc_progress_label = ttk.Label(parent, text="Ready")
        self.preproc_progress_label.pack()

        # Log
        log_frame = ttk.LabelFrame(parent, text="Preprocessing Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.preproc_log = tk.Text(log_frame, height=6, state="disabled", font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.preproc_log.yview)
        self.preproc_log.configure(yscrollcommand=log_scroll.set)
        self.preproc_log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    # ==================================================================
    # TAB 3 — Extraction
    # ==================================================================

    def _setup_extraction_tab(self, parent: ttk.Frame) -> None:
        # --- Pairing mode ---
        pair_frame = ttk.LabelFrame(parent, text="Control Pairing", padding=10)
        pair_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(pair_frame, text="Pairing mode:").pack(side="left")
        self.pairing_mode_var = tk.StringVar(value="temperature")
        ttk.Combobox(pair_frame, textvariable=self.pairing_mode_var,
                      values=["temperature", "time"], state="readonly", width=14).pack(side="left", padx=5)

        # --- Extraction params ---
        params_frame = ttk.LabelFrame(parent, text="Extraction Parameters", padding=10)
        params_frame.pack(fill="x", padx=5, pady=5)

        row1 = ttk.Frame(params_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Temp bin (°C):").pack(side="left")
        self.temp_bin_var = tk.StringVar(value="1.0")
        ttk.Entry(row1, textvariable=self.temp_bin_var, width=6).pack(side="left", padx=(2, 10))

        ttk.Label(row1, text="Triggered window (s):").pack(side="left")
        self.trig_window_var = tk.StringVar(value="30.0")
        ttk.Entry(row1, textvariable=self.trig_window_var, width=6).pack(side="left", padx=(2, 10))

        ttk.Label(row1, text="PSTH bin (s):").pack(side="left")
        self.psth_bin_var = tk.StringVar(value="10.0")
        ttk.Entry(row1, textvariable=self.psth_bin_var, width=6).pack(side="left", padx=(2, 10))

        ttk.Label(row1, text="PSTH window (s):").pack(side="left")
        self.psth_window_var = tk.StringVar(value="60.0")
        ttk.Entry(row1, textvariable=self.psth_window_var, width=6).pack(side="left", padx=(2, 10))

        row2 = ttk.Frame(params_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Moving avg window (s):").pack(side="left")
        self.ma_window_var = tk.StringVar(value="30.0")
        ttk.Entry(row2, textvariable=self.ma_window_var, width=6).pack(side="left", padx=(2, 10))

        ttk.Label(row2, text="Moving avg step (s):").pack(side="left")
        self.ma_step_var = tk.StringVar(value="5.0")
        ttk.Entry(row2, textvariable=self.ma_step_var, width=6).pack(side="left", padx=(2, 10))

        ttk.Label(row2, text="Pre-ictal temp range (°C):").pack(side="left")
        self.preictal_range_var = tk.StringVar(value="10.0")
        ttk.Entry(row2, textvariable=self.preictal_range_var, width=6).pack(side="left", padx=(2, 10))

        ttk.Label(row2, text="Spike trig window (s):").pack(side="left")
        self.spike_trig_var = tk.StringVar(value="30.0")
        ttk.Entry(row2, textvariable=self.spike_trig_var, width=6).pack(side="left", padx=2)

        # --- Run button + progress ---
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=5, pady=10)
        ttk.Button(btn_frame, text="Run Extraction", command=self._run_extraction, width=25).pack(anchor="center")

        self.extract_progress = ttk.Progressbar(parent, mode="determinate", length=500)
        self.extract_progress.pack(pady=5)
        self.extract_progress_label = ttk.Label(parent, text="Ready")
        self.extract_progress_label.pack()

        # Log
        log_frame = ttk.LabelFrame(parent, text="Extraction Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.extract_log = tk.Text(log_frame, height=10, state="disabled", font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.extract_log.yview)
        self.extract_log.configure(yscrollcommand=log_scroll.set)
        self.extract_log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    # ------------------------------------------------------------------
    # Tab 1 actions
    # ------------------------------------------------------------------

    def _browse_ppd(self) -> None:
        path = filedialog.askopenfilename(title="Select PPD file", filetypes=[("PPD files", "*.ppd")])
        if path:
            self.ppd_path_var.set(path)

    def _browse_oep(self) -> None:
        path = filedialog.askdirectory(title="Select Open Ephys recording folder")
        if path:
            self.oep_path_var.set(path)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir_var.set(path)

    def _log(self, widget: tk.Text, msg: str) -> None:
        widget.config(state="normal")
        widget.insert(tk.END, msg + "\n")
        widget.see(tk.END)
        widget.config(state="disabled")

    def _log_loading(self, msg: str) -> None:
        self.root.after(0, lambda: self._log(self.loading_log, msg))

    def _log_preproc(self, msg: str) -> None:
        self.root.after(0, lambda: self._log(self.preproc_log, msg))

    def _log_extract(self, msg: str) -> None:
        self.root.after(0, lambda: self._log(self.extract_log, msg))

    def _parse_float_or_none(self, var: tk.StringVar) -> Optional[float]:
        val = var.get().strip()
        if not val:
            return None
        return float(val)

    def _load_session(self) -> None:
        """Load a single session from PPD + OEP files and add to session list."""
        ppd_path = self.ppd_path_var.get().strip()
        oep_path = self.oep_path_var.get().strip()
        mouse_id = self.mouse_id_var.get().strip()

        if not ppd_path or not oep_path or not mouse_id:
            messagebox.showerror("Error", "PPD file, OEP folder, and Mouse ID are required.")
            return

        heating_start = self._parse_float_or_none(self.heating_start_var)
        if heating_start is None:
            messagebox.showerror("Error", "Heating start time is required.")
            return

        self._log_loading(f"Loading {mouse_id}...")

        def worker():
            try:
                ppd = read_ppd(ppd_path)
                self._log_loading(f"  PPD loaded: {len(ppd.signal_470)} samples, fs={ppd.fs} Hz")

                oep = read_oep(oep_path)
                self._log_loading(f"  OEP loaded: {len(oep.ecog)} samples, fs={oep.fs_ecog} Hz")

                sync = synchronize(ppd, oep)
                self._log_loading(
                    f"  Sync: {sync.n_matched} pulses matched, "
                    f"drift={sync.drift_ppm:.1f} ppm, residual={sync.residual_ms:.3f} ms"
                )

                landmarks = SessionLandmarks(
                    heating_start_time=heating_start,
                    eec_time=self._parse_float_or_none(self.eec_var),
                    ueo_time=self._parse_float_or_none(self.ueo_var),
                    behavioral_onset_time=self._parse_float_or_none(self.behav_var),
                    off_time=self._parse_float_or_none(self.off_var),
                )

                n_seizures = int(self.n_seizures_var.get())
                cohort = self.cohort_var.get()

                raw = RawData(
                    signal_470=sync.signal_470,
                    signal_405=sync.signal_405,
                    ecog=sync.ecog,
                    emg=sync.emg,
                    temperature_raw=sync.temperature_raw,
                    temp_bit_volts=sync.temp_bit_volts,
                    time=sync.time,
                    fs=sync.fs,
                )

                session = Session(
                    mouse_id=mouse_id,
                    genotype=self.genotype_var.get(),
                    heating_session=int(self.heating_session_var.get()),
                    n_seizures=n_seizures,
                    survived=True,
                    experiment_label="",
                    landmarks=landmarks,
                    raw=raw,
                )

                self.sessions.append(session)

                # Update treeview
                self.root.after(0, lambda: self.session_tree.insert(
                    "", "end", values=(mouse_id, self.genotype_var.get(), cohort, n_seizures, "loaded"),
                ))
                self._log_loading(f"  {mouse_id} loaded successfully.")

            except Exception as e:
                self._log_loading(f"  ERROR loading {mouse_id}: {e}")
                self.root.after(0, lambda err=e: messagebox.showerror("Load Error", str(err)))

        threading.Thread(target=worker, daemon=True).start()

    def _remove_session(self) -> None:
        selected = self.session_tree.selection()
        if not selected:
            return
        for item in selected:
            idx = self.session_tree.index(item)
            self.session_tree.delete(item)
            if idx < len(self.sessions):
                self.sessions.pop(idx)

    # ------------------------------------------------------------------
    # Tab 2 actions
    # ------------------------------------------------------------------

    def _read_preprocessing_config(self) -> PreprocessingConfig:
        """Build a PreprocessingConfig from the GUI fields."""
        return PreprocessingConfig(
            ecog=ECoGConfig(
                bandpass_low=float(self.bp_low_var.get()),
                bandpass_high=float(self.bp_high_var.get()),
                notch_freq=float(self.notch_var.get()),
                notch_q=float(self.notch_q_var.get()),
            ),
            photometry=PhotometryConfig(
                strategy=self.strategy_var.get(),
                gaussian_sigma=int(self.gaussian_sigma_var.get()),
                lowpass_cutoff_b=float(self.lp_cutoff_b_var.get()),
                lowpass_cutoff_c=float(self.lp_cutoff_c_var.get()),
                irls_tuning_c=float(self.irls_c_var.get()),
                hpf_cutoff=float(self.hpf_cutoff_var.get()),
            ),
            transient=TransientConfig(
                method=self.transient_method_var.get(),
                min_prominence=float(self.min_prominence_var.get()),
                max_width_s=float(self.max_width_var.get()),
                mad_k=float(self.mad_k_var.get()),
            ),
            spike_detection=SpikeDetectionConfig(
                tmul=float(self.tmul_var.get()),
                abs_threshold=float(self.abs_thresh_var.get()),
                spkdur_min_ms=float(self.spk_min_var.get()),
                spkdur_max_ms=float(self.spk_max_var.get()),
            ),
        )

    def _run_preprocessing(self) -> None:
        if not self.sessions:
            messagebox.showerror("Error", "No sessions loaded.")
            return
        if self.running:
            messagebox.showwarning("Warning", "Already running.")
            return

        self.preprocessing_config = self._read_preprocessing_config()
        self.output_dir = Path(self.output_dir_var.get())

        def worker():
            self.running = True
            total = len(self.sessions)

            for i, session in enumerate(self.sessions):
                self._log_preproc(f"[{i+1}/{total}] Preprocessing {session.mouse_id}...")
                self.root.after(0, lambda v=int((i / total) * 100): self.preproc_progress.configure(value=v))
                self.root.after(0, lambda s=session.mouse_id, idx=i: self.preproc_progress_label.configure(
                    text=f"Processing {s} ({idx+1}/{total})"))

                config = self.preprocessing_config
                session.preprocessing_config = config
                raw = session.raw

                # Set baseline_end_s from heating start
                config.photometry.baseline_end_s = session.landmarks.heating_start_time

                # 1. Temperature
                temp_result = process_temperature(
                    raw.temperature_raw, raw.temp_bit_volts, raw.fs, config.temperature)
                self._log_preproc(f"  Temperature: baseline={temp_result.baseline_temp:.1f}°C, "
                                  f"max={temp_result.max_temp:.1f}°C")

                # Update landmarks with computed temps
                session.landmarks.baseline_temp = temp_result.baseline_temp
                session.landmarks.max_temp = temp_result.max_temp
                session.landmarks.max_temp_time = temp_result.max_temp_time
                session.landmarks.terminal_temp = temp_result.terminal_temp
                session.landmarks.terminal_time = raw.time[-1]

                # 2. ECoG filtering
                ecog_filt = filter_ecog(raw.ecog, raw.fs, config.ecog)
                self._log_preproc(f"  ECoG filtered ({config.ecog.bandpass_low}-{config.ecog.bandpass_high} Hz)")

                # 3. Photometry
                strategy_cls = STRATEGY_MAP[config.photometry.strategy]
                strategy = strategy_cls()
                phot_result = strategy.preprocess(raw.signal_470, raw.signal_405, raw.fs, config.photometry)

                # z-score and HPF
                phot_result.dff_zscore = z_score_baseline(
                    phot_result.dff, raw.fs, session.landmarks.heating_start_time)
                phot_result.dff_hpf = highpass_filter(
                    phot_result.dff, raw.fs, config.photometry.hpf_cutoff, config.photometry.hpf_order)
                self._log_preproc(f"  Photometry: strategy {config.photometry.strategy}")

                # 4. Store processed data
                session.processed = ProcessedData(
                    photometry=phot_result,
                    ecog_filtered=ecog_filt,
                    temperature_c=temp_result.temperature_c,
                    temperature_smooth=temp_result.temperature_smooth,
                    time=raw.time,
                    fs=raw.fs,
                )

                # 5. Transient detection
                transients = detect_transients(
                    phot_result.dff_hpf, raw.fs, config.transient,
                    temp_result.temperature_smooth)
                session.transients = transients
                self._log_preproc(f"  Transients: {len(transients)} detected")

                # 6. Spike detection
                exclusion_zones = []
                if session.landmarks.ueo_time is not None and session.landmarks.off_time is not None:
                    exclusion_zones.append((session.landmarks.ueo_time, session.landmarks.off_time))
                spikes = detect_spikes(
                    ecog_filt, raw.fs, session.landmarks.heating_start_time, config.spike_detection,
                    exclusion_zones=exclusion_zones if exclusion_zones else None)
                self._log_preproc(f"  Spikes: {len(spikes)} detected")
                # Store spikes on session (will be used by spike_triggered)
                session._spikes = spikes

                # 7. Save sanity check plots
                out = str(self.output_dir / "plots" / "sanity")
                path_v1 = plot_sanity_check(session, out, transients)
                path_v2 = plot_zoomed(session, out)
                self._log_preproc(f"  Plots saved: {path_v1}, {path_v2}")

                # Update treeview status
                self.root.after(0, lambda idx=i: self._update_session_status(idx, "preprocessed"))

            self.root.after(0, lambda: self.preproc_progress.configure(value=100))
            self.root.after(0, lambda: self.preproc_progress_label.configure(
                text=f"Done — {total} sessions preprocessed", foreground="green"))
            self._log_preproc("Preprocessing complete.")
            self.running = False

        threading.Thread(target=worker, daemon=True).start()

    def _update_session_status(self, idx: int, status: str) -> None:
        items = self.session_tree.get_children()
        if idx < len(items):
            item = items[idx]
            values = list(self.session_tree.item(item, "values"))
            values[4] = status
            self.session_tree.item(item, values=values)

    # ------------------------------------------------------------------
    # Tab 3 actions
    # ------------------------------------------------------------------

    def _read_analysis_config(self) -> AnalysisConfig:
        """Build an AnalysisConfig from the GUI fields."""
        return AnalysisConfig(
            temp_bin_size=float(self.temp_bin_var.get()),
            triggered_window_s=float(self.trig_window_var.get()),
            psth_bin_size_s=float(self.psth_bin_var.get()),
            psth_window_s=float(self.psth_window_var.get()),
            moving_avg_window_s=float(self.ma_window_var.get()),
            moving_avg_step_s=float(self.ma_step_var.get()),
            preictal_temp_range=float(self.preictal_range_var.get()),
            spike_triggered_window_s=float(self.spike_trig_var.get()),
        )

    def _run_extraction(self) -> None:
        # Check that sessions are preprocessed
        if not self.sessions:
            messagebox.showerror("Error", "No sessions loaded.")
            return
        if any(s.processed is None for s in self.sessions):
            messagebox.showerror("Error", "Run preprocessing first — some sessions are not preprocessed.")
            return
        if self.running:
            messagebox.showwarning("Warning", "Already running.")
            return

        self.analysis_config = self._read_analysis_config()
        self.output_dir = Path(self.output_dir_var.get())

        def worker():
            self.running = True
            config = self.analysis_config
            out_plots = str(self.output_dir / "plots" / "group")
            steps = 10  # pairing + 8 modules + done
            step = 0

            def progress(msg: str):
                nonlocal step
                step += 1
                pct = int((step / steps) * 100)
                self.root.after(0, lambda: self.extract_progress.configure(value=pct))
                self.root.after(0, lambda: self.extract_progress_label.configure(text=msg))
                self._log_extract(msg)

            # --- 1. Pairing ---
            progress("Assigning control equivalent landmarks...")
            mode = self.pairing_mode_var.get()
            seizure_sessions = [s for s in self.sessions if s.n_seizures > 0]
            control_sessions = [s for s in self.sessions if s.n_seizures == 0]
            if seizure_sessions and control_sessions:
                assign_all_controls(seizure_sessions, control_sessions, mode=mode)
                self._log_extract(f"  Pairing ({mode}): {len(seizure_sessions)} seizure, "
                                  f"{len(control_sessions)} control sessions")
            else:
                self._log_extract("  Pairing skipped (need both seizure and control sessions)")

            # Group sessions by cohort for group plots
            cohort_groups: Dict[str, List[Session]] = {}
            for s in self.sessions:
                if s.n_seizures > 0:
                    key = "seizure"
                elif s.genotype == "Scn1a":
                    key = "failed_seizure"
                else:
                    key = "wt"
                cohort_groups.setdefault(key, []).append(s)

            # --- 2–9. Run 8 analysis modules ---
            module_names = [
                "Cohort characteristics",
                "Baseline transients",
                "Pre-ictal mean",
                "Pre-ictal transients",
                "Ictal mean",
                "Ictal transients",
                "Postictal",
                "Spike-triggered averages",
            ]

            compute_fns = [
                compute_cohort_characteristics,
                compute_baseline_transients,
                compute_preictal_mean,
                compute_preictal_transients,
                compute_ictal_mean,
                compute_ictal_transients,
                compute_postictal,
                None,  # spike_triggered needs special handling
            ]

            plot_fns = [
                plot_cohort_characteristics,
                plot_baseline_transients,
                plot_preictal_mean,
                plot_preictal_transients,
                plot_ictal_mean,
                plot_ictal_transients,
                plot_postictal,
                plot_spike_triggered,
            ]

            for name, compute_fn, plot_fn in zip(module_names, compute_fns, plot_fns):
                progress(f"Running {name}...")
                try:
                    # Compute results per cohort
                    results = {}
                    for cohort_name, sessions in cohort_groups.items():
                        if name == "Spike-triggered averages":
                            spike_times_list = []
                            for s in sessions:
                                spikes = getattr(s, "_spikes", [])
                                spike_times_list.append(
                                    np.array([sp.time for sp in spikes]) if spikes else np.array([]))
                            results[cohort_name] = compute_spike_triggered_average(
                                sessions, spike_times_list, config)
                        else:
                            results[cohort_name] = compute_fn(sessions, config)

                    # Plot
                    fig_path = plot_fn(results, out_plots)
                    self._log_extract(f"  {name}: saved {fig_path}")

                except Exception as e:
                    self._log_extract(f"  {name}: ERROR — {e}")
                    logger.exception(f"Error in {name}")

            # Update session statuses
            for i in range(len(self.sessions)):
                self.root.after(0, lambda idx=i: self._update_session_status(idx, "extracted"))

            self.root.after(0, lambda: self.extract_progress.configure(value=100))
            self.root.after(0, lambda: self.extract_progress_label.configure(
                text="Extraction complete", foreground="green"))
            self._log_extract("All extraction modules complete.")
            self.running = False

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = FiberPhotometryApp()
    app.run()


if __name__ == "__main__":
    main()
