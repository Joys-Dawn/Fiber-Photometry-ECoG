"""
Fiber Photometry & ECoG Analysis Tool — Tkinter GUI.

Three-tab interface following the EphysAutomatedAnalysis pattern:
  Tab 1 — Data Loading & Cohort Management
  Tab 2 — Preprocessing
  Tab 3 — Extraction
"""

import logging
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from .core.config import (
    AnalysisConfig,
    PhotometryConfig,
    PreprocessingConfig,
)
from .core.data_models import (
    RawData,
    Session,
    SessionLandmarks,
)
from .core.session_io import (
    save_session, load_session, get_sessions_dir,
    find_saved_sessions, find_available_strategies,
)
from .data_loading import (
    read_ppd, read_oep, synchronize,
    scan_experiment_folder, extract_date_from_oep, read_data_log,
)
from .preprocessing import preprocess_session
from .preprocessing.photometry import STRATEGY_NAMES, strategy_folder_name
from .core.output_layout import ensure_layout
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
    plot_preictal_mean_diagnostic,
    plot_preictal_transients,
    plot_ictal_mean,
    plot_ictal_transients,
    plot_postictal,
    plot_spike_triggered,
    plot_ueo_per_cohort,
    plot_ueo_aligned_heatmaps,
    plot_experimental_vs_isosbestic_spike_triggered,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
        # --- Experiment folder ---
        folder_frame = ttk.LabelFrame(parent, text="Experiment Folder", padding=10)
        folder_frame.pack(fill="x", padx=5, pady=5)

        row1 = ttk.Frame(folder_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Experiment folder:").pack(side="left")
        self.experiment_path_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.experiment_path_var, width=60).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row1, text="Browse...", command=self._browse_experiment_folder).pack(side="right", padx=(0, 5))
        ttk.Button(row1, text="Scan & Load", command=self._scan_and_populate).pack(side="right")

        # Channel defaults (editable, applied to all sessions)
        ch_frame = ttk.Frame(folder_frame)
        ch_frame.pack(fill="x", pady=2)
        ttk.Label(ch_frame, text="Channels —").pack(side="left")
        ttk.Label(ch_frame, text="ECoG:").pack(side="left", padx=(5, 0))
        self.ecog_ch_var = tk.StringVar(value="2")
        ttk.Entry(ch_frame, textvariable=self.ecog_ch_var, width=4).pack(side="left", padx=(2, 10))
        ttk.Label(ch_frame, text="EMG:").pack(side="left")
        self.emg_ch_var = tk.StringVar(value="3")
        ttk.Entry(ch_frame, textvariable=self.emg_ch_var, width=4).pack(side="left", padx=(2, 10))

        # --- Session list ---
        list_frame = ttk.LabelFrame(parent, text="Sessions (from data log)", padding=10)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        cols = ("cohort", "mouse_id", "genotype", "seizure", "sudep",
                "include", "heat_start", "status")
        self.session_tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=12)
        for col, w in zip(cols, [120, 80, 70, 60, 50, 55, 75, 140]):
            self.session_tree.heading(col, text=col.replace("_", " ").title())
            self.session_tree.column(col, width=w, anchor="center")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.session_tree.yview)
        self.session_tree.configure(yscrollcommand=scrollbar.set)
        self.session_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_frame, text="Load All (from raw)", command=self._load_all_sessions).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Load Saved (preprocessed)", command=self._load_saved_sessions).pack(side="left", padx=5)
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

        # Internal: discovered session info (list of dicts with metadata from Excel)
        self._discovered_sessions: List[Dict] = []

    # ==================================================================
    # TAB 2 — Preprocessing
    # ==================================================================

    def _setup_preprocessing_tab(self, parent: ttk.Frame) -> None:
        # --- User decisions (per xlsx) ---
        config_frame = ttk.LabelFrame(parent, text="Preprocessing", padding=10)
        config_frame.pack(fill="x", padx=5, pady=5)

        row0 = ttk.Frame(config_frame)
        row0.pack(fill="x", pady=2)
        ttk.Label(row0, text="Photometry strategies (each checked one is run):").pack(side="left")
        # IRLS is the chosen default; the others are opt-in for pilot comparison.
        self.strategy_checks: Dict[str, tk.BooleanVar] = {
            "A": tk.BooleanVar(value=False),
            "B": tk.BooleanVar(value=False),
            "C": tk.BooleanVar(value=True),
            "D": tk.BooleanVar(value=False),
        }
        for letter in ("A", "B", "C", "D"):
            ttk.Checkbutton(
                row0, text=f"{letter} ({STRATEGY_NAMES[letter]})",
                variable=self.strategy_checks[letter],
            ).pack(side="left", padx=4)
        # Single-strategy var kept for the load-saved/extract flows which still
        # operate on one strategy at a time.
        self.strategy_var = tk.StringVar(value="C")

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
        log_frame.pack(fill="x", padx=5, pady=5)
        self.preproc_log = tk.Text(log_frame, height=4, state="disabled", font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.preproc_log.yview)
        self.preproc_log.configure(yscrollcommand=log_scroll.set)
        self.preproc_log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        # --- Interactive seizure marking ---
        mark_frame = ttk.LabelFrame(parent, text="Seizure Landmark Marking (click on ECoG trace)", padding=5)
        mark_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Landmark selector
        mark_ctrl = ttk.Frame(mark_frame)
        mark_ctrl.pack(fill="x", pady=2)
        ttk.Label(mark_ctrl, text="Placing:").pack(side="left")
        self.landmark_placing_var = tk.StringVar(value="EEC")
        for lm_name in ["EEC", "UEO", "Behavioral onset", "OFF"]:
            ttk.Radiobutton(mark_ctrl, text=lm_name, variable=self.landmark_placing_var,
                            value=lm_name).pack(side="left", padx=5)

        # Current landmark values display
        lm_vals = ttk.Frame(mark_frame)
        lm_vals.pack(fill="x", pady=2)
        self.eec_val_var = tk.StringVar(value="EEC: —")
        self.ueo_val_var = tk.StringVar(value="UEO: —")
        self.behav_val_var = tk.StringVar(value="Behavioral onset: —")
        self.off_val_var = tk.StringVar(value="OFF: —")
        for var in [self.eec_val_var, self.ueo_val_var, self.behav_val_var, self.off_val_var]:
            ttk.Label(lm_vals, textvariable=var, width=22).pack(side="left", padx=5)

        ttk.Button(mark_ctrl, text="Apply & Next", command=self._apply_and_next).pack(side="right", padx=5)
        ttk.Button(mark_ctrl, text="Next", command=self._next_session_marking).pack(side="right", padx=2)
        ttk.Button(mark_ctrl, text="Prev", command=self._prev_session_marking).pack(side="right", padx=2)

        # Matplotlib canvas for ECoG trace
        self._mark_fig, self._mark_ax = plt.subplots(1, 1, figsize=(10, 2.5))
        self._mark_ax.set_xlabel("Time (s)")
        self._mark_ax.set_ylabel("ECoG (filtered)")
        self._mark_ax.set_title("Select a preprocessed session to mark landmarks")
        self._mark_canvas = FigureCanvasTkAgg(self._mark_fig, master=mark_frame)
        self._mark_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._mark_canvas.mpl_connect("button_press_event", self._on_ecog_click)

        # Store landmark lines
        self._landmark_lines: Dict[str, Optional[plt.Line2D]] = {
            "EEC": None, "UEO": None, "Behavioral onset": None, "OFF": None,
        }
        self._landmark_times: Dict[str, Optional[float]] = {
            "EEC": None, "UEO": None, "Behavioral onset": None, "OFF": None,
        }
        self._marking_session_idx: Optional[int] = None

    # ==================================================================
    # TAB 3 — Extraction
    # ==================================================================

    def _setup_extraction_tab(self, parent: ttk.Frame) -> None:
        # --- Pairing mode (per xlsx: user decision #2) ---
        pair_frame = ttk.LabelFrame(parent, text="Control Pairing", padding=10)
        pair_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(pair_frame, text="Define controls by:").pack(side="left")
        self.pairing_mode_var = tk.StringVar(value="temperature")
        ttk.Combobox(pair_frame, textvariable=self.pairing_mode_var,
                      values=["temperature", "time"], state="readonly", width=14).pack(side="left", padx=5)

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

    def _browse_experiment_folder(self) -> None:
        path = filedialog.askdirectory(title="Select experiment folder")
        if path:
            self.experiment_path_var.set(path)

    def _scan_experiment_folder(self, experiment_dir: str) -> List[Dict]:
        return scan_experiment_folder(experiment_dir)

    def _read_data_log(self, experiment_dir: str) -> Optional[dict]:
        return read_data_log(experiment_dir)

    def _extract_date_from_oep(self, oep_path: str) -> Optional[str]:
        return extract_date_from_oep(oep_path)

    def _scan_and_populate(self) -> None:
        """Scan experiment folder, read data log, and populate session treeview."""
        exp_path = self.experiment_path_var.get().strip()
        if not exp_path:
            messagebox.showerror("Error", "Select an experiment folder first.")
            return

        # Scan folder structure
        discovered = self._scan_experiment_folder(exp_path)
        if not discovered:
            messagebox.showinfo("No sessions", "No sessions found in the selected folder.")
            return

        # Read data log
        log_lookup = self._read_data_log(exp_path)

        # Match each discovered session to its log entry
        for d in discovered:
            # Extract mouse ID: leading digits if present (e.g. "1938het" -> "1938"),
            # else everything before first "_" (e.g. "3993_session1" -> "3993"),
            # else the full session name.
            name = d["session_name"]
            digits_match = re.match(r"^(\d+)", name)
            if digits_match:
                mouse_id = digits_match.group(1)
            elif "_" in name:
                mouse_id = name.split("_")[0]
            else:
                mouse_id = name
            d["mouse_id"] = mouse_id

            # Extract date from OEP folder
            date_str = None
            if d["oep_path"]:
                date_str = self._extract_date_from_oep(d["oep_path"])
            d["date"] = date_str

            # Look up metadata from log
            if log_lookup and (mouse_id, date_str) in log_lookup:
                d.update(log_lookup[(mouse_id, date_str)])
            else:
                # Defaults if not in log
                cohort = d["cohort"]
                d["genotype"] = "WT" if "wt" in cohort.lower() else "Scn1a"
                d["seizure"] = 1 if "seizure" in cohort.lower() and "no" not in cohort.lower() else 0
                d["sudep"] = False
                d["include"] = True
                d["exclusion_reason"] = None
                d["heating_start"] = None

            # Determine cohort from metadata
            if d["genotype"] == "WT":
                d["cohort"] = "wt"
            elif d["seizure"] > 0:
                d["cohort"] = "seizure"
            else:
                d["cohort"] = "failed_seizure"

        self._discovered_sessions = discovered

        # Clear and populate treeview
        for item in self.session_tree.get_children():
            self.session_tree.delete(item)
        for d in discovered:
            include_str = "yes" if d["include"] else "no"
            has_data = "ready" if d["oep_path"] and d["ppd_path"] else "missing"
            heat = str(int(d["heating_start"])) if d["heating_start"] else "?"
            self.session_tree.insert(
                "", "end",
                values=(d["cohort"], d["mouse_id"], d["genotype"],
                        d["seizure"], "yes" if d["sudep"] else "",
                        include_str, heat, has_data),
            )

        n_excluded = sum(1 for d in discovered if not d["include"])
        log_status = "with data log" if log_lookup else "no data log found"
        self._log_loading(
            f"Scanned: {len(discovered)} sessions ({log_status}), "
            f"{n_excluded} excluded"
        )

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

    def _load_all_sessions(self) -> None:
        """Batch-load all discovered sessions (skipping excluded ones)."""
        if not self._discovered_sessions:
            messagebox.showerror("Error", "Scan an experiment folder first.")
            return
        if self.running:
            messagebox.showwarning("Warning", "Already running.")
            return

        ecog_ch = int(self.ecog_ch_var.get())
        emg_ch_str = self.emg_ch_var.get().strip()
        emg_ch = int(emg_ch_str) if emg_ch_str else None

        to_load = [d for d in self._discovered_sessions
                    if d["include"] and d["oep_path"] and d["ppd_path"]
                    and d.get("heating_start") is not None]

        if not to_load:
            messagebox.showerror("Error", "No loadable sessions (check include status and heating start times).")
            return

        self.running = True
        self._log_loading(f"Loading {len(to_load)} sessions...")

        def worker():
            n_ok = 0
            n_fail = 0
            for i, d in enumerate(to_load):
                mouse_id = d["mouse_id"]
                try:
                    self._log_loading(f"[{i+1}/{len(to_load)}] Loading {mouse_id}...")

                    ppd = read_ppd(d["ppd_path"])
                    # Per-session channel overrides from the data log, else the
                    # UI defaults. Temperature: None means auto-detect.
                    sess_ecog_ch = d.get("ecog_channel") or ecog_ch
                    sess_emg_ch = d.get("emg_channel") if d.get("emg_channel") is not None else emg_ch
                    sess_temp_ch = d.get("temperature_channel")
                    oep = read_oep(
                        d["oep_path"],
                        ecog_channel=sess_ecog_ch,
                        emg_channel=sess_emg_ch,
                        temperature_channel=sess_temp_ch,
                    )
                    sync = synchronize(ppd, oep)
                    self._log_loading(
                        f"  Sync: {sync.n_matched} pulses, "
                        f"drift={sync.drift_ppm:.1f} ppm"
                    )

                    landmarks = SessionLandmarks(
                        heating_start_time=d["heating_start"],
                        eec_time=d.get("eec_time"),
                        ueo_time=d.get("ueo_time"),
                        off_time=d.get("off_time"),
                    )

                    raw = RawData(
                        signal_470=sync.signal_470,
                        signal_405=sync.signal_405,
                        ecog=sync.ecog,
                        emg=sync.emg,
                        temperature_raw=sync.temperature_raw,
                        temp_bit_volts=sync.temp_bit_volts,
                        temp_slope=sync.temp_slope,
                        temp_intercept=sync.temp_intercept,
                        time=sync.time,
                        fs=sync.fs,
                    )

                    session = Session(
                        mouse_id=mouse_id,
                        genotype=d["genotype"],
                        heating_session=d.get("heating_session", 1),
                        n_seizures=d["seizure"],
                        sudep=d["sudep"],
                        include_session=d["include"],
                        exclusion_reason=d.get("exclusion_reason"),
                        landmarks=landmarks,
                        raw=raw,
                    )
                    session.cohort = d["cohort"]
                    session.date = d.get("date")
                    session.session_name = d.get("session_name")
                    self.sessions.append(session)
                    n_ok += 1

                    # Update treeview status
                    items = self.session_tree.get_children()
                    full_idx = self._discovered_sessions.index(d)
                    if full_idx < len(items):
                        self.root.after(0, lambda idx=full_idx: self._update_session_status(idx, "loaded"))

                except Exception as e:
                    n_fail += 1
                    msg = str(e)
                    low = msg.lower()
                    if "temperature" in low:
                        status = "failed: no temp"
                    elif "sync" in low or "ttl" in low or "pulse" in low:
                        status = "failed: sync"
                    elif "ppd" in low:
                        status = "failed: ppd"
                    elif "oep" in low or "record" in low:
                        status = "failed: oep"
                    else:
                        status = "failed"
                    self._log_loading(f"  ERROR: {mouse_id}: {msg}")
                    try:
                        full_idx = self._discovered_sessions.index(d)
                        self.root.after(
                            0,
                            lambda idx=full_idx, s=status: self._update_session_status(idx, s),
                        )
                    except ValueError:
                        pass

            if n_fail > 0:
                self._log_loading(
                    f"Done: {n_ok} loaded, {n_fail} failed — "
                    "see status column and errors above."
                )
            else:
                self._log_loading(f"Done: {n_ok} sessions loaded.")
            self.running = False

        threading.Thread(target=worker, daemon=True).start()

    def _load_saved_sessions(self) -> None:
        """Load previously preprocessed sessions from .sessions/<strategy>/.

        If multiple strategies are available, prompts the user to choose one.
        """
        exp_path = self.experiment_path_var.get().strip()
        if not exp_path:
            messagebox.showerror("Error", "Select an experiment folder first.")
            return

        strategies = find_available_strategies(exp_path)
        if not strategies:
            messagebox.showinfo("No saved sessions",
                                "No preprocessed sessions found. Run 'Load All' and preprocess first.")
            return

        # Let user choose strategy if multiple are available
        if len(strategies) == 1:
            chosen = strategies[0]
        else:
            chosen = self._ask_strategy_choice(strategies)
            if not chosen:
                return  # user cancelled

        saved = find_saved_sessions(exp_path, strategy=chosen)
        if not saved:
            messagebox.showinfo("No saved sessions",
                                f"No sessions found for strategy {chosen}.")
            return

        self.sessions.clear()
        for item in self.session_tree.get_children():
            self.session_tree.delete(item)

        self._log_loading(f"Loading {len(saved)} sessions (strategy {chosen})...")
        self.output_dir = Path(self.output_dir_var.get())
        strategy_root = self.output_dir / strategy_folder_name(chosen)
        layout = ensure_layout(strategy_root)
        sanity_out = str(layout["quality_check"])
        for path in saved:
            try:
                session = load_session(path)
                self.sessions.append(session)
                cohort = session.cohort
                include_str = "yes" if session.include_session else "no"
                status = "preprocessed" if session.processed else "loaded"
                self.session_tree.insert(
                    "", "end",
                    values=(cohort, session.mouse_id, session.genotype,
                            session.n_seizures, "yes" if session.sudep else "",
                            include_str, "", status),
                )
                # Regenerate sanity check plots so loading cached data produces
                # the same verification outputs as a fresh preprocessing run.
                if session.processed is not None and session.raw is not None:
                    try:
                        plot_sanity_check(session, sanity_out, session.transients)
                        plot_zoomed(session, sanity_out)
                    except Exception as plot_err:
                        self._log_loading(f"  WARN: sanity plots failed for "
                                          f"{session.mouse_id}: {plot_err}")
            except Exception as e:
                self._log_loading(f"  ERROR loading {path.name}: {e}")

        self._log_loading(f"Loaded {len(self.sessions)} sessions (strategy {chosen}). "
                          f"Quality check plots in {sanity_out}")
        # Set strategy dropdown to match loaded data
        self.strategy_var.set(chosen)
        if self.sessions and self.sessions[0].processed:
            self._show_ecog_for_marking(0)

    def _ask_strategy_choice(self, strategies: List[str]) -> Optional[str]:
        """Show a simple dialog asking which strategy to load."""
        labels = {"A": "A (Chandni)", "B": "B (Meiling)", "C": "C (IRLS/Keevers)"}
        options = [labels.get(s, s) for s in strategies]
        dialog = tk.Toplevel(self.root)
        dialog.title("Choose preprocessing strategy")
        dialog.resizable(False, False)
        dialog.grab_set()

        ttk.Label(dialog, text="Multiple strategies found. Choose one to load:").pack(padx=15, pady=(15, 5))

        choice_var = tk.StringVar(value=strategies[0])
        for s, label in zip(strategies, options):
            ttk.Radiobutton(dialog, text=label, variable=choice_var, value=s).pack(anchor="w", padx=25)

        result = [None]

        def on_ok():
            result[0] = choice_var.get()
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Load", command=on_ok).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=5)

        dialog.wait_window()
        return result[0]

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
    # Tab 2 actions — seizure marking
    # ------------------------------------------------------------------

    def _show_ecog_for_marking(self, session_idx: int) -> None:
        """Display preprocessed ECoG trace for landmark marking."""
        if session_idx >= len(self.sessions):
            return
        session = self.sessions[session_idx]
        if session.processed is None or session.processed.ecog_filtered is None:
            self._log_preproc("Session not preprocessed yet — run preprocessing first.")
            return

        self._marking_session_idx = session_idx
        ecog = session.processed.ecog_filtered
        fs = session.processed.fs
        time = np.arange(len(ecog)) / fs

        self._mark_ax.clear()
        self._mark_ax.plot(time, ecog, color="gray", linewidth=0.5)
        self._mark_ax.set_xlabel("Time (s)")
        self._mark_ax.set_ylabel("ECoG (filtered)")
        self._mark_ax.set_title(f"Session: {session.mouse_id} — click to place landmarks")

        # Reset landmarks
        for key in self._landmark_lines:
            self._landmark_lines[key] = None
            self._landmark_times[key] = None

        # Show existing landmarks if present
        lm = session.landmarks
        if lm:
            for name, t in [("EEC", lm.eec_time), ("UEO", lm.ueo_time),
                            ("Behavioral onset", lm.behavioral_onset_time), ("OFF", lm.off_time)]:
                if t is not None:
                    line = self._mark_ax.axvline(t, color=self._landmark_color(name),
                                                  linestyle="--", linewidth=1.5, label=name)
                    self._landmark_lines[name] = line
                    self._landmark_times[name] = t
            if any(v is not None for v in self._landmark_lines.values()):
                self._mark_ax.legend(fontsize=7, loc="upper right")

        self._update_landmark_labels()
        self._mark_canvas.draw()

    def _landmark_color(self, name: str) -> str:
        return {"EEC": "orange", "UEO": "red", "Behavioral onset": "purple", "OFF": "blue"}.get(name, "black")

    def _on_ecog_click(self, event) -> None:
        """Handle click on ECoG trace to place a landmark."""
        if event.inaxes != self._mark_ax or self._marking_session_idx is None:
            return
        t = event.xdata
        if t is None:
            return

        name = self.landmark_placing_var.get()

        # Remove old line if exists
        if self._landmark_lines[name] is not None:
            self._landmark_lines[name].remove()

        line = self._mark_ax.axvline(t, color=self._landmark_color(name),
                                      linestyle="--", linewidth=1.5, label=name)
        self._landmark_lines[name] = line
        self._landmark_times[name] = t

        self._mark_ax.legend(fontsize=7, loc="upper right")
        self._mark_canvas.draw()
        self._update_landmark_labels()

    def _update_landmark_labels(self) -> None:
        for name, var in [("EEC", self.eec_val_var), ("UEO", self.ueo_val_var),
                          ("Behavioral onset", self.behav_val_var), ("OFF", self.off_val_var)]:
            t = self._landmark_times[name]
            var.set(f"{name}: {t:.2f}s" if t is not None else f"{name}: —")

    def _next_session_marking(self) -> None:
        """Show next session for landmark marking."""
        if self._marking_session_idx is None:
            return
        next_idx = self._marking_session_idx + 1
        if next_idx < len(self.sessions):
            self._show_ecog_for_marking(next_idx)

    def _prev_session_marking(self) -> None:
        """Show previous session for landmark marking."""
        if self._marking_session_idx is None:
            return
        prev_idx = self._marking_session_idx - 1
        if prev_idx >= 0:
            self._show_ecog_for_marking(prev_idx)

    def _apply_and_next(self) -> None:
        """Apply landmarks to current session and move to next."""
        self._apply_landmarks()
        self._next_session_marking()

    def _apply_landmarks(self) -> None:
        """Write marked landmarks back to the selected session."""
        if self._marking_session_idx is None or self._marking_session_idx >= len(self.sessions):
            messagebox.showinfo("Info", "No session selected for marking.")
            return
        session = self.sessions[self._marking_session_idx]
        if session.landmarks is None:
            return
        session.landmarks.eec_time = self._landmark_times["EEC"]
        session.landmarks.ueo_time = self._landmark_times["UEO"]
        session.landmarks.behavioral_onset_time = self._landmark_times["Behavioral onset"]
        session.landmarks.off_time = self._landmark_times["OFF"]
        self._log_preproc(f"Landmarks applied for {session.mouse_id}: "
                          f"EEC={self._landmark_times['EEC']}, "
                          f"UEO={self._landmark_times['UEO']}, "
                          f"OFF={self._landmark_times['OFF']}")

        # Auto-save updated session
        exp_path = self.experiment_path_var.get().strip()
        if exp_path:
            try:
                strategy = session.preprocessing_config.photometry.strategy
                save_session(session, get_sessions_dir(exp_path, strategy))
            except Exception as e:
                self._log_preproc(f"  WARNING: could not save {session.mouse_id}: {e}")

    # ------------------------------------------------------------------
    # Tab 2 actions — preprocessing
    # ------------------------------------------------------------------

    def _read_preprocessing_config(self, strategy: str) -> PreprocessingConfig:
        """Build a PreprocessingConfig for *strategy*.

        Only the strategy letter is user-selectable; everything else uses
        scientifically validated defaults from PreprocessingConfig.
        """
        return PreprocessingConfig(
            photometry=PhotometryConfig(strategy=strategy),
        )

    def _selected_strategies(self) -> List[str]:
        """Return the list of strategy letters whose checkbox is ticked."""
        return [s for s in ("A", "B", "C", "D") if self.strategy_checks[s].get()]

    def _run_preprocessing(self) -> None:
        if not self.sessions:
            messagebox.showerror("Error", "No sessions loaded.")
            return
        if self.running:
            messagebox.showwarning("Warning", "Already running.")
            return

        strategies = self._selected_strategies()
        if not strategies:
            messagebox.showerror("Error", "Select at least one photometry strategy.")
            return

        self.output_dir = Path(self.output_dir_var.get())
        # Active strategy for downstream tabs follows the first selected.
        self.strategy_var.set(strategies[0])

        def worker():
            self.running = True
            total_sessions = len(self.sessions)
            total_steps = total_sessions * len(strategies)
            step = 0

            for strategy in strategies:
                config = self._read_preprocessing_config(strategy)
                strategy_label = STRATEGY_NAMES[strategy]
                strategy_root = self.output_dir / strategy_folder_name(strategy)
                layout = ensure_layout(strategy_root)
                qc_dir = str(layout["quality_check"])
                self._log_preproc(f"=== Strategy {strategy} ({strategy_label}) ===")

                for i, session in enumerate(self.sessions):
                    step += 1
                    self._log_preproc(
                        f"  [{i+1}/{total_sessions}] {session.mouse_id} ({strategy})")
                    pct = int((step / total_steps) * 100)
                    self.root.after(0, lambda v=pct: self.preproc_progress.configure(value=v))
                    self.root.after(
                        0, lambda s=session.mouse_id, lbl=strategy_label, idx=i, t=total_sessions:
                            self.preproc_progress_label.configure(
                                text=f"{lbl}: {s} ({idx+1}/{t})"))

                    session.preprocessing_config = config
                    config.photometry.baseline_end_s = session.landmarks.heating_start_time

                    try:
                        preprocess_session(session, config)
                    except ValueError as exc:
                        self._log_preproc(f"    SKIPPED: {exc}")
                        self.root.after(0, lambda idx=i: self._update_session_status(idx, "skipped"))
                        continue

                    self._log_preproc(
                        f"    Temp baseline={session.landmarks.baseline_temp:.1f}°C, "
                        f"max={session.landmarks.max_temp:.1f}°C; "
                        f"transients={len(session.transients)}, spikes={len(session.spikes)}")

                    path_v1 = plot_sanity_check(session, qc_dir, session.transients)
                    path_v2 = plot_zoomed(session, qc_dir)
                    self._log_preproc(f"    QC plots: {Path(path_v1).name}, {Path(path_v2).name}")

                    self.root.after(0, lambda idx=i: self._update_session_status(idx, "preprocessed"))

                # Auto-save when multiple strategies are selected (the user is
                # building a side-by-side comparison and clearly wants both on
                # disk); fall back to the existing prompt for the single-
                # strategy case.
                if len(strategies) > 1:
                    self.root.after(0, lambda c=config: self._save_preprocessed_silent(c))
                else:
                    self.root.after(0, lambda c=config: self._prompt_save_preprocessed(c))

            self.root.after(0, lambda: self.preproc_progress.configure(value=100))
            self.root.after(0, lambda: self.preproc_progress_label.configure(
                text=f"Done — {total_sessions} sessions × {len(strategies)} strategy/strategies",
                foreground="green"))
            self._log_preproc("Preprocessing complete. Mark landmarks via the trace viewer below.")
            if self.sessions:
                self.root.after(0, lambda: self._show_ecog_for_marking(0))

            self.running = False

        threading.Thread(target=worker, daemon=True).start()

    def _save_preprocessed_silent(self, config) -> None:
        """Save preprocessed sessions for *config*'s strategy without prompting.

        Used when running multi-strategy preprocessing: each strategy auto-saves
        so the user gets a complete side-by-side comparison on disk.
        """
        exp_path = self.experiment_path_var.get().strip()
        if not exp_path:
            return
        strategy = config.photometry.strategy
        sessions_dir = get_sessions_dir(exp_path, strategy)
        existing = list(sessions_dir.glob("*.npz")) if sessions_dir.exists() else []
        for f in existing:
            f.unlink()

        def save_worker():
            for s in self.sessions:
                try:
                    save_session(s, sessions_dir)
                except Exception as e:
                    self._log_preproc(f"  WARN: could not save {s.mouse_id}: {e}")
            self._log_preproc(f"  Saved {len(self.sessions)} sessions (strategy {strategy})")

        threading.Thread(target=save_worker, daemon=True).start()

    def _prompt_save_preprocessed(self, config) -> None:
        """Ask user whether to save preprocessed data, then save with progress."""
        exp_path = self.experiment_path_var.get().strip()
        if not exp_path:
            return

        strategy = config.photometry.strategy
        sessions_dir = get_sessions_dir(exp_path, strategy)

        # Check if data already exists for this strategy
        existing = list(sessions_dir.glob("*.npz")) if sessions_dir.exists() else []
        if existing:
            msg = (f"Save preprocessed data (strategy {strategy})?\n\n"
                   f"{len(existing)} files already exist for strategy {strategy} "
                   f"and will be overwritten.")
        else:
            msg = f"Save preprocessed data (strategy {strategy})?"

        if not messagebox.askyesno("Save preprocessed data", msg):
            self._log_preproc("Preprocessed data not saved.")
            return

        # Clear existing files for this strategy before saving
        for f in existing:
            f.unlink()

        # Save in background thread with progress
        def save_worker():
            total = len(self.sessions)
            for i, s in enumerate(self.sessions):
                self.root.after(0, lambda v=int((i / total) * 100): self.preproc_progress.configure(value=v))
                self.root.after(0, lambda name=s.mouse_id, idx=i, t=total:
                    self.preproc_progress_label.configure(
                        text=f"Saving {name} ({idx+1}/{t})..."))
                try:
                    save_session(s, sessions_dir)
                except Exception as e:
                    self._log_preproc(f"  WARNING: could not save {s.mouse_id}: {e}")

            self.root.after(0, lambda: self.preproc_progress.configure(value=100))
            self.root.after(0, lambda: self.preproc_progress_label.configure(
                text=f"Saved {total} sessions (strategy {strategy})", foreground="green"))
            self._log_preproc(f"Saved {total} sessions to {sessions_dir}")

        threading.Thread(target=save_worker, daemon=True).start()

    def _update_session_status(self, idx: int, status: str) -> None:
        items = self.session_tree.get_children()
        if idx < len(items):
            item = items[idx]
            values = list(self.session_tree.item(item, "values"))
            values[7] = status
            self.session_tree.item(item, values=values)

    # ------------------------------------------------------------------
    # Tab 3 actions
    # ------------------------------------------------------------------

    def _read_analysis_config(self) -> AnalysisConfig:
        """Build an AnalysisConfig with defaults per xlsx spec."""
        return AnalysisConfig(
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

            # Output layout follows the strategy that's currently loaded —
            # session.preprocessing_config carries the strategy used at preprocess time.
            chosen_strategy = (self.sessions[0].preprocessing_config.photometry.strategy
                               if self.sessions and self.sessions[0].preprocessing_config
                               else self.strategy_var.get())
            strategy_root = self.output_dir / strategy_folder_name(chosen_strategy)
            layout = ensure_layout(strategy_root)
            means_dir = str(layout["means"])
            transients_dir = str(layout["transients"])
            param_dir = str(layout["parameter_detection"])
            qc_dir = str(layout["quality_check"])

            steps = 11  # pairing + 8 modules + diagnostic + done
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
                try:
                    assign_all_controls(seizure_sessions, control_sessions, mode=mode)
                    self._log_extract(f"  Pairing ({mode}): {len(seizure_sessions)} seizure, "
                                      f"{len(control_sessions)} control sessions")
                except Exception as e:
                    self._log_extract(f"  Pairing FAILED — aborting extraction: {e}")
                    logger.exception("Pairing failed")
                    self.root.after(0, lambda: self.extract_progress_label.configure(
                        text="Extraction failed (pairing error)", foreground="red"))
                    self.running = False
                    return
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

            # --- 2–9. Run 8 analysis modules, each routed to means/ or transients/. ---
            # (name, compute_fn, plot_fn, dest_dir)
            modules = [
                ("Cohort characteristics", compute_cohort_characteristics, plot_cohort_characteristics, means_dir),
                ("Baseline transients", compute_baseline_transients, plot_baseline_transients, transients_dir),
                ("Pre-ictal mean", compute_preictal_mean, plot_preictal_mean, means_dir),
                ("Pre-ictal transients", compute_preictal_transients, plot_preictal_transients, transients_dir),
                ("Ictal mean", compute_ictal_mean, plot_ictal_mean, means_dir),
                ("Ictal transients", compute_ictal_transients, plot_ictal_transients, transients_dir),
                ("Postictal", compute_postictal, plot_postictal, means_dir),
                ("Spike-triggered averages", None, plot_spike_triggered, means_dir),
            ]

            failed_modules = []
            # Capture cross-module data for follow-up plots (Phase 2f, 4e).
            ictal_results_by_cohort = {}
            spike_times_by_cohort: Dict[str, List[np.ndarray]] = {}
            for name, compute_fn, plot_fn, dest_dir in modules:
                progress(f"Running {name}...")
                try:
                    # Compute results per cohort
                    results = {}
                    for cohort_name, sessions in cohort_groups.items():
                        if name == "Spike-triggered averages":
                            spike_times_list = []
                            for s in sessions:
                                spike_times_list.append(
                                    np.array([sp.time for sp in s.spikes]) if s.spikes else np.array([]))
                            spike_times_by_cohort[cohort_name] = spike_times_list
                            results[cohort_name] = compute_spike_triggered_average(
                                sessions, spike_times_list, config)
                        elif name == "Ictal mean":
                            # OFF AUC fixed at 100 s (spec D5); others use config default.
                            results[cohort_name] = compute_ictal_mean(
                                sessions, config,
                                auc_end_s_per_landmark={"OFF": 100.0},
                            )
                        else:
                            results[cohort_name] = compute_fn(sessions, config)

                    # Plot
                    fig_path = plot_fn(results, dest_dir)
                    self._log_extract(f"  {name}: saved {fig_path}")
                    if name == "Ictal mean":
                        ictal_results_by_cohort = results

                except Exception as e:
                    failed_modules.append(name)
                    self._log_extract(f"  {name}: ERROR — {e}")
                    logger.exception("Error in %s", name)

            # --- 9b. Follow-up multi-source plots ---
            # UEO-aligned heatmaps split by cohort (Phase 4e)
            if ictal_results_by_cohort:
                progress("Writing UEO-aligned heatmaps...")
                try:
                    paths = plot_ueo_aligned_heatmaps(ictal_results_by_cohort, means_dir)
                    for c, p in paths.items():
                        self._log_extract(f"  UEO heatmap [{c}]: {p}")
                except Exception as e:
                    failed_modules.append("UEO-aligned heatmaps")
                    self._log_extract(f"  UEO heatmaps: ERROR — {e}")
                    logger.exception("Error in UEO heatmaps")

            # Experimental (470) vs isosbestic (405) spike-triggered overlay
            # — motion-artifact diagnostic (Phase 2f). Quality-check, not a
            # signal metric, so it lives under the quality check folder.
            if spike_times_by_cohort:
                progress("Writing 470 vs 405 spike-triggered overlay...")
                try:
                    p = plot_experimental_vs_isosbestic_spike_triggered(
                        cohort_groups, spike_times_by_cohort, qc_dir
                    )
                    if p:
                        self._log_extract(f"  470/405 overlay: {p}")
                except Exception as e:
                    failed_modules.append("470/405 spike-triggered overlay")
                    self._log_extract(f"  470/405 overlay: ERROR — {e}")
                    logger.exception("Error in 470/405 overlay")

            # --- 10. Per-session parameter detection diagnostic ---
            progress("Writing parameter detection diagnostics...")
            try:
                n_done = 0
                for cohort_sessions in cohort_groups.values():
                    for s in cohort_sessions:
                        lm = s.landmarks
                        if lm is None or lm.heating_start_time is None:
                            continue
                        if lm.ueo_time is None and getattr(lm, "equiv_ueo_time", None) is None:
                            continue
                        if plot_preictal_mean_diagnostic(s, param_dir):
                            n_done += 1
                self._log_extract(f"  Parameter detection: wrote {n_done} session plots")
            except Exception as e:
                failed_modules.append("Parameter detection")
                self._log_extract(f"  Parameter detection: ERROR — {e}")
                logger.exception("Error in parameter detection")

            # Update session statuses
            for i in range(len(self.sessions)):
                self.root.after(0, lambda idx=i: self._update_session_status(idx, "extracted"))

            self.root.after(0, lambda: self.extract_progress.configure(value=100))
            if failed_modules:
                summary = f"Extraction done with {len(failed_modules)} error(s): {', '.join(failed_modules)}"
                self.root.after(0, lambda: self.extract_progress_label.configure(
                    text=summary, foreground="orange"))
                self._log_extract(summary)
            else:
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
