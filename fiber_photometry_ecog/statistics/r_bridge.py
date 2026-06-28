"""
R bridge: call lmerTest + emmeans via subprocess for LMM inference.

Ported from EphysAutomatedAnalysis with minor trimming. All R communication
flows through this module. Each function writes a DataFrame to a temp CSV,
generates an R script, runs it via ``Rscript.exe``, and parses structured
stdout lines back into Python dataclasses.

This module is responsible for:
 * Auto-detecting an R installation on Windows / macOS / Linux.
 * Auto-installing ``lmerTest`` and ``emmeans`` to the user library on
   first use (no admin privileges required).
 * Caching availability across calls so we don't re-probe R on every fit.

The caller (`compare_paired_lmm`) is responsible for the actual model
choice and the fallback path when R is unavailable, the fit fails, or
the model is reported singular.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

if sys.platform == "win32":
    import winreg

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OmnibusEffect:
    """One row from lmerTest::anova (Satterthwaite)."""
    effect: str
    f_value: float
    df1: float
    df2: float
    p_value: float


@dataclass
class PairwiseResult:
    """One row from emmeans::pairs."""
    contrast: str
    level_a: str
    level_b: str
    estimate: float
    se: float
    df: float
    t_value: float
    p_value: float


@dataclass
class VarianceComponent:
    group: str
    variance: float
    std_dev: float


@dataclass
class LMMResult:
    """Bundle of one R LMM fit's outputs."""
    omnibus: List[OmnibusEffect] = field(default_factory=list)
    pairwise: List[PairwiseResult] = field(default_factory=list)
    variance_components: List[VarianceComponent] = field(default_factory=list)
    singular: bool = False
    converged: bool = True


# ---------------------------------------------------------------------------
# R auto-detection (cached at module level)
# ---------------------------------------------------------------------------

_RSCRIPT_PATH: Optional[str] = None
_R_CHECKED = False
_R_PACKAGES_OK: Optional[bool] = None

_REQUIRED_R_PACKAGES = ("lmerTest", "emmeans")


def find_r() -> Optional[str]:
    """Locate Rscript on this system. Cached after first call."""
    global _RSCRIPT_PATH, _R_CHECKED
    if _R_CHECKED:
        return _RSCRIPT_PATH
    _R_CHECKED = True

    # 1) R_HOME environment variable
    r_home = os.environ.get("R_HOME")
    if r_home:
        p = _rscript_from_home(r_home)
        if p:
            _RSCRIPT_PATH = p
            return _RSCRIPT_PATH

    # 2) Windows registry
    if sys.platform == "win32":
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for sub in (r"SOFTWARE\R-core\R64", r"SOFTWARE\R-core\R"):
                try:
                    key = winreg.OpenKey(hive, sub)
                    install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                    winreg.CloseKey(key)
                    p = _rscript_from_home(install_path)
                    if p:
                        _RSCRIPT_PATH = p
                        return _RSCRIPT_PATH
                except OSError:
                    continue

    # 3) PATH lookup
    p = shutil.which("Rscript")
    if p:
        _RSCRIPT_PATH = p
        return _RSCRIPT_PATH

    # 4) Common install dirs
    if sys.platform == "win32":
        search_patterns = (
            r"C:\R\R-*", r"C:\Program Files\R\R-*",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\R\R-*"),
        )
    elif sys.platform == "darwin":
        search_patterns = (
            "/Library/Frameworks/R.framework/Versions/*",
            "/usr/local/Cellar/r/*/lib/R",
        )
    else:
        search_patterns = ("/usr/lib/R", "/usr/local/lib/R", "/opt/R/*/lib/R")

    for pattern in search_patterns:
        parent = Path(pattern).parent
        if parent.exists():
            dirs = sorted(parent.glob(Path(pattern).name), reverse=True)
            for d in dirs:
                p = _rscript_from_home(str(d))
                if p:
                    _RSCRIPT_PATH = p
                    return _RSCRIPT_PATH

    logger.warning(
        "R installation not found. LMM will fall back to the classical paired test. "
        "Install R from https://cloud.r-project.org/ to enable Satterthwaite LMM."
    )
    return None


def _rscript_from_home(r_home: str) -> Optional[str]:
    """Return path to Rscript inside an R home directory, or None."""
    for sub in ("bin/x64/Rscript.exe", "bin/Rscript.exe",
                "bin/x64/Rscript", "bin/Rscript"):
        p = os.path.join(r_home, sub)
        if os.path.isfile(p):
            return p
    return None


def _r_packages_present(rscript: str) -> bool:
    """Return True iff every required R package is loadable."""
    check = " && ".join(
        f'requireNamespace("{pkg}", quietly=TRUE)' for pkg in _REQUIRED_R_PACKAGES
    )
    try:
        res = subprocess.run(
            [rscript, "-e", f"cat({check})"],
            capture_output=True, text=True, timeout=15,
        )
        return res.stdout.strip() == "TRUE"
    except Exception:
        return False


def _install_r_packages(rscript: str) -> bool:
    """Install required R packages to the user library. No admin required."""
    pkgs = ", ".join(f'"{pkg}"' for pkg in _REQUIRED_R_PACKAGES)
    script = (
        'lib <- Sys.getenv("R_LIBS_USER"); '
        'if (lib == "") lib <- file.path(Sys.getenv("HOME"), "R", "library"); '
        'dir.create(lib, recursive=TRUE, showWarnings=FALSE); '
        f'install.packages(c({pkgs}), lib=lib, '
        f'repos="https://cloud.r-project.org", quiet=TRUE); '
        f'cat(all(sapply(c({pkgs}), requireNamespace, quietly=TRUE)))'
    )
    try:
        logger.info("Installing required R packages (lmerTest, emmeans). "
                    "First run may take a few minutes...")
        res = subprocess.run(
            [rscript, "-e", script],
            capture_output=True, text=True, timeout=600,
        )
        # install.packages spams stdout/stderr with progress lines, so the
        # final `cat(TRUE/FALSE)` may not be the last whitespace-stripped
        # line. Re-probe via a clean session instead of parsing the noisy
        # install output.
        if _r_packages_present(rscript):
            logger.info("R packages installed successfully.")
            return True
        logger.warning(
            "R package installation did not succeed. stderr tail: %s",
            res.stderr[-500:],
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning("R package installation timed out (>10 min).")
        return False
    except Exception as exc:
        logger.warning("R package installation failed: %s", exc)
        return False


def check_r_packages() -> bool:
    """Verify required R packages are installed, auto-installing if needed."""
    global _R_PACKAGES_OK
    if _R_PACKAGES_OK is not None:
        return _R_PACKAGES_OK

    rscript = find_r()
    if rscript is None:
        _R_PACKAGES_OK = False
        return False

    if _r_packages_present(rscript):
        _R_PACKAGES_OK = True
        return True

    if _install_r_packages(rscript) and _r_packages_present(rscript):
        _R_PACKAGES_OK = True
        return True

    _R_PACKAGES_OK = False
    logger.warning(
        "R packages lmerTest/emmeans not available. LMM disabled — "
        "falling back to classical paired test."
    )
    return False


def is_r_available() -> bool:
    """One-shot check: R installed AND required packages present."""
    return find_r() is not None and check_r_packages()


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

def _run_r_script(script: str, timeout: int = 60) -> Optional[str]:
    """Write script to a temp file, run via Rscript, return stdout or None."""
    rscript = find_r()
    if rscript is None:
        return None

    fd, path = tempfile.mkstemp(suffix=".R", prefix="lmm_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)
        result = subprocess.run(
            [rscript, "--vanilla", path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            stderr = "\n".join(
                line for line in result.stderr.splitlines()
                if not any(x in line for x in
                           ("Loading", "Attaching", "package", "mask"))
            ).strip()
            if stderr:
                logger.debug("R stderr: %s", stderr[:300])
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning("R script timed out (>%ss)", timeout)
        return None
    except Exception as exc:
        logger.warning("R subprocess error: %s", exc)
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _write_temp_csv(data: pd.DataFrame) -> str:
    """Write DataFrame to temp CSV; return forward-slash path for R."""
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="lmm_data_")
    os.close(fd)
    data.to_csv(path, index=False)
    return path.replace("\\", "/")


def _cleanup_csv(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_lmm_pairwise(
    data: pd.DataFrame,
    formula: str,
    random_effects: str,
    factor: str,
    ddf: str = "Satterthwaite",
) -> Optional[LMMResult]:
    """Fit an LMM and return Satterthwaite omnibus + emmeans pairwise contrasts.

    Parameters
    ----------
    data : DataFrame with columns referenced in formula/random_effects.
    formula : R fixed-effects formula, e.g. ``"value ~ cohort"``.
    random_effects : R random-effects terms, e.g. ``"(1|mouse_id) + (1|pair_id)"``.
    factor : name of the factor whose pairwise emmeans contrast is wanted.
    ddf : ``"Satterthwaite"`` (default) or ``"Kenward-Roger"``.

    Returns
    -------
    LMMResult on success, None on fit failure, convergence warning, or
    singular fit (caller should fall back). ``LMMResult.singular`` is
    True only in the path where the fit succeeded but lme4's
    ``isSingular`` reports True; in our pipeline both that and
    ``None`` are treated the same (loud-warning fallback).
    """
    csv_path = _write_temp_csv(data)
    try:
        full_formula = f"{formula} + {random_effects}"
        script = _PAIRWISE_TEMPLATE.format(
            csv=csv_path, formula=full_formula, factor=factor, ddf=ddf,
        )
        stdout = _run_r_script(script)
        if stdout is None:
            return None
        return _parse_pairwise_output(stdout)
    finally:
        _cleanup_csv(csv_path)


# ---------------------------------------------------------------------------
# R script template
# ---------------------------------------------------------------------------

_FIT_BLOCK = '''\
conv_warn <- NULL
fit <- tryCatch(
    withCallingHandlers(
        lmer({formula}, data=df, REML=TRUE),
        warning = function(w) {{
            if (grepl("converge|gradient|singular", conditionMessage(w), ignore.case=TRUE)) {{
                conv_warn <<- conditionMessage(w)
            }}
            invokeRestart("muffleWarning")
        }}
    ),
    error = function(e) NULL
)

if (is.null(fit)) {{
    cat("STATUS: FIT_FAILED\\n")
}} else {{
    is_sing <- isSingular(fit)
    conv_msgs <- fit@optinfo$conv$lme4$messages
    has_conv_issue <- !is.null(conv_warn) || !is.null(conv_msgs)
    if (is_sing) {{
        cat("STATUS: SINGULAR\\n")
    }} else if (has_conv_issue) {{
        cat("STATUS: CONVERGENCE_WARNING\\n")
    }} else {{
        cat("STATUS: OK\\n")
    }}
'''


_PAIRWISE_TEMPLATE = '''\
suppressPackageStartupMessages({{
    library(lmerTest)
    library(emmeans)
}})

df <- read.csv("{csv}")
for (col in names(df)) {{
    if (is.character(df[[col]])) df[[col]] <- factor(df[[col]])
}}

''' + _FIT_BLOCK + '''
    # Variance components
    vc <- as.data.frame(VarCorr(fit))
    for (i in 1:nrow(vc)) {{
        cat(sprintf("VC: group=%s variance=%.10e stddev=%.10e\\n",
            vc$grp[i], vc$vcov[i], vc$sdcor[i]))
    }}

    # Satterthwaite ANOVA
    aov <- anova(fit, ddf="{ddf}")
    for (i in 1:nrow(aov)) {{
        cat(sprintf("OMNIBUS: effect=%s F=%.10e df1=%.4f df2=%.4f p=%.10e\\n",
            rownames(aov)[i], aov[["F value"]][i],
            aov[["NumDF"]][i], aov[["DenDF"]][i], aov[["Pr(>F)"]][i]))
    }}

    # emmeans pairwise contrast (uncorrected)
    emm <- emmeans(fit, as.formula(paste("~", "{factor}")), lmer.df="{ddf}")
    pw <- as.data.frame(pairs(emm, adjust="none"))
    for (i in 1:nrow(pw)) {{
        cat(sprintf("PAIR: contrast=%s estimate=%.10e SE=%.10e df=%.4f t=%.10e p=%.10e\\n",
            pw$contrast[i], pw$estimate[i], pw$SE[i],
            pw$df[i], pw$t.ratio[i], pw$p.value[i]))
    }}
}}
'''


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_status(stdout: str) -> tuple:
    """Return (ok, singular). ok=False indicates caller should fall back."""
    for line in stdout.splitlines():
        if line.startswith("STATUS:"):
            status = line.split(":", 1)[1].strip()
            if status == "FIT_FAILED":
                return False, False
            if status == "SINGULAR":
                return False, True
            if status == "CONVERGENCE_WARNING":
                logger.warning("LMM convergence warning")
                return False, False
            return True, False
    return False, False


def _parse_vc(stdout: str) -> List[VarianceComponent]:
    out: List[VarianceComponent] = []
    for line in stdout.splitlines():
        if line.startswith("VC:"):
            parts = _parse_kv(line[3:].strip())
            out.append(VarianceComponent(
                group=parts.get("group", ""),
                variance=float(parts.get("variance", 0)),
                std_dev=float(parts.get("stddev", 0)),
            ))
    return out


def _parse_omnibus(stdout: str) -> List[OmnibusEffect]:
    out: List[OmnibusEffect] = []
    for line in stdout.splitlines():
        if line.startswith("OMNIBUS:"):
            parts = _parse_kv(line[8:].strip())
            out.append(OmnibusEffect(
                effect=parts.get("effect", ""),
                f_value=float(parts.get("F", 0)),
                df1=float(parts.get("df1", 0)),
                df2=float(parts.get("df2", 0)),
                p_value=float(parts.get("p", 1)),
            ))
    return out


def _parse_pairs(stdout: str) -> List[PairwiseResult]:
    out: List[PairwiseResult] = []
    for line in stdout.splitlines():
        if line.startswith("PAIR:"):
            parts = _parse_kv(line[5:].strip())
            contrast = parts.get("contrast", "")
            la, lb = _split_contrast(contrast)
            out.append(PairwiseResult(
                contrast=contrast, level_a=la, level_b=lb,
                estimate=float(parts.get("estimate", 0)),
                se=float(parts.get("SE", 0)),
                df=float(parts.get("df", 0)),
                t_value=float(parts.get("t", 0)),
                p_value=float(parts.get("p", 1)),
            ))
    return out


def _parse_pairwise_output(stdout: str) -> Optional[LMMResult]:
    ok, singular = _parse_status(stdout)
    if not ok:
        return None
    return LMMResult(
        omnibus=_parse_omnibus(stdout),
        pairwise=_parse_pairs(stdout),
        variance_components=_parse_vc(stdout),
        singular=singular,
    )


def _parse_kv(text: str) -> Dict[str, str]:
    """Parse ``key=value key2=value2 ...`` allowing spaces inside values."""
    result: Dict[str, str] = {}
    pattern = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")
    for m in pattern.finditer(text):
        result[m.group(1)] = m.group(2).strip()
    return result


def _split_contrast(contrast: str) -> tuple:
    parts = contrast.split(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return contrast.strip(), ""
