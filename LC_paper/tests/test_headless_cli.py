import sys
from pathlib import Path

import pytest

from LC_paper import run_hyperthermia as hyper
from LC_paper import run_opto as opto
from LC_paper import run_threshold_sweep as sweep


@pytest.mark.parametrize("module,dataset", [
    (hyper, "LC_GCaMP"),
    (opto, "Hotspot_GANE/CPP_treatment"),
])
def test_cli_paths_and_success(module, dataset, tmp_path, monkeypatch):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(sys, "argv", [
        module.__file__, "--only", dataset,
        "--data-root", str(data), "--cache-root", str(cache),
        "--output-root", str(output),
    ])
    if module is hyper:
        monkeypatch.setattr(hyper, "HYPER_ROOT", Path("unused"))
        monkeypatch.setattr(sweep, "CACHE_DIR", Path("unused"))
        monkeypatch.setattr(hyper, "run_one_subfolder", lambda *a, **k: calls.append((a, k)))
    else:
        monkeypatch.setattr(opto, "OPTO_BASE", Path("unused"))
        monkeypatch.setattr(opto, "OPTO_CACHE_DIR", Path("unused"))
        monkeypatch.setattr(opto, "run_one_root", lambda *a, **k: calls.append((a, k)))
    module.main()
    assert len(calls) == 1
    assert calls[0][0][0] == dataset
    assert calls[0][1]["output_root"] == output
    assert (hyper.HYPER_ROOT if module is hyper else opto.OPTO_BASE) == data
    assert (sweep.CACHE_DIR if module is hyper else opto.OPTO_CACHE_DIR) == cache


@pytest.mark.parametrize("module,dataset", [
    (hyper, "LC_GCaMP"),
    (opto, "Hotspot_GANE/CPP_treatment"),
])
def test_missing_dataset_exits_nonzero(module, dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        module.__file__, "--only", dataset,
        "--data-root", str(tmp_path / "missing"),
        "--cache-root", str(tmp_path / "cache"),
    ])
    monkeypatch.setattr(hyper, "HYPER_ROOT", hyper.HYPER_ROOT)
    monkeypatch.setattr(sweep, "CACHE_DIR", sweep.CACHE_DIR)
    monkeypatch.setattr(opto, "OPTO_BASE", opto.OPTO_BASE)
    monkeypatch.setattr(opto, "OPTO_CACHE_DIR", opto.OPTO_CACHE_DIR)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1


def test_hyperthermia_empty_dataset_raises(tmp_path, monkeypatch):
    (tmp_path / "LC_GCaMP").mkdir()
    monkeypatch.setattr(hyper, "HYPER_ROOT", tmp_path)
    monkeypatch.setattr(hyper, "load_sessions", lambda *a, **k: [])
    with pytest.raises(ValueError, match="No loadable sessions"):
        hyper.run_one_subfolder("LC_GCaMP", "lc", 0.015, True)


def test_opto_empty_dataset_raises(tmp_path, monkeypatch):
    (tmp_path / "CPP").mkdir()
    monkeypatch.setattr(opto, "OPTO_BASE", tmp_path)
    monkeypatch.setattr(opto, "load_sessions", lambda *a, **k: [])
    with pytest.raises(ValueError, match="No loadable sessions"):
        opto.run_one_root("CPP", "cpp", None, zscore=False, dff_label="dF/F")
