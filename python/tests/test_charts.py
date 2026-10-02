"""Smoke tests for charts.py: every chart is written into a temporary folder and python/outputs is never touched."""
import shutil
from pathlib import Path

import pytest

from simm_margin import charts, config

# ===== CONFIG (user inputs) =====
REAL_OUT = config.OUT_DIR
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MIN_CHARTS = 12
MIN_BYTES = 5_000
COPY_SUFFIXES = (".csv", ".json")
# ===== END CONFIG =====

pytestmark = pytest.mark.skipif(not (REAL_OUT / charts.FILE_SERIES).exists(), reason="outputs not built")


@pytest.fixture(scope="module")
def tmp_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    for f in REAL_OUT.iterdir():
        if f.is_file() and f.suffix in COPY_SUFFIXES:
            shutil.copy(f, out / f.name)
    shutil.copytree(REAL_OUT / charts.DETAIL_DIR, out / charts.DETAIL_DIR)
    return out


def _snapshot(folder: Path):
    return {p.name: p.stat().st_mtime_ns for p in folder.glob("*.png")}


def test_charts_written_to_temp_dir(tmp_out):
    before = _snapshot(REAL_OUT / "charts") if (REAL_OUT / "charts").exists() else {}
    written = charts.make_all_charts(out_dir=tmp_out, chart_dir=tmp_out / "charts", build_profile=False)
    assert len(written) >= MIN_CHARTS
    for path in written.values():
        assert Path(path).parent == tmp_out / "charts"
        data = Path(path).read_bytes()
        assert data.startswith(PNG_MAGIC) and len(data) > MIN_BYTES
    after = _snapshot(REAL_OUT / "charts") if (REAL_OUT / "charts").exists() else {}
    assert before == after, "the real chart folder must not be touched by tests"


def test_missing_inputs_are_skipped(tmp_path):
    assert charts.make_all_charts(out_dir=tmp_path, chart_dir=tmp_path / "c", build_profile=False) == {}


def test_partial_inputs_still_chart(tmp_out, tmp_path):
    for name in (charts.FILE_UAT, charts.FILE_SUMMARY):
        shutil.copy(tmp_out / name, tmp_path / name)
    written = charts.make_all_charts(out_dir=tmp_path, chart_dir=tmp_path / "c", build_profile=False)
    assert "chart_uat" in written and "chart_im_breakdown" in written and "chart_pvalues" not in written


def test_cva_profile_written_in_temp_dir(tmp_path):
    for name in (charts.FILE_SERIES, charts.FILE_META):
        shutil.copy(REAL_OUT / name, tmp_path / name)
    charts.write_cva_profile(tmp_path)
    assert (tmp_path / charts.FILE_CVA_PROFILE).exists()
    written = charts.make_all_charts(out_dir=tmp_path, chart_dir=tmp_path / "c", build_profile=False)
    assert "chart_cva_profile" in written
