"""UAT framework for the EUR/USD cross-currency swap. SYNTHETIC data; SIMM-style, not ISDA-certified."""
import json

import pandas as pd
import pytest

from simm_margin import uat
from simm_margin.market_history import generate_history

# ===== CONFIG (user inputs) =====
MIN_TESTS = 15
ALLOWED_RESULTS = {"PASS", "FAIL", "SKIPPED"}
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("uat")
    res = uat.run_uat(out, out / "suite.csv", generate_history(quick=True))
    return out, res


def test_suite_file_is_machine_readable(tmp_path):
    path = uat.write_suite(tmp_path / "s" / "suite.csv")
    s = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(s.columns) == uat.SUITE_COLUMNS
    assert len(s) >= MIN_TESTS and s["test_id"].is_unique
    assert set(s["result"]) == {uat.NOT_RUN} and (s["actual"] == "").all()
    assert set(s["test_id"]) == set(uat.TESTS)


def test_results_are_complete_and_honest(run):
    out, res = run
    assert len(res) >= MIN_TESTS
    assert set(res["result"]) <= ALLOWED_RESULTS
    assert (res["actual"].str.len() > 0).all() and (res["executed_at"] != "").all()
    assert (out / uat.RESULTS_FILE).exists()
    assert all((out / f).exists() for f in res["evidence_file"])
    skipped = set(res.loc[res["result"] == "SKIPPED", "test_id"])
    assert skipped <= {"U15"}                                 # only the Excel check may be skipped
    assert not (res["result"] == "FAIL").any(), res.loc[res["result"] == "FAIL", ["test_id", "actual"]].to_string()


def test_core_numbers_in_actuals(run):
    _, res = run
    a = res.set_index("test_id")["actual"]
    assert a["U01"].startswith("PV = 0 USD") or a["U01"].startswith("PV = ")
    assert "rank 1 = missing_trade" in a["U14"]
    assert "basis RW 21, CR 1" in a["U06"]


def test_excel_check_skips_without_workbook(run, monkeypatch, tmp_path):
    monkeypatch.setattr(uat, "EXCEL_DIR", tmp_path / "no_excel")
    ok, msg = uat._u15(uat._Fixture(generate_history(quick=True), tmp_path))
    assert ok is None and "not built" in msg


def test_excel_check_reads_reconciliation_report(monkeypatch, tmp_path):
    (tmp_path / "excel").mkdir()
    (tmp_path / "excel" / "m.xlsx").write_bytes(b"x")
    monkeypatch.setattr(uat, "EXCEL_DIR", tmp_path / "excel")
    f = uat._Fixture(generate_history(quick=True), tmp_path)
    (tmp_path / uat.EXCEL_RECON_FILE).write_text(json.dumps({"checks": 10, "n_failed": 0}))
    assert uat._u15(f)[0] is True
    (tmp_path / uat.EXCEL_RECON_FILE).write_text(json.dumps({"checks": 10, "failures": ["a", "b"]}))
    assert uat._u15(f)[0] is False
    (tmp_path / uat.EXCEL_RECON_FILE).write_text(json.dumps({"hello": 1}))
    assert uat._u15(f)[0] is None


def test_count_failures_helper():
    assert uat._count_failures({"a": {"n_fail": 2}, "b": [{"failed": 1}]}) == 3
    assert uat._count_failures({"ok": True}) is None


def test_failing_test_is_recorded_as_fail(tmp_path, monkeypatch):
    monkeypatch.setitem(uat.TESTS, "U01", lambda f: (False, "forced"))
    res = uat.run_uat(tmp_path, tmp_path / "suite.csv", generate_history(quick=True))
    row = res.set_index("test_id").loc["U01"]
    assert row["result"] == "FAIL" and row["actual"] == "forced"


def test_erroring_test_is_recorded_as_fail(tmp_path, monkeypatch):
    def boom(f):
        raise RuntimeError("broken")
    monkeypatch.setitem(uat.TESTS, "U02", boom)
    res = uat.run_uat(tmp_path, tmp_path / "suite.csv", generate_history(quick=True))
    row = res.set_index("test_id").loc["U02"]
    assert row["result"] == "FAIL" and "RuntimeError" in row["actual"]
