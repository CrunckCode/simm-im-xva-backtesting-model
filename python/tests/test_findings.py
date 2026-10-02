"""Findings log: mappings, evidence and file existence. SYNTHETIC data; SIMM-style, not ISDA-certified."""
from pathlib import Path

import pandas as pd
import pytest

from simm_margin import columns as C
from simm_margin import config
from simm_margin import findings as F
from simm_margin.instruments import sample_portfolio
from simm_margin.market_history import generate_history

# ===== CONFIG (user inputs) =====
MODULES = ("dispute.py", "attribution.py", "uat.py", "findings.py")
FORBIDDEN = (chr(0x2014), chr(0x2013), " " + "-" * 2 + " ")        # em dash, en dash, double-hyphen dash substitute
# ===== END CONFIG =====


def _bt(rows):
    cols = ["test", "model", "split", "value", "threshold", "light", "n", "exceptions", "note", "data_source"]
    return pd.DataFrame([dict(zip(cols, r + ("", "synthetic"))) for r in rows], columns=cols)


@pytest.fixture(scope="module")
def inputs():
    return generate_history(quick=True), sample_portfolio()


@pytest.fixture()
def out(tmp_path):
    bt = _bt([
        ("overall", "hs_im_eu_1plus3", "1d", 18, "t", "AMBER", 1327, 18),
        ("overall", "hs_im_eu_1plus3", "10d_overlapping", 31, "t", "RED", 1318, 31),     # ignored: over-sized design
        ("overall", "hs_im_eu_1plus3", "10d_nonoverlap", 2, "t", "PASS", 132, 2),
        ("rolling250_max", "hs_im_eu_1plus3", "1d", 12, "t", "RED", 250, 12),
        ("overall", "hs_im_ewma", "1d", 31, "t", "RED", 1327, 31),                        # challenger: not a finding
        ("overall", "xva_var", "10d_nonoverlap", 4, "t", "AMBER", 132, 4),
        ("shortfall_multiplier", "hs_im_eu_1plus3", "10d_overlapping", 1.14, "k", "Info", 1318, 31),
    ])
    bt.to_csv(tmp_path / F.BACKTEST_FILE, index=False)
    pd.DataFrame([{"scenario": "D3", "seeded_cause": "zero_rate_delta", "rank_of_seeded_cause": 2, "seeded_ranks": "2",
                   "im_a": 5e6, "im_b": 5.1e6, "gap": 1e5, "top_ranked_cause": "stale_curve", "residual_after_top_fix": 0.0,
                   "accepted": False}]).to_csv(tmp_path / F.DISPUTE_FILE, index=False)
    pd.DataFrame([{"test_id": "U03", "area": "sensitivities", "description": "ladder", "expected": "x", "tolerance": "t",
                   "actual": "ladder 1 vs 2", "result": "FAIL"}]).to_csv(tmp_path / F.UAT_FILE, index=False)
    pd.DataFrame([{"label": "scenario_day", "date": "2026-07-10", "simm_im": 9e6, "schedule_gross": 8e6, "schedule_ngr": 1.0,
                   "schedule_net": 8e6, "simm_over_schedule_net": 1.125}]).to_csv(tmp_path / F.SCHEDULE_FILE, index=False)
    pd.DataFrame([{"risk_class": "IR", "margin_type": "Curvature", "n_checks": 60, "n_violations": 3, "max_excess_usd": 1234.0}]
                 ).to_csv(tmp_path / F.SUBADD_FILE, index=False)
    return tmp_path


def test_automatic_mappings_and_columns(out, inputs):
    h, trades = inputs
    log = F.build_log(out, h, trades)
    assert list(log.columns) == list(F.LOG_COLUMNS)
    auto = log[log[C.SOURCE] == "auto"]
    titles = " | ".join(auto[C.TITLE])
    sev = dict(zip(auto[C.TITLE], auto[C.SEVERITY]))
    assert sev["rolling250_max RED for hs_im_eu_1plus3"] == "High"
    assert sev["overall AMBER for hs_im_eu_1plus3"] == "Medium"
    assert sev["overall AMBER for xva_var"] == "Medium"
    assert "ewma" not in titles                                              # challengers are comparators
    assert not any("RED for hs_im_eu_1plus3" in t and t.startswith("overall") for t in auto[C.TITLE])   # overlap RED ignored
    assert sev["UAT U03 failed: ladder"] == "High"
    assert any(t.startswith("Dispute D3") and s == "High" for t, s in sev.items())
    assert sev["SIMM-style IM exceeds the schedule IM on the sample book"] == "Medium"
    assert sev["IR Curvature margin is not sub-additive on random book splits"] == "Low"
    ev = auto.set_index(C.TITLE).loc["overall AMBER for hs_im_eu_1plus3", C.EVIDENCE]
    assert "18 exceptions in 1327" in ev and "1.140" in ev


def test_static_findings_present_and_sorted(out, inputs):
    h, trades = inputs
    log = F.build_log(out, h, trades)
    rev = log[log[C.SOURCE] == "review"]
    assert len(rev) >= 11
    key = " ".join(rev[C.TITLE]).lower()
    for phrase in ("single ois-style curve", "inflation or credit", "normal approximation", "hypothetical p&l", "one simm version",
                   "kupiec", "237.8", "vega correlation", "credit, equity and commodity", "synthetic", "overlap"):
        assert phrase in key, phrase
    order = log[C.SEVERITY].map(F.SEVERITY_ORDER)
    assert order.is_monotonic_increasing
    assert log[C.FINDING_ID].is_unique and log[C.FINDING_ID].iloc[0] == "F-01"
    assert (log[C.TRAFFIC_LIGHT] == log[C.SEVERITY].map(F.SEVERITY_TO_LIGHT)).all()
    assert set(log[C.SEVERITY]) <= set(F.SEVERITY_ORDER)


def test_every_evidence_file_exists(out, inputs):
    h, trades = inputs
    log = F.build_log(out, h, trades)
    for name in log[C.EVIDENCE_FILE]:
        assert name and ((out / name).exists() or (config.ROOT / name).exists()), name
    assert (out / F.LOG_FILE).exists()


def test_evidence_numbers_come_from_the_files(out, inputs):
    h, trades = inputs
    log = F.build_log(out, h, trades).set_index(C.TITLE)
    assert "USD 9,000,000" in log.loc["SIMM-style IM exceeds the schedule IM on the sample book", C.EVIDENCE]
    assert "3 of 60" in log.loc["IR Curvature margin is not sub-additive on random book splits", C.EVIDENCE]
    assert "31 exceptions in 1318 overlapping" in log.loc["Backtest windows overlap for the 10-day horizon", C.EVIDENCE]


def test_no_outputs_still_builds_static_and_writes_helper_tables(tmp_path, inputs):
    h, trades = inputs
    log = F.build_log(tmp_path, h, trades)
    assert (log[C.SOURCE] == "review").sum() >= 11
    assert (tmp_path / F.SCHEDULE_FILE).exists() and (tmp_path / F.SUBADD_FILE).exists()
    for name in log[C.EVIDENCE_FILE]:
        assert (tmp_path / name).exists() or (config.ROOT / name).exists(), name


def test_no_dash_substitutes_in_sources_or_output(out, inputs):
    h, trades = inputs
    log = F.build_log(out, h, trades)
    text = (out / F.LOG_FILE).read_text(encoding="utf-8")
    pkg = Path(F.__file__).parent
    texts = [text] + [(pkg / m).read_text(encoding="utf-8") for m in MODULES]
    for t in texts:
        for bad in FORBIDDEN:
            assert bad not in t, repr(bad)
    assert len(log) > 0
