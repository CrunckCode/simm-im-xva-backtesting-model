"""IM change attribution: Shapley, bridge, one-at-a-time, netting, scan. SYNTHETIC data; SIMM-style, not ISDA-certified."""
import dataclasses
import itertools

import numpy as np
import pandas as pd
import pytest

from simm_margin import attribution as A
from simm_margin import columns as C
from simm_margin import config
from simm_margin import params as P
from simm_margin.instruments import sample_portfolio
from simm_margin.market_history import generate_history

# ===== CONFIG (user inputs) =====
TOL = 1e-6                    # USD
SCAN_DAYS = 25
MIN_INTERACTION_USD = 1.0
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def setup():
    h = generate_history(quick=True)
    return h, sample_portfolio(), P.load(config.SIMM_VERSION_PRIOR), P.load(config.SIMM_VERSION)


@pytest.fixture(scope="module")
def scenario(setup):
    h, trades, _, _ = setup
    inputs = A.scenario_inputs(h, trades)
    return inputs, A._attribute(*inputs)


def _drivers(frame):
    return frame[frame["level"] == "driver"].set_index("driver")


def test_toy_shapley_symmetry_null_and_efficiency():
    names = ("a", "b", "c", "d")
    w = {"a": 2.0, "b": 2.0, "c": 5.0, "d": 0.0}        # a and b identical, d null
    v = {frozenset(s): sum(w[x] for x in s) ** 2 for k in range(5) for s in itertools.combinations(names, k)}
    phi = A.shapley_values(v, names)
    assert phi["a"] == pytest.approx(phi["b"], abs=1e-12)
    assert phi["d"] == pytest.approx(0.0, abs=1e-12)
    assert sum(phi.values()) == pytest.approx(v[frozenset(names)] - v[frozenset()], abs=1e-12)


def test_scenario_contributions_sum_to_total(scenario):
    _, res = scenario
    f = res.frame
    total = res.im1 - res.im0
    d = _drivers(f)
    assert d["shapley"].sum() == pytest.approx(total, abs=TOL)
    assert d["bridge"].sum() == pytest.approx(total, abs=TOL)
    assert f.set_index("driver").loc["residual", ["shapley", "bridge"]].abs().max() <= TOL
    steps = f[f["level"] == "substep"].set_index("driver")["bridge"]
    assert steps.sum() == pytest.approx(d.loc[A.DRV_MARKET, "bridge"], abs=TOL)
    assert d["shapley_share"].sum() == pytest.approx(1.0, abs=1e-9)
    assert len(res.subset_ims) == 16


def test_one_at_a_time_residual_is_the_interaction(scenario):
    _, res = scenario
    f = res.frame.set_index("driver")
    assert f.loc["residual", "one_at_a_time"] == pytest.approx(f.loc["total_dIM", "one_at_a_time"] - f.loc["sum_of_drivers", "one_at_a_time"], abs=TOL)
    assert abs(f.loc["residual", "one_at_a_time"]) > MIN_INTERACTION_USD     # the scenario has real interactions


def test_scenario_design(scenario):
    (p0, m0, th0, p1, m1, th1), res = scenario
    assert th0.version == config.SIMM_VERSION_PRIOR and th1.version == config.SIMM_VERSION
    assert set(p0[C.TRADE_ID]) - set(p1[C.TRADE_ID]) == set(A.MATURED_ON_SCENARIO_DAY)
    assert set(p1[C.TRADE_ID]) - set(p0[C.TRADE_ID]) == {A.NEW_TRADE["id"]}
    assert m1.par["USD"][0, 0] - m0.par["USD"][0, 0] == pytest.approx(35e-4, abs=1e-12)
    assert m1.fx_spot["EUR"][0] == pytest.approx(m0.fx_spot["EUR"][0] * (1 - 0.035))
    assert res.netting["netting_effect"] == pytest.approx(res.netting["incremental_im"] - res.netting["standalone_im"])
    assert res.netting["netting_effect"] < 0                              # a hedging-side trade nets against the book


def test_no_change_is_zero(setup):
    h, trades, _, th1 = setup
    m = h.market.row(10)
    f = A.attribute(trades, m, th1, trades, m, th1)
    nums = f[["shapley", "bridge", "one_at_a_time"]].fillna(0.0).to_numpy()
    assert np.abs(nums).max() <= TOL


def test_null_drivers_get_zero_when_only_market_moves(setup):
    h, trades, _, th1 = setup
    m0 = h.market.row(10)
    m1 = A.stress_market(m0)
    f = A.attribute(trades, m0, th1, trades, m1, th1)
    d = _drivers(f)
    total = f.set_index("driver").loc["total_dIM", "shapley"]
    assert d.loc[A.DRV_MARKET, "shapley"] == pytest.approx(total, abs=TOL)
    for null in (A.DRV_MATURED, A.DRV_NEW, A.DRV_PARAMS):
        assert abs(d.loc[null, "shapley"]) <= TOL and abs(d.loc[null, "bridge"]) <= TOL
    assert d.loc[A.DRV_MARKET, "one_at_a_time"] == pytest.approx(total, abs=TOL)


def test_netting_effect_for_a_perfect_hedge(setup):
    h, trades, _, th1 = setup
    m = h.market.row(10)
    hedge = trades[trades[C.TRADE_ID] == "IRS_USD_10Y"].copy()
    hedge[C.TRADE_ID], hedge[C.DIRECTION] = "HEDGE", -hedge[C.DIRECTION]
    p1 = pd.concat([trades, hedge], ignore_index=True)
    f = A.attribute(trades, m, th1, p1, m, th1)
    n = f.attrs["netting"]
    assert n["netting_effect"] < 0 and n["incremental_im"] < n["standalone_im"]


def test_changed_terms_for_same_id_rejected(setup):
    h, trades, _, th1 = setup
    p1 = trades.copy()
    p1.loc[0, C.NOTIONAL] *= 2
    with pytest.raises(ValueError):
        A.attribute(trades, h.market.row(10), th1, p1, h.market.row(10), th1)


def test_run_scenario_day_writes_files(setup, tmp_path):
    h, trades, _, _ = setup
    A.run_scenario_day(h, trades, tmp_path)
    for key in ("scenario", "methods", "subsets", "netting"):
        assert (tmp_path / A.FILES[key]).exists()
    assert len(pd.read_csv(tmp_path / A.FILES["subsets"])) == 16
    assert str(config.SCENARIO_DAY.date()) in pd.read_csv(tmp_path / A.FILES["scenario"])["date"].iloc[0]


def test_scan_flags_match_thresholds(setup, tmp_path, monkeypatch):
    h, trades, _, _ = setup
    mini = dataclasses.replace(h, dates=h.dates[:SCAN_DAYS], regime=h.regime[:SCAN_DAYS], market=h.market.take(np.arange(SCAN_DAYS)))
    monkeypatch.setattr(config, "BIG_MOVE_REL", 0.002)
    monkeypatch.setattr(config, "BIG_MOVE_ABS", 1.0e12)
    flagged = A.run_scan(mini, trades, tmp_path, workers=1)
    allday = pd.read_csv(tmp_path / A.FILES["scan"])
    expect = allday["dim_rel"].abs() > 0.002
    assert expect.sum() > 0
    assert set(flagged["date"].astype(str)) == set(allday.loc[expect, "date"].astype(str))
    assert flagged["d_rates"].notna().any()
    top = flagged.loc[flagged["dim"].abs().idxmax()]
    assert top["d_rates"] + top["d_fx"] + top["d_vol"] == pytest.approx(top["dim"], rel=1e-9, abs=1e-6)   # market-only day
