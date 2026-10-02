"""Backtest statistics against closed forms, scipy references, hand cases and a size simulation. SYNTHETIC data only."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, chi2

from simm_margin import backtest as B
from simm_margin import columns as C
from simm_margin import config
from simm_margin.market_history import generate_history

# ===== CONFIG (user inputs) =====
P = 0.01
N_SIM, SIM_N, SIM_SEED = 2000, 1000, 7
SIM_TOL_SE = 4.0
DOC_CUM = {4: 0.8922, 5: 0.9588, 9: 0.99975, 10: 0.99995}      # BCBS 22 Table 2 cumulative probabilities
DOC_CUM_TOL = 6e-5
PLUS = {0: 0.0, 4: 0.0, 5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85, 10: 1.0, 15: 1.0}
# ===== END CONFIG =====


def test_zone_boundaries_table2():
    assert [B.basel_zone(x, 250) for x in (0, 4)] == ["Green", "Green"]
    assert [B.basel_zone(x, 250) for x in (5, 9)] == ["Yellow", "Yellow"]
    assert [B.basel_zone(x, 250) for x in (10, 25)] == ["Red", "Red"]


@pytest.mark.parametrize("x,cum", list(DOC_CUM.items()))
def test_cumulative_probabilities_table2(x, cum):
    assert abs(binom.cdf(x, 250, P) - cum) < DOC_CUM_TOL


@pytest.mark.parametrize("x,plus", list(PLUS.items()))
def test_plus_factors_from_file(x, plus):
    zone, factor, src = B.basel_table_zone(x)
    assert factor == pytest.approx(plus)
    assert zone == B.basel_zone(x, 250)


def test_table_fallback_used_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "BCBS_TABLE_FILE", tmp_path / "missing.csv")
    df, src = B.bcbs22_table()
    assert "fallback" in src and df.loc[df["exceptions"] == 5, "plus_factor"].iloc[0] == 0.40
    assert B.basel_table_zone(12)[:2] == ("Red", 1.0)


@pytest.mark.parametrize("x,n", [(0, 250), (1, 250), (5, 250), (13, 1327), (40, 1327), (250, 250)])
def test_kupiec_vs_scipy(x, n):
    phat = x / n
    lr = 2.0 * (binom.logpmf(x, n, phat) - binom.logpmf(x, n, P))      # binomial coefficient cancels
    out = B.kupiec_pof(x, n, P)
    assert out["LR"] == pytest.approx(lr, abs=1e-9)
    assert out["p_value"] == pytest.approx(chi2.sf(lr, 1), abs=1e-12)


def test_kupiec_closed_forms():
    assert B.kupiec_pof(0, 250, P)["LR"] == pytest.approx(-2 * 250 * math.log(1 - P), rel=1e-12)       # x = 0
    assert B.kupiec_pof(250, 250, P)["LR"] == pytest.approx(-2 * 250 * math.log(P), rel=1e-12)         # x = n
    exact = B.kupiec_pof(25, 2500, P)                                                                   # x = p n
    assert exact["LR"] == pytest.approx(0.0, abs=1e-9) and exact["p_value"] == pytest.approx(1.0)
    assert B.kupiec_pof(0, 250, P)["p_value"] == pytest.approx(0.02498, abs=1e-5)                       # 2*(1-Phi(sqrt(5.0252)))


def _rejection_exact(n):
    xs = np.arange(n + 1)
    rej = np.array([B.kupiec_pof(int(k), n, P)["p_value"] < 0.05 for k in xs])
    return float((binom.pmf(xs, n, P) * rej).sum())


def test_kupiec_size_simulation():
    rng = np.random.default_rng(SIM_SEED)
    rate = np.mean([B.kupiec_pof(int(k), SIM_N, P)["p_value"] < 0.05 for k in rng.binomial(SIM_N, P, N_SIM)])
    exact = _rejection_exact(SIM_N)
    assert abs(rate - exact) < SIM_TOL_SE * math.sqrt(exact * (1 - exact) / N_SIM)
    assert abs(exact - 0.05) < 0.02                   # close to nominal at n = 1000 (discreteness at n = 250 is larger)


def test_kupiec_size_simulation_raw_bernoulli_series():
    rng = np.random.default_rng(SIM_SEED + 1)
    rate = np.mean([B.kupiec_pof(int(s.sum()), 250, P)["p_value"] < 0.05 for s in rng.random((N_SIM, 250)) < P])
    exact = _rejection_exact(250)                     # exact finite-sample size at n = 250
    assert abs(rate - exact) < SIM_TOL_SE * math.sqrt(exact * (1 - exact) / N_SIM)


def _ind_reference(e):
    """Independent plain-python Christoffersen independence LR with 0 ln 0 = 0."""
    c = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for a, b in zip(e[:-1], e[1:]):
        c[(a, b)] += 1
    xl = lambda k, q: 0.0 if k == 0 else k * math.log(q)
    pi01 = c[(0, 1)] / (c[(0, 0)] + c[(0, 1)])
    pi11 = c[(1, 1)] / (c[(1, 0)] + c[(1, 1)]) if c[(1, 0)] + c[(1, 1)] else 0.0
    pi = (c[(0, 1)] + c[(1, 1)]) / sum(c.values())
    la = xl(c[(0, 0)], 1 - pi01) + xl(c[(0, 1)], pi01) + xl(c[(1, 0)], 1 - pi11) + xl(c[(1, 1)], pi11)
    l0 = xl(c[(0, 0)] + c[(1, 0)], 1 - pi) + xl(c[(0, 1)] + c[(1, 1)], pi)
    return 2 * (la - l0)


def test_christoffersen_hand_case():
    e = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]          # n00=4 n01=2 n10=2 n11=1
    out = B.christoffersen(e, P)
    assert (out["n00"], out["n01"], out["n10"], out["n11"]) == (4, 2, 2, 1)
    assert out["pi01"] == pytest.approx(2 / 6) and out["pi11"] == pytest.approx(1 / 3)
    assert out["LR_ind"] == pytest.approx(_ind_reference(e), abs=1e-12)
    assert out["LR_cc"] == pytest.approx(B.kupiec_pof(3, 10, P)["LR"] + out["LR_ind"], abs=1e-12)
    assert out["p_ind"] == pytest.approx(chi2.sf(out["LR_ind"], 1)) and out["p_cc"] == pytest.approx(chi2.sf(out["LR_cc"], 2))


def test_christoffersen_zero_cells():
    isolated = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]            # n11 = 0
    out = B.christoffersen(isolated, P)
    assert out["n11"] == 0 and np.isfinite(out["LR_ind"]) and out["LR_ind"] == pytest.approx(_ind_reference(isolated), abs=1e-12)
    none = B.christoffersen([0] * 50, P)                  # no exceptions at all
    assert none["LR_ind"] == 0.0 and none["p_ind"] == 1.0 and np.isfinite(none["LR_cc"])
    clustered = B.christoffersen([0] * 40 + [1] * 5 + [0] * 40, P)
    assert clustered["p_ind"] < 0.01                      # clustering is detected


def test_christoffersen_size_simulation():
    rng = np.random.default_rng(SIM_SEED + 2)
    rate = np.mean([B.christoffersen(s.astype(int), P)["p_ind"] < 0.05 for s in rng.random((N_SIM, 1000)) < P])
    assert 0.0 < rate < 0.12                             # sparse hit sequences: close to but not exactly nominal


def test_exceptions_strict_and_nan():
    loss = pd.Series([1.0, 2.0, np.nan, 3.0])
    out = B.exceptions(loss, pd.Series([1.0, 1.0, 0.0, 5.0]))
    assert out.tolist() == [0, 1, 0, 0]


def test_rolling_and_by_regime():
    exc = pd.Series([1, 0, 1, 1, 0, 0])
    assert B.rolling(exc, 3).dropna().tolist() == [2, 2, 2, 1]
    r = B.by_regime([1, 0, 1, 1], [0, 0, 1, 1]).set_index(C.REGIME)
    assert r.loc["calm", "exceptions"] == 1 and r.loc["stress", "exceptions"] == 2 and r.loc["stress", "n"] == 2


@pytest.mark.parametrize("n,seed", [(250, 1), (1000, 2), (1327, 3)])
def test_shortfall_multiplier_property(n, seed):
    rng = np.random.default_rng(seed)
    im = rng.uniform(0.8, 1.2, n)
    loss = rng.standard_t(4, n) * 0.6
    k = B.shortfall_multiplier(loss, im)
    zone = lambda kk: B.basel_zone(int(B.exceptions(loss, kk * im).sum()), n)
    assert k > 0 and zone(k) == "Green"
    assert zone(k * (1 - 1e-6)) != "Green"


def test_shortfall_multiplier_is_zero_when_always_green():
    assert B.shortfall_multiplier(np.full(250, -1.0), np.ones(250)) == 0.0


def test_battery_rows_and_lights():
    rng = np.random.default_rng(5)
    good = (rng.random(1000) < P).astype(int)
    rows = pd.DataFrame(B.battery_rows(good, "m", "1d"))
    assert set(rows[C.LIGHT]) <= {"PASS", "AMBER", "RED"}
    bad = pd.DataFrame(B.battery_rows((rng.random(1000) < 0.05).astype(int), "m", "1d")).set_index(C.TEST)
    assert bad.loc["overall", C.LIGHT] == "RED" and bad.loc["kupiec_pof", C.LIGHT] == "RED"
    assert B.worst("PASS", "AMBER") == "AMBER" and B.light_p(0.03) == "AMBER" and B.light_p(0.005) == "RED"


@pytest.fixture(scope="module")
def quick_battery(tmp_path_factory):
    out = tmp_path_factory.mktemp("battery")
    res = B.run_battery(generate_history(quick=True), out_dir=out)
    return res, out


def test_run_battery_outputs(quick_battery):
    res, out = quick_battery
    assert out.joinpath(B.RESULTS_FILE).exists() and out.joinpath(B.SERIES_FILE).exists()
    assert list(res.columns[:6]) == [C.TEST, C.MODEL, "split", "value", "threshold", C.LIGHT]
    assert (res["data_source"] == config.DATA_SOURCE).all()
    assert {"hs_im_eu_1plus3", "hs_im_ewma", "hs_im_plain250", "hs_im_scaled", "xva_var"} <= set(res[C.MODEL])
    over = res[(res["split"] == "10d_overlapping") & (res[C.TEST] != "shortfall_multiplier")]
    assert over["note"].str.contains("BCBS 22 section II").all()
    ser = pd.read_csv(out / B.SERIES_FILE)
    assert {C.DATE, C.MODEL, C.LOSS, C.VAR_FORECAST, C.EXCEPTION, C.REGIME} <= set(ser.columns)


def test_scaled_challenger_has_more_exceptions(quick_battery):
    res, _ = quick_battery
    o = res[(res[C.TEST] == "overall") & (res["split"] == "1d")].set_index(C.MODEL)["exceptions"]
    assert o["hs_im_scaled"] >= o["hs_im_eu_1plus3"]
