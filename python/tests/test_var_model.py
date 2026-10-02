"""HS IM tests: independent revaluation, monotonicity, zero book, stress-window rule, SIMM series stub. SYNTHETIC data."""
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest

from simm_margin import columns as C
from simm_margin import config
from simm_margin import var_model as V
from simm_margin.instruments import sample_portfolio
from simm_margin.market_history import MarketBatch, generate_history
from simm_margin.pricing import price_portfolio

# ===== CONFIG (user inputs) =====
HORIZONS = (1, 10)
STRESS_TOL = 1e-9
PERF_SECONDS_QUICK = 30.0
N_STUB_DATES = 3
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def full():
    return generate_history()


@pytest.fixture(scope="module")
def quick():
    return generate_history(quick=True)


@pytest.fixture(scope="module")
def trades():
    return sample_portfolio()


def test_independent_revaluation_plain250(quick, trades):
    """Rebuild the 250-day scenario set by explicit loops and compare with the vectorised engine."""
    h, t = 10, V.first_valid_index(quick, 10) + 5
    m = quick.market
    srcs = np.arange(t - 249, t + 1)
    v0 = price_portfolio(trades, m.row(t)).sum()
    loss = []
    for s in srcs:
        one = m.row(t)
        z = {c: one.zero[c] + (m.zero[c][s] - m.zero[c][s - h]) for c in m.zero}
        p = {c: one.par[c] + (m.par[c][s] - m.par[c][s - h]) for c in m.par}
        fx = {c: one.fx_spot[c] * m.fx_spot[c][s] / m.fx_spot[c][s - h] for c in m.fx_spot}
        iv = {c: one.ir_nvol[c] * m.ir_nvol[c][s] / m.ir_nvol[c][s - h] for c in m.ir_nvol}
        fv = {c: one.fx_vol[c] * m.fx_vol[c][s] / m.fx_vol[c][s - h] for c in m.fx_vol}
        sc = MarketBatch(p, z, fx, iv, fv, one.cds, {k: v + (m.xccy_basis[k][s] - m.xccy_basis[k][s - h]) for k, v in one.xccy_basis.items()})
        loss.append(-(price_portfolio(trades, sc).sum() - v0))
    ref = float(np.quantile(loss, config.CONF, method="inverted_cdf"))
    assert V.hs_im(quick, trades, t, 10, config.CONF, V.MODEL_PLAIN250) == pytest.approx(ref, rel=1e-9)


def test_series_matches_scalar(quick, trades):
    s = V.im_series(quick, trades, 10)
    t = int(s.index.get_indexer([s.index[40]])[0]) + V.first_valid_index(quick, 10)
    for w in (V.MODEL_CHAMPION, V.MODEL_EWMA, V.MODEL_SCALED):
        assert V.hs_im(quick, trades, s.index[40], 10, config.CONF, w) == pytest.approx(
            V.im_series(quick, trades, 10, model=w).iloc[40], rel=1e-10)
    assert t > 0


def test_monotone_in_confidence_and_horizon(quick, trades):
    t = quick.dates[-5]
    ims = [V.hs_im(quick, trades, t, 10, c) for c in (0.90, 0.95, 0.99, 0.995)]
    assert ims == sorted(ims) and ims[0] < ims[-1]
    one, ten = V.im_series(quick, trades, 1), V.im_series(quick, trades, 10)
    assert (ten > one).all()                                    # 10-day IM above 1-day IM on every date
    five = V.im_series(quick, trades, 5)
    assert ((one < five) & (five < ten * 1.05)).mean() > 0.9    # in between on nearly all dates (finite-sample HS is not exactly monotone)


def test_zero_notional_book_has_zero_im(quick, trades):
    zero = trades.copy()
    zero[C.NOTIONAL] = 0.0
    assert V.hs_im(quick, zero, quick.dates[-3], 10) == 0.0
    assert (V.realized_loss_series(quick, zero, 1).dropna() == 0.0).all()


def test_scaled_challenger_is_scaled_champion(quick, trades):
    a, b = V.im_series(quick, trades, 1), V.im_series(quick, trades, 1, model=V.MODEL_SCALED)
    assert np.allclose(b, config.CHALLENGER_SCALE * a)


def test_ewma_weights():
    w = V.ewma_weights(250)
    assert w.sum() == pytest.approx(1.0) and w[-1] / w[-2] == pytest.approx(1 / config.EWMA_LAMBDA) and w[-1] == w.max()


def test_weighted_var_equal_weights_is_inverted_cdf():
    x = np.random.default_rng(0).standard_normal((3, 750))
    ref = np.quantile(x, 0.99, axis=1, method="inverted_cdf")
    assert np.allclose(V.weighted_var(x, np.full(750, 1 / 750), 0.99), ref)


@pytest.mark.parametrize("h", HORIZONS)
def test_stress_share_at_least_minimum_full_history(full, h):
    comp = V.window_composition(full, h)
    assert (comp["stressed_share"] >= config.MIN_STRESS_SHARE - STRESS_TOL).all()
    assert (comp["n_recent"] + comp["n_stress_window"] == config.WINDOW_DAYS).all()
    assert comp["n_stress_window"].max() > 0                    # replacement happens once the stress window has aged out
    first = comp[comp["date"] <= full.stress_dates[1] + pd.Timedelta(days=700)]
    assert (first["n_stress_window"] == 0).any()                # and is not needed while stress is still in the window


def test_stress_window_replaces_oldest_only(full, trades):
    comp = V.window_composition(full, 10)
    ts = np.array([V.first_valid_index(full, 10) + int(comp["n_stress_window"].values.argmax())])
    m = V.stress_replacement(full, ts, 10)[0]
    assert 0 < m <= full.stress_batch.n_states - 10
    cs = np.cumsum(full.regime)
    kept = cs[ts[0]] - cs[ts[0] - (config.WINDOW_DAYS - m)]
    assert kept + m >= int(np.ceil(config.MIN_STRESS_SHARE * config.WINDOW_DAYS))
    if m > 0:                                                   # minimal: one fewer replaced day would fall short
        kept1 = cs[ts[0]] - cs[ts[0] - (config.WINDOW_DAYS - m + 1)]
        assert kept1 + m - 1 < int(np.ceil(config.MIN_STRESS_SHARE * config.WINDOW_DAYS))


def test_realized_loss_series(quick, trades):
    l1, l10 = V.realized_loss_series(quick, trades, 1), V.realized_loss_series(quick, trades, 10)
    v = price_portfolio(trades, quick.market).sum(1)
    assert l1.iloc[5] == pytest.approx(-(v[6] - v[5])) and l10.iloc[5] == pytest.approx(-(v[15] - v[5]))
    assert l1.iloc[-1:].isna().all() and l10.iloc[-10:].isna().all() and l10.iloc[:-10].notna().all()


def test_date_without_window_raises(quick, trades):
    with pytest.raises(ValueError):
        V.hs_im(quick, trades, quick.dates[3], 10)


def test_quick_series_runtime(quick, trades):
    V.clear_cache()
    t0 = time.time()
    V.im_series(quick, trades, 1)
    V.im_series(quick, trades, 10)
    assert time.time() - t0 < PERF_SECONDS_QUICK


def test_daily_simm_series_with_stub(quick, trades, monkeypatch):
    """SIMM is stubbed on the real modules: total IM = sum of absolute USD sensitivities."""
    calls = []
    monkeypatch.setattr("simm_margin.params.load", lambda version: calls.append(version) or "P")
    monkeypatch.setattr("simm_margin.simm.simm",
                        lambda crif, p: types.SimpleNamespace(total=float(crif[C.AMOUNT_USD].abs().sum())))
    dates = quick.dates[-N_STUB_DATES:]
    out = V.daily_simm_series(quick, trades, dates, "2.8+2512")
    assert len(out) == N_STUB_DATES and (out > 0).all() and calls == ["2.8+2512"]
    assert list(out.index) == list(dates)
