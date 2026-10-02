"""Pricing tests on synthetic market states: closed forms, parity, vectorisation, performance."""
import time

import numpy as np
import pytest
from scipy.stats import norm

from simm_margin import columns as C
from simm_margin.instruments import (BASIS_SPREAD, new_product_xccy, sample_portfolio, _row)
from simm_margin.market_history import generate_history
from simm_margin.pricing import (bachelier, bachelier_vega, fx_forward_rate, gk_delta_spot, gk_price, gk_vega,
                                 ir_vol, fx_vol, price_one, price_portfolio, price_trade)

# ===== CONFIG (user inputs) =====
N_PERF_STATES, PERF_SECONDS = 750, 2.0
PAR_PV_TOL_USD = 1e-3      # on a USD 100mm swap
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def hist():
    return generate_history()


@pytest.fixture(scope="module")
def row(hist):
    return hist.market.row(len(hist) - 1)


def _trade(**kw):
    import pandas as pd
    return pd.Series(_row(**{C.TRADE_ID: "T", **kw}))


def test_portfolio_composition():
    tr = sample_portfolio()
    counts = tr[C.PRODUCT].value_counts()
    assert len(tr) == 20 and counts[C.PRODUCT_IRS] == 9 and counts[C.PRODUCT_SWAPTION] == 3
    assert counts[C.PRODUCT_FXFWD] == 4 and counts[C.PRODUCT_FXOPT] == 4
    assert set(tr[C.PORTFOLIO_ID]) == {"NS_A"} and set(tr[C.DIRECTION]) == {-1, 1}
    assert set(tr[tr[C.PRODUCT] == C.PRODUCT_IRS][C.DIRECTION]) == {-1, 1}       # partly offsetting


def test_at_par_swap_pv_is_zero(row):
    par5 = float(row.par["USD"][0, 7])
    t = _trade(**{C.PRODUCT: C.PRODUCT_IRS, C.CCY: "USD", C.NOTIONAL: 1e8, C.DIRECTION: 1, C.MATURITY: 5.0, C.STRIKE: par5})
    assert abs(price_one(t, row)) < PAR_PV_TOL_USD


def test_bachelier_vega_closed_form(row):
    F, K, s, T = 0.04, 0.042, 0.0095, 2.0
    d = (F - K) / (s * np.sqrt(T))
    assert np.isclose(bachelier_vega(F, K, s, T), np.sqrt(T) * norm.pdf(d))
    h = 1e-7
    fd = (bachelier(F, K, s + h, T, True) - bachelier(F, K, s - h, T, True)) / (2 * h)
    assert np.isclose(fd, bachelier_vega(F, K, s, T), rtol=1e-6)
    # swaption PV vega equals N * A * sqrt(T) * phi(d) * fx
    t = _trade(**{C.PRODUCT: C.PRODUCT_SWAPTION, C.CCY: "EUR", C.NOTIONAL: 1e7, C.DIRECTION: 1, C.OPT_TYPE: "payer",
                  C.START: 2.0, C.EXPIRY: 2.0, C.MATURITY: 7.0, C.STRIKE: 0.03})
    Fw, A = row.curve("EUR").forward_swap(2.0, 7.0)
    sig = ir_vol(row, "EUR", 2.0, 5.0)[0]
    b = row.copy_with(); b.ir_nvol["EUR"] = row.ir_nvol["EUR"] + 1e-6
    b2 = row.copy_with(); b2.ir_nvol["EUR"] = row.ir_nvol["EUR"] - 1e-6
    fd = (price_one(t, b) - price_one(t, b2)) / 2e-6
    exp = 1e7 * A[0] * bachelier_vega(Fw[0], 0.03, sig, 2.0) * row.fx_spot["EUR"][0]
    assert np.isclose(fd, exp, rtol=1e-6)


def test_payer_minus_receiver_equals_forward_swap(row):
    kw = {C.PRODUCT: C.PRODUCT_SWAPTION, C.CCY: "GBP", C.NOTIONAL: 2e7, C.DIRECTION: 1, C.START: 3.0, C.EXPIRY: 3.0,
          C.MATURITY: 8.0, C.STRIKE: 0.041}
    pay = price_one(_trade(**kw, **{C.OPT_TYPE: "payer"}), row)
    rec = price_one(_trade(**kw, **{C.OPT_TYPE: "receiver"}), row)
    fwd = _trade(**{C.PRODUCT: C.PRODUCT_IRS, C.CCY: "GBP", C.NOTIONAL: 2e7, C.DIRECTION: 1, C.START: 3.0, C.MATURITY: 8.0,
                    C.STRIKE: 0.041})
    assert np.isclose(pay - rec, price_one(fwd, row), rtol=1e-10)


def test_gk_put_call_parity_and_forward_parity(row):
    kw = {C.CCY: "EUR", C.CCY2: "USD", C.NOTIONAL: 1e7, C.DIRECTION: 1, C.EXPIRY: 0.5, C.MATURITY: 0.5, C.STRIKE: 1.1}
    call = price_one(_trade(**kw, **{C.PRODUCT: C.PRODUCT_FXOPT, C.OPT_TYPE: "call"}), row)
    put = price_one(_trade(**kw, **{C.PRODUCT: C.PRODUCT_FXOPT, C.OPT_TYPE: "put"}), row)
    fwd = price_one(_trade(**kw, **{C.PRODUCT: C.PRODUCT_FXFWD}), row)
    assert np.isclose(call - put, fwd, rtol=1e-10)
    F = fx_forward_rate(row, "EUR", "USD", 0.5)[0]
    S = row.pair_spot("EURUSD")[0]
    dfe, dfu = row.curve("EUR").df(0.5)[0, 0], row.curve("USD").df(0.5)[0, 0]
    assert np.isclose(F, S * dfe / dfu)
    at_fwd = _trade(**{**kw, C.PRODUCT: C.PRODUCT_FXFWD, C.STRIKE: F})
    assert abs(price_one(at_fwd, row)) < 1e-6


def test_gk_analytic_delta_and_vega_vs_bumps():
    F, K, s, T, dfq, dfb = 1.12, 1.10, 0.09, 0.75, 0.98, 0.99
    h = 1e-6
    v = (gk_price(F, K, s + h, T, dfq, True) - gk_price(F, K, s - h, T, dfq, True)) / (2 * h)
    assert np.isclose(v, gk_vega(F, K, s, T, dfq), rtol=1e-6)
    # spot bump moves F = S*dfb/dfq
    S = F * dfq / dfb
    f = lambda S_: gk_price(S_ * dfb / dfq, K, s, T, dfq, True)
    assert np.isclose((f(S + h) - f(S - h)) / (2 * h), gk_delta_spot(F, K, s, T, dfb, True), rtol=1e-6)
    g = lambda S_: gk_price(S_ * dfb / dfq, K, s, T, dfq, False)
    assert np.isclose((g(S + h) - g(S - h)) / (2 * h), gk_delta_spot(F, K, s, T, dfb, False), rtol=1e-6)


def test_vectorised_equals_single_trade_loop(hist):
    tr = sample_portfolio()
    idx = np.linspace(0, len(hist) - 1, 25).astype(int)
    big = hist.market.take(idx)
    pv = price_portfolio(tr, big)
    for i, j in enumerate(idx):
        r = hist.market.row(j)
        single = np.array([price_one(t, r) for _, t in tr.iterrows()])
        assert np.allclose(pv[i], single, rtol=1e-12, atol=1e-6)


def test_performance_750_scenarios(hist):
    tr = sample_portfolio()
    b = hist.market.take(np.arange(N_PERF_STATES))
    t0 = time.perf_counter()
    pv = price_portfolio(tr, b)
    assert pv.shape == (N_PERF_STATES, 20) and np.isfinite(pv).all()
    assert time.perf_counter() - t0 < PERF_SECONDS


def test_xccy_near_zero_at_par_basis_and_linear_in_spread(row):
    fx0 = float(row.fx_spot["EUR"][0])
    mkt = float(row.xccy_basis["EURUSD"][0])
    x = new_product_xccy(initial_fx=fx0, basis=mkt).iloc[0]
    assert abs(price_one(x, row)) < 1e-6
    x2 = x.copy(); x2[BASIS_SPREAD] = mkt + 1e-4
    ann = row.curve("EUR").annuity(0.0, 5.0)[0]
    assert np.isclose(price_one(x2, row), x[C.NOTIONAL] * ann * 1e-4 * fx0, rtol=1e-9)
    assert abs(price_trade(x, row, include_exchange=False)[0]) > 1e5      # excluding principal leaves coupon PVs


def test_ir_vol_interpolation_flat_and_exact_on_grid(row):
    assert np.isclose(ir_vol(row, "USD", 2.0, 10.0)[0], row.ir_nvol["USD"][0, 1, 1])
    assert np.isclose(ir_vol(row, "USD", 0.1, 3.0)[0], row.ir_nvol["USD"][0, 0, 0])
    assert np.isclose(fx_vol(row, "EURUSD", 0.25)[0], row.fx_vol["EURUSD"][0, 1])
