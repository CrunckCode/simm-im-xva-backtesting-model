"""Sensitivity tests: Jacobian vs full re-bootstrap, parallel PV01, translation FX delta, vega allocation."""
import numpy as np
import pandas as pd
import pytest

from simm_margin import columns as C
from simm_margin.instruments import _row, new_product_xccy, sample_portfolio
from simm_margin.market_history import generate_history
from simm_margin.pricing import bachelier_vega, ir_vol, price_one
from simm_margin.sensitivities import (curvature_cvr, curvature_scaling, fx_delta, fx_vega, fx_vega_ladder,
                                       ir_delta, ir_delta_full_bootstrap, ir_vega, ir_vega_ladder, parallel_pv01,
                                       rebucket_weights, xccy_basis_delta)

# ===== CONFIG (user inputs) =====
JAC_VS_FULL_REL_TOL = 1e-4        # max ladder difference relative to the largest ladder entry
PV01_REL_TOL = 2e-3               # parallel PV01 vs sum of the ladder (second-order cross terms)
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def row():
    h = generate_history()
    return h.market.row(len(h) - 1)


def _t(**kw):
    return pd.Series(_row(**{C.TRADE_ID: "T", **kw}))


def _ccys(t):
    return {t[C.CCY], t[C.CCY2] or t[C.CCY]}


def test_jacobian_ladder_matches_full_rebootstrap_for_whole_portfolio(row):
    for _, t in sample_portfolio().iterrows():
        for ccy in _ccys(t):
            a, b = ir_delta(t, row, ccy), ir_delta_full_bootstrap(t, row, ccy)
            scale = max(np.abs(b).max(), 1.0)
            assert np.abs(a - b).max() / scale < JAC_VS_FULL_REL_TOL, t[C.TRADE_ID]


def test_parallel_pv01_equals_sum_of_ladder(row):
    book = sample_portfolio().set_index(C.TRADE_ID, drop=False)
    for tid in ("IRS_USD_10Y", "IRS_GBP_7Y", "SWPT_USD_1Yx5Y", "FXF_EURUSD_6M"):
        t = book.loc[tid]
        for ccy in _ccys(t):
            s = ir_delta(t, row, ccy).sum()
            p = parallel_pv01(t, row, ccy)
            assert abs(s - p) <= PV01_REL_TOL * max(abs(p), abs(s), 1.0), (tid, ccy, s, p)


def test_par_swap_pv01_is_notional_times_annuity(row):
    par5 = float(row.par["USD"][0, 7])
    t = _t(**{C.PRODUCT: C.PRODUCT_IRS, C.CCY: "USD", C.NOTIONAL: 1e8, C.DIRECTION: 1, C.MATURITY: 5.0, C.STRIKE: par5})
    ann = row.curve("USD").annuity(0.0, 5.0)[0]
    assert np.isclose(parallel_pv01(t, row, "USD"), 1e8 * ann * 1e-4, rtol=0.01)
    lad = ir_delta(t, row, "USD")
    assert np.argmax(np.abs(lad)) == 7 and lad[7] > 0       # payer gains when the 5y rate rises


def test_fx_delta_includes_translation_and_matches_linear_forms(row):
    t = _t(**{C.PRODUCT: C.PRODUCT_IRS, C.CCY: "EUR", C.NOTIONAL: 2e7, C.DIRECTION: 1, C.MATURITY: 5.0, C.STRIKE: 0.03})
    assert np.isclose(fx_delta(t, row, "EUR"), 0.01 * price_one(t, row), rtol=1e-9)      # USD PV is linear in spot
    f = _t(**{C.PRODUCT: C.PRODUCT_FXFWD, C.CCY: "EUR", C.CCY2: "USD", C.NOTIONAL: 1e7, C.DIRECTION: 1, C.MATURITY: 0.5,
              C.STRIKE: 1.1})
    dfb = row.curve("EUR").df(0.5)[0, 0]
    assert np.isclose(fx_delta(f, row, "EUR"), 1e7 * row.fx_spot["EUR"][0] * dfb * 0.01, rtol=1e-9)


def test_ir_vega_matches_closed_form_and_allocates_linearly(row):
    t = _t(**{C.PRODUCT: C.PRODUCT_SWAPTION, C.CCY: "USD", C.NOTIONAL: 1e7, C.DIRECTION: 1, C.OPT_TYPE: "payer",
              C.START: 4.0, C.EXPIRY: 4.0, C.MATURITY: 9.0, C.STRIKE: 0.04})
    per_bp, sig_bp, amount = ir_vega(t, row)
    F, A = row.curve("USD").forward_swap(4.0, 9.0)
    sig = ir_vol(row, "USD", 4.0, 5.0)[0]
    assert np.isclose(per_bp, 1e7 * A[0] * bachelier_vega(F[0], 0.04, sig, 4.0) * 1e-4, rtol=1e-4)   # central 1bp bump has O(h^2) error
    assert np.isclose(amount, per_bp * sig * 1e4)
    lad = ir_vega_ladder(t, row)
    assert np.isclose(lad.sum(), amount) and np.isclose(lad[6], 0.5 * amount) and np.isclose(lad[7], 0.5 * amount)


def test_rebucket_weights_sum_to_one_and_seven_year_rule():
    w = rebucket_weights(7.0)
    assert [i for i, _ in w] == [7, 8] and np.allclose([b for _, b in w], [0.6, 0.4])
    for x in (0.01, 0.4, 1.0, 2.5, 12.0, 29.0, 31.0, 50.0):
        w = rebucket_weights(x)
        assert np.isclose(sum(b for _, b in w), 1.0) and all(b >= 0 for _, b in w)


def test_fx_vega_and_ladder(row):
    t = _t(**{C.PRODUCT: C.PRODUCT_FXOPT, C.CCY: "EUR", C.CCY2: "USD", C.NOTIONAL: 1e7, C.DIRECTION: 1, C.OPT_TYPE: "call",
              C.EXPIRY: 0.5, C.MATURITY: 0.5, C.STRIKE: 1.1})
    per_pt, sig, amount = fx_vega(t, row)
    assert per_pt > 0 and np.isclose(amount, per_pt * 100 * sig)
    lad = fx_vega_ladder(t, row)
    assert np.isclose(lad[3], amount) and np.isclose(lad.sum(), amount)         # 6m is a vertex
    t[C.EXPIRY] = 0.75
    lad2 = fx_vega_ladder(t, row)
    assert lad2[3] > 0 and lad2[4] > 0 and np.isclose(lad2[3] / lad2.sum(), 0.5)  # 9m is halfway between 6m and 1y


def test_curvature_scaling_and_cvr():
    assert np.isclose(curvature_scaling(14 / 365), 0.5) and np.isclose(curvature_scaling(1 / 365), 0.5)
    assert np.isclose(curvature_scaling(1.0), 0.5 * 14 / 365)
    parts, cvr = curvature_cvr(np.arange(1.0, 13.0))
    assert np.isclose(cvr, parts.sum()) and parts[0] == 0.5 * 1.0


def test_xccy_basis_delta_and_exchange_exclusion(row):
    fx0, mkt = float(row.fx_spot["EUR"][0]), float(row.xccy_basis["EURUSD"][0])
    elig = new_product_xccy(initial_fx=fx0, basis=mkt, settle_notional_exchange="eligible").iloc[0]
    full = new_product_xccy(initial_fx=fx0, basis=mkt, settle_notional_exchange="none").iloc[0]
    ann = row.curve("EUR").annuity(0.0, 5.0)[0]
    assert np.isclose(xccy_basis_delta(elig, row), elig[C.NOTIONAL] * ann * fx0 * 1e-4, rtol=1e-9)
    assert np.isclose(fx_delta(full, row, "EUR"), full[C.NOTIONAL] * fx0 * 0.01, rtol=1e-3)   # principal re-exchange
    assert abs(fx_delta(elig, row, "EUR")) < 0.3 * abs(fx_delta(full, row, "EUR"))
    assert np.abs(ir_delta(full, row, "EUR")).max() < 1.0                                    # floating legs at par
    assert np.abs(ir_delta(elig, row, "EUR")).max() > 1e3
