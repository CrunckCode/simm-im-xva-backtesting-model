"""Schedule IM: rates, buckets, NGR cases and the net formula (hand-computed)."""
import numpy as np
import pandas as pd
import pytest

from simm_margin import columns as C
from simm_margin import instruments as I
from simm_margin import schedule_im as SI

# ===== CONFIG (user inputs) =====
SPOT = {"USD": 1.0, "EUR": 1.10, "GBP": 1.25, "JPY": 0.01, "MXN": 0.05}
# ===== END CONFIG =====


def tbl(*rows) -> pd.DataFrame:
    """rows: (id, product, ccy, notional, maturity)."""
    return pd.DataFrame([{C.TRADE_ID: i, C.PRODUCT: p, C.CCY: c, C.NOTIONAL: n, C.MATURITY: m} for i, p, c, n, m in rows])


def test_rates_by_asset_class_and_bucket():
    t = tbl(("a", C.PRODUCT_IRS, "USD", 100e6, 1.0), ("b", C.PRODUCT_IRS, "USD", 100e6, 2.0), ("c", C.PRODUCT_IRS, "USD", 100e6, 5.0),
            ("d", C.PRODUCT_FXFWD, "EUR", 10e6, 0.5), ("e", C.PRODUCT_FXOPT, "GBP", 10e6, 1.0),
            ("f", C.PRODUCT_XCCY, "EUR", 10e6, 3.0), ("g", C.PRODUCT_IRS, "MXN", 1e9, 10.0))
    out = SI.schedule_im(t, np.zeros(7), fx_spot=SPOT)
    r = out["by_row"].set_index(C.TRADE_ID)
    assert r[SI.RATE_COL].to_dict() == {"a": 0.01, "b": 0.02, "c": 0.04, "d": 0.06, "e": 0.06, "f": 0.02, "g": 0.04}
    assert r.loc["d", SI.NOTIONAL_USD_COL] == pytest.approx(11e6) and r.loc["g", SI.NOTIONAL_USD_COL] == pytest.approx(50e6)
    gross = 1e6 + 2e6 + 4e6 + 0.06 * 11e6 + 0.06 * 12.5e6 + 0.02 * 11e6 + 0.04 * 50e6
    assert out["gross"] == pytest.approx(gross) and r.loc["f", SI.ASSET_COL] == SI.ASSET_XCCY


def test_bcbs_source_uses_interest_rate_rows_for_xccy():
    t = tbl(("f", C.PRODUCT_XCCY, "EUR", 10e6, 6.0))
    out = SI.schedule_im(t, [0.0], fx_spot=SPOT, source=SI.SOURCE_BCBS)
    assert out["by_row"][SI.ASSET_COL].iloc[0] == SI.ASSET_IR and out["gross"] == pytest.approx(0.04 * 11e6)


def test_swaption_duration_is_expiry_plus_tenor():
    tr = I.sample_portfolio()
    sw = tr[tr[C.PRODUCT] == C.PRODUCT_SWAPTION]
    out = SI.schedule_im(sw, np.zeros(len(sw)), fx_spot=SPOT)
    assert out["by_row"][SI.DURATION_COL].tolist() == (sw[C.EXPIRY] + (sw[C.MATURITY] - sw[C.EXPIRY])).tolist()
    short = tbl(("s", C.PRODUCT_SWAPTION, "USD", 1e6, 1.5))     # 1y expiry x 6m underlying
    assert SI.schedule_im(short, [0.0])["gross"] == pytest.approx(0.01 * 1e6)


def test_ngr_cases():
    assert SI.ngr([5.0, 3.0, 2.0]) == 1.0                  # all positive
    assert SI.ngr([5.0, -5.0]) == 0.0                      # fully offsetting
    assert SI.ngr([10.0, -6.0]) == pytest.approx(0.4)
    assert SI.ngr([0.0, 0.0]) == 1.0 and SI.ngr([-3.0, -1.0]) == 1.0 and SI.ngr([]) == 1.0   # gross replacement cost 0
    assert SI.ngr([4.0, -9.0, 3.0]) == 0.0                 # net negative floors at 0 before dividing


def test_net_formula_and_row_checks():
    t = tbl(("a", C.PRODUCT_IRS, "USD", 100e6, 10.0), ("b", C.PRODUCT_FXFWD, "USD", 50e6, 1.0))
    g = 0.04 * 100e6 + 0.06 * 50e6
    for mtm, n in (([1.0, 2.0], 1.0), ([1.0, -1.0], 0.0), ([3.0, -1.0], 2 / 3), ([-1.0, -2.0], 1.0)):
        out = SI.schedule_im(t, mtm)
        assert out["ngr"] == pytest.approx(n) and out["gross"] == pytest.approx(g)
        assert out["net"] == pytest.approx(0.4 * g + 0.6 * n * g)
    with pytest.raises(ValueError):
        SI.schedule_im(t, [1.0])


def test_sample_portfolio_net_between_bounds():
    tr = I.sample_portfolio()
    out = SI.schedule_im(tr, np.linspace(-1, 2, len(tr)))
    assert len(out["by_row"]) == len(tr) == 20
    assert 0.4 * out["gross"] <= out["net"] <= out["gross"] and 0 <= out["ngr"] <= 1
