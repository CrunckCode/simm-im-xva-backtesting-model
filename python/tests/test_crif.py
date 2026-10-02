from simm_margin.simm import SIGMA_MARKET_COL
"""CRIF schema and content tests (synthetic trades and market)."""
import numpy as np
import pytest

from simm_margin import columns as C
from simm_margin import config
from simm_margin.crif import CRIF_COLUMNS, build_crif
from simm_margin.curves import CURRENCIES
from simm_margin.instruments import new_product_xccy, sample_portfolio
from simm_margin.market_history import generate_history
from simm_margin.sensitivities import parallel_pv01, rebucket_weights

# ===== CONFIG (user inputs) =====
RISK_TYPES = {C.RISK_IRCURVE, C.RISK_XCCYBASIS, C.RISK_IRVOL, C.RISK_FX, C.RISK_FXVOL}
PV01_REL_TOL = 2e-3
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def setup():
    h = generate_history()
    return sample_portfolio(), h.market.row(len(h) - 1), h


def test_schema_and_risk_types(setup):
    tr, row, _ = setup
    crif = build_crif(tr, row)
    assert list(crif.columns) == CRIF_COLUMNS + [SIGMA_MARKET_COL] and len(crif) > 50
    assert set(crif[C.RISK_TYPE]) <= RISK_TYPES
    assert {C.RISK_IRCURVE, C.RISK_IRVOL, C.RISK_FX, C.RISK_FXVOL} <= set(crif[C.RISK_TYPE])
    assert set(crif[C.PRODUCT_CLASS]) == {C.RATES_FX} and set(crif[C.CRIF_TRADE_ID]) <= set(tr[C.TRADE_ID])
    assert crif[[C.AMOUNT, C.AMOUNT_USD]].notna().all().all()


def test_labels_and_qualifiers(setup):
    tr, row, _ = setup
    crif = build_crif(tr, row)
    ir = crif[crif[C.RISK_TYPE] == C.RISK_IRCURVE]
    assert set(ir[C.QUALIFIER]) <= set(CURRENCIES) and set(ir[C.LABEL1]) <= set(config.IR_TENORS)
    assert set(ir[C.LABEL2]) == {"OIS"}
    fx = crif[crif[C.RISK_TYPE] == C.RISK_FX]
    assert set(fx[C.QUALIFIER]) <= {"EUR", "GBP", "JPY", "MXN"} and set(fx[C.AMOUNT_CCY]) == {"USD"}
    fv = crif[crif[C.RISK_TYPE] == C.RISK_FXVOL]
    assert set(fv[C.QUALIFIER]) <= {"EURUSD", "GBPUSD", "USDJPY", "USDMXN"}
    assert set(fv[C.LABEL1]) <= set(config.VEGA_EXPIRIES)


def test_amount_usd_conversion(setup):
    tr, row, _ = setup
    crif = build_crif(tr, row)
    for ccy in CURRENCIES:
        sub = crif[crif[C.AMOUNT_CCY] == ccy]
        assert np.allclose(sub[C.AMOUNT] * float(row.fx_spot[ccy][0]), sub[C.AMOUNT_USD], rtol=1e-12)


def test_ir_delta_sums_match_parallel_pv01(setup):
    tr, row, _ = setup
    crif = build_crif(tr, row)
    for ccy in ("USD", "EUR", "JPY"):
        tot = crif[(crif[C.RISK_TYPE] == C.RISK_IRCURVE) & (crif[C.QUALIFIER] == ccy)][C.AMOUNT_USD].sum()
        pv01 = sum(parallel_pv01(t, row, ccy) for _, t in tr.iterrows() if ccy in (t[C.CCY], t[C.CCY2]))
        assert abs(tot - pv01) <= PV01_REL_TOL * max(abs(pv01), 1.0) + 1.0


def test_vega_rebucketing_weights_sum_to_one(setup):
    tr, row, _ = setup
    crif = build_crif(tr, row)
    rows = crif[(crif[C.CRIF_TRADE_ID] == "SWPT_EUR_4Yx10Y") & (crif[C.RISK_TYPE] == C.RISK_IRVOL)]
    assert set(rows[C.LABEL1]) == {"3y", "5y"}
    assert np.isclose(rows[C.AMOUNT_USD].iloc[0], rows[C.AMOUNT_USD].iloc[1])      # 4y = 50% 3y + 50% 5y
    assert np.isclose(sum(w for _, w in rebucket_weights(4.0)), 1.0)


def test_xccy_crif_rows_and_exchange_flag(setup):
    _, row, _ = setup
    fx0, mkt = float(row.fx_spot["EUR"][0]), float(row.xccy_basis["EURUSD"][0])
    elig = build_crif(new_product_xccy(initial_fx=fx0, basis=mkt), row)
    none = build_crif(new_product_xccy(initial_fx=fx0, basis=mkt, settle_notional_exchange="none"), row)
    for c in (elig, none):
        b = c[c[C.RISK_TYPE] == C.RISK_XCCYBASIS]
        assert len(b) == 1 and b[C.QUALIFIER].iloc[0] == "EUR" and b[C.AMOUNT_USD].iloc[0] > 0
    assert (none[C.RISK_TYPE] == C.RISK_IRCURVE).sum() == 0 and (elig[C.RISK_TYPE] == C.RISK_IRCURVE).sum() > 0
    fx_none = none[none[C.RISK_TYPE] == C.RISK_FX][C.AMOUNT_USD].abs().iloc[0]
    fx_elig = elig[elig[C.RISK_TYPE] == C.RISK_FX][C.AMOUNT_USD].abs().iloc[0]
    assert fx_none > 3 * fx_elig


def test_build_crif_needs_single_state(setup):
    tr, _, h = setup
    with pytest.raises(ValueError):
        build_crif(tr, h.market.take([0, 1]))


def test_crif_deterministic(setup):
    tr, row, _ = setup
    assert build_crif(tr, row).equals(build_crif(tr, row))
