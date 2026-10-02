"""Euler allocation: contributions sum to IM when CR = 1, differ when CR > 1."""
import pandas as pd
import pytest

from simm_margin import allocation as A
from simm_margin import columns as C
from simm_margin import params as P
from simm_margin import simm as S
from simm_margin.crif import build_crif
from simm_margin.instruments import sample_portfolio
from simm_margin.market_history import generate_history

# ===== CONFIG (user inputs) =====
TOL = 1e-6
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def sample():
    h = generate_history(quick=True)
    row = h.market.row(len(h) - 1)
    tr = sample_portfolio()
    return S.attach_fx_sigma(build_crif(tr, row), tr, row)


@pytest.mark.parametrize("version", ["2.8+2512", "2.8+2506"])
@pytest.mark.parametrize("by", ["TRADE_ID", "risk_factor"])
def test_contributions_sum_to_total(sample, by, version):
    p = P.load(version)
    chk = A.euler_check(sample, p, by)
    assert chk["sum_contributions"] == pytest.approx(chk["total"], rel=TOL)
    c = A.euler_contributions(sample, p, by)
    assert c.sum() == pytest.approx(S.simm(sample, p).total, rel=TOL)
    if by == "TRADE_ID":
        assert set(c.index) == set(sample[C.CRIF_TRADE_ID])


def test_single_trade_gets_everything_and_offset_is_negative(sample):
    p = P.load()
    one = sample[sample[C.CRIF_TRADE_ID] == "IRS_USD_10Y"]
    assert A.euler_contributions(one, p)["IRS_USD_10Y"] == pytest.approx(S.simm(one, p).total, rel=TOL)
    t = one.copy()
    t[C.AMOUNT_USD] = -0.5 * t[C.AMOUNT_USD]
    t[C.CRIF_TRADE_ID] = "HEDGE"
    c = A.euler_contributions(pd.concat([one, t]), p)
    assert c["HEDGE"] < 0 < c["IRS_USD_10Y"] and c.sum() == pytest.approx(S.simm(pd.concat([one, t]), p).total, rel=TOL)


def test_gap_when_concentration_above_one():
    rows = [{C.CRIF_TRADE_ID: "A", C.PRODUCT_CLASS: C.RATES_FX, C.RISK_TYPE: C.RISK_IRCURVE, C.QUALIFIER: "USD", C.LABEL1: "2y",
             C.LABEL2: "OIS", C.AMOUNT_USD: 4 * 220e6}]
    chk = A.euler_check(pd.DataFrame(rows), P.load())
    assert abs(chk["gap"]) > 1.0               # CR = sqrt(|s|/T) makes the IM degree 1.5 in the amount, not 1


def test_bad_by_and_empty(sample):
    with pytest.raises(ValueError):
        A.euler_contributions(sample, P.load(), by="bucket")
    assert len(A.euler_contributions(sample.iloc[0:0], P.load())) == 0
