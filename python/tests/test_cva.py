"""CVA and xVA VaR tests: hazard bootstrap, exposure profile limits, monotonicity in spread, VaR sign. SYNTHETIC data."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from simm_margin import columns as C
from simm_margin import config
from simm_margin import cva as X
from simm_margin.instruments import new_product_xccy, sample_portfolio
from simm_margin.market_history import generate_history
from simm_margin.pricing import price_trade

# ===== CONFIG (user inputs) =====
DATE_IDX = 400
SPREAD_SCALES = (0.5, 1.0, 1.5, 2.5)
CS01_SHIFT = 5.0e-4
CS01_TOL = 0.05
REPRICE_TOL = 1e-9
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def quick():
    return generate_history(quick=True)


@pytest.fixture(scope="module")
def trades():
    return sample_portfolio()


@pytest.fixture(scope="module")
def row(quick):
    return quick.market.row(DATE_IDX)


@pytest.fixture(scope="module")
def cov(quick):
    return X.rolling_factor_cov(quick)[DATE_IDX]


def test_hazard_reprices_cds(row):
    lam = X.bootstrap_hazard(row.cds, row.curve("USD"))
    for j, T in enumerate(X.CDS_PILLARS):
        assert X.cds_par_spread(lam, T, row.curve("USD"))[0] == pytest.approx(row.cds[0, j], abs=REPRICE_TOL)
    assert (lam > 0).all()


def test_flat_spread_gives_credit_triangle(row):
    flat = np.full((1, 5), 0.01)
    lam = X.bootstrap_hazard(flat, row.curve("USD"))
    assert np.allclose(lam, 0.01 / (1 - X.RECOVERY), rtol=0.03)          # s = (1 - R) lambda, to first order


def test_remaining_pv_matches_pricer_at_time_zero(row, trades):
    book = trades.copy()
    for _, t in X.linear_trades(book).iterrows():
        assert X.remaining_pv(t, row, np.array([0.0]))[0, 0] == pytest.approx(price_trade(t, row)[0], rel=1e-9, abs=1e-6)
    xc = new_product_xccy().iloc[0]
    assert X.remaining_pv(xc, row, np.array([0.0]))[0, 0] == pytest.approx(price_trade(xc, row, True)[0], rel=1e-9, abs=1e-6)
    assert X.remaining_pv(xc, row, np.array([5.0, 6.0])).sum() == 0.0      # nothing after maturity


def test_ee_limits_and_formula(row, trades):
    zero_cov = np.zeros((X.N_FACTORS, X.N_FACTORS))
    p0 = X.ee_profile(trades, row, zero_cov)
    assert np.allclose(p0["ee"], np.maximum(p0["mu"], 0.0), atol=1e-4)     # no diffusion: EE = max(mu, 0)


def test_ee_matches_quadrature(row, trades, cov):
    p = X.ee_profile(trades, row, cov)
    k = 8
    mu, s = p["mu"][0, k], p["s"][0, k]
    ref = quad(lambda v: max(v, 0.0) * norm.pdf(v, mu, s), mu - 12 * s, mu + 12 * s, points=[0.0] if mu - 12 * s < 0 < mu + 12 * s else None)[0]
    assert p["ee"][0, k] == pytest.approx(ref, rel=1e-6)
    assert (p["ee"] >= np.maximum(p["mu"], 0) - 1e-6).all() and (p["s"] > 0).all()


def test_cva_positive_and_increasing_in_spread(row, trades, cov):
    cvas = []
    for k in SPREAD_SCALES:
        b = row.copy_with()
        cvas.append(X.cva_state(trades, b, cov, cds=row.cds * k)["cva"][0])
    assert cvas[0] > 0 and cvas == sorted(cvas) and cvas[-1] > 2 * cvas[0]


def test_cva_zero_with_zero_lgd(row, trades, cov):
    res = X.cva_state(trades, row, cov)
    assert X.cva_from_profile(res["profile"], res["lam"], lgd=0.0)[0] == 0.0 and res["cva"][0] > 0


def test_cva_value_summary(quick, trades):
    out = X.cva_value(quick, trades, quick.dates[DATE_IDX])
    assert out["cva"] > 0 and out["peak_ee"] >= out["epe"] > 0
    assert set(out["vols"]) >= {"sigma_r_USD_bp", "sigma_fx_EUR_pct"}


def test_rolling_cov_is_psd_and_annualised(quick):
    c = X.rolling_factor_cov(quick)[DATE_IDX]
    assert np.allclose(c, c.T) and np.linalg.eigvalsh(c).min() > -1e-12
    assert 0.001 < np.sqrt(c[0, 0]) < 0.03                    # USD parallel-rate vol of a few tens of bp per annum


@pytest.fixture(scope="module")
def state(quick, trades):
    return X.xva_state(quick, trades)


def test_cs01_positive_and_matches_full_reval(quick, trades, state):
    r = 30
    cs01 = state["sens"][r, :5]
    assert cs01.sum() > 0                                        # parallel spread up, CVA up (single pillars can be negative)
    t = state["ts"][r]
    base = state["cva"][r]
    b = quick.market.row(t)
    shifted = X.cva_state(trades, b, state["cov"][r], cds=b.cds + CS01_SHIFT)["cva"][0]
    assert (shifted - base) == pytest.approx(cs01.sum() * CS01_SHIFT, rel=CS01_TOL)


@pytest.mark.parametrize("h", X.XVA_HORIZONS)
def test_xva_var_positive_and_aligned(quick, trades, h):
    v = X.xva_var_series(quick, trades, h)
    l = X.realized_cva_change(quick, trades, h)
    assert (v > 0).all() and v.index.equals(l.index)
    assert l.dropna().abs().mean() > 0 and l.iloc[-h:].isna().all()
    assert (X.xva_var_series(quick, trades, 10) > X.xva_var_series(quick, trades, 1)).mean() > 0.9
