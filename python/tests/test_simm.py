"""SIMM-style engine: hand-computed cases (arithmetic written out independently of the engine) and invariants."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from simm_margin import columns as C
from simm_margin import config
from simm_margin import params as P
from simm_margin import simm as S
from simm_margin.crif import build_crif
from simm_margin.instruments import sample_portfolio
from simm_margin.market_history import generate_history

# ===== CONFIG (user inputs) =====
V12, V06 = "2.8+2512", "2.8+2506"
REL = 1e-12
NEG_TOL = 1e-6
CRIF_COLS = [C.CRIF_TRADE_ID, C.PORTFOLIO_ID, C.PRODUCT_CLASS, C.RISK_TYPE, C.QUALIFIER, C.BUCKET, C.LABEL1, C.LABEL2,
             C.AMOUNT, C.AMOUNT_CCY, C.AMOUNT_USD]
IRC, FXR, IRV, FXV, XB = C.RISK_IRCURVE, C.RISK_FX, C.RISK_IRVOL, C.RISK_FXVOL, C.RISK_XCCYBASIS
TEN = config.IR_TENORS
# ===== END CONFIG =====


def mk(*rows) -> pd.DataFrame:
    """rows: (risk_type, qualifier, label1, label2, amount_usd[, trade_id])."""
    recs = []
    for r in rows:
        rt, q, l1, l2, a = r[:5]
        recs.append({C.CRIF_TRADE_ID: r[5] if len(r) > 5 else "T1", C.PORTFOLIO_ID: "NS", C.PRODUCT_CLASS: C.RATES_FX,
                     C.RISK_TYPE: rt, C.QUALIFIER: q, C.BUCKET: "", C.LABEL1: l1, C.LABEL2: l2, C.AMOUNT: a,
                     C.AMOUNT_CCY: "USD", C.AMOUNT_USD: a})
    return pd.DataFrame(recs, columns=CRIF_COLS)


@pytest.fixture(scope="module")
def p12():
    return P.load(V12)


@pytest.fixture(scope="module")
def p06():
    return P.load(V06)


@pytest.fixture(scope="module")
def sample():
    h = generate_history(quick=True)
    row = h.market.row(len(h) - 1)
    tr = sample_portfolio()
    return S.attach_fx_sigma(build_crif(tr, row), tr, row)


def rho(a, b, p):
    return p.ir_corr[TEN.index(a), TEN.index(b)]


# == hand-computed examples ==
def test_two_tenor_usd_bucket(p12):
    s2, s5 = 3.0e5, -1.5e5
    w2, w5 = 69 * s2, 61 * s5
    k = np.sqrt(w2 ** 2 + w5 ** 2 + 2 * rho("2y", "5y", p12) * w2 * w5)
    res = S.simm(mk((IRC, "USD", "2y", "OIS", s2), (IRC, "USD", "5y", "OIS", s5)), p12)
    assert res.total == pytest.approx(k, rel=REL) and res.by_margin_type[(C.RC_IR, C.MT_DELTA)] == pytest.approx(k, rel=REL)
    assert 0.92 == rho("2y", "5y", p12)


def test_two_currency_gamma(p12):
    s_usd, s_eur = 1.0e5, -6.0e4
    k1, k2 = abs(61 * s_usd), abs(61 * s_eur)
    s1, s2 = 61 * s_usd, 61 * s_eur
    expected = np.sqrt(k1 ** 2 + k2 ** 2 + 2 * 0.35 * s1 * s2)
    c = mk((IRC, "USD", "5y", "OIS", s_usd), (IRC, "EUR", "5y", "OIS", s_eur))
    assert S.simm(c, p12).total == pytest.approx(expected, rel=REL)
    assert S.simm(c, P.load(V06)).total == pytest.approx(expected, rel=REL)     # IR delta is identical in both versions


def test_concentration_above_threshold_and_g_bc(p12):
    # |s| / 1e6 / T = 4 for USD (T = 220), so CR = 2; EUR sits exactly at its threshold so CR = 1
    s_usd, s_eur = 4 * 220e6, 220e6
    w1, w2 = 69 * s_usd * 2.0, 69 * s_eur
    g = 1.0 / 2.0
    expected = np.sqrt(w1 ** 2 + w2 ** 2 + 2 * 0.35 * g * w1 * w2)
    c = mk((IRC, "USD", "2y", "OIS", s_usd), (IRC, "EUR", "2y", "OIS", s_eur))
    assert S.simm(c, p12).total == pytest.approx(expected, rel=REL)
    # high volatility currency: MXN threshold is 71, 9 times the threshold gives CR = 3
    s_mxn = 9 * 71e6
    assert S.simm(mk((IRC, "MXN", "2y", "OIS", s_mxn)), p12).total == pytest.approx(93 * s_mxn * 3.0, rel=REL)


def test_phi_two_subcurves(p12):
    a, b = 1.0e5, 2.0e5
    k_same_tenor = np.sqrt((61 * a) ** 2 + (61 * b) ** 2 + 2 * 0.981 * (61 * a) * (61 * b))
    c = mk((IRC, "USD", "5y", "OIS", a), (IRC, "USD", "5y", "Libor3m", b))
    assert S.simm(c, p12).total == pytest.approx(k_same_tenor, rel=REL)
    k_diff = np.sqrt((69 * a) ** 2 + (61 * b) ** 2 + 2 * 0.981 * rho("2y", "5y", p12) * (69 * a) * (61 * b))
    c2 = mk((IRC, "USD", "2y", "OIS", a), (IRC, "USD", "5y", "Libor3m", b))
    assert S.simm(c2, p12).total == pytest.approx(k_diff, rel=REL)


def test_inflation_and_xccy_basis(p12):
    y, i, x = 1.0e5, 5.0e4, 1.0e5
    w = np.array([61 * y, 51 * i, 21 * x])
    m = np.array([[1, 0.42, -0.01], [0.42, 1, -0.01], [-0.01, -0.01, 1]])
    c = mk((IRC, "USD", "5y", "OIS", y), ("Risk_Inflation", "USD", "", "", i), (XB, "USD", "", "", x))
    assert S.simm(c, p12).total == pytest.approx(np.sqrt(w @ m @ w), rel=REL)


def test_xccy_basis_not_in_cr_sum_and_not_scaled(p12):
    y, x = 4 * 220e6, -4 * 220e6                         # yield alone gives CR = 2; basis must neither offset nor be scaled
    w = np.array([61 * y * 2.0, 21 * x])
    expected = np.sqrt(w @ np.array([[1, -0.01], [-0.01, 1]]) @ w)
    c = mk((IRC, "USD", "5y", "OIS", y), (XB, "USD", "", "", x))
    assert S.simm(c, p12).total == pytest.approx(expected, rel=REL)


def test_inflation_included_in_cr_sum(p12):
    y, i = 2 * 220e6, 2 * 220e6                          # sum 4*220e6 gives CR = 2 for both
    w = np.array([61 * y * 2, 51 * i * 2])
    expected = np.sqrt(w @ np.array([[1, 0.42], [0.42, 1]]) @ w)
    c = mk((IRC, "USD", "5y", "OIS", y), ("Risk_Inflation", "USD", "", "", i))
    assert S.simm(c, p12).total == pytest.approx(expected, rel=REL)


def test_fx_delta_hand_and_concentration(p12):
    e, m, a = 2.0e5, -1.0e5, 5.0e4
    w = np.array([7.4 * e, 7.4 * m, 31.7 * a])
    corr = np.array([[1, 0.5, 0.12], [0.5, 1, 0.12], [0.12, 0.12, 1]])
    c = mk((FXR, "EUR", "", "", e), (FXR, "MXN", "", "", m), (FXR, "ARS", "", "", a))
    assert S.simm(c, p12).total == pytest.approx(np.sqrt(w @ corr @ w), rel=REL)
    s_eur = 4 * 2100e6                                   # CR = 2 for Category 1
    assert S.simm(mk((FXR, "EUR", "", "", s_eur)), p12).total == pytest.approx(7.4 * s_eur * 2, rel=REL)


def test_ir_vega_and_curvature_hand(p12):
    a1, a5 = 1.0e5, 2.0e5
    c = mk((IRV, "USD", "1y", "", a1), (IRV, "USD", "5y", "", a5))
    vr = np.array([0.2 * a1, 0.2 * a5])
    m = np.array([[1, rho("1y", "5y", p12)], [rho("1y", "5y", p12), 1]])
    vega, _ = S.vega_margin_ir(c, p12)
    assert vega == pytest.approx(np.sqrt(vr @ m @ vr), rel=REL)
    days = {"1y": 365.0, "5y": 5 * 365.0}
    cvr = np.array([0.5 * min(1, 14 / days["1y"]) * a1, 0.5 * min(1, 14 / days["5y"]) * a5])
    m2 = np.array([[1, rho("1y", "5y", p12) ** 2], [rho("1y", "5y", p12) ** 2, 1]])
    k = np.sqrt(cvr @ m2 @ cvr)
    lam = norm.ppf(0.995) ** 2 - 1                       # theta = 0 for positive CVR
    curv, _ = S.curvature_margin(c, p12, C.RC_IR)
    assert curv == pytest.approx((cvr.sum() + lam * k) / 0.74 ** 2, rel=REL)


def test_ir_vega_concentration_and_cross_currency(p12):
    big = 4 * 3800e6 / 0.2 * 0.2                         # |sum VR_ik| / 1e6 / 3800 = 4 -> VCR = 2
    c = mk((IRV, "USD", "5y", "", big), (IRV, "EUR", "5y", "", 1.0e6))
    vr_usd, vr_eur = 0.2 * big * 2.0, 0.2 * 1.0e6 * 1.0
    expected = np.sqrt(vr_usd ** 2 + vr_eur ** 2 + 2 * 0.35 * 0.5 * vr_usd * vr_eur)
    assert S.vega_margin_ir(c, p12)[0] == pytest.approx(expected, rel=REL)


def test_fx_vega_hand_both_amount_conventions(p12):
    sig_simm_pts = 7.4 * np.sqrt(365 / 14) / norm.ppf(0.99)
    per_pt = 1.0e4
    vr = 0.33 * 0.67 * sig_simm_pts * per_pt
    c = mk((FXV, "EURUSD", "1y", "", per_pt))
    assert S.vega_margin_fx(c, p12)[0] == pytest.approx(vr, rel=REL)
    sig_mkt = 0.09                                       # crif.py convention: Amount = vega per unit vol * market vol
    c2 = mk((FXV, "EURUSD", "1y", "", per_pt / 0.01 * sig_mkt))
    c2[S.SIGMA_MARKET_COL] = sig_mkt
    assert S.vega_margin_fx(c2, p12)[0] == pytest.approx(vr, rel=REL)
    # VCR above 1: threshold for Category 1 pair is 2900
    big = 4 * 2900e6 / (0.67 * sig_simm_pts)
    vcr = 2.0
    assert S.vega_margin_fx(mk((FXV, "EURUSD", "1y", "", big)), p12)[0] == pytest.approx(0.33 * 4 * 2900e6 * vcr, rel=REL)


def test_fx_vega_two_pairs_correlation_and_f(p12):
    sig = 7.4 * np.sqrt(365 / 14) / norm.ppf(0.99)
    a1 = 4 * 2900e6 / (0.67 * sig)                       # EURUSD: VCR = 2 (Cat1-Cat1 threshold 2900)
    a2 = 1.0e4                                           # USDMXN: tiny, VCR = 1 (Cat1-Cat2 threshold 1500)
    v1, v2 = 0.33 * 0.67 * sig * a1 * 2.0, 0.33 * 0.67 * sig * a2
    expected = np.sqrt(v1 ** 2 + v2 ** 2 + 2 * 0.5 * (1 / 2) * v1 * v2)
    c = mk((FXV, "EURUSD", "1y", "", a1), (FXV, "USDMXN", "1y", "", a2))
    assert S.vega_margin_fx(c, p12)[0] == pytest.approx(expected, rel=REL)


def test_fx_curvature_long_option_theta_zero(p12):
    sig = 7.4 * np.sqrt(365 / 14) / norm.ppf(0.99)
    per_pt = 1.0e4
    cvr = 0.5 * 14 / 365 * sig * per_pt
    lam = norm.ppf(0.995) ** 2 - 1
    m, rows = S.curvature_margin(mk((FXV, "EURUSD", "1y", "", per_pt)), p12, C.RC_FX)
    assert m == pytest.approx(cvr + lam * cvr, rel=REL)
    assert [r[C.VALUE] for r in rows if r[C.LEVEL] == "theta"] == [0.0]


def test_fx_curvature_single_short_option_theta_negative(p12):
    # short option: CVR < 0, theta = -1, lambda = 1, K = |CVR| so the single-factor margin is exactly zero
    m, rows = S.curvature_margin(mk((FXV, "EURUSD", "1y", "", -1.0e4)), p12, C.RC_FX)
    theta = [r[C.VALUE] for r in rows if r[C.LEVEL] == "theta"][0]
    lam = [r[C.VALUE] for r in rows if r[C.LEVEL] == "lambda"][0]
    assert theta == -1.0 and lam == pytest.approx(1.0) and m == pytest.approx(0.0, abs=1e-6)


def test_fx_curvature_two_pairs_theta_between(p12):
    sig = 7.4 * np.sqrt(365 / 14) / norm.ppf(0.99)
    a, b = 3.0e4, -5.0e4                                  # per vol point, 1y vertex
    c1, c2 = 0.5 * 14 / 365 * sig * a, 0.5 * 14 / 365 * sig * b
    theta = (c1 + c2) / (abs(c1) + abs(c2))
    lam = (norm.ppf(0.995) ** 2 - 1) * (1 + theta) - theta
    k = np.sqrt(c1 ** 2 + c2 ** 2 + 2 * 0.25 * c1 * c2)
    expected = max(c1 + c2 + lam * k, 0.0)
    m, rows = S.curvature_margin(mk((FXV, "EURUSD", "1y", "", a), (FXV, "GBPUSD", "1y", "", b)), p12, C.RC_FX)
    assert theta < 0 and m == pytest.approx(expected, rel=REL) and m > 0
    assert [r[C.VALUE] for r in rows if r[C.LEVEL] == "theta"][0] == pytest.approx(theta)


def test_ir_curvature_two_currencies_gamma_squared(p12):
    a, b = 1.0e5, 3.0e4
    w = 0.5 * 14 / 365
    c1, c2 = a * w, b * w
    lam = norm.ppf(0.995) ** 2 - 1
    agg = np.sqrt(c1 ** 2 + c2 ** 2 + 2 * 0.35 ** 2 * c1 * c2)
    expected = (c1 + c2 + lam * agg) / 0.74 ** 2
    cr = mk((IRV, "USD", "1y", "", a), (IRV, "EUR", "1y", "", b))
    assert S.curvature_margin(cr, p12, C.RC_IR)[0] == pytest.approx(expected, rel=REL)


def test_product_class_and_risk_class_formulas(p12):
    assert S.risk_class_im(1.0, 2.0, 3.0) == 6.0
    assert S.product_class_simm({"IR": 3.0, "FX": 4.0}, p12) == pytest.approx(np.sqrt(9 + 16 + 2 * 0.15 * 12))
    assert S.product_class_simm({"IR": 3.0, "FX": 4.0}, p12, psi_override=0.0) == pytest.approx(5.0)
    assert S.product_class_simm({"IR": 3.0}, p12) == pytest.approx(3.0)


# == invariants ==
def test_empty_portfolio_is_zero(p12):
    r = S.simm(mk(), p12)
    assert r.total == 0.0 and list(r.breakdown.columns) == [C.LEVEL, C.RISK_CLASS, C.MARGIN_TYPE, "bucket", C.RISK_FACTOR, C.VALUE]
    assert S.simm(mk((IRC, "USD", "5y", "OIS", 0.0)), p12).total == 0.0


def test_bad_inputs_raise(p12):
    with pytest.raises(ValueError):
        S.simm(mk(("Risk_Equity", "X", "", "", 1.0)), p12)
    with pytest.raises(KeyError):
        S.simm(mk((IRC, "USD", "5y", "OIS", 1.0)).drop(columns=[C.LABEL2]), p12)
    assert S.simm(mk((FXR, "USD", "", "", 1.0e6)), p12).total == 0.0      # no FX factor for the calculation currency


def test_negation_symmetry_of_delta_and_vega(sample, p12):
    neg = sample.copy()
    neg[C.AMOUNT_USD] = -neg[C.AMOUNT_USD]
    ws, wn = S.weighted_sensitivities(sample, p12), S.weighted_sensitivities(neg, p12)
    for f in (S.delta_margin_ir, S.delta_margin_fx):
        assert f(wn, p12)[0] == pytest.approx(f(ws, p12)[0], rel=1e-12)
    for f in (S.vega_margin_ir, S.vega_margin_fx):
        assert f(neg, p12)[0] == pytest.approx(f(sample, p12)[0], rel=1e-12)
    delta_only = sample[~sample[C.RISK_TYPE].isin([IRV, FXV])]
    dn = delta_only.copy()
    dn[C.AMOUNT_USD] = -dn[C.AMOUNT_USD]
    assert S.simm(dn, p12).total == pytest.approx(S.simm(delta_only, p12).total, rel=1e-12)


def test_curvature_is_not_sign_symmetric_documented(sample, p12):
    neg = sample.copy()
    neg[C.AMOUNT_USD] = -neg[C.AMOUNT_USD]
    a = S.simm(sample, p12).by_margin_type
    b = S.simm(neg, p12).by_margin_type
    assert a[(C.RC_FX, C.MT_CURVATURE)] != b[(C.RC_FX, C.MT_CURVATURE)] or a[(C.RC_IR, C.MT_CURVATURE)] != b[(C.RC_IR, C.MT_CURVATURE)]


def test_homogeneity_when_cr_is_one(sample, p12):
    base = S.simm(sample, p12).total
    for k in (0.5, 2.0, 7.0):
        scaled = sample.copy()
        scaled[C.AMOUNT_USD] = scaled[C.AMOUNT_USD] * k
        assert S.simm(scaled, p12).total == pytest.approx(k * base, rel=1e-10)


def test_superlinear_when_cr_above_one(p12):
    base = mk((IRC, "USD", "2y", "OIS", 4 * 220e6), (FXR, "EUR", "", "", 4 * 2100e6))
    one = S.simm(base, p12).total
    dbl = base.copy()
    dbl[C.AMOUNT_USD] *= 2
    assert S.simm(dbl, p12).total > 2 * one * (1 + 1e-6)
    assert S.simm(dbl, p12).total == pytest.approx(2 * np.sqrt(2) * one, rel=1e-10)    # CR grows with sqrt(k)


def test_offsetting_trades_reduce_im(sample, p12):
    t = sample[sample[C.CRIF_TRADE_ID] == "IRS_USD_10Y"]
    opp = t.copy()
    opp[C.AMOUNT_USD] = -opp[C.AMOUNT_USD]
    opp[C.CRIF_TRADE_ID] = "OPP"
    assert S.simm(pd.concat([t, opp]), p12).total == pytest.approx(0.0, abs=1e-6)
    half = opp.copy()
    half[C.AMOUNT_USD] *= 0.5
    assert S.simm(pd.concat([t, half]), p12).total < S.simm(t, p12).total


def _split(sample, n_groups=2, seed=0):
    ids = sorted(sample[C.CRIF_TRADE_ID].unique())
    rng = np.random.default_rng(seed)
    g = dict(zip(ids, rng.integers(0, n_groups, len(ids))))
    lab = sample[C.CRIF_TRADE_ID].map(g)
    return [sample[lab == k] for k in range(n_groups)]


def test_delta_vega_subadditive_within_risk_class(sample, p12):
    for seed in range(6):
        a, b = _split(sample, 2, seed)
        for fn in (lambda x: S.delta_margin_ir(S.weighted_sensitivities(x, p12), p12)[0],
                   lambda x: S.delta_margin_fx(S.weighted_sensitivities(x, p12), p12)[0],
                   lambda x: S.vega_margin_ir(x, p12)[0], lambda x: S.vega_margin_fx(x, p12)[0]):
            assert fn(sample) <= fn(a) + fn(b) + NEG_TOL


def test_curvature_subadditivity_reported_not_asserted(sample, p12, record_property):
    viol = 0
    for seed in range(6):
        a, b = _split(sample, 2, seed)
        for rc in (C.RC_IR, C.RC_FX):
            whole = S.curvature_margin(sample, p12, rc)[0]
            parts = S.curvature_margin(a, p12, rc)[0] + S.curvature_margin(b, p12, rc)[0]
            viol += int(whole > parts + NEG_TOL)
    record_property("curvature_subadditivity_violations_of_12", viol)
    assert 0 <= viol <= 12


def test_psi_bounds(sample, p12):
    r = S.simm(sample, p12)
    zero = S.simm(sample, p12, psi_override=0.0).total
    assert zero <= r.total <= sum(r.by_risk_class.values()) + 1e-9
    assert zero == pytest.approx(np.sqrt(sum(v ** 2 for v in r.by_risk_class.values())), rel=1e-12)


def test_breakdown_is_consistent(sample, p12):
    r = S.simm(sample, p12)
    b = r.breakdown
    assert r.total == b[b[C.LEVEL] == "total"][C.VALUE].iloc[0]
    for rc in (C.RC_IR, C.RC_FX):
        parts = b[(b[C.LEVEL] == "margin_type") & (b[C.RISK_CLASS] == rc)][C.VALUE].sum()
        assert parts == pytest.approx(b[(b[C.LEVEL] == "risk_class") & (b[C.RISK_CLASS] == rc)][C.VALUE].iloc[0], rel=1e-12)
    assert set(b[C.LEVEL]) >= {"factor", "K", "S", "margin_type", "risk_class", "product_class", "total"}


def test_version_differences_direction(p12, p06):
    # FX regular currency: 7.4 (2512) versus 7.1 (2506), high-vol currency 31.7 versus 18.0
    fx = mk((FXR, "EUR", "", "", 1.0e6), (FXR, "GBP", "", "", -3.0e5))
    assert S.simm(fx, p12).total > S.simm(fx, p06).total
    ars = mk((FXR, "ARS", "", "", 1.0e5))
    assert S.simm(ars, p12).total == pytest.approx(S.simm(ars, p06).total * 31.7 / 18.0, rel=1e-12)
    # high-vol IR delta threshold 71 (2512) versus 51 (2506): same position is more concentrated under 2506
    mxn = mk((IRC, "MXN", "5y", "OIS", 5.0e8))
    assert S.simm(mxn, p06).total > S.simm(mxn, p12).total
    # IR-FX psi 15% versus 10%
    both = mk((IRC, "USD", "5y", "OIS", 1.0e5), (FXR, "EUR", "", "", 1.0e6))
    assert S.simm(both, p12).total > S.simm(both, p06).total
    # FX vega threshold change (2900 vs 2800) and HVR/VRW changes
    v = mk((FXV, "EURUSD", "1y", "", 1.0e4))
    assert S.simm(v, p06).total != S.simm(v, p12).total


def test_sample_portfolio_runs_and_is_positive(sample, p12):
    r = S.simm(sample, p12)
    assert r.total > 0 and all(v > 0 for v in r.by_risk_class.values())
    ws = S.weighted_sensitivities(sample, p12)
    assert (ws[C.CR] == 1.0).all()                       # sample portfolio is far below every concentration threshold
