"""Vectorised pricing of the RatesFX sample trades over many market states. SYNTHETIC curves and vols.

All prices are in USD (calculation currency). Single OIS-style curve per currency for discounting and
projection; swaptions use Bachelier (normal vol), FX options Garman-Kohlhagen; FX forwards follow interest-rate
parity; the XCCY swap is two floating legs on their own curves plus (contract basis - market basis) * annuity.
"""
import numpy as np
from scipy.stats import norm

from . import columns as C
from .curves import interp_weights
from .instruments import (BASIS_SPREAD, EXCHANGE_ELIGIBLE, OPT_CALL, OPT_PAYER, pair_name)
from .market_history import FX_VOL_EXPIRIES, IR_VOL_EXPIRIES, IR_VOL_TENORS, MarketBatch

# ===== CONFIG (user inputs) =====
MIN_SQRT_T = 1.0e-8        # guards the option formulas against zero expiry
# ===== END CONFIG =====

_IR_EXP, _IR_TEN = np.array(IR_VOL_EXPIRIES), np.array(IR_VOL_TENORS)
_FX_EXP = np.array(FX_VOL_EXPIRIES)


def ir_vol(batch: MarketBatch, ccy: str, expiry: float, tenor: float) -> np.ndarray:
    """Normal vol [n] by bilinear interpolation on the (expiry, tenor) grid, flat outside."""
    g = batch.ir_nvol[ccy]
    i, w = interp_weights(expiry, _IR_EXP)
    j, v = interp_weights(tenor, _IR_TEN)
    i, w, j, v = i[0], w[0], j[0], v[0]
    return ((g[:, i, j] * (1 - v) + g[:, i, j + 1] * v) * (1 - w) + (g[:, i + 1, j] * (1 - v) + g[:, i + 1, j + 1] * v) * w)


def fx_vol(batch: MarketBatch, pair: str, expiry: float) -> np.ndarray:
    """Lognormal vol [n] linear in expiry, flat outside."""
    i, w = interp_weights(expiry, _FX_EXP)
    g = batch.fx_vol[pair]
    return g[:, i[0]] * (1 - w[0]) + g[:, i[0] + 1] * w[0]


def bachelier(F, K, sigma, T, payer: bool):
    """Undiscounted Bachelier price per unit annuity."""
    s = sigma * np.sqrt(max(T, MIN_SQRT_T))
    d = (F - K) / s
    return ((F - K) * norm.cdf(d) if payer else (K - F) * norm.cdf(-d)) + s * norm.pdf(d)


def bachelier_vega(F, K, sigma, T):
    """dPrice/dsigma per unit annuity = sqrt(T) * phi(d)."""
    s = sigma * np.sqrt(max(T, MIN_SQRT_T))
    return np.sqrt(max(T, MIN_SQRT_T)) * norm.pdf((F - K) / s)


def _gk_d(F, K, sigma, T):
    s = sigma * np.sqrt(max(T, MIN_SQRT_T))
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    return d1, d1 - s


def gk_price(F, K, sigma, T, df_quote, call: bool):
    """Garman-Kohlhagen price in quote currency per unit of base notional (F is the forward)."""
    d1, d2 = _gk_d(F, K, sigma, T)
    if call:
        return df_quote * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df_quote * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def gk_vega(F, K, sigma, T, df_quote):
    """dPrice/dsigma (per unit vol) in quote currency per unit base notional."""
    d1, _ = _gk_d(F, K, sigma, T)
    return df_quote * F * norm.pdf(d1) * np.sqrt(max(T, MIN_SQRT_T))


def gk_delta_spot(F, K, sigma, T, df_base, call: bool):
    """dPrice/dSpot (quote per base) per unit base notional: DF_base * Phi(+/-d1)."""
    d1, _ = _gk_d(F, K, sigma, T)
    return df_base * (norm.cdf(d1) if call else -norm.cdf(-d1))


def _irs(t, batch):
    c, s_, e = batch.curve(t[C.CCY]), t[C.START], t[C.MATURITY]
    d = c.df(np.array([s_, e]))
    return t[C.NOTIONAL] * t[C.DIRECTION] * ((d[:, 0] - d[:, 1]) - t[C.STRIKE] * c.annuity(s_, e)) * batch.fx_spot[t[C.CCY]]


def _swaption(t, batch):
    c = batch.curve(t[C.CCY])
    exp, mat = t[C.EXPIRY], t[C.MATURITY]
    F, A = c.forward_swap(exp, mat)
    sig = ir_vol(batch, t[C.CCY], exp, mat - exp)
    px = bachelier(F, t[C.STRIKE], sig, exp, t[C.OPT_TYPE] == OPT_PAYER)
    return t[C.NOTIONAL] * t[C.DIRECTION] * A * px * batch.fx_spot[t[C.CCY]]


def _fxfwd(t, batch):
    b, q, T = t[C.CCY], t[C.CCY2], t[C.MATURITY]
    dfb, dfq = batch.curve(b).df(T)[:, 0], batch.curve(q).df(T)[:, 0]
    return t[C.NOTIONAL] * t[C.DIRECTION] * (batch.fx_spot[b] * dfb - t[C.STRIKE] * batch.fx_spot[q] * dfq)


def fx_forward_rate(batch, b, q, T):
    """Forward FX (quote per base) by interest-rate parity, [n]."""
    dfb, dfq = batch.curve(b).df(T)[:, 0], batch.curve(q).df(T)[:, 0]
    return batch.fx_spot[b] / batch.fx_spot[q] * dfb / dfq


def _fxopt(t, batch):
    b, q, T = t[C.CCY], t[C.CCY2], t[C.EXPIRY]
    F = fx_forward_rate(batch, b, q, T)
    dfq = batch.curve(q).df(T)[:, 0]
    sig = fx_vol(batch, pair_name(b, q), T)
    px = gk_price(F, t[C.STRIKE], sig, T, dfq, t[C.OPT_TYPE] == OPT_CALL)
    return t[C.NOTIONAL] * t[C.DIRECTION] * px * batch.fx_spot[q]


def _xccy(t, batch, include_exchange):
    e_, u_ = t[C.CCY], t[C.CCY2]
    ce, cu = batch.curve(e_), batch.curve(u_)
    s_, m = t[C.START], t[C.MATURITY]
    n_e = t[C.NOTIONAL]
    n_u = n_e * t[C.STRIKE]
    de, du = ce.df(np.array([s_, m])), cu.df(np.array([s_, m]))
    mkt = batch.xccy_basis.get(pair_name(e_, u_), 0.0)
    leg_e = n_e * (de[:, 0] - de[:, 1]) + (t[BASIS_SPREAD] - mkt) * n_e * ce.annuity(s_, m)
    leg_u = n_u * (du[:, 0] - du[:, 1])
    if include_exchange:
        leg_e = leg_e + n_e * de[:, 1] - (n_e * de[:, 0] if s_ > 0 else 0.0)
        leg_u = leg_u + n_u * du[:, 1] - (n_u * du[:, 0] if s_ > 0 else 0.0)
    return t[C.DIRECTION] * (leg_e * batch.fx_spot[e_] - leg_u * batch.fx_spot[u_])


def xccy_include_exchange(trade) -> bool:
    return trade[C.EXCHANGE_FLAG] != EXCHANGE_ELIGIBLE


def price_trade(trade, batch: MarketBatch, include_exchange: bool = True) -> np.ndarray:
    """USD PV of one trade (row of the trade table) for every state in the batch, shape [n_states].
    include_exchange only matters for XCCY (full MTM includes the principal exchange)."""
    p = trade[C.PRODUCT]
    if p == C.PRODUCT_IRS:
        return _irs(trade, batch)
    if p == C.PRODUCT_SWAPTION:
        return _swaption(trade, batch)
    if p == C.PRODUCT_FXFWD:
        return _fxfwd(trade, batch)
    if p == C.PRODUCT_FXOPT:
        return _fxopt(trade, batch)
    if p == C.PRODUCT_XCCY:
        return _xccy(trade, batch, include_exchange)
    raise ValueError(f"unknown product {p}")


def price_one(trade, batch_row: MarketBatch) -> float:
    """Scalar PV for a single trade on a single-state batch (reference for the vectorised path)."""
    return float(price_trade(trade, batch_row)[0])


def price_portfolio(trades, batch: MarketBatch) -> np.ndarray:
    """USD PVs, shape [n_states, n_trades], columns in trade-table order (full MTM, XCCY includes the exchange)."""
    out = np.empty((batch.n_states, len(trades)))
    for k, (_, t) in enumerate(trades.iterrows()):
        out[:, k] = price_trade(t, batch)
    return out
