"""Unilateral CVA on the linear trades of the netting set plus a sensitivity-based xVA VaR model. ALL DATA IS SYNTHETIC.

Credit: piecewise-constant hazard bootstrapped from CDS par spreads at 1/3/5/7/10y (history.cds), recovery 40%.
Exposure: for the linear trades (IRS, FX forwards, XCCY; options are excluded, see findings) the netting-set value at
grid time t_k is Gaussian with mean mu_k = forward value of the remaining cash flows (today's PV of flows after t_k
divided by the USD discount factor to t_k) and variance s_k^2 = t_k * d_k' S d_k. The vector d_k holds the sensitivities
of mu_k to nine factors (a parallel shift of each of the 5 zero curves, per unit rate; log spot of the 4 non-USD
currencies) and S is their ANNUALISED covariance estimated from the trailing ROLLING_COV_DAYS daily changes of the
history (parallel shift = change of the mean zero rate over the 12 pillars; FX = log spot change), so sigma_r, sigma_fx and
the cross terms all come from the history. Then EE_k = mu*Phi(mu/s) + s*phi(mu/s) and
CVA = LGD * sum 0.5*(EE_{k-1}DF_{k-1} + EE_k DF_k) * (Q_{k-1} - Q_k) (trapezoid, adapted from the CCR sibling project).
xVA VaR: historical simulation of CS01 ladder (5), IR parallel delta (5) and FX delta (4) sensitivities against the
realized full-revaluation change in CVA (model parameters held at their date-t values), 1-day and 10-day.
Trade times are frozen (no ageing), like the IM backtest.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import columns as C
from . import config
from .curves import CURRENCIES, payment_times
from .instruments import BASIS_SPREAD, pair_name
from .market_history import CCY_FX_PAIR, CDS_TENORS, MarketBatch, MarketHistory
from .var_model import first_valid_index, trades_digest, weighted_var, window_days_for

# ===== CONFIG (user inputs) =====
RECOVERY = 0.40
LGD = 1.0 - RECOVERY
CDS_PILLARS = np.array(CDS_TENORS)
CDS_FREQ = 4                       # quarterly premium grid in the CDS bootstrap
HAZARD_MAX, BISECT_ITERS = 5.0, 60
GRID_STEP = 0.25                   # exposure grid spacing in years
ROLLING_COV_DAYS = config.BACKTEST_WINDOW
TRADING_DAYS = 252
RATE_BUMP, FX_BUMP_REL, CDS_BUMP = 1.0e-4, 0.01, 1.0e-4
LINEAR_PRODUCTS = (C.PRODUCT_IRS, C.PRODUCT_FXFWD, C.PRODUCT_XCCY)
XVA_HORIZONS = (1, 10)
CACHE_SIZE = 4
# ===== END CONFIG =====

FACTOR_CCYS = CURRENCIES                       # parallel zero shifts, USD included
FX_CCYS = tuple(c for c in CURRENCIES if c != config.CALC_CCY)
N_RATE, N_FX = len(FACTOR_CCYS), len(FX_CCYS)
N_FACTORS = N_RATE + N_FX
_CACHE: "OrderedDict" = OrderedDict()


def linear_trades(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[trades[C.PRODUCT].isin(LINEAR_PRODUCTS)].reset_index(drop=True)


# ---------- hazard curve ----------
def survival(lam: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Survival probabilities [n, len(t)] for piecewise-constant hazards lam [n, 5] on the CDS pillars (flat after 10y)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    starts = np.concatenate([[0.0], CDS_PILLARS[:-1]])
    ends = np.concatenate([CDS_PILLARS[:-1], [np.inf]])
    span = np.clip(np.minimum(t[:, None], ends[None, :]) - starts[None, :], 0.0, None)      # [len(t), 5]
    return np.exp(-lam @ span.T)


def cds_par_spread(lam: np.ndarray, T: float, usd_curve, recovery: float = RECOVERY) -> np.ndarray:
    """Par spread [n] of a T-year CDS with quarterly premiums (accrual on default approximated by the half-period rule)."""
    u = np.arange(0, int(round(T * CDS_FREQ)) + 1) / CDS_FREQ
    q = survival(lam, u)
    df = usd_curve.df(u[1:])
    protection = (1.0 - recovery) * (df * (q[:, :-1] - q[:, 1:])).sum(1)
    rpv01 = (df * 0.5 * (q[:, :-1] + q[:, 1:])).sum(1) / CDS_FREQ
    return protection / rpv01


def bootstrap_hazard(cds: np.ndarray, usd_curve, recovery: float = RECOVERY) -> np.ndarray:
    """Hazards [n, 5] repricing the 1/3/5/7/10y par spreads exactly (vectorised bisection over states)."""
    cds = np.atleast_2d(cds)
    lam = np.zeros((cds.shape[0], len(CDS_PILLARS)))
    for j, T in enumerate(CDS_PILLARS):
        lo, hi = np.zeros(len(cds)), np.full(len(cds), HAZARD_MAX)
        for _ in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            lam[:, j] = mid
            below = cds_par_spread(lam, T, usd_curve, recovery) < cds[:, j]
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        lam[:, j] = 0.5 * (lo + hi)
    return lam


# ---------- exposure ----------
def exposure_grid(trades: pd.DataFrame) -> np.ndarray:
    return np.arange(0.0, float(trades[C.MATURITY].max()) + 1e-9, GRID_STEP)


def remaining_pv(trade, batch: MarketBatch, grid: np.ndarray) -> np.ndarray:
    """USD PV today [n, K] of the trade's cash flows strictly after each grid time (linear products only)."""
    p, T, N, d = trade[C.PRODUCT], trade[C.MATURITY], trade[C.NOTIONAL], trade[C.DIRECTION]
    alive = (grid < T)[None, :]
    if p == C.PRODUCT_FXFWD:
        b, q = trade[C.CCY], trade[C.CCY2]
        v = N * d * (batch.fx_spot[b] * batch.curve(b).df(T)[:, 0] - trade[C.STRIKE] * batch.fx_spot[q] * batch.curve(q).df(T)[:, 0])
        return v[:, None] * alive
    tp = payment_times(0.0, T)
    acc = np.diff(np.concatenate(([0.0], tp)))
    mask = (tp[:, None] > grid[None, :]) * acc[:, None]                    # [P, K]

    def leg(ccy):
        c = batch.curve(ccy)
        return c.df(grid), c.df(T)[:, [0]], c.df(tp) @ mask               # DF(t_k), DF(T), annuity after t_k
    if p == C.PRODUCT_IRS:
        dfg, dfT, ann = leg(trade[C.CCY])
        v = N * d * ((dfg - dfT) - trade[C.STRIKE] * ann) * batch.fx_spot[trade[C.CCY]][:, None]
        return v * alive
    if p == C.PRODUCT_XCCY:
        e, u = trade[C.CCY], trade[C.CCY2]
        dfe, dfeT, anne = leg(e)
        dfu, _, _ = leg(u)
        mkt = batch.xccy_basis.get(pair_name(e, u), 0.0)
        mkt = mkt[:, None] if isinstance(mkt, np.ndarray) else mkt
        n_u = N * trade[C.STRIKE]
        leg_e = N * (dfe - dfeT) + (trade[BASIS_SPREAD] - mkt) * N * anne + N * dfeT      # principal at maturity
        leg_u = n_u * dfu                                                                 # float leg plus principal
        return d * (leg_e * batch.fx_spot[e][:, None] - leg_u * batch.fx_spot[u][:, None]) * alive
    raise ValueError(f"not a linear product: {p}")


def _book_mu(lin: pd.DataFrame, batch: MarketBatch, grid: np.ndarray) -> np.ndarray:
    """Forward value mu [n, K] of the netting set (USD at t_k)."""
    pv = sum(remaining_pv(t, batch, grid) for _, t in lin.iterrows())
    return pv / batch.curve(config.CALC_CCY).df(grid)


def _bumped(batch: MarketBatch, kind: str, ccy: str, size: float) -> MarketBatch:
    b = batch.copy_with()
    if kind == "rate":
        b.zero[ccy] = batch.zero[ccy] + size
    else:
        b.fx_spot[ccy] = batch.fx_spot[ccy] * (1.0 + size)
    return b


def factor_sensitivities(lin: pd.DataFrame, batch: MarketBatch, grid: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """d mu_k / d factor [n, K, 9]: per unit parallel rate shift (5 ccys) then per unit log spot (4 ccys)."""
    cols = []
    for c in FACTOR_CCYS:
        cols.append((_book_mu(lin, _bumped(batch, "rate", c, RATE_BUMP), grid) - mu) / RATE_BUMP)
    for c in FX_CCYS:
        cols.append((_book_mu(lin, _bumped(batch, "fx", c, FX_BUMP_REL), grid) - mu) / np.log1p(FX_BUMP_REL))
    return np.stack(cols, axis=2)


def ee_profile(trades: pd.DataFrame, batch: MarketBatch, cov: np.ndarray, grid: np.ndarray = None) -> dict:
    """Normal-approximation EE profile of the linear trades. cov: annualised factor covariance [n,9,9] or [9,9]."""
    lin = linear_trades(trades)
    grid = exposure_grid(lin) if grid is None else grid
    mu = _book_mu(lin, batch, grid)
    d = factor_sensitivities(lin, batch, grid, mu)
    cov = np.broadcast_to(cov, (batch.n_states, N_FACTORS, N_FACTORS))
    var = np.einsum("nki,nij,nkj->nk", d, cov, d) * grid[None, :]
    s = np.sqrt(np.maximum(var, 1e-18))
    z = mu / s
    ee = mu * norm.cdf(z) + s * norm.pdf(z)
    ee[:, 0] = np.maximum(mu[:, 0], 0.0)                                    # no diffusion at t = 0
    return dict(grid=grid, mu=mu, s=s, ee=ee, df=batch.curve(config.CALC_CCY).df(grid))


def cva_from_profile(prof: dict, lam: np.ndarray, lgd: float = LGD) -> np.ndarray:
    """CVA [n] = LGD * sum 0.5*(EE DF)_{k-1,k} * (Q_{k-1} - Q_k)."""
    q = survival(lam, prof["grid"])
    de = prof["ee"] * prof["df"]
    return lgd * (0.5 * (de[:, :-1] + de[:, 1:]) * (q[:, :-1] - q[:, 1:])).sum(1)


def cva_state(trades: pd.DataFrame, batch: MarketBatch, cov: np.ndarray, recovery: float = RECOVERY,
              cds: np.ndarray = None) -> dict:
    """CVA [n] for market states (USD), with the profile and hazards that produced it."""
    prof = ee_profile(trades, batch, cov)
    lam = bootstrap_hazard(batch.cds if cds is None else cds, batch.curve(config.CALC_CCY), recovery)
    return dict(cva=cva_from_profile(prof, lam, 1.0 - recovery), profile=prof, lam=lam)


# ---------- factor history ----------
def factor_levels(history: MarketHistory) -> np.ndarray:
    """[T, 14] levels: CDS spreads (5), parallel zero rate per ccy (5), log spot per non-USD ccy (4)."""
    m = history.market
    zs = [m.zero[c].mean(1) for c in FACTOR_CCYS]
    fx = [np.log(m.fx_spot[c]) for c in FX_CCYS]
    return np.column_stack([m.cds] + zs + fx)


def rolling_factor_cov(history: MarketHistory, window: int = ROLLING_COV_DAYS) -> np.ndarray:
    """Annualised covariance [T, 9, 9] of daily (parallel rate, log FX) changes over the trailing window; NaN before."""
    lv = factor_levels(history)[:, len(CDS_PILLARS):]                  # drop the 5 CDS columns
    d = np.vstack([np.zeros((1, lv.shape[1])), np.diff(lv, axis=0)])
    T = len(d)
    out = np.full((T, N_FACTORS, N_FACTORS), np.nan)
    c1 = np.vstack([np.zeros((1, N_FACTORS)), np.cumsum(d, axis=0)])
    c2 = np.concatenate([np.zeros((1, N_FACTORS, N_FACTORS)), np.cumsum(d[:, :, None] * d[:, None, :], axis=0)])
    for t in range(window, T):                                                     # day 0 change is a placeholder zero
        s1, s2 = c1[t + 1] - c1[t + 1 - window], c2[t + 1] - c2[t + 1 - window]
        out[t] = (s2 - np.outer(s1, s1) / window) / (window - 1) * TRADING_DAYS
    return out


def factor_vols(cov: np.ndarray) -> dict:
    """Annualised sigma per factor (for reporting): rate vols in bp, FX vols in percent."""
    sd = np.sqrt(np.diag(cov))
    return {**{f"sigma_r_{c}_bp": sd[i] * 1e4 for i, c in enumerate(FACTOR_CCYS)},
            **{f"sigma_fx_{c}_pct": sd[N_RATE + i] * 100 for i, c in enumerate(FX_CCYS)}}


# ---------- CVA level and series ----------
def cva_value(history: MarketHistory, trades: pd.DataFrame, t=None) -> dict:
    """CVA (USD) at date t (default last date) with the profile summary."""
    ti = len(history) - 1 if t is None else int(history.index_of([pd.Timestamp(t)])[0])
    cov = rolling_factor_cov(history)[ti]
    res = cva_state(trades, history.market.row(ti), cov)
    ee = res["profile"]["ee"][0]
    return dict(cva=float(res["cva"][0]), epe=float(np.trapezoid(ee, res["profile"]["grid"]) / res["profile"]["grid"][-1]),
                peak_ee=float(ee.max()), lam=res["lam"][0], vols=factor_vols(cov))


def xva_state(history: MarketHistory, trades: pd.DataFrame) -> dict:
    """Per valid date: model CVA, sensitivities [nT, 14] (CS01 x5, IR parallel x5, FX x4) and cached CVA(t+h) changes."""
    key = (id(history), len(history), trades_digest(trades))
    if key in _CACHE:
        return _CACHE[key]
    ts = np.arange(first_valid_index(history), len(history))
    cov_all = rolling_factor_cov(history)
    cov = cov_all[ts]
    m = history.market.take(ts)
    base = cva_state(trades, m, cov)
    sens = np.empty((len(ts), N_FACTORS + len(CDS_PILLARS)))
    usd = m.curve(config.CALC_CCY)
    for j in range(len(CDS_PILLARS)):                                              # CS01 per unit spread, profile reused
        cds = m.cds.copy()
        cds[:, j] += CDS_BUMP
        lam = bootstrap_hazard(cds, usd)
        sens[:, j] = (cva_from_profile(base["profile"], lam) - base["cva"]) / CDS_BUMP
    k = len(CDS_PILLARS)
    for i, c in enumerate(FACTOR_CCYS):
        up = cva_state(trades, _bumped(m, "rate", c, RATE_BUMP), cov)["cva"]
        sens[:, k + i] = (up - base["cva"]) / RATE_BUMP
    for i, c in enumerate(FX_CCYS):
        up = cva_state(trades, _bumped(m, "fx", c, FX_BUMP_REL), cov)["cva"]
        sens[:, k + N_RATE + i] = (up - base["cva"]) / np.log1p(FX_BUMP_REL)
    out = dict(ts=ts, cov=cov, cva=base["cva"], sens=sens, real={})
    _CACHE[key] = out
    while len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)
    return out


def realized_cva_change(history, trades, horizon: int) -> pd.Series:
    """Full-revaluation CVA(t+h) - CVA(t) (positive = loss), model parameters (covariance) held at date t; NaN at the end."""
    st = xva_state(history, trades)
    if horizon not in st["real"]:
        ts = st["ts"]
        ok = ts + horizon < len(history)
        later = cva_state(trades, history.market.take(ts[ok] + horizon), st["cov"][ok])["cva"]
        chg = np.full(len(ts), np.nan)
        chg[ok] = later - st["cva"][ok]
        st["real"][horizon] = chg
    return pd.Series(st["real"][horizon], index=history.dates[st["ts"]], name=f"xva_loss_h{horizon}")


def xva_var_series(history, trades, horizon: int = 1, conf: float = config.CONF) -> pd.Series:
    """Sensitivity-based HS xVA VaR at each valid date: quantile of sens . (h-day factor changes) over the newest window."""
    st = xva_state(history, trades)
    lv = factor_levels(history)
    chg = np.zeros_like(lv)
    chg[horizon:] = lv[horizon:] - lv[:-horizon]
    W = window_days_for(history)
    var = np.empty(len(st["ts"]))
    for r, t in enumerate(st["ts"]):
        sc = chg[t - W + 1:t + 1] @ st["sens"][r]                                  # CVA change per scenario = loss
        var[r] = weighted_var(sc, np.full(W, 1.0 / W), conf)[0]
    return pd.Series(var, index=history.dates[st["ts"]], name=f"xva_var_h{horizon}")
