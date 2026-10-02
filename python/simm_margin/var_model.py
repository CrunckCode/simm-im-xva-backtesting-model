"""Historical-simulation (HS) VaR initial margin with full revaluation, plus realized losses. ALL DATA IS SYNTHETIC.

Design (cost: about 3.5 microseconds per scenario for the whole 20-trade book, so a full 1-day plus 10-day run over
~1,300 backtest days is about 2M revaluations, roughly 20 seconds): the scenario set is a window of historical
h-day factor changes (overlapping) applied to the market state at date t. Changes are zero-rate and par level
changes, log FX spot, log IR normal vol and log FX vol changes. The revaluation matrix of all scenarios for all dates
is computed once per horizon and cached; the champion and the challengers are different weightings or subsets of that
same matrix (EWMA and the plain 250-day model use the newest 250 recent changes, the scaled model multiplies the champion).
Window rule (EU 2016/2251 Art 16 style): the newest WINDOW_DAYS changes, with the oldest ones replaced by changes from
the forced stress window until at least MIN_STRESS_SHARE of the window is stressed (regime label 1 or stress window).
Loss = -(V(t+h) - V(t)) on the frozen portfolio (no ageing, so hypothetical P&L net of cash flows).
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import columns as C
from . import config
from .market_history import MarketBatch, MarketHistory
from .pricing import price_portfolio

# ===== CONFIG (user inputs) =====
MODEL_CHAMPION, MODEL_EWMA, MODEL_PLAIN250, MODEL_SCALED = "eu_1plus3", "ewma", "plain250", "scaled"
MODELS = (MODEL_CHAMPION, MODEL_EWMA, MODEL_PLAIN250, MODEL_SCALED)
PLAIN_WINDOW = 250                 # recent changes used by the EWMA and plain challengers
CHUNK_STATES = 30000               # revaluation batch size (memory bound)
CACHE_SIZE = 6
QUICK_HISTORY_DAYS = 800           # shorter histories use QUICK_SCENARIOS as the window
# ===== END CONFIG =====

_CACHE: "OrderedDict" = OrderedDict()


def clear_cache():
    _CACHE.clear()


def window_days_for(history: MarketHistory) -> int:
    """Window length: config.WINDOW_DAYS on the full history, QUICK_SCENARIOS on a short (quick) history."""
    return config.WINDOW_DAYS if len(history) >= QUICK_HISTORY_DAYS else config.QUICK_SCENARIOS


def trades_digest(trades: pd.DataFrame) -> int:
    return int(pd.util.hash_pandas_object(trades.astype(str), index=False).sum() % (2 ** 62))


def _to_index(history: MarketHistory, t) -> int:
    if isinstance(t, (int, np.integer)):
        return int(t)
    return int(history.index_of([pd.Timestamp(t)])[0])


# ---------- historical change bank ----------
def _levels(m: MarketBatch) -> dict:
    """Factor arrays as (kind, key, array [T, ...]); changes are differences of these (logs for multiplicative)."""
    out = {("zero", c): v for c, v in m.zero.items()}
    out.update({("par", c): v for c, v in m.par.items()})
    out.update({("lfx", c): np.log(v) for c, v in m.fx_spot.items()})
    out.update({("lirv", c): np.log(v) for c, v in m.ir_nvol.items()})
    out.update({("lfxv", p): np.log(v) for p, v in m.fx_vol.items()})
    out.update({("xb", p): v for p, v in m.xccy_basis.items()})
    return out


def _diff(x: np.ndarray, h: int) -> np.ndarray:
    d = np.zeros_like(x)
    d[h:] = x[h:] - x[:-h]
    return d


def change_bank(history: MarketHistory, horizon: int) -> dict:
    """Bank of h-day changes: rows 0..T-1 are the history (row s = change ending at s), rows T.. are the stress window."""
    rec, st = _levels(history.market), _levels(history.stress_batch)
    return {k: np.concatenate([_diff(rec[k], horizon), _diff(st[k], horizon)], axis=0) for k in rec}


def _shocked(m, base_idx, bank, bank_idx) -> MarketBatch:
    """States m[base_idx] shocked by bank rows: zero/par additive, FX and vols multiplicative (log changes)."""
    zero = {k: v[base_idx] + bank[("zero", k)][bank_idx] for k, v in m.zero.items()}
    par = {k: v[base_idx] + bank[("par", k)][bank_idx] for k, v in m.par.items()}
    fx = {k: v[base_idx] * np.exp(bank[("lfx", k)][bank_idx]) for k, v in m.fx_spot.items()}
    irv = {k: v[base_idx] * np.exp(bank[("lirv", k)][bank_idx]) for k, v in m.ir_nvol.items()}
    fxv = {k: v[base_idx] * np.exp(bank[("lfxv", k)][bank_idx]) for k, v in m.fx_vol.items()}
    xb = {k: v[base_idx] + bank[("xb", k)][bank_idx] for k, v in m.xccy_basis.items()}
    return MarketBatch(par, zero, fx, irv, fxv, m.cds[base_idx], xb, None)


# ---------- window composition ----------
def _stress_cap(history, horizon) -> int:
    return max(history.stress_batch.n_states - horizon, 0)


def first_valid_index(history: MarketHistory, horizon: int = None) -> int:
    """Earliest date index with a full window of h-day changes for every horizon in use."""
    return window_days_for(history) + max(horizon or 0, max(config.HORIZON_DAYS, 1))


def stress_replacement(history: MarketHistory, ts: np.ndarray, horizon: int, W: int = None) -> np.ndarray:
    """Number m of newest-window slots replaced by stress-window changes, per date index in ts."""
    W = W or window_days_for(history)
    cap = _stress_cap(history, horizon)
    cs = np.concatenate([[0], np.cumsum(history.regime.astype(int))])      # cs[i] = stressed days among first i
    need = int(np.ceil(config.MIN_STRESS_SHARE * W - 1e-9))
    m_grid = np.arange(cap + 1)
    n_r = W - m_grid                                                       # recent days kept
    stressed_recent = cs[ts[:, None] + 1] - cs[ts[:, None] + 1 - n_r[None, :]]
    ok = stressed_recent + m_grid[None, :] >= need
    first = np.where(ok.any(1), ok.argmax(1), cap)
    return first.astype(int)


def window_composition(history: MarketHistory, horizon: int = config.HORIZON_DAYS, ts=None) -> pd.DataFrame:
    """Per date: recent days kept, stress days added, and the stressed share of the window."""
    W = window_days_for(history)
    ts = np.arange(first_valid_index(history, horizon), len(history)) if ts is None else np.asarray(ts)
    m = stress_replacement(history, ts, horizon, W)
    cs = np.concatenate([[0], np.cumsum(history.regime.astype(int))])
    stressed = cs[ts + 1] - cs[ts + 1 - (W - m)] + m
    return pd.DataFrame({C.DATE: history.dates[ts], "n_recent": W - m, "n_stress_window": m,
                         "stressed_share": stressed / W})


# ---------- scenario revaluation ----------
def _scenario_pnl(history, trades, horizon, ts):
    """(recent [nT, W] pnl with column c = change ending at t-(W-1-c); stress [nT, cap] pnl, NaN beyond m; m)."""
    W, cap = window_days_for(history), _stress_cap(history, horizon)
    T = len(history)
    bank = change_bank(history, horizon)
    m_ = stress_replacement(history, ts, horizon, W)
    v0 = price_portfolio(trades, history.market).sum(1)
    recent = np.full((len(ts), W), np.nan)
    stress = np.full((len(ts), cap), np.nan)
    per = W + m_                                                           # scenarios per date
    start = 0
    while start < len(ts):
        stop, tot = start, 0
        while stop < len(ts) and (tot + per[stop] <= CHUNK_STATES or stop == start):
            tot += per[stop]
            stop += 1
        blk = np.arange(start, stop)
        base, bidx = [], []
        for r in blk:
            t = ts[r]
            bidx.append(np.concatenate([t - (W - 1 - np.arange(W)), T + horizon + np.arange(m_[r])]))
            base.append(np.full(per[r], t))
        base, bidx = np.concatenate(base), np.concatenate(bidx)
        pnl = price_portfolio(trades, _shocked(history.market, base, bank, bidx)).sum(1) - v0[base]
        off = 0
        for r in blk:
            recent[r] = pnl[off:off + W]
            stress[r, :m_[r]] = pnl[off + W:off + per[r]]
            off += per[r]
        start = stop
    return recent, stress, m_


def _engine(history, trades, horizon):
    """Cached revaluation matrices for every valid date at one horizon."""
    key = (id(history), len(history), trades_digest(trades), horizon)
    if key not in _CACHE:
        ts = np.arange(first_valid_index(history, horizon), len(history))
        recent, stress, m = _scenario_pnl(history, trades, horizon, ts)
        _CACHE[key] = dict(ts=ts, recent=recent, stress=stress, m=m)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def _champion_pnl(recent, stress, m) -> np.ndarray:
    W = recent.shape[1]
    cols = np.arange(W)[None, :]
    cap = stress.shape[1]
    st = np.zeros_like(recent)
    st[:, :min(W, cap)] = np.nan_to_num(stress[:, :min(W, cap)])
    return np.where(cols < m[:, None], st, recent)


def weighted_var(loss: np.ndarray, weights: np.ndarray, conf: float) -> np.ndarray:
    """Weighted empirical quantile (inverted CDF) of losses per row; weights [W] or [n, W] summing to 1."""
    loss = np.atleast_2d(loss)
    w = np.broadcast_to(weights, loss.shape)
    order = np.argsort(loss, axis=1)
    sl, sw = np.take_along_axis(loss, order, 1), np.take_along_axis(w, order, 1)
    cum = np.cumsum(sw, axis=1)
    idx = (cum >= conf - 1e-12).argmax(1)
    return sl[np.arange(len(sl)), idx]


def ewma_weights(n: int, lam: float = config.EWMA_LAMBDA) -> np.ndarray:
    """Weights for columns oldest..newest, proportional to lam^age, summing to 1."""
    w = lam ** np.arange(n - 1, -1, -1.0)
    return w / w.sum()


def _model_var(eng, model, conf, rows=slice(None)):
    recent, stress, m = eng["recent"][rows], eng["stress"][rows], eng["m"][rows]
    n = min(PLAIN_WINDOW, recent.shape[1])
    if model in (MODEL_CHAMPION, MODEL_SCALED):
        loss = -_champion_pnl(recent, stress, m)
        v = weighted_var(loss, np.full(loss.shape[1], 1.0 / loss.shape[1]), conf)
        return v * (config.CHALLENGER_SCALE if model == MODEL_SCALED else 1.0)
    loss = -recent[:, -n:]
    if model == MODEL_EWMA:
        return weighted_var(loss, ewma_weights(n), conf)
    if model == MODEL_PLAIN250:
        return weighted_var(loss, np.full(n, 1.0 / n), conf)
    raise ValueError(f"unknown model {model}")


def im_series(history, trades, horizon: int = config.HORIZON_DAYS, conf: float = config.CONF,
              model: str = MODEL_CHAMPION) -> pd.Series:
    """HS IM forecast made at each valid date t (information up to t), indexed by date."""
    eng = _engine(history, trades, horizon)
    return pd.Series(_model_var(eng, model, conf), index=history.dates[eng["ts"]], name=f"{model}_h{horizon}")


def hs_im(history: MarketHistory, trades: pd.DataFrame, t, horizon: int = config.HORIZON_DAYS,
          conf: float = config.CONF, window: str = "eu_1plus3") -> float:
    """HS IM (USD) at date t: full revaluation of the frozen portfolio under the historical window of h-day changes.
    window: eu_1plus3 (champion), ewma, plain250, scaled."""
    ti = _to_index(history, t)
    if ti < first_valid_index(history, horizon) or ti >= len(history):
        raise ValueError("date has no full calibration window")
    ts = np.array([ti])
    recent, stress, m = _scenario_pnl(history, trades, horizon, ts)
    return float(_model_var(dict(recent=recent, stress=stress, m=m), window, conf)[0])


# ---------- realized losses and SIMM ----------
def realized_loss_series(history: MarketHistory, trades: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Hypothetical loss -(V(t+h) - V(t)) of the frozen book, indexed by forecast date t (NaN for the last h dates).
    Times to maturity are constant, so there are no cash flows and the P&L is hypothetical, net of cash flows."""
    v = price_portfolio(trades, history.market).sum(1)
    loss = np.full(len(v), np.nan)
    loss[:-horizon] = -(v[horizon:] - v[:-horizon])
    return pd.Series(loss, index=history.dates, name=f"loss_h{horizon}")


def portfolio_value_series(history, trades) -> pd.Series:
    return pd.Series(price_portfolio(trades, history.market).sum(1), index=history.dates, name="value")


def daily_simm_series(history: MarketHistory, trades: pd.DataFrame, dates=None, version: str = config.SIMM_VERSION) -> pd.Series:
    """SIMM-style total IM (USD) per date from a CRIF built on that date's market (Jacobian method per day)."""
    from . import params, simm                     # Task B modules, imported lazily
    from .crif import build_crif
    p = params.load(version)
    dates = history.dates if dates is None else pd.DatetimeIndex(dates)
    out = [simm.simm(build_crif(trades, history.batch([d])), p).total for d in dates]
    return pd.Series(out, index=dates, name=f"simm_{version}")
