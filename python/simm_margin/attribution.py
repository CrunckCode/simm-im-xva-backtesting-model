"""IM change attribution: exact Shapley over four drivers, a sequential bridge, one-at-a-time and netting effect.

SIMM-style, not ISDA-licensed or certified. ALL TRADES AND MARKET DATA ARE SYNTHETIC.
Drivers: matured trades, new trades, market move, parameter version. IM(S) is the SIMM-style IM of the state in which
the drivers in S have been switched from their day-0 to their day-1 setting; the 16 subset IMs feed
  Shapley   phi_i = sum over S without i of |S|!(n-|S|-1)!/n! * [IM(S+i) - IM(S)]   (adds up to dIM exactly),
  bridge    matured -> new -> rates -> FX -> vol -> parameters (each step valued after the earlier ones; telescopes),
  one-at-a-time  IM({i}) - IM({}) with the interaction residual dIM - sum of those.
Bridge order rationale: trade population changes are valued at the old market and parameters, market moves on the new
population, and the methodology change last on today's portfolio and market (how a version-change impact is quoted).
A trade ID present on both days must have identical terms (an amendment is booked as matured plus new under a new ID).
Netting effect of the new trades = their incremental IM on the final book minus their standalone IM.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

from . import columns as C
from . import config
from . import params as params_mod
from .crif import build_crif
from .market_history import MarketBatch, MarketHistory
from .simm import simm

# ===== CONFIG (user inputs) =====
DRV_MATURED, DRV_NEW, DRV_MARKET, DRV_PARAMS = "matured_trades", "new_trades", "market_move", "parameter_version"
DRIVERS = (DRV_MATURED, DRV_NEW, DRV_MARKET, DRV_PARAMS)
SUB_RATES, SUB_FX, SUB_VOL = "market_rates", "market_fx", "market_vol"
BRIDGE_ORDER = (DRV_MATURED, DRV_NEW, SUB_RATES, SUB_FX, SUB_VOL, DRV_PARAMS)
MATURED_ON_SCENARIO_DAY = ("FXF_USDJPY_3M", "FXO_EURUSD_C3M")      # 3-month trades that expire on the scenario day
NEW_TRADE = {"template": "IRS_USD_10Y", "id": "IRS_USD_10Y_NEW", "notional": 150e6, "direction": -1}
# designed stress-like move relative to the prior business day (parallel par shift in bp, FX % vs USD, vol multipliers)
SCENARIO_SHOCK = {"par_bp": {"USD": 35.0, "EUR": 28.0, "GBP": 30.0, "JPY": 8.0, "MXN": 70.0},
                  "fx_pct": {"EUR": -0.035, "GBP": -0.040, "JPY": 0.030, "MXN": -0.070},
                  "ir_vol_mult": 1.40, "fx_vol_mult": 1.60, "xccy_basis_bp": -5.0}
SCAN_WORKERS = 6                      # processes for the history scan (1 = serial)
SCAN_EXPLAIN_TOP = 10                 # flagged days that also get a rates/FX/vol split
FILES = {"scenario": "attribution_scenario_day.csv", "methods": "attribution_methods_comparison.csv",
         "subsets": "attribution_subset_ims.csv", "netting": "attribution_netting.csv", "big": "big_moves.csv",
         "scan": "daily_im_scan.csv"}
DRIVER_COL, LEVEL_COL = "driver", "level"
SHAPLEY, SHAPLEY_SHARE, BRIDGE, BRIDGE_SHARE, OAAT = "shapley", "shapley_share", "bridge", "bridge_share", "one_at_a_time"
# ===== END CONFIG =====


# ---------- market construction ----------
def stress_market(m0: MarketBatch, shock: dict = None) -> MarketBatch:
    """Day-1 market = day-0 market plus a designed stress-like shock (single-state batch in, single-state batch out)."""
    shock = shock or SCENARIO_SHOCK
    m = m0.copy_with()
    m = m.rebootstrapped({c: m0.par[c] + bp * 1e-4 for c, bp in shock["par_bp"].items()})
    for c, pct in shock["fx_pct"].items():
        m.fx_spot[c] = m0.fx_spot[c] * (1.0 + pct)
    for c in m.ir_nvol:
        m.ir_nvol[c] = m0.ir_nvol[c] * shock["ir_vol_mult"]
    for p in m.fx_vol:
        m.fx_vol[p] = m0.fx_vol[p] * shock["fx_vol_mult"]
    m.xccy_basis = {k: v + shock.get("xccy_basis_bp", 0.0) * 1e-4 for k, v in m0.xccy_basis.items()}
    return m


def sub_markets(m0: MarketBatch, m1: MarketBatch):
    """Cumulative markets for the bridge: rates (curves and basis), then FX, then vols; the last equals m1."""
    r = m0.copy_with()
    for c in r.par:
        r.par[c], r.zero[c] = m1.par[c], m1.zero[c]
    r.xccy_basis = dict(m1.xccy_basis)
    f = r.copy_with()
    f.fx_spot = dict(m1.fx_spot)
    v = f.copy_with()
    v.ir_nvol, v.fx_vol, v.cds = dict(m1.ir_nvol), dict(m1.fx_vol), m1.cds
    return r, f, v


# ---------- engine ----------
class _Engine:
    """Caches CRIF rows per (market, trade) because a CRIF is the union of independent per-trade rows."""

    def __init__(self, p0, p1, m0, m1, th0, th1):
        for t0 in (p0, p1):
            if t0[C.TRADE_ID].duplicated().any():
                raise ValueError("duplicate trade IDs in a portfolio")
        both = p0.merge(p1, on=C.TRADE_ID, suffixes=("_0", "_1"))
        for col in p0.columns:
            if col == C.TRADE_ID:
                continue
            a, b = both[col + "_0"], both[col + "_1"]
            if not ((a == b) | (a.isna() & b.isna())).all():
                raise ValueError("a trade ID changes terms between the two days; book it as matured plus a new ID")
        ids0, ids1 = set(p0[C.TRADE_ID]), set(p1[C.TRADE_ID])
        self.matured, self.new, self.common = ids0 - ids1, ids1 - ids0, ids0 & ids1
        self.trades = pd.concat([p0, p1[p1[C.TRADE_ID].isin(self.new)]], ignore_index=True)
        self.m0, self.m1, self.th0, self.th1 = m0, m1, th0, th1
        self._crif = {}
        self._markets = {"m0": m0, "m1": m1}

    def market(self, key, market=None):
        if market is not None:
            self._markets[key] = market
        return self._markets[key]

    def crif(self, key) -> pd.DataFrame:
        if key not in self._crif:
            self._crif[key] = build_crif(self.trades, self._markets[key])
        return self._crif[key]

    def ids(self, matured_applied: bool, new_applied: bool) -> set:
        s = set(self.common)
        if not matured_applied:
            s |= self.matured
        if new_applied:
            s |= self.new
        return s

    def im(self, ids, market_key, params) -> float:
        c = self.crif(market_key)
        return simm(c[c[C.CRIF_TRADE_ID].isin(ids)], params).total if len(ids) else 0.0

    def state_im(self, on: frozenset) -> float:
        return self.im(self.ids(DRV_MATURED in on, DRV_NEW in on), "m1" if DRV_MARKET in on else "m0",
                       self.th1 if DRV_PARAMS in on else self.th0)


def _subsets():
    for k in range(len(DRIVERS) + 1):
        for s in combinations(DRIVERS, k):
            yield frozenset(s)


def shapley_values(value: dict, names=DRIVERS) -> dict:
    """Exact Shapley value of each name from a dict {frozenset of names: value of the coalition}."""
    n = len(names)
    out = {}
    for i in names:
        rest = [x for x in names if x != i]
        phi = 0.0
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for s in combinations(rest, k):
                s = frozenset(s)
                phi += w * (value[s | {i}] - value[s])
        out[i] = phi
    return out


def _share(x, total):
    return x / total if total not in (0, 0.0) and not np.isnan(total) else np.nan


@dataclass
class AttributionResult:
    frame: pd.DataFrame
    subset_ims: dict
    im0: float
    im1: float
    bridge_steps: pd.DataFrame
    netting: dict


def _attribute(p0, m0, th0, p1, m1, th1) -> AttributionResult:
    eng = _Engine(p0, p1, m0, m1, th0, th1)
    v = {s: eng.state_im(s) for s in _subsets()}
    im0, im1 = v[frozenset()], v[frozenset(DRIVERS)]
    total = im1 - im0
    shap = shapley_values(v)
    oaat = {d: v[frozenset([d])] - im0 for d in DRIVERS}
    # sequential bridge: portfolio, rates, FX, vol, parameters
    r, f, vv = sub_markets(m0, m1)
    eng.market("rates", r), eng.market("fx", f), eng.market("vol", vv)
    steps = [("start", eng.ids(False, False), "m0", th0),
             (DRV_MATURED, eng.ids(True, False), "m0", th0),
             (DRV_NEW, eng.ids(True, True), "m0", th0),
             (SUB_RATES, eng.ids(True, True), "rates", th0),
             (SUB_FX, eng.ids(True, True), "fx", th0),
             (SUB_VOL, eng.ids(True, True), "vol", th0),
             (DRV_PARAMS, eng.ids(True, True), "vol", th1)]
    levels = [eng.im(ids, mk, th) for _, ids, mk, th in steps]
    bridge_steps = pd.DataFrame({"step": [s[0] for s in steps], "im_after": levels})
    bridge_steps["delta"] = bridge_steps["im_after"].diff()
    br = dict(zip(bridge_steps["step"][1:], bridge_steps["delta"][1:]))
    br[DRV_MARKET] = br[SUB_RATES] + br[SUB_FX] + br[SUB_VOL]
    rows = []
    for d in DRIVERS:
        rows.append({DRIVER_COL: d, LEVEL_COL: "driver", SHAPLEY: shap[d], SHAPLEY_SHARE: _share(shap[d], total),
                     BRIDGE: br[d], BRIDGE_SHARE: _share(br[d], total), OAAT: oaat[d]})
    for s in (SUB_RATES, SUB_FX, SUB_VOL):
        rows.append({DRIVER_COL: s, LEVEL_COL: "substep", SHAPLEY: np.nan, SHAPLEY_SHARE: np.nan, BRIDGE: br[s],
                     BRIDGE_SHARE: _share(br[s], total), OAAT: np.nan})
    sums = {SHAPLEY: sum(shap.values()), BRIDGE: sum(br[d] for d in DRIVERS), OAAT: sum(oaat.values())}
    rows.append({DRIVER_COL: "sum_of_drivers", LEVEL_COL: "sum", SHAPLEY: sums[SHAPLEY], BRIDGE: sums[BRIDGE], OAAT: sums[OAAT]})
    rows.append({DRIVER_COL: "total_dIM", LEVEL_COL: "total", SHAPLEY: total, BRIDGE: total, OAAT: total})
    rows.append({DRIVER_COL: "residual", LEVEL_COL: "residual", SHAPLEY: total - sums[SHAPLEY], BRIDGE: total - sums[BRIDGE],
                 OAAT: total - sums[OAAT]})
    frame = pd.DataFrame(rows)
    # netting effect of the new trades on the final book
    final = eng.ids(True, True)
    standalone = eng.im(eng.new, "m1", th1)
    incremental = eng.im(final, "m1", th1) - eng.im(final - eng.new, "m1", th1)
    netting = {"new_trades": len(eng.new), "incremental_im": incremental, "standalone_im": standalone,
               "netting_effect": incremental - standalone}
    frame.attrs.update(netting=netting, im0=im0, im1=im1)
    return AttributionResult(frame, v, im0, im1, bridge_steps, netting)


def attribute(p0, m0, th0, p1, m1, th1) -> pd.DataFrame:
    """Attribution frame (columns driver, level, shapley, shapley_share, bridge, bridge_share, one_at_a_time). p = trade
    table, m = single-state MarketBatch, th = SimmParams on day 0 and day 1; frame.attrs holds netting, im0 and im1."""
    return _attribute(p0, m0, th0, p1, m1, th1).frame


def subset_table(res: AttributionResult) -> pd.DataFrame:
    rows = [{**{d: int(d in s) for d in DRIVERS}, "im": val} for s, val in res.subset_ims.items()]
    return pd.DataFrame(rows).sort_values(list(DRIVERS)).reset_index(drop=True)


def methods_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side Shapley, bridge and one-at-a-time with the differences against Shapley."""
    d = frame[[DRIVER_COL, LEVEL_COL, SHAPLEY, BRIDGE, OAAT]].copy()
    d["bridge_minus_shapley"], d["one_at_a_time_minus_shapley"] = d[BRIDGE] - d[SHAPLEY], d[OAAT] - d[SHAPLEY]
    return d


# ---------- designed scenario day ----------
def scenario_inputs(history: MarketHistory, trades: pd.DataFrame = None):
    """(p0, m0, th0, p1, m1, th1) for config.SCENARIO_DAY: prior-business-day history, shocked to day-1 market,
    two 3-month trades matured, one large new trade, parameter set 2506 -> 2512."""
    from .instruments import sample_portfolio
    p0 = trades.copy() if trades is not None else sample_portfolio()
    i = int(history.index_of([config.SCENARIO_DAY])[0])
    m0 = history.market.row(i - 1)
    new = p0[p0[C.TRADE_ID] == NEW_TRADE["template"]].copy()
    new[C.TRADE_ID], new[C.NOTIONAL], new[C.DIRECTION] = NEW_TRADE["id"], NEW_TRADE["notional"], NEW_TRADE["direction"]
    p1 = pd.concat([p0[~p0[C.TRADE_ID].isin(MATURED_ON_SCENARIO_DAY)], new], ignore_index=True)
    return (p0, m0, params_mod.load(config.SIMM_VERSION_PRIOR), p1, stress_market(m0),
            params_mod.load(config.SIMM_VERSION))


def run_scenario_day(history: MarketHistory = None, trades: pd.DataFrame = None, out_dir=None) -> AttributionResult:
    """Attribute the designed scenario day and write the scenario, methods, subset and netting csv files."""
    from .market_history import generate_history
    history = history if history is not None else generate_history()
    out = Path(out_dir) if out_dir is not None else config.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    res = _attribute(*scenario_inputs(history, trades))
    f = res.frame.copy()
    f.insert(0, "date", config.SCENARIO_DAY.date().isoformat())
    f["im0"], f["im1"], f["data_source"] = res.im0, res.im1, config.DATA_SOURCE
    f.to_csv(out / FILES["scenario"], index=False)
    methods_comparison(res.frame).assign(data_source=config.DATA_SOURCE).to_csv(out / FILES["methods"], index=False)
    subset_table(res).assign(data_source=config.DATA_SOURCE).to_csv(out / FILES["subsets"], index=False)
    pd.DataFrame([res.netting]).assign(data_source=config.DATA_SOURCE).to_csv(out / FILES["netting"], index=False)
    return res


# ---------- history scan ----------
def _im_chunk(args):
    batch, trades, version = args
    p = params_mod.load(version)
    return [simm(build_crif(trades, batch.row(i)), p).total for i in range(batch.n_states)]


def daily_im(history: MarketHistory, trades: pd.DataFrame, version: str = config.SIMM_VERSION, workers: int = SCAN_WORKERS) -> pd.Series:
    """SIMM-style IM of the frozen portfolio on every history date (process pool over contiguous chunks)."""
    n = len(history)
    k = max(1, min(workers, n))
    edges = np.linspace(0, n, k + 1).astype(int)
    jobs = [(history.market.take(np.arange(a, b)), trades, version) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if len(jobs) > 1:
        try:
            with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
                parts = list(ex.map(_im_chunk, jobs))
        except Exception:                       # e.g. no process spawning available; serial gives the same numbers
            parts = [_im_chunk(j) for j in jobs]
    else:
        parts = [_im_chunk(j) for j in jobs]
    return pd.Series(np.concatenate(parts), index=history.dates, name="im")


def scan_big_moves(history: MarketHistory, trades: pd.DataFrame, version: str = config.SIMM_VERSION, workers: int = SCAN_WORKERS,
                   explain_top: int = SCAN_EXPLAIN_TOP, series: pd.Series = None):
    """(flagged days, all days). A day is flagged when |dIM|/IM(prev) > BIG_MOVE_REL or |dIM| > BIG_MOVE_ABS for the frozen
    portfolio; the largest days also get the rates, FX and vol split of the market-only move."""
    im = series if series is not None else daily_im(history, trades, version, workers)
    d = pd.DataFrame({"date": history.dates, "im": im.values, "regime": history.regime})
    d["im_prev"] = d["im"].shift(1)
    d["dim"] = d["im"] - d["im_prev"]
    d["dim_rel"] = d["dim"] / d["im_prev"]
    d["flag_rel"] = d["dim_rel"].abs() > config.BIG_MOVE_REL
    d["flag_abs"] = d["dim"].abs() > config.BIG_MOVE_ABS
    d["flagged"] = d["flag_rel"] | d["flag_abs"]
    d["d_rates"] = d["d_fx"] = d["d_vol"] = np.nan
    p = params_mod.load(version)
    top = d[d["flagged"]].reindex(d.loc[d["flagged"], "dim"].abs().sort_values(ascending=False).index).head(explain_top).index
    for i in top:                              # market-only sub-steps, same portfolio and parameters
        m0, m1 = history.market.row(i - 1), history.market.row(i)
        ms = [m0, *sub_markets(m0, m1)]
        ims = [simm(build_crif(trades, m), p).total for m in ms]
        d.loc[i, ["d_rates", "d_fx", "d_vol"]] = np.diff(ims)
    d["data_source"] = config.DATA_SOURCE
    return d[d["flagged"]].reset_index(drop=True), d


def run_scan(history: MarketHistory = None, trades: pd.DataFrame = None, out_dir=None, workers: int = SCAN_WORKERS) -> pd.DataFrame:
    """Write big_moves.csv (flagged days) and daily_im_scan.csv (all days); return the flagged days."""
    from .instruments import sample_portfolio
    from .market_history import generate_history
    history = history if history is not None else generate_history()
    trades = trades if trades is not None else sample_portfolio()
    out = Path(out_dir) if out_dir is not None else config.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    flagged, allday = scan_big_moves(history, trades, workers=workers)
    flagged.to_csv(out / FILES["big"], index=False)
    allday.to_csv(out / FILES["scan"], index=False)
    return flagged
