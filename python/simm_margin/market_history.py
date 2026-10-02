"""Synthetic market history and the MarketBatch container. ALL DATA IS SYNTHETIC (no market data).

Model: 2-state Markov regime (calm/stress), multivariate Student-t shocks (df 5), 3 PCA-style factors
(level, slope, curvature) per currency with cross-currency correlation, FX log-returns, log-AR(1) normal IR
vols and lognormal FX vols tied to the regime, CDS log-spreads with jumps on entry to stress, and one forced
stress window. Units: par/zero/vol/CDS are decimals (0.01 = 1%); IR normal vols are absolute rate vols
(0.0095 = 95bp per annum); fx_spot[ccy] is USD per 1 unit of ccy.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import columns as C
from . import config
from .curves import CURRENCIES, TENOR_YEARS, base_par, bootstrap_curve, Curve

# ===== CONFIG (user inputs) =====
FX_PAIRS = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "USDJPY": ("USD", "JPY"), "USDMXN": ("USD", "MXN")}
FX_BASE = {"EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 148.0, "USDMXN": 18.0}          # market quote
FX_DAILY_VOL = {"EURUSD": 0.0055, "GBPUSD": 0.0060, "USDJPY": 0.0060, "USDMXN": 0.0080}
FX_STRESS_DRIFT = {"EURUSD": -0.05, "GBPUSD": -0.05, "USDJPY": -0.05, "USDMXN": 0.08}    # in units of daily vol
FX_USD_LOAD = {"EURUSD": -0.6, "GBPUSD": -0.5, "USDJPY": 0.3, "USDMXN": 0.5}         # loading on common USD factor
FX_KAPPA = 0.004                                                                     # mean reversion of log spot
IR_VOL_EXPIRIES = (1.0, 2.0, 5.0, 10.0)
IR_VOL_TENORS = (5.0, 10.0)
IR_VOL_BASE_BP = {"USD": 100, "EUR": 75, "GBP": 85, "JPY": 45, "MXN": 120}
IR_VOL_EXPIRY_SHAPE = (1.06, 1.02, 0.97, 0.90)
IR_VOL_TENOR_SHAPE = (1.0, 0.97)
IR_VOL_PHI, IR_VOL_ETA, IR_VOL_STRESS_SHIFT = 0.985, 0.03, np.log(1.5)
FX_VOL_EXPIRIES = (1 / 12, 0.25, 0.5, 1.0)
FX_VOL_BASE = {"EURUSD": 0.075, "GBPUSD": 0.085, "USDJPY": 0.095, "USDMXN": 0.140}
FX_VOL_EXPIRY_SHAPE = (0.95, 1.0, 1.03, 1.05)
FX_VOL_PHI, FX_VOL_ETA, FX_VOL_STRESS_SHIFT = 0.98, 0.04, np.log(1.7)
CDS_TENORS = (1.0, 3.0, 5.0, 7.0, 10.0)
CDS_BASE = (0.0040, 0.0060, 0.0080, 0.0095, 0.0110)
CDS_STRESS_LOAD = (1.2, 1.1, 1.0, 0.9, 0.8)
CDS_PHI, CDS_ETA, CDS_STRESS_SHIFT, CDS_JUMP_MEAN, CDS_JUMP_SD = 0.97, 0.03, np.log(2.5), 0.5, 0.1
XCCY_BASIS_BASE, XCCY_BASIS_STRESS = -0.0015, -0.0020
# IR factor model: daily vol in bp for level, slope, curvature; mean reversion; loading on the global factor
IR_FACTOR_VOL_BP = {"USD": (4.0, 2.5, 1.2), "EUR": (3.5, 2.2, 1.0), "GBP": (4.0, 2.5, 1.2),
                    "JPY": (1.2, 0.8, 0.4), "MXN": (6.0, 3.5, 1.8)}
IR_FACTOR_GLOBAL_LOAD = {"USD": (0.8, 0.6, 0.3), "EUR": (0.7, 0.5, 0.3), "GBP": (0.7, 0.5, 0.3),
                         "JPY": (0.35, 0.25, 0.2), "MXN": (0.45, 0.3, 0.2)}
IR_FACTOR_KAPPA = 0.005
IR_STRESS_LEVEL_DRIFT = {"USD": -0.08, "EUR": -0.08, "GBP": -0.06, "JPY": -0.03, "MXN": 0.1}   # in units of daily level vol
FX_COMMON_RATES_CORR = 0.3
PAR_FLOOR = -0.004
STRESS_WINDOW_START = pd.Timestamp("2019-03-01")        # early so every later day can use it without look-ahead
# ===== END CONFIG =====

CCY_FX_PAIR = {"EUR": "EURUSD", "GBP": "GBPUSD", "JPY": "USDJPY", "MXN": "USDMXN"}
_N_FACTORS = 3


@dataclass
class HistoryConfig:
    start: pd.Timestamp = config.HIST_START
    end: pd.Timestamp = config.HIST_END
    quick_start: pd.Timestamp = config.HIST_QUICK_START
    seed: int = config.SEED
    persistence: tuple = config.REGIME_PERSISTENCE
    stress_mult: float = config.STRESS_VOL_MULT
    dof: int = config.T_DOF
    stress_days: int = config.STRESS_WINDOW_DAYS
    stress_start: pd.Timestamp = STRESS_WINDOW_START
    save_dir: object = None          # when set, the regime path csv is written there


@dataclass
class MarketBatch:
    """Market states with leading dimension n_states."""
    par: dict
    zero: dict
    fx_spot: dict
    ir_nvol: dict
    fx_vol: dict
    cds: np.ndarray
    xccy_basis: dict = field(default_factory=dict)      # {"EURUSD": [n]} market cross-currency basis spread
    dates: object = None

    @property
    def n_states(self) -> int:
        return self.cds.shape[0]

    def curve(self, ccy: str) -> Curve:
        return Curve(self.par[ccy], self.zero[ccy])

    def pair_spot(self, pair: str) -> np.ndarray:
        """Spot in market quote (quote ccy per base ccy) for a pair like EURUSD or USDJPY."""
        base, quote = FX_PAIRS[pair] if pair in FX_PAIRS else (pair[:3], pair[3:])
        return self.fx_spot[base] / self.fx_spot[quote]

    def take(self, idx) -> "MarketBatch":
        idx = np.atleast_1d(np.asarray(idx))
        sub = lambda d: {k: v[idx] for k, v in d.items()}
        dates = None if self.dates is None else self.dates[idx]
        return MarketBatch(sub(self.par), sub(self.zero), sub(self.fx_spot), sub(self.ir_nvol), sub(self.fx_vol),
                           self.cds[idx], sub(self.xccy_basis), dates)

    def row(self, i: int) -> "MarketBatch":
        return self.take([i])

    def tile(self, m: int) -> "MarketBatch":
        """Repeat every state m times (state-major), e.g. to stack bumped copies of one row."""
        rep = lambda d: {k: np.repeat(v, m, axis=0) for k, v in d.items()}
        return MarketBatch(rep(self.par), rep(self.zero), rep(self.fx_spot), rep(self.ir_nvol), rep(self.fx_vol),
                           np.repeat(self.cds, m, axis=0), rep(self.xccy_basis), None)

    def copy_with(self, **changes) -> "MarketBatch":
        """Shallow copy with dict fields copied so that item assignment on the copy is safe."""
        base = replace(self, **{k: dict(getattr(self, k)) for k in
                                ("par", "zero", "fx_spot", "ir_nvol", "fx_vol", "xccy_basis")})
        return replace(base, **changes)

    def rebootstrapped(self, par: dict) -> "MarketBatch":
        """Copy with new par quotes for the given currencies and zeros re-bootstrapped (full re-bootstrap)."""
        out = self.copy_with()
        for ccy, p in par.items():
            out.par[ccy] = np.asarray(p, dtype=float)
            out.zero[ccy] = bootstrap_curve(out.par[ccy]).zero
        return out


@dataclass
class MarketHistory:
    dates: pd.DatetimeIndex
    regime: np.ndarray                 # 0 calm, 1 stress, per date
    market: MarketBatch
    stress_slice: object               # (i0, i1) into this history, or None when outside (quick mode)
    stress_batch: MarketBatch          # the forced stress window states (always present)
    stress_dates: tuple
    cfg: HistoryConfig
    data_source: str = config.DATA_SOURCE

    par = property(lambda s: s.market.par)
    zero = property(lambda s: s.market.zero)
    fx_spot = property(lambda s: s.market.fx_spot)
    ir_nvol = property(lambda s: s.market.ir_nvol)
    fx_vol = property(lambda s: s.market.fx_vol)
    cds = property(lambda s: s.market.cds)

    def __len__(self):
        return len(self.dates)

    def index_of(self, dates) -> np.ndarray:
        idx = self.dates.get_indexer(pd.DatetimeIndex(np.atleast_1d(dates)))
        if (idx < 0).any():
            raise KeyError("date not in history (non-business day or out of range)")
        return idx

    def batch(self, dates=None) -> MarketBatch:
        """MarketBatch for the given dates (None = every date), leading dimension n_states."""
        return self.market if dates is None else self.market.take(self.index_of(dates))

    def regime_frame(self) -> pd.DataFrame:
        return pd.DataFrame({C.DATE: self.dates, C.REGIME: self.regime, "data_source": self.data_source})

    def save(self, out_dir) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "regime_path.csv"
        self.regime_frame().to_csv(path, index=False)
        return path


def _regime_path(rng, T, cfg, i0, i1):
    u = rng.random(T)
    reg = np.zeros(T, dtype=int)
    s = 0
    for t in range(T):
        if i0 <= t < i1:
            s = 1
        elif u[t] > cfg.persistence[s]:
            s = 1 - s
        reg[t] = s
    return reg


def _ar1(u, phi):
    return lfilter([1.0], [1.0, -phi], u, axis=0)


def _vol_grid(base, shape_a, shape_b, x_lev, x_tilt, tilt_a):
    """Log-AR vol surface [T, len(a), len(b)] = base * shape * exp(level + tilt_e * tilt)."""
    shape = np.outer(shape_a, shape_b)
    return base * shape[None] * np.exp(x_lev[:, None, None] + tilt_a[None, :, None] * x_tilt[:, None, None])


def generate_history(cfg: HistoryConfig = None, quick: bool = False) -> MarketHistory:
    """Simulate the synthetic history. Quick mode returns dates from cfg.quick_start only (all days are simulated
    on the full calendar first so the quick data is a strict subset of the full run)."""
    cfg = cfg or HistoryConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, cfg.end)
    T = len(dates)
    i0 = int(np.clip(dates.searchsorted(cfg.stress_start), 0, max(T - cfg.stress_days, 0)))
    i1 = min(i0 + cfg.stress_days, T)
    regime = _regime_path(rng, T, cfg, i0, i1)
    s = regime.astype(float)
    entry = np.r_[0.0, np.diff(s)] > 0
    mult = 1.0 + (cfg.stress_mult - 1.0) * s                       # shock scale by regime
    ccys, pairs = CURRENCIES, tuple(FX_PAIRS)

    # shocks: global factors + idiosyncratic, one multivariate-t mixing draw per day (unit variance)
    G = rng.standard_normal((T, _N_FACTORS))
    E = rng.standard_normal((T, len(ccys), _N_FACTORS))
    L = np.array([IR_FACTOR_GLOBAL_LOAD[c] for c in ccys])
    X = L[None] * G[:, None, :] + np.sqrt(1 - L ** 2)[None] * E
    U = FX_COMMON_RATES_CORR * G[:, 0] + np.sqrt(1 - FX_COMMON_RATES_CORR ** 2) * rng.standard_normal(T)
    cu = np.array([FX_USD_LOAD[p] for p in pairs])
    Z = cu[None] * U[:, None] + np.sqrt(1 - cu ** 2)[None] * rng.standard_normal((T, len(pairs)))
    mix = np.sqrt((cfg.dof - 2) / rng.chisquare(cfg.dof, T))[:, None]
    X, Z = X * mix[:, :, None], Z * mix

    # rates: par = base + loadings @ factors (bp), factors mean-reverting
    par, zero = {}, {}
    Tn = TENOR_YEARS
    loadings = np.stack([np.ones_like(Tn), np.exp(-Tn / 4.0) - 0.35, (Tn / 3.0) * np.exp(-Tn / 3.0) / np.exp(-1.0)], axis=1)
    for a, c in enumerate(ccys):
        vol = np.array(IR_FACTOR_VOL_BP[c])[None] * mult[:, None]
        drift = np.zeros((T, _N_FACTORS))
        drift[:, 0] = IR_STRESS_LEVEL_DRIFT[c] * IR_FACTOR_VOL_BP[c][0] * s
        y = _ar1(vol * X[:, a, :] + drift, 1 - IR_FACTOR_KAPPA)
        par[c] = np.maximum(base_par(c)[None] + 1e-4 * y @ loadings.T, PAR_FLOOR)
        zero[c] = bootstrap_curve(par[c]).zero

    # FX: market-quote log returns, mean reverting to base; fx_spot is USD per unit of ccy
    quote = {}
    for j, p in enumerate(pairs):
        r = FX_DAILY_VOL[p] * (mult * Z[:, j] + FX_STRESS_DRIFT[p] * s)
        quote[p] = FX_BASE[p] * np.exp(_ar1(r, 1 - FX_KAPPA))
    fx_spot = {"USD": np.ones(T), "EUR": quote["EURUSD"], "GBP": quote["GBPUSD"],
               "JPY": 1.0 / quote["USDJPY"], "MXN": 1.0 / quote["USDMXN"]}

    # IR normal vols [T, expiry, tenor], decimal
    ir_nvol = {}
    tilt = np.array([0.5, 0.2, -0.2, -0.5])
    for c in ccys:
        lev = _ar1((1 - IR_VOL_PHI) * IR_VOL_STRESS_SHIFT * s + IR_VOL_ETA * rng.standard_normal(T), IR_VOL_PHI)
        til = _ar1(0.015 * rng.standard_normal(T), 0.98)
        ir_nvol[c] = _vol_grid(IR_VOL_BASE_BP[c] * 1e-4, IR_VOL_EXPIRY_SHAPE, IR_VOL_TENOR_SHAPE, lev, til, tilt)

    # FX lognormal vols [T, expiry]
    fx_vol = {}
    for p in pairs:
        lev = _ar1((1 - FX_VOL_PHI) * FX_VOL_STRESS_SHIFT * s + FX_VOL_ETA * rng.standard_normal(T), FX_VOL_PHI)
        fx_vol[p] = FX_VOL_BASE[p] * np.array(FX_VOL_EXPIRY_SHAPE)[None] * np.exp(lev)[:, None]

    # CDS log-spreads with jumps on entry to stress and occasional jumps within it
    jump = entry * rng.normal(CDS_JUMP_MEAN, CDS_JUMP_SD, T) + (s > 0) * (rng.random(T) < 0.02) * rng.normal(0, 0.2, T)
    x = _ar1((1 - CDS_PHI) * CDS_STRESS_SHIFT * s + CDS_ETA * rng.standard_normal(T) + jump, CDS_PHI)
    cds = np.array(CDS_BASE)[None] * np.exp(x[:, None] * np.array(CDS_STRESS_LOAD)[None])

    # market cross-currency basis (EUR/USD), decimal
    xb = XCCY_BASIS_BASE + _ar1((1 - 0.98) * XCCY_BASIS_STRESS * s + 0.00005 * rng.standard_normal(T), 0.98)

    full = MarketBatch(par, zero, fx_spot, ir_nvol, fx_vol, cds, {"EURUSD": xb}, dates)
    stress_batch = full.take(np.arange(i0, i1))
    k0 = int(dates.searchsorted(cfg.quick_start)) if quick else 0
    keep = np.arange(k0, T)
    market = full.take(keep) if quick else full
    sl = (i0 - k0, i1 - k0) if i0 >= k0 else None
    hist = MarketHistory(dates[keep], regime[keep], market, sl, stress_batch, (dates[i0], dates[i1 - 1]), cfg)
    if cfg.save_dir is not None:
        hist.save(cfg.save_dir)
    return hist
