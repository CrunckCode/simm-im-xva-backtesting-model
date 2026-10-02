"""Backtesting statistics and the test battery for the IM and xVA VaR models. ALL DATA IS SYNTHETIC.

Statistics: Kupiec proportion-of-failures (POF) LR, Christoffersen independence and conditional-coverage LR (Markov
transition counts), Basel (BCBS 22) traffic light, rolling exception counts, regime split and SIMM shortfall multiplier.
Formulas follow Kupiec (1995) and Christoffersen (1998) as restated in Federal Reserve publications (see
docs/sources/backtest_formula_verification.md); the originals were not retrievable.
Lights: PASS if p >= 0.05 and Green; AMBER if 0.01 <= p < 0.05 or Yellow; RED if p < 0.01 or Red (worst governs).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2

from . import columns as C
from . import config
from . import cva as xva
from . import var_model as vm
from .market_history import generate_history
from .instruments import sample_portfolio

# ===== CONFIG (user inputs) =====
P_EXC = 1.0 - config.CONF
P_PASS, P_AMBER = 0.05, 0.01
YELLOW_CUM, RED_CUM = 0.95, 0.9999                  # BCBS 22 cumulative-probability rule behind Table 2
TABLE_N = 250
BCBS_TABLE_FILE = config.PARAM_DIR / "bcbs22_traffic_light.csv"
FALLBACK_TABLE = [(0, "Green", 0.0), (1, "Green", 0.0), (2, "Green", 0.0), (3, "Green", 0.0), (4, "Green", 0.0),
                  (5, "Yellow", 0.40), (6, "Yellow", 0.50), (7, "Yellow", 0.65), (8, "Yellow", 0.75),
                  (9, "Yellow", 0.85), (10, "Red", 1.00)]
NONOVERLAP_STEP = config.HORIZON_DAYS
LIGHT_PASS, LIGHT_AMBER, LIGHT_RED, LIGHT_INFO = "PASS", "AMBER", "RED", "Info"
ZONE_GREEN, ZONE_YELLOW, ZONE_RED = "Green", "Yellow", "Red"
SPLIT_COL, N_COL, EXC_COL, NOTE_COL, SOURCE_COL = "split", "n", "exceptions", "note", "data_source"
RESULT_COLUMNS = [C.TEST, C.MODEL, SPLIT_COL, "value", "threshold", C.LIGHT, N_COL, EXC_COL, NOTE_COL, SOURCE_COL]
OVERLAP_NOTE = ("overlapping 10-day windows share days so exceptions cluster and the tests are over-sized; "
                "BCBS 22 section II warns against comparing multi-day measures with overlapping outcomes")
P_THRESHOLD_TEXT = "PASS p>=0.05; AMBER 0.01<=p<0.05; RED p<0.01"
RESULTS_FILE, SERIES_FILE, REGIME_FILE = "backtest_results.csv", "backtest_series.csv", "backtest_by_regime.csv"
# ===== END CONFIG =====

_ORDER = {LIGHT_PASS: 0, LIGHT_AMBER: 1, LIGHT_RED: 2}


# ---------- basic statistics ----------
def exceptions(loss, var):
    """1 where loss > var (strictly), else 0; NaN inputs give 0. Returns a Series when `loss` is one."""
    l, v = np.asarray(loss, dtype=float), np.asarray(var, dtype=float)
    out = (l > v).astype(int)
    return pd.Series(out, index=loss.index) if isinstance(loss, pd.Series) else out


def kupiec_pof(x: int, n: int, p: float = P_EXC) -> dict:
    """Kupiec POF: LR = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x], 0 ln 0 = 0, chi-square(1) p-value."""
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = max(-2.0 * ll0 + 2.0 * ll1, 0.0)
    return dict(LR=float(lr), p_value=float(chi2.sf(lr, 1)), x=int(x), n=int(n), p=p)


def christoffersen(exc, p: float = P_EXC) -> dict:
    """Independence LR (first-order Markov alternative) and conditional coverage LR_cc = LR_uc + LR_ind (chi-square(2))."""
    e = np.asarray(exc, dtype=int)
    a, b = e[:-1], e[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_a = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    ll_0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    lr_ind = max(2.0 * (ll_a - ll_0), 0.0)
    uc = kupiec_pof(int(e.sum()), len(e), p)
    lr_cc = uc["LR"] + lr_ind
    return dict(n00=n00, n01=n01, n10=n10, n11=n11, pi01=pi01, pi11=pi11, LR_ind=float(lr_ind),
                p_ind=float(chi2.sf(lr_ind, 1)), LR_uc=uc["LR"], LR_cc=float(lr_cc), p_cc=float(chi2.sf(lr_cc, 2)))


def bcbs22_table():
    """(DataFrame[exceptions, zone, plus_factor], source text). Reads the parameter file if present, else the known table."""
    if Path(BCBS_TABLE_FILE).exists():
        raw = pd.read_csv(BCBS_TABLE_FILE)
        ex = raw["exceptions"].astype(int)
        df = pd.DataFrame({"exceptions": ex, "zone": raw["zone"].str.capitalize(), "plus_factor": raw["plus_factor"].astype(float)})
        return df, f"{Path(BCBS_TABLE_FILE).name} (BCBS 22 Table 2)"
    df = pd.DataFrame(FALLBACK_TABLE, columns=["exceptions", "zone", "plus_factor"])
    return df, "built-in BCBS 22 Table 2 fallback (parameter file missing)"


def basel_zone(x: int, n: int, p: float = P_EXC) -> str:
    """Generic binomial rule: Yellow once P(X <= x) >= 95%, Red once it is >= 99.99%."""
    cum = binom.cdf(x, n, p)
    return ZONE_RED if cum >= RED_CUM else (ZONE_YELLOW if cum >= YELLOW_CUM else ZONE_GREEN)


def basel_table_zone(x: int):
    """(zone, plus factor, source) for 250 observations at 99% from BCBS 22 Table 2 (10 or more is Red, factor 1.00)."""
    df, src = bcbs22_table()
    row = df[df["exceptions"] == min(int(x), int(df["exceptions"].max()))].iloc[0]
    return row["zone"], float(row["plus_factor"]), src


def rolling(series, window: int = config.BACKTEST_WINDOW) -> pd.Series:
    """Rolling sum (exception counts over the trailing window)."""
    s = series if isinstance(series, pd.Series) else pd.Series(np.asarray(series))
    return s.rolling(window).sum()


def by_regime(exc, regime) -> pd.DataFrame:
    """Exceptions, observations, rate and Kupiec p-value per regime label (0 calm, 1 stress)."""
    e, r = np.asarray(exc, dtype=int), np.asarray(regime, dtype=int)
    rows = []
    for lab, name in ((C.REGIME_CALM, "calm"), (C.REGIME_STRESS, "stress")):
        n, x = int((r == lab).sum()), int(e[r == lab].sum())
        rows.append({C.REGIME: name, N_COL: n, EXC_COL: x, "rate": x / n if n else np.nan,
                     "kupiec_p": kupiec_pof(x, n)["p_value"] if n else np.nan})
    return pd.DataFrame(rows)


def shortfall_multiplier(loss, im, p: float = P_EXC) -> float:
    """Smallest k such that k*IM leaves the exception count in the Green zone (generic binomial rule)."""
    l, m = np.asarray(loss, dtype=float), np.asarray(im, dtype=float)
    ok = ~(np.isnan(l) | np.isnan(m))
    l, m = l[ok], m[ok]
    n, xmax = len(l), -1
    while basel_zone(xmax + 1, n, p) == ZONE_GREEN:
        xmax += 1
    ratio = np.where(m > 0, l / np.where(m > 0, m, 1.0), np.where(l > 0, np.inf, -np.inf))
    ratio = np.sort(ratio)[::-1]
    return float(max(ratio[xmax], 0.0)) if 0 <= xmax < len(ratio) else 0.0


# ---------- lights ----------
def light_p(pv: float) -> str:
    return LIGHT_PASS if pv >= P_PASS else (LIGHT_AMBER if pv >= P_AMBER else LIGHT_RED)


def light_zone(zone: str) -> str:
    return {ZONE_GREEN: LIGHT_PASS, ZONE_YELLOW: LIGHT_AMBER, ZONE_RED: LIGHT_RED}[zone]


def worst(*lights) -> str:
    return max(lights, key=lambda l: _ORDER[l])


def _row(test, model, split, value, threshold, light, n, x, note=""):
    return {C.TEST: test, C.MODEL: model, SPLIT_COL: split, "value": value, "threshold": threshold, C.LIGHT: light,
            N_COL: n, EXC_COL: x, NOTE_COL: note, SOURCE_COL: config.DATA_SOURCE}


def battery_rows(exc, model: str, split: str, note: str = "") -> list:
    """Battery rows for one exception series: Kupiec, Christoffersen (ind, cc), Basel zone and the overall light."""
    e = np.asarray(exc, dtype=int)
    n, x = len(e), int(e.sum())
    ku, ch = kupiec_pof(x, n), christoffersen(e)
    zone = basel_zone(x, n)
    l_ku, l_ind, l_zone = light_p(ku["p_value"]), light_p(ch["p_ind"]), light_zone(zone)
    return [
        _row("kupiec_pof", model, split, ku["p_value"], P_THRESHOLD_TEXT, l_ku, n, x, note),
        _row("christoffersen_ind", model, split, ch["p_ind"], P_THRESHOLD_TEXT, l_ind, n, x, note),
        _row("christoffersen_cc", model, split, ch["p_cc"], P_THRESHOLD_TEXT, light_p(ch["p_cc"]), n, x, note),
        _row("basel_zone", model, split, x, f"{zone}: Yellow once cum. binomial prob >= {YELLOW_CUM:.0%}, Red >= {RED_CUM:.2%}",
             l_zone, n, x, note),
        _row("overall", model, split, x, "worst of Kupiec, Christoffersen independence, Basel zone",
             worst(l_ku, l_ind, l_zone), n, x, note),
    ]


# ---------- battery ----------
def _align(loss: pd.Series, var: pd.Series):
    idx = var.index.intersection(loss.dropna().index)
    return loss.loc[idx], var.loc[idx]


def _designs(loss: pd.Series, var: pd.Series, horizon: int):
    """Yield (split, loss, var, note): 1-day, or 10-day on non-overlapping and overlapping windows."""
    l, v = _align(loss, var)
    if horizon == 1:
        yield "1d", l, v, ""
    else:
        yield "10d_nonoverlap", l.iloc[::NONOVERLAP_STEP], v.iloc[::NONOVERLAP_STEP], ""
        yield "10d_overlapping", l, v, OVERLAP_NOTE


def run_battery(history=None, trades=None, out_dir=None, simm_series: pd.Series = None, quick: bool = False,
                with_xva: bool = True) -> pd.DataFrame:
    """Run every backtest and write backtest_results.csv, backtest_series.csv and backtest_by_regime.csv to out_dir."""
    history = history if history is not None else generate_history(quick=quick)
    trades = trades if trades is not None else sample_portfolio()
    out = Path(out_dir) if out_dir is not None else config.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    regime = pd.Series(history.regime, index=history.dates)
    rows, long = [], []

    def add(model, loss, var, horizon):
        for split, l, v, note in _designs(loss, var, horizon):
            e = exceptions(l, v)
            rows.extend(battery_rows(e.values, model, split, note))
            long.append(pd.DataFrame({C.DATE: l.index, C.MODEL: model, SPLIT_COL: split, C.LOSS: l.values,
                                      C.VAR_FORECAST: v.values, C.EXCEPTION: e.values,
                                      C.REGIME: regime.reindex(l.index).values}))
        return exceptions(*_align(loss, var))

    loss = {h: vm.realized_loss_series(history, trades, h) for h in (1, 10)}
    models = {}
    for m in vm.MODELS:
        for h in (1, 10):
            var = vm.im_series(history, trades, h, model=m)
            models[(m, h)] = var
            exc = add(f"hs_im_{m}", loss[h], var, h)
            if h == 1 and m == vm.MODEL_CHAMPION:
                reg = by_regime(exc.values, regime.reindex(exc.index).values)
                reg.insert(0, C.MODEL, f"hs_im_{m}")
                reg.to_csv(out / REGIME_FILE, index=False)
    # BCBS 22 table check on the last 250 one-day observations, plus rolling 250-day maximum
    for m in vm.MODELS:
        l, v = _align(loss[1], models[(m, 1)])
        e = exceptions(l, v)
        last = int(e.iloc[-TABLE_N:].sum())
        zone, plus, src = basel_table_zone(last)
        rows.append(_row("basel_table_last250", f"hs_im_{m}", "1d", last,
                         f"{zone}, plus factor {plus:.2f} ({src})", light_zone(zone), TABLE_N, last))
        mx = int(rolling(e, TABLE_N).max())
        zone, plus, src = basel_table_zone(mx)
        rows.append(_row("rolling250_max", f"hs_im_{m}", "1d", mx, f"{zone}, plus factor {plus:.2f} ({src})",
                         light_zone(zone), TABLE_N, mx))
    if simm_series is not None:
        for split, l, v, note in _designs(loss[10], simm_series, 10):
            e = exceptions(l, v)
            rows.extend(battery_rows(e.values, "simm_style", split, note))
            long.append(pd.DataFrame({C.DATE: l.index, C.MODEL: "simm_style", SPLIT_COL: split, C.LOSS: l.values,
                                      C.VAR_FORECAST: v.values, C.EXCEPTION: e.values, C.REGIME: regime.reindex(l.index).values}))
        l, v = _align(loss[10], simm_series)
        rows.append(_row("shortfall_multiplier", "simm_style", "10d_overlapping", shortfall_multiplier(l, v),
                         "smallest k with k*IM in the Green zone", LIGHT_INFO, len(l), int(exceptions(l, v).sum())))
    l, v = _align(loss[10], models[(vm.MODEL_CHAMPION, 10)])
    rows.append(_row("shortfall_multiplier", f"hs_im_{vm.MODEL_CHAMPION}", "10d_overlapping", shortfall_multiplier(l, v),
                     "smallest k with k*IM in the Green zone", LIGHT_INFO, len(l), int(exceptions(l, v).sum())))
    if with_xva:
        for h in xva.XVA_HORIZONS:
            add("xva_var", xva.realized_cva_change(history, trades, h), xva.xva_var_series(history, trades, h), h)
    res = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    res.to_csv(out / RESULTS_FILE, index=False)
    ser = pd.concat(long, ignore_index=True)
    ser[SOURCE_COL] = config.DATA_SOURCE
    ser.to_csv(out / SERIES_FILE, index=False)
    return res
