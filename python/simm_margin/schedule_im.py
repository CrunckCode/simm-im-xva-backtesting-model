"""Standardised (schedule) initial margin: gross = sum of rate x notional, net = 0.4 G + 0.6 NGR G. SYNTHETIC trades.

Rates come from data/parameters/schedule_im.csv (BCBS-IOSCO Appendix A and 12 CFR Part 45 Appendix A). Assumptions:
bucket edges are lower-inclusive (a 2 year trade is in the 2-5 bucket); a swaption's duration is expiry plus underlying
tenor (the MATURITY column of the trade table); notionals are converted to USD with fx_spot (default: the synthetic
history base levels); FX forwards and options use the FX rate on their notional in the first currency.
"""
import numpy as np
import pandas as pd

from . import columns as C
from . import parameter_io as pio
from .market_history import FX_BASE, FX_PAIRS

# ===== CONFIG (user inputs) =====
SOURCE_US, SOURCE_BCBS = "12 CFR Part 45", "BCBS-IOSCO"
DEFAULT_SOURCE = SOURCE_US                    # has explicit cross-currency swap rows
ASSET_IR, ASSET_FX, ASSET_XCCY = "Interest rate", "Foreign exchange", "Cross-currency swap"
ASSET_OF_PRODUCT = {C.PRODUCT_IRS: ASSET_IR, C.PRODUCT_SWAPTION: ASSET_IR, C.PRODUCT_FXFWD: ASSET_FX,
                    C.PRODUCT_FXOPT: ASSET_FX, C.PRODUCT_XCCY: ASSET_XCCY}
OPEN_BUCKET = "inf"
CONST_GROSS, CONST_NGR, CONST_ZERO_RC = "schedule_gross_weight", "schedule_ngr_weight", "schedule_ngr_when_gross_rc_zero"
RATE_COL, ASSET_COL, DURATION_COL, NOTIONAL_USD_COL, GROSS_IM_COL = "rate", "asset_class", "duration_years", "notional_usd", "gross_im"
# ===== END CONFIG =====


def default_fx_spot() -> dict:
    """USD per unit of each currency from the synthetic base levels (market quote converted)."""
    spot = {"USD": 1.0}
    for pair, (b, q) in FX_PAIRS.items():
        if q == "USD":
            spot[b] = FX_BASE[pair]
        elif b == "USD":
            spot[q] = 1.0 / FX_BASE[pair]
    return spot


def _rates(source: str, param_dir=None) -> dict:
    """{asset class: [(low, high, rate fraction)]} for one source document."""
    out = {}
    for r in pio.read_csv(pio.SCHEDULE_FILE, param_dir):
        if source not in r["source_doc"]:
            continue
        lo = float(r["maturity_bucket_low_years"]) if r["maturity_bucket_low_years"] != "" else None
        hi = None if r["maturity_bucket_high_years"] == "" else (np.inf if r["maturity_bucket_high_years"] == OPEN_BUCKET
                                                                 else float(r["maturity_bucket_high_years"]))
        out.setdefault(r["asset_class"], []).append((lo, hi, float(r["rate_pct"]) / 100.0))
    return out


def _constant(name: str, param_dir=None) -> float:
    for r in pio.read_csv(pio.REG_CONSTANTS_FILE, param_dir):
        if r["constant"] == name and r["jurisdiction"] in (SOURCE_BCBS, "US"):
            return float(r["value"])
    raise KeyError(name)


def _lookup(table: dict, asset: str, years: float) -> float:
    rows = table[asset]
    if len(rows) == 1 and rows[0][0] is None:
        return rows[0][2]                               # flat rate, not duration based
    for lo, hi, rate in rows:
        if lo <= years < hi:
            return rate
    raise ValueError(f"no schedule bucket for {asset} at {years} years")


def ngr(mtm) -> float:
    """Net-to-gross ratio = max(sum MTM, 0) / sum max(MTM, 0); equals 1 when gross replacement cost is 0."""
    m = np.asarray(mtm, dtype=float)
    gross_rc = float(np.maximum(m, 0.0).sum())
    return 1.0 if gross_rc <= 0.0 else float(max(m.sum(), 0.0) / gross_rc)


def schedule_im(trades: pd.DataFrame, mtm, fx_spot: dict = None, source: str = DEFAULT_SOURCE, param_dir=None) -> dict:
    """dict(gross, ngr, net, by_row): by_row is a DataFrame with one row per trade (asset class, duration, rate, USD
    notional, gross IM). mtm is the per-trade value in the same order as trades (netting set level NGR)."""
    mtm = np.asarray(mtm, dtype=float)
    if len(mtm) != len(trades):
        raise ValueError("mtm must have one value per trade")
    spot = fx_spot or default_fx_spot()
    table = _rates(source, param_dir)
    w_gross, w_ngr = _constant(CONST_GROSS, param_dir), _constant(CONST_NGR, param_dir)
    rows = []
    for _, t in trades.iterrows():
        asset = ASSET_OF_PRODUCT[t[C.PRODUCT]]
        if asset not in table:                          # BCBS-IOSCO has no cross-currency rows: use the interest rate rows
            asset = ASSET_IR
        years = float(t[C.MATURITY])
        usd = abs(float(t[C.NOTIONAL])) * spot[t[C.CCY]]
        rows.append({C.TRADE_ID: t[C.TRADE_ID], ASSET_COL: asset, DURATION_COL: years, NOTIONAL_USD_COL: usd,
                     RATE_COL: _lookup(table, asset, years)})
    by_row = pd.DataFrame(rows, columns=[C.TRADE_ID, ASSET_COL, DURATION_COL, NOTIONAL_USD_COL, RATE_COL])
    by_row[GROSS_IM_COL] = by_row[RATE_COL] * by_row[NOTIONAL_USD_COL]
    gross = float(by_row[GROSS_IM_COL].sum())
    r = ngr(mtm)
    return {"gross": gross, "ngr": r, "net": w_gross * gross + w_ngr * r * gross, "by_row": by_row}
