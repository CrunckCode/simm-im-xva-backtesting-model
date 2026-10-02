"""CRIF-style sensitivity table (RatesFX) for one market state. SYNTHETIC trades and market data.

Risk types: Risk_IRCurve (per 1bp par quote at the 12 vertices, Jacobian method), Risk_XCcyBasis (per 1bp basis),
Risk_IRVol (vega * sigma by expiry vertex), Risk_FX (per 1% relative move, USD), Risk_FXVol (vega * sigma by expiry).
"""
import numpy as np
import pandas as pd

from . import columns as C
from . import config
from .curves import par_to_zero_jacobian
from .instruments import BASIS_SPREAD
from .market_history import MarketBatch
from .simm import SIGMA_MARKET_COL
from .sensitivities import (fx_delta, fx_vega, fx_vega_ladder, ir_delta, ir_vega_ladder, trade_currencies, xccy_basis_delta)
from .instruments import pair_name

# ===== CONFIG (user inputs) =====
CRIF_COLUMNS = [C.CRIF_TRADE_ID, C.PORTFOLIO_ID, C.PRODUCT_CLASS, C.RISK_TYPE, C.QUALIFIER, C.BUCKET,
                C.LABEL1, C.LABEL2, C.AMOUNT, C.AMOUNT_CCY, C.AMOUNT_USD]
IR_BUCKET = {"USD": "regular_well_traded", "EUR": "regular_well_traded", "GBP": "regular_well_traded",
             "JPY": "regular_low_vol", "MXN": "high_vol"}      # informational volatility-group label
SUBCURVE = "OIS"
MIN_ABS_USD = 1.0e-6                                           # drop numerically empty rows
# ===== END CONFIG =====


def _rec(trade, risk_type, qualifier, bucket, label1, label2, amount, amount_ccy, usd):
    return {C.CRIF_TRADE_ID: trade[C.TRADE_ID], C.PORTFOLIO_ID: trade[C.PORTFOLIO_ID], C.PRODUCT_CLASS: C.RATES_FX,
            C.RISK_TYPE: risk_type, C.QUALIFIER: qualifier, C.BUCKET: bucket, C.LABEL1: label1, C.LABEL2: label2,
            C.AMOUNT: amount, C.AMOUNT_CCY: amount_ccy, C.AMOUNT_USD: usd}


def build_crif(trades: pd.DataFrame, batch_row: MarketBatch) -> pd.DataFrame:
    """CRIF rows for every trade at one market state (batch_row must have exactly one state)."""
    if batch_row.n_states != 1:
        raise ValueError("build_crif needs a single-state batch (use batch.row(i))")
    spot = {k: float(v[0]) for k, v in batch_row.fx_spot.items()}
    jac = {}
    rows = []
    for _, t in trades.iterrows():
        for ccy in trade_currencies(t):
            if ccy not in jac:
                jac[ccy] = par_to_zero_jacobian(batch_row.par[ccy][0])
            lad = ir_delta(t, batch_row, ccy, jac[ccy])
            for tenor, usd in zip(config.IR_TENORS, lad):
                if abs(usd) > MIN_ABS_USD:
                    rows.append(_rec(t, C.RISK_IRCURVE, ccy, IR_BUCKET[ccy], tenor, SUBCURVE, usd / spot[ccy], ccy, usd))
        if t[C.PRODUCT] == C.PRODUCT_XCCY:
            usd = xccy_basis_delta(t, batch_row)
            ccy = t[C.CCY]
            rows.append(_rec(t, C.RISK_XCCYBASIS, ccy, IR_BUCKET[ccy], "", "", usd / spot[ccy], ccy, usd))
        for ccy in trade_currencies(t):
            if ccy != config.CALC_CCY:
                usd = fx_delta(t, batch_row, ccy)
                if abs(usd) > MIN_ABS_USD:
                    rows.append(_rec(t, C.RISK_FX, ccy, "", "", "", usd, config.CALC_CCY, usd))
        if t[C.PRODUCT] == C.PRODUCT_SWAPTION:
            ccy = t[C.CCY]
            for lab, usd in zip(config.VEGA_EXPIRIES, ir_vega_ladder(t, batch_row)):
                if abs(usd) > MIN_ABS_USD:
                    rows.append(_rec(t, C.RISK_IRVOL, ccy, IR_BUCKET[ccy], lab, "", usd / spot[ccy], ccy, usd))
        if t[C.PRODUCT] == C.PRODUCT_FXOPT:
            pair = pair_name(t[C.CCY], t[C.CCY2])
            for lab, usd in zip(config.VEGA_EXPIRIES, fx_vega_ladder(t, batch_row)):
                if abs(usd) > MIN_ABS_USD:
                    rows.append(_rec(t, C.RISK_FXVOL, pair, "", lab, "", usd, config.CALC_CCY, usd))
    out = pd.DataFrame(rows, columns=CRIF_COLUMNS)
    # market vol of the option's expiry lets the SIMM engine turn vega*sigma amounts back into vega
    sigma = {t[C.TRADE_ID]: fx_vega(t, batch_row)[1] for _, t in trades.iterrows() if t[C.PRODUCT] == C.PRODUCT_FXOPT}
    out[SIGMA_MARKET_COL] = np.where(out[C.RISK_TYPE] == C.RISK_FXVOL, out[C.CRIF_TRADE_ID].map(sigma), np.nan)
    return out
