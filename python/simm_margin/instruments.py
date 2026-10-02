"""Trade tables: the ~20-trade sample portfolio and the new EUR/USD cross-currency swap. ALL TRADES ARE SYNTHETIC.

Times (START, MATURITY, EXPIRY) are year fractions from the valuation date, i.e. the portfolio is frozen with
constant time to maturity (no ageing). Direction: +1 = pay fixed (IRS), long (options), buy base ccy (FX forward),
receive the EUR leg (XCCY). Notionals are in the first currency (CCY), FX pairs are base/quote (CCY/CCY2).
"""
import numpy as np
import pandas as pd

from . import columns as C
from .curves import ns_curve
from .market_history import FX_BASE, FX_PAIRS

# ===== CONFIG (user inputs) =====
NETTING_SET, COUNTERPARTY = "NS_A", "CPTY_B"
BASIS_SPREAD, COUNTERPARTY_COL = "basis_spread", "counterparty"     # extra columns (columns.py is frozen)
EXCHANGE_ELIGIBLE, EXCHANGE_NONE = "eligible", "none"
OPT_PAYER, OPT_RECEIVER, OPT_CALL, OPT_PUT = "payer", "receiver", "call", "put"
VOL_NORMAL, VOL_LOGNORMAL = "normal", "lognormal"
# (id, ccy, notional, direction, maturity years, strike offset bp from base par)
IRS_SPEC = [
    ("IRS_USD_2Y", "USD", 60e6, +1, 2, 6), ("IRS_USD_5Y", "USD", 40e6, -1, 5, -9),
    ("IRS_USD_10Y", "USD", 25e6, +1, 10, 4), ("IRS_USD_30Y", "USD", 10e6, -1, 30, 15),
    ("IRS_EUR_5Y", "EUR", 30e6, +1, 5, -12), ("IRS_EUR_10Y", "EUR", 20e6, -1, 10, 8),
    ("IRS_GBP_7Y", "GBP", 25e6, +1, 7, 5), ("IRS_JPY_10Y", "JPY", 4e9, -1, 10, -6),
    ("IRS_MXN_3Y", "MXN", 500e6, +1, 3, 20),
]
# (id, ccy, notional, direction, option type, expiry, underlying tenor, strike offset bp from forward swap)
SWAPTION_SPEC = [
    ("SWPT_USD_1Yx5Y", "USD", 50e6, +1, OPT_PAYER, 1.0, 5.0, 10),
    ("SWPT_EUR_4Yx10Y", "EUR", 30e6, +1, OPT_RECEIVER, 4.0, 10.0, -15),
    ("SWPT_GBP_5Yx5Y", "GBP", 25e6, -1, OPT_PAYER, 5.0, 5.0, 0),
]
# (id, base, quote, notional in base, direction, maturity, strike offset relative to forward)
FXFWD_SPEC = [
    ("FXF_EURUSD_6M", "EUR", "USD", 25e6, +1, 0.5, 0.004), ("FXF_GBPUSD_1Y", "GBP", "USD", 15e6, -1, 1.0, -0.006),
    ("FXF_USDJPY_3M", "USD", "JPY", 20e6, +1, 0.25, 0.010), ("FXF_USDMXN_6M", "USD", "MXN", 10e6, -1, 0.5, 0.015),
]
# (id, base, quote, notional in base, direction, option type, expiry, strike offset relative to forward)
FXOPT_SPEC = [
    ("FXO_EURUSD_C3M", "EUR", "USD", 20e6, +1, OPT_CALL, 0.25, 0.01),
    ("FXO_USDMXN_C1Y", "USD", "MXN", 10e6, +1, OPT_CALL, 1.0, 0.05),
    ("FXO_GBPUSD_STRADDLE_C", "GBP", "USD", 15e6, +1, OPT_CALL, 0.5, 0.0),
    ("FXO_GBPUSD_STRADDLE_P", "GBP", "USD", 15e6, +1, OPT_PUT, 0.5, 0.0),
]
XCCY_DEFAULT_NOTIONAL, XCCY_DEFAULT_MATURITY, XCCY_DEFAULT_BASIS = 50e6, 5.0, -0.0015
# ===== END CONFIG =====

TRADE_COLUMNS = [C.TRADE_ID, C.PORTFOLIO_ID, COUNTERPARTY_COL, C.PRODUCT, C.CCY, C.CCY2, C.NOTIONAL, C.DIRECTION,
                 C.START, C.MATURITY, C.EXPIRY, C.STRIKE, C.VOL_TYPE, C.OPT_TYPE, C.EXCHANGE_FLAG, BASIS_SPREAD]


def _row(**kw) -> dict:
    base = {C.PORTFOLIO_ID: NETTING_SET, COUNTERPARTY_COL: COUNTERPARTY, C.CCY2: "", C.START: 0.0, C.EXPIRY: np.nan,
            C.STRIKE: np.nan, C.VOL_TYPE: "", C.OPT_TYPE: "", C.EXCHANGE_FLAG: "", BASIS_SPREAD: np.nan}
    base.update(kw)
    return base


def _swap_rate(ccy: str, t0: float, t1: float) -> float:
    """Forward par swap rate on the base synthetic NS curve (annual fixed leg, single curve)."""
    ns = ns_curve(ccy)
    tp = np.minimum(t0 + np.arange(1, int(np.ceil(t1 - t0 - 1e-9)) + 1), t1)
    acc = np.diff(np.concatenate(([t0], tp)))
    return float((ns.discount(t0) - ns.discount(t1)) / (ns.discount(tp) * acc).sum())


def _base_forward(base: str, quote: str, t: float) -> float:
    """Forward FX (quote per base) from the base NS curves and base spot levels."""
    px = FX_BASE[base + quote] if base + quote in FX_BASE else 1.0 / FX_BASE[quote + base]
    return px * float(ns_curve(quote).discount(t) ** -1 * ns_curve(base).discount(t))


def sample_portfolio() -> pd.DataFrame:
    """9 IRS, 3 European swaptions, 4 FX forwards, 4 FX European options (incl. a straddle); netting set NS_A."""
    rows = []
    for tid, ccy, n, d, mat, off in IRS_SPEC:
        rows.append(_row(**{C.TRADE_ID: tid, C.PRODUCT: C.PRODUCT_IRS, C.CCY: ccy, C.NOTIONAL: n, C.DIRECTION: d,
                            C.MATURITY: float(mat), C.STRIKE: _swap_rate(ccy, 0.0, mat) + off * 1e-4}))
    for tid, ccy, n, d, ot, exp, ten, off in SWAPTION_SPEC:
        rows.append(_row(**{C.TRADE_ID: tid, C.PRODUCT: C.PRODUCT_SWAPTION, C.CCY: ccy, C.NOTIONAL: n, C.DIRECTION: d,
                            C.OPT_TYPE: ot, C.START: exp, C.EXPIRY: exp, C.MATURITY: exp + ten, C.VOL_TYPE: VOL_NORMAL,
                            C.STRIKE: _swap_rate(ccy, exp, exp + ten) + off * 1e-4}))
    for tid, b, q, n, d, mat, off in FXFWD_SPEC:
        rows.append(_row(**{C.TRADE_ID: tid, C.PRODUCT: C.PRODUCT_FXFWD, C.CCY: b, C.CCY2: q, C.NOTIONAL: n,
                            C.DIRECTION: d, C.MATURITY: mat, C.STRIKE: _base_forward(b, q, mat) * (1 + off)}))
    for tid, b, q, n, d, ot, exp, off in FXOPT_SPEC:
        rows.append(_row(**{C.TRADE_ID: tid, C.PRODUCT: C.PRODUCT_FXOPT, C.CCY: b, C.CCY2: q, C.NOTIONAL: n,
                            C.DIRECTION: d, C.OPT_TYPE: ot, C.EXPIRY: exp, C.MATURITY: exp, C.VOL_TYPE: VOL_LOGNORMAL,
                            C.STRIKE: _base_forward(b, q, exp) * (1 + off)}))
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def new_product_xccy(notional: float = XCCY_DEFAULT_NOTIONAL, maturity: float = XCCY_DEFAULT_MATURITY,
                     basis: float = XCCY_DEFAULT_BASIS, initial_fx: float = None,
                     settle_notional_exchange: str = EXCHANGE_ELIGIBLE, direction: int = +1,
                     trade_id: str = "XCCY_EURUSD_5Y") -> pd.DataFrame:
    """EUR/USD cross-currency basis swap (EUR notional, USD notional = notional * initial_fx), one-row table.

    basis: contractual spread on the EUR leg (decimal); initial_fx: USD per EUR at inception (STRIKE column).
    settle_notional_exchange="eligible" means the principal exchange is excluded from IM sensitivities."""
    fx0 = FX_BASE["EURUSD"] if initial_fx is None else initial_fx
    r = _row(**{C.TRADE_ID: trade_id, C.PRODUCT: C.PRODUCT_XCCY, C.CCY: "EUR", C.CCY2: "USD", C.NOTIONAL: notional,
                C.DIRECTION: direction, C.MATURITY: float(maturity), C.STRIKE: fx0,
                C.EXCHANGE_FLAG: settle_notional_exchange, BASIS_SPREAD: basis})
    return pd.DataFrame([r], columns=TRADE_COLUMNS)


def pair_name(base: str, quote: str) -> str:
    """Market pair name for a base/quote currency pair (either order of the market convention is accepted)."""
    p = base + quote
    if p in FX_PAIRS:
        return p
    if quote + base in FX_PAIRS:
        return quote + base
    raise KeyError(f"no market pair for {base}/{quote}")
