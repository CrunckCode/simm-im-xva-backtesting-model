"""Euler allocation of SIMM-style IM to trades or risk factors by a central directional derivative. SYNTHETIC inputs.

Contribution_g = d IM(crif with group g scaled by (1 + e)) / d e at e = 0 (central difference). When every concentration
factor is 1 the IM is positively homogeneous of degree 1 in the CRIF amounts, so contributions sum to the total IM
(Euler's theorem). With CR above 1 the sum differs from the total; the shortfall is reported by euler_check.
"""
import numpy as np
import pandas as pd

from . import columns as C
from .params import SimmParams, load
from .simm import simm

# ===== CONFIG (user inputs) =====
BY_TRADE, BY_FACTOR = "TRADE_ID", C.RISK_FACTOR
BUMP = 1.0e-5                       # relative scaling used on each side of the central difference
FACTOR_KEYS = [C.RISK_TYPE, C.QUALIFIER, C.LABEL1, C.LABEL2]
# ===== END CONFIG =====


def _group_keys(crif: pd.DataFrame, by: str) -> pd.Series:
    key = by.lower().replace("_", "")
    if key == "tradeid":
        return crif[C.CRIF_TRADE_ID].astype(str)
    if key == "riskfactor":
        return crif[FACTOR_KEYS].fillna("").astype(str).agg("|".join, axis=1)
    raise ValueError("by must be 'TRADE_ID' or 'risk_factor'")


def _scaled(crif: pd.DataFrame, mask: np.ndarray, factor: float) -> pd.DataFrame:
    out = crif.copy()
    out.loc[mask, C.AMOUNT_USD] = out.loc[mask, C.AMOUNT_USD] * factor
    return out


def euler_contributions(crif: pd.DataFrame, p: SimmParams = None, by: str = BY_TRADE, bump: float = BUMP) -> pd.Series:
    """IM contribution per trade (by='TRADE_ID') or per risk factor (by='risk_factor'), in USD.
    Scaling a trade also scales any SigmaMarket weights consistently because only AmountUSD is bumped."""
    p = p or load()
    if len(crif) == 0:
        return pd.Series(dtype=float)
    keys = _group_keys(crif, by)
    out = {}
    for g in keys.unique():
        mask = (keys == g).to_numpy()
        up = simm(_scaled(crif, mask, 1.0 + bump), p).total
        dn = simm(_scaled(crif, mask, 1.0 - bump), p).total
        out[g] = (up - dn) / (2.0 * bump)
    return pd.Series(out, name="im_contribution")


def euler_check(crif: pd.DataFrame, p: SimmParams = None, by: str = BY_TRADE) -> dict:
    """Total IM, sum of contributions and their gap (zero up to finite-difference error when all CR = 1)."""
    p = p or load()
    contrib = euler_contributions(crif, p, by)
    total = simm(crif, p).total
    return {"total": total, "sum_contributions": float(contrib.sum()), "gap": total - float(contrib.sum())}
