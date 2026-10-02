"""Bump-and-reprice sensitivities in USD on a single-state MarketBatch row. SYNTHETIC market data.

IR delta: PV change per 1bp parallel move of ONE par quote (central difference +/- 0.5bp), by the Jacobian method
(zero-pillar bumps times d zero/d par) with a slow full re-bootstrap version for testing. FX delta: PV change per
1% relative move in USD per unit of the currency (central +/- 0.5%), which includes translation risk. Vega: IR per
1bp of normal vol and FX per vol point; CRIF vega amounts are vega * sigma allocated linearly to expiry vertices.
"""
import numpy as np

from . import columns as C
from . import config
from .curves import TENOR_YEARS, interp_weights, par_to_zero_jacobian
from .market_history import MarketBatch
from .pricing import fx_vol, ir_vol, price_trade, xccy_include_exchange
from .instruments import BASIS_SPREAD, pair_name

# ===== CONFIG (user inputs) =====
IR_BUMP = config.IR_BUMP_BP * 1e-4          # width of the central difference on par/zero (1bp)
FX_BUMP = config.FX_BUMP_REL                # width of the relative central difference (1%)
IR_VOL_BUMP = config.VOL_BUMP_ABS           # width of the normal-vol bump (1bp)
FX_VOL_POINT = 0.01                         # one vol point of lognormal vol
SF_DAYS = 14.0                              # curvature scaling SF(t) = 0.5 * min(1, 14 / t_days)
DAYS_PER_YEAR = 365.0
# ===== END CONFIG =====

N_V = len(TENOR_YEARS)


def trade_currencies(trade) -> list:
    """Currencies whose curves and spot affect the trade (USD included; it still has rate risk)."""
    return [trade[C.CCY]] + ([trade[C.CCY2]] if trade[C.CCY2] else [])


def _inc(trade) -> bool:
    return xccy_include_exchange(trade) if trade[C.PRODUCT] == C.PRODUCT_XCCY else True


def _stack_zero_bumps(row: MarketBatch, ccy: str) -> MarketBatch:
    b = row.tile(2 * N_V).copy_with()
    z = np.repeat(row.zero[ccy], 2 * N_V, axis=0).copy()
    idx = np.arange(N_V)
    z[idx, idx] += IR_BUMP / 2
    z[N_V + idx, idx] -= IR_BUMP / 2
    b.zero[ccy] = z
    return b


def zero_pillar_deltas(trade, row: MarketBatch, ccy: str) -> np.ndarray:
    """dV/d(zero_i) per 1bp zero bump, USD, shape [12]."""
    v = price_trade(trade, _stack_zero_bumps(row, ccy), _inc(trade))
    return (v[:N_V] - v[N_V:]) * (1e-4 / IR_BUMP)


def ir_delta(trade, row: MarketBatch, ccy: str, jac: np.ndarray = None) -> np.ndarray:
    """USD per 1bp par-quote bump at each of the 12 vertices, via the Jacobian method."""
    jac = par_to_zero_jacobian(row.par[ccy][0]) if jac is None else jac
    return jac.T @ zero_pillar_deltas(trade, row, ccy)


def ir_delta_full_bootstrap(trade, row: MarketBatch, ccy: str) -> np.ndarray:
    """Slow reference: re-bootstrap the curve for every +/- half-bp par bump (24 full re-bootstraps)."""
    par = np.repeat(row.par[ccy], 2 * N_V, axis=0).copy()
    idx = np.arange(N_V)
    par[idx, idx] += IR_BUMP / 2
    par[N_V + idx, idx] -= IR_BUMP / 2
    v = price_trade(trade, row.tile(2 * N_V).rebootstrapped({ccy: par}), _inc(trade))
    return (v[:N_V] - v[N_V:]) * (1e-4 / IR_BUMP)


def parallel_pv01(trade, row: MarketBatch, ccy: str) -> float:
    """PV change for a +1bp parallel move of all par quotes of ccy (full re-bootstrap, central difference)."""
    par = np.repeat(row.par[ccy], 2, axis=0).copy()
    par[0] += 0.5e-4
    par[1] -= 0.5e-4
    v = price_trade(trade, row.tile(2).rebootstrapped({ccy: par}), _inc(trade))
    return float(v[0] - v[1])


def fx_delta(trade, row: MarketBatch, ccy: str) -> float:
    """USD PV change for a 1% relative rise of ccy against USD (central +/- 0.5%), incl. translation."""
    b = row.tile(2).copy_with()
    b.fx_spot[ccy] = row.fx_spot[ccy][0] * np.array([1 + FX_BUMP / 2, 1 - FX_BUMP / 2])
    v = price_trade(trade, b, _inc(trade))
    return float(v[0] - v[1])


def rebucket_weights(t_years: float, vertices=TENOR_YEARS):
    """Linear rebucketing of an off-vertex time onto neighbouring vertices: [(vertex index, weight)], weights sum to 1.
    Example: 7y -> 60% of 5y and 40% of 10y; outside the range everything goes to the end vertex."""
    i, w = interp_weights(t_years, np.asarray(vertices, dtype=float))
    i, w = int(i[0]), float(w[0])
    return [(i, 1.0 - w), (i + 1, w)] if w > 0 else [(i, 1.0)]


def _allocate(amount: float, t_years: float) -> np.ndarray:
    out = np.zeros(N_V)
    for k, w in rebucket_weights(t_years):
        out[k] += w * amount
    return out


def ir_vega(trade, row: MarketBatch):
    """(vega per 1bp normal vol in USD, sigma in bp, CRIF amount = vega_per_bp * sigma_bp) for a swaption."""
    ccy = trade[C.CCY]
    b = row.tile(2).copy_with()
    b.ir_nvol[ccy] = row.ir_nvol[ccy] + np.array([IR_VOL_BUMP / 2, -IR_VOL_BUMP / 2])[:, None, None]
    v = price_trade(trade, b)
    per_bp = float((v[0] - v[1]) * (1e-4 / IR_VOL_BUMP))
    sig_bp = float(ir_vol(row, ccy, trade[C.EXPIRY], trade[C.MATURITY] - trade[C.EXPIRY])[0]) * 1e4
    return per_bp, sig_bp, per_bp * sig_bp


def ir_vega_ladder(trade, row: MarketBatch) -> np.ndarray:
    """CRIF IR vega amounts (USD) on the 12 expiry vertices, linear in expiry."""
    return _allocate(ir_vega(trade, row)[2], trade[C.EXPIRY])


def fx_vega(trade, row: MarketBatch):
    """(vega per vol point in USD, sigma as a fraction, CRIF amount = vega_per_unit_vol * sigma) for an FX option."""
    pair = pair_name(trade[C.CCY], trade[C.CCY2])
    b = row.tile(2).copy_with()
    b.fx_vol[pair] = row.fx_vol[pair] + np.array([FX_VOL_POINT / 2, -FX_VOL_POINT / 2])[:, None]
    v = price_trade(trade, b)
    per_pt = float(v[0] - v[1])
    sig = float(fx_vol(row, pair, trade[C.EXPIRY])[0])
    return per_pt, sig, per_pt / FX_VOL_POINT * sig


def fx_vega_ladder(trade, row: MarketBatch) -> np.ndarray:
    return _allocate(fx_vega(trade, row)[2], trade[C.EXPIRY])


def xccy_basis_delta(trade, row: MarketBatch) -> float:
    """USD PV change per +1bp of the contractual basis spread (equals notional * annuity * FX * 1bp)."""
    t0, t1 = trade.copy(), trade.copy()
    t0[BASIS_SPREAD], t1[BASIS_SPREAD] = trade[BASIS_SPREAD] + 0.5e-4, trade[BASIS_SPREAD] - 0.5e-4
    return float(price_trade(t0, row, _inc(trade))[0] - price_trade(t1, row, _inc(trade))[0])


def curvature_scaling(t_years) -> np.ndarray:
    """SF(t) = 0.5 * min(1, 14 / t_days) per the verified SIMM v2.8 structure (VERIFICATION_LOG row 1)."""
    t_days = np.maximum(np.asarray(t_years, dtype=float) * DAYS_PER_YEAR, 1e-12)
    return 0.5 * np.minimum(1.0, SF_DAYS / t_days)


def curvature_cvr(vega_ladder: np.ndarray, expiries=TENOR_YEARS):
    """(per-vertex SF(t_j) * sigma*vega_j, their sum CVR) for one risk factor's vega ladder; the SIMM engine
    applies theta, lambda and the correlations on top of CVR."""
    parts = curvature_scaling(expiries) * np.asarray(vega_ladder, dtype=float)
    return parts, float(parts.sum())
