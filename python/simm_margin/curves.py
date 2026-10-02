"""OIS-style curve bootstrapped from par quotes at the 12 SIMM vertices. ALL CURVE DATA IS SYNTHETIC.

Vertices: cash rates (simple) at 2w, 1m, 3m, 6m and annual-pay par swap rates from 1y.
Zero rates (continuous compounding) are linear in time between vertices with flat extrapolation.
Everything is vectorised over a leading state dimension n.
"""
from dataclasses import dataclass

import numpy as np

from . import config

# ===== CONFIG (user inputs) =====
TENORS = config.IR_TENORS
TENOR_YEARS = np.array(config.IR_TENOR_YEARS, dtype=float)
N_VERTEX = len(TENORS)
N_CASH = 4                      # 2w, 1m, 3m, 6m are simple cash rates; 1y onwards annual swaps
FIRST_SWAP_IDX = 4              # 1y
BOOT_ITERS = 12                 # Newton iterations per swap vertex
JAC_BUMP = 1.0e-4               # par bump width (1bp) for the Jacobian; central +/- half
NS_MIN_T = 1.0e-8
CURRENCIES = ("USD", "EUR", "GBP", "JPY", "MXN")
# synthetic Nelson-Siegel zero curves (beta0, beta1, beta2, tau); different levels by currency
NS_PARAMS = {
    "USD": (0.0450, -0.0080, -0.0050, 2.5),
    "EUR": (0.0290, -0.0070, -0.0020, 3.0),
    "GBP": (0.0420, -0.0060, 0.0020, 3.0),
    "JPY": (0.0210, -0.0145, -0.0040, 4.0),
    "MXN": (0.0900, -0.0150, 0.0060, 2.5),
}
# ===== END CONFIG =====


def interp_weights(t, nodes=TENOR_YEARS):
    """Index of the left node and weight of the right node for linear interpolation, flat outside."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    tc = np.clip(t, nodes[0], nodes[-1])
    i = np.clip(np.searchsorted(nodes, tc, side="right") - 1, 0, len(nodes) - 2)
    return i, (tc - nodes[i]) / (nodes[i + 1] - nodes[i])


def payment_times(t0: float, t1: float) -> np.ndarray:
    """Annual fixed-leg payment times after t0 up to t1, with a final stub if t1 - t0 is not whole years."""
    k = int(np.ceil(t1 - t0 - 1e-9))
    return np.minimum(t0 + np.arange(1, k + 1, dtype=float), t1)


@dataclass
class Curve:
    """Bootstrapped curve for n states: par[n,12] (may be None for bumped-zero curves) and zero[n,12]."""
    par: np.ndarray
    zero: np.ndarray

    @property
    def n(self) -> int:
        return self.zero.shape[0]

    def zero_rate(self, t) -> np.ndarray:
        i, w = interp_weights(t)
        return self.zero[:, i] * (1.0 - w) + self.zero[:, i + 1] * w

    def df(self, t) -> np.ndarray:
        """Discount factors [n, len(t)]; DF(0) = 1."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.exp(-self.zero_rate(t) * t)

    def annuity(self, t0: float, t1: float) -> np.ndarray:
        """Annual-accrual annuity of a swap running t0..t1, shape [n]."""
        tp = payment_times(t0, t1)
        acc = np.diff(np.concatenate(([t0], tp)))
        return self.df(tp) @ acc

    def forward_swap(self, t0: float, t1: float):
        """(forward par swap rate, annuity), each [n]; float leg is single-curve DF(t0) - DF(t1)."""
        a = self.annuity(t0, t1)
        d = self.df(np.array([t0, t1]))
        return (d[:, 0] - d[:, 1]) / a, a

    def with_zero(self, zero: np.ndarray) -> "Curve":
        return Curve(None, zero)


def bootstrap_curve(par_quotes) -> Curve:
    """Bootstrap zeros that reprice every par quote exactly. par_quotes: [12] or [n,12] in decimals."""
    par = np.atleast_2d(np.asarray(par_quotes, dtype=float))
    n, T = par.shape[0], TENOR_YEARS
    z = np.empty((n, N_VERTEX))
    for k in range(FIRST_SWAP_IDX + 1):       # cash vertices and the 1y single-period swap
        z[:, k] = np.log1p(par[:, k] * T[k]) / T[k]
    known = np.exp(-z[:, FIRST_SWAP_IDX] * T[FIRST_SWAP_IDX])   # annuity of annual dates already solved
    for k in range(FIRST_SWAP_IDX + 1, N_VERTEX):
        tp, tk = T[k - 1], T[k]
        tn = np.arange(tp + 1.0, tk + 1e-9)
        w = (tn - tp) / (tk - tp)
        zp, p = z[:, k - 1][:, None], par[:, k]
        zk = np.log1p(p)
        for _ in range(BOOT_ITERS):
            dfi = np.exp(-(zp + (zk[:, None] - zp) * w) * tn)
            f = p * (known + dfi.sum(1)) - 1.0 + dfi[:, -1]
            d = -(dfi * tn * w).sum(1) * p - tk * dfi[:, -1]
            zk = zk - f / d
        z[:, k] = zk
        dfi = np.exp(-(zp + (zk[:, None] - zp) * w) * tn)
        known = known + dfi.sum(1)
    return Curve(par, z)


def par_to_zero_jacobian(par_row, bump: float = JAC_BUMP) -> np.ndarray:
    """J[i, j] = d zero_i / d par_j from 12 central-difference re-bootstraps (one batched call)."""
    par = np.asarray(par_row, dtype=float).reshape(N_VERTEX)
    bumped = np.tile(par, (2 * N_VERTEX, 1))
    idx = np.arange(N_VERTEX)
    bumped[idx, idx] += bump / 2
    bumped[N_VERTEX + idx, idx] -= bump / 2
    z = bootstrap_curve(bumped).zero
    return ((z[:N_VERTEX] - z[N_VERTEX:]) / bump).T


def par_rates_of(curve: Curve) -> np.ndarray:
    """Par quotes implied by a curve's zeros at the 12 vertices (for self-consistency checks)."""
    out = np.empty((curve.n, N_VERTEX))
    for k, T in enumerate(TENOR_YEARS):
        if k <= FIRST_SWAP_IDX:
            out[:, k] = (1.0 / curve.df(T)[:, 0] - 1.0) / T
        else:
            tp = np.arange(1.0, T + 1e-9)
            out[:, k] = (1.0 - curve.df(T)[:, 0]) / curve.df(tp).sum(1)
    return out


@dataclass(frozen=True)
class NelsonSiegelCurve:
    """Synthetic Nelson-Siegel zero curve (continuous compounding), used only to generate par levels."""
    beta0: float
    beta1: float
    beta2: float
    tau: float
    shift: float = 0.0

    def zero_rate(self, T):
        T = np.asarray(T, dtype=float)
        x = np.maximum(T, NS_MIN_T) / self.tau
        load1 = -np.expm1(-x) / x
        return self.beta0 + self.beta1 * load1 + self.beta2 * (load1 - np.exp(-x)) + self.shift

    def discount(self, T):
        T = np.asarray(T, dtype=float)
        return np.where(T <= 0.0, 1.0, np.exp(-self.zero_rate(T) * T))

    def par_quotes(self) -> np.ndarray:
        """Cash and annual-swap par quotes at the 12 SIMM vertices implied by this curve."""
        out = np.empty(N_VERTEX)
        for k, T in enumerate(TENOR_YEARS):
            d = float(self.discount(T))
            if k <= FIRST_SWAP_IDX:
                out[k] = (1.0 / d - 1.0) / T
            else:
                out[k] = (1.0 - d) / self.discount(np.arange(1.0, T + 1e-9)).sum()
        return out


def ns_curve(ccy: str) -> NelsonSiegelCurve:
    return NelsonSiegelCurve(*NS_PARAMS[ccy])


def base_par(ccy: str) -> np.ndarray:
    """Base (long-run) par levels for a currency, [12]."""
    return ns_curve(ccy).par_quotes()
