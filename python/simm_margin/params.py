"""Typed SIMM-style parameter set built from data/parameters/simm_parameters.csv (verified numbers, SYNTHETIC use only).

Every accessor reads the parameter file; nothing is hard-coded here. Matrices are built once, checked for symmetry and
positive semidefiniteness, and an unknown version raises. Not ISDA-licensed or certified.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.stats import norm

from . import columns as C
from . import config
from . import parameter_io as pio

# ===== CONFIG (user inputs) =====
PSI_CLASSES = ("IR", "CreditQ", "CreditNonQ", "Equity", "Commodity", "FX")
EIG_TOL = 1e-9                  # smallest eigenvalue allowed when checking positive semidefiniteness
OTHER_KEY = "ALL_OTHER"         # parameter-file marker for "every currency not listed"
GROUP_REGULAR, GROUP_LOW, GROUP_HIGH = "regular", "low", "high"
CT_WELL, CT_LESS, CT_LOW, CT_HIGH = "regular_well_traded", "regular_less_well_traded", "low", "high"
CATEGORIES = ("Category 1", "Category 2", "Category 3")
DEFAULT_CATEGORY = "Category 3"
# ===== END CONFIG =====


def check_matrix(m: np.ndarray, name: str, unit_diagonal: bool = True) -> np.ndarray:
    """Raise unless m is symmetric, (optionally) has a unit diagonal and is positive semidefinite."""
    if not np.allclose(m, m.T, atol=0, rtol=0):
        raise ValueError(f"{name}: matrix is not symmetric")
    if unit_diagonal and not np.allclose(np.diag(m), 1.0):
        raise ValueError(f"{name}: diagonal is not 1")
    if np.linalg.eigvalsh(m).min() < -EIG_TOL:
        raise ValueError(f"{name}: matrix is not positive semidefinite")
    return m


@dataclass
class SimmParams:
    """Parameters of one SIMM-style version. Build with load(version)."""
    version: str
    calc_ccy: str
    values: dict = field(repr=False)            # (risk_class, param_name, key1, key2) -> float
    members: dict = field(repr=False)           # (risk_class, param_name, key1) -> set of key2
    tenors: tuple = config.IR_TENORS
    ir_corr: np.ndarray = None                  # 12x12 tenor correlation
    psi_matrix: np.ndarray = None               # 6x6 risk class correlation

    # == generic ==
    def v(self, risk_class, name, key1="", key2="") -> float:
        try:
            return self.values[(risk_class, name, key1, key2)]
        except KeyError:
            raise KeyError(f"parameter {self.version}/{risk_class}/{name}/{key1}/{key2} not in the parameter file")

    # == interest rate ==
    def ir_rw_group(self, ccy: str) -> str:
        """Risk-weight group of a currency: low (JPY), regular (14 listed) or high (all others)."""
        for g in (GROUP_REGULAR, GROUP_LOW):
            if ccy in self.members[(C.RC_IR, "rw_group_member", g)]:
                return g
        return GROUP_HIGH

    def ir_ct_group(self, ccy: str) -> str:
        for g in (CT_WELL, CT_LESS, CT_LOW):
            if ccy in self.members[(C.RC_IR, "ct_group_member", g)]:
                return g
        return CT_HIGH

    def ir_rw(self, ccy: str) -> np.ndarray:
        """Delta risk weights by tenor (IR_TENORS order) for the currency's volatility group."""
        g = self.ir_rw_group(ccy)
        return np.array([self.v(C.RC_IR, "delta_rw", g, t) for t in self.tenors])

    @property
    def ir_rw_inflation(self) -> float:
        return self.v(C.RC_IR, "delta_rw_inflation", "any")

    @property
    def ir_rw_xccy(self) -> float:
        return self.v(C.RC_IR, "delta_rw_xccy_basis", "any")

    @property
    def phi(self) -> float:
        return self.v(C.RC_IR, "subcurve_phi")

    @property
    def infl_corr(self) -> float:
        return self.v(C.RC_IR, "inflation_yield_corr")

    @property
    def infl_vol_corr(self) -> float:
        return self.v(C.RC_IR, "inflation_vol_ir_vol_corr")

    @property
    def xccy_corr(self) -> float:
        return self.v(C.RC_IR, "xccy_basis_yield_corr")

    @property
    def ir_gamma(self) -> float:
        return self.v(C.RC_IR, "gamma")

    @property
    def ir_hvr(self) -> float:
        return self.v(C.RC_IR, "hvr")

    @property
    def ir_vrw(self) -> float:
        return self.v(C.RC_IR, "vega_rw")

    def ir_delta_ct(self, ccy: str) -> float:
        """Delta concentration threshold, USD mm per bp."""
        return self.v(C.RC_IR, "delta_ct", self.ir_ct_group(ccy))

    def ir_vega_ct(self, ccy: str) -> float:
        """Vega concentration threshold, USD mm."""
        return self.v(C.RC_IR, "vega_ct", self.ir_ct_group(ccy))

    @property
    def ir_curv_scale(self) -> float:
        """Scale applied to the IR curvature margin: HVR_IR raised to the file's exponent (-2)."""
        return self.ir_hvr ** self.v(C.RC_IR, "curvature_hvr_exponent")

    # == foreign exchange ==
    def fx_group(self, ccy: str) -> str:
        return GROUP_HIGH if ccy in self.members[(C.RC_FX, "high_vol_member", GROUP_HIGH)] else GROUP_REGULAR

    def fx_rw(self, ccy: str, other: str = None) -> float:
        """FX delta risk weight: row = group of ccy, column = group of `other` (default calculation currency)."""
        col = self.fx_group(other or self.calc_ccy)
        return self.v(C.RC_FX, "delta_rw", self.fx_group(ccy), col)

    def fx_corr(self, ccy_a: str, ccy_b: str) -> float:
        """Correlation between the FX delta factors of two distinct currencies (table chosen by calc ccy group)."""
        name = "delta_corr_calc_high" if self.fx_group(self.calc_ccy) == GROUP_HIGH else "delta_corr_calc_regular"
        return self.v(C.RC_FX, name, self.fx_group(ccy_a), self.fx_group(ccy_b))

    def fx_corr_matrix(self, ccys) -> np.ndarray:
        """Factor-level FX correlation matrix: unit diagonal, table value for every distinct pair. Checked PSD."""
        n = len(ccys)
        m = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                m[i, j] = m[j, i] = self.fx_corr(ccys[i], ccys[j])
        return check_matrix(m, f"FX correlation {self.version}")

    def fx_category(self, ccy: str) -> str:
        for c in CATEGORIES[:2]:
            if ccy in self.members[(C.RC_FX, "ct_category_member", c)]:
                return c
        return DEFAULT_CATEGORY

    def fx_delta_ct(self, ccy: str) -> float:
        """FX delta concentration threshold, USD mm per 1%."""
        return self.v(C.RC_FX, "delta_ct", self.fx_category(ccy))

    def fx_vega_ct(self, ccy_a: str, ccy_b: str) -> float:
        """FX vega concentration threshold for a currency pair (category pair in file order), USD mm."""
        ca, cb = sorted((self.fx_category(ccy_a), self.fx_category(ccy_b)))
        return self.v(C.RC_FX, "vega_ct", ca, cb)

    @property
    def fx_hvr(self) -> float:
        return self.v(C.RC_FX, "hvr")

    @property
    def fx_vrw(self) -> float:
        return self.v(C.RC_FX, "vega_rw")

    @property
    def fx_vol_corr(self) -> float:
        """Correlation between FX vega factors and between FX curvature factors."""
        return self.v(C.RC_FX, "vol_curvature_corr")

    def fx_sigma(self, ccy_a: str, ccy_b: str) -> float:
        """Vol used for FX vega as a fraction: RW(pair) * sqrt(days_num / days_den) / Phi^-1(quantile) / 100."""
        rw = self.fx_rw(ccy_a, ccy_b)
        ratio = self.v("ALL", "vega_sigma_days_numerator") / self.v("ALL", "vega_sigma_days_denominator")
        return rw * np.sqrt(ratio) / norm.ppf(self.v("ALL", "vega_sigma_quantile")) / 100.0

    # == curvature ==
    def sf(self, days) -> np.ndarray:
        """Curvature scaling function SF(t) = scale * min(1, ref_days / t_days), t in calendar days."""
        d = np.maximum(np.asarray(days, dtype=float), 1e-12)
        return self.v("ALL", "curvature_sf_scale") * np.minimum(1.0, self.v("ALL", "curvature_sf_ref_days") / d)

    def tenor_days(self) -> np.ndarray:
        """Calendar days of the 12 vertices: 2w is 14 days, other tenors are pro rata on a 365 day year."""
        yrs = np.array(config.IR_TENOR_YEARS)
        days = yrs * self.v("ALL", "tenor_days_per_year")
        days[0] = self.v("ALL", "curvature_sf_ref_days")
        return days

    def curv_lambda(self, theta: float) -> float:
        q = norm.ppf(self.v("ALL", "curvature_lambda_quantile"))
        return (q * q - 1.0) * (1.0 + theta) - theta

    # == risk class aggregation ==
    def psi(self, rc_a: str, rc_b: str) -> float:
        i, j = PSI_CLASSES.index(rc_a), PSI_CLASSES.index(rc_b)
        return float(self.psi_matrix[i, j])

    @property
    def multiplicative_scale(self) -> float:
        return self.v("ALL", "multiplicative_scale_default", C.RATES_FX)


@lru_cache(maxsize=None)
def _load(version: str, param_dir) -> SimmParams:
    rows = pio.simm_rows(version, param_dir=param_dir)
    if not rows:
        raise ValueError(f"unknown SIMM parameter version {version!r}")
    values, members = {}, {}
    for r in rows:
        k = (r[C.RISK_CLASS], r[C.PARAM_NAME], r[C.KEY1], r[C.KEY2])
        if k in values:
            raise ValueError(f"duplicate parameter {k}")
        values[k] = float(r[C.PARAM_VALUE])
        members.setdefault((k[0], k[1], k[2]), set()).add(k[3])
    ir = check_matrix(pio.square_matrix(version, C.RC_IR, "delta_tenor_corr", config.IR_TENORS, param_dir), "IR tenor")
    psi = check_matrix(pio.square_matrix(version, "ALL", "psi", PSI_CLASSES, param_dir), "psi")
    p = SimmParams(version=version, calc_ccy=config.CALC_CCY, values=values, members=members, ir_corr=ir, psi_matrix=psi)
    for pn in ("delta_corr_calc_regular", "delta_corr_calc_high"):      # expanded FX matrices must be PSD
        t = np.array([[p.v(C.RC_FX, pn, a, b) for b in (GROUP_REGULAR, GROUP_HIGH)] for a in (GROUP_REGULAR, GROUP_HIGH)])
        if not np.allclose(t, t.T):
            raise ValueError(f"{pn} table is not symmetric")
    p.fx_corr_matrix([config.CALC_CCY, "EUR", "MXN", "ARS"])
    return p


def load(version: str = None, param_dir=None) -> SimmParams:
    """Parameters for a version such as '2.8+2512' (default config.SIMM_VERSION); unknown versions raise ValueError."""
    return _load(version or config.SIMM_VERSION, None if param_dir is None else str(param_dir))


def available_versions(param_dir=None) -> list:
    return sorted({r[C.PARAM_SET] for r in pio.read_csv(pio.SIMM_PARAMS_FILE, param_dir)})
