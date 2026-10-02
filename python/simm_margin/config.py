"""Global configuration shared by all modules. Change values here, not inside modules."""
from pathlib import Path
import pandas as pd

# ===== CONFIG (user inputs) =====
ROOT = Path(__file__).resolve().parents[2]
PARAM_DIR = ROOT / "data" / "parameters"
OUT_DIR = ROOT / "python" / "outputs"
CHART_DIR, EXCEL_INPUT_DIR = OUT_DIR / "charts", OUT_DIR / "excel_inputs"
DATA_SOURCE = "synthetic (no market data)"
SEED = 20261002

SIMM_VERSION = "2.8+2512"            # primary parameter set
SIMM_VERSION_PRIOR = "2.8+2506"      # used in attribution and dispute scenarios
CALC_CCY = "USD"

# synthetic history
HIST_START, HIST_END = pd.Timestamp("2018-10-01"), pd.Timestamp("2026-09-30")
HIST_QUICK_START = pd.Timestamp("2024-10-01")
STRESS_WINDOW_DAYS = 250
REGIME_PERSISTENCE = (0.99, 0.97)    # calm, stress
STRESS_VOL_MULT, T_DOF = 2.8, 5

# VaR IM / backtest settings
CONF, HORIZON_DAYS, MPOR_DAYS = 0.99, 10, 10
WINDOW_DAYS, MIN_STRESS_SHARE = 750, 0.25       # EU 2016/2251 Art 16 style calibration window
BACKTEST_WINDOW = 250
EWMA_LAMBDA = 0.97
CHALLENGER_SCALE = 0.7
QUICK_SCENARIOS = 250
DISPUTE_TOL_ABS, DISPUTE_TOL_REL = 1.0, 1e-9
BIG_MOVE_REL, BIG_MOVE_ABS = 0.10, 1.0e6
SCENARIO_DAY = pd.Timestamp("2026-07-10")        # COB where SIMM v2.8+2512 first applies

# SIMM sensitivity conventions
IR_TENORS = ("2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y")
IR_TENOR_YEARS = (14 / 365, 1 / 12, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30)
VEGA_EXPIRIES = ("2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y")
IR_BUMP_BP, FX_BUMP_REL, VOL_BUMP_ABS = 1.0, 0.01, 0.0001   # central differences use +/- half of these
# ===== END CONFIG =====
