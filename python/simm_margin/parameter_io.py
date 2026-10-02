"""Tiny readers for the verified parameter CSVs (data/parameters). No SIMM logic here."""
import csv
from pathlib import Path

import numpy as np

from simm_margin import columns as C
from simm_margin.config import PARAM_DIR, IR_TENORS

# ===== CONFIG (user inputs) =====
SIMM_PARAMS_FILE = "simm_parameters.csv"
SCHEDULE_FILE = "schedule_im.csv"
TRAFFIC_LIGHT_FILE = "bcbs22_traffic_light.csv"
REG_CONSTANTS_FILE = "regulatory_constants.csv"
# ===== END CONFIG =====


def read_csv(name, param_dir=None):
    """Return a CSV in the parameter folder as a list of dict rows (all values as strings)."""
    path = Path(param_dir or PARAM_DIR) / name
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def simm_rows(param_set, risk_class=None, param_name=None, param_dir=None):
    """Filter simm_parameters.csv rows by param_set and optionally risk class and parameter name."""
    out = []
    for r in read_csv(SIMM_PARAMS_FILE, param_dir):
        if r[C.PARAM_SET] != param_set:
            continue
        if risk_class is not None and r[C.RISK_CLASS] != risk_class:
            continue
        if param_name is not None and r[C.PARAM_NAME] != param_name:
            continue
        out.append(r)
    return out


def scalar(param_set, risk_class, param_name, key1="", key2="", param_dir=None):
    """Single numeric value; raises if the key is missing or ambiguous."""
    hits = [r for r in simm_rows(param_set, risk_class, param_name, param_dir)
            if r[C.KEY1] == key1 and r[C.KEY2] == key2]
    if len(hits) != 1:
        raise KeyError((param_set, risk_class, param_name, key1, key2, len(hits)))
    return float(hits[0][C.PARAM_VALUE])


def square_matrix(param_set, risk_class, param_name, keys, param_dir=None):
    """Dense matrix over the ordered keys from long-format key1/key2 rows."""
    pos = {k: i for i, k in enumerate(keys)}
    m = np.full((len(keys), len(keys)), np.nan)
    for r in simm_rows(param_set, risk_class, param_name, param_dir):
        m[pos[r[C.KEY1]], pos[r[C.KEY2]]] = float(r[C.PARAM_VALUE])
    if np.isnan(m).any():
        raise ValueError("incomplete matrix %s %s" % (param_set, param_name))
    return m


def ir_tenor_corr(param_set, param_dir=None):
    """12x12 IR tenor correlation matrix in IR_TENORS order."""
    return square_matrix(param_set, C.RC_IR, "delta_tenor_corr", IR_TENORS, param_dir)
