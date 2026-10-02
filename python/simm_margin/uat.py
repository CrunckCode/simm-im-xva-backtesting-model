"""New-product onboarding UAT for the EUR/USD cross-currency basis swap (SIMM-style, not ISDA-licensed or certified).

ALL TRADES AND MARKET DATA ARE SYNTHETIC. The suite is machine readable (data/uat/uat_suite.csv: test_id, area, description,
preconditions, input, expected, tolerance, actual, result, evidence_file, executed_at, executed_by). run_uat executes every
test against the synthetic market of config.SCENARIO_DAY, fills actual and result honestly (PASS, FAIL or SKIPPED) and
writes outputs/uat_results.csv. Hand-case expectations use the verified numbers in docs/VERIFICATION_LOG.md (row 1), not
the parameter loader, so they are independent of the code under test.
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import columns as C
from . import config
from . import dispute
from . import params as params_mod
from . import schedule_im
from . import sensitivities as sens
from . import simm as simm_mod
from .crif import CRIF_COLUMNS, build_crif
from .curves import TENOR_YEARS, par_to_zero_jacobian, payment_times
from .instruments import BASIS_SPREAD, EXCHANGE_NONE, new_product_xccy, sample_portfolio
from .market_history import generate_history
from .pricing import price_trade

# ===== CONFIG (user inputs) =====
SUITE_DIR = config.ROOT / "data" / "uat"
SUITE_FILE = "uat_suite.csv"
RESULTS_FILE = "uat_results.csv"
EVIDENCE_CRIF, EVIDENCE_DISPUTE = "uat_evidence_xccy_crif.csv", "uat_evidence_dispute_xccy.csv"
EXCEL_DIR, EXCEL_RECON_FILE = config.ROOT / "excel", "excel_reconciliation.json"
SUITE_COLUMNS = ["test_id", "area", "description", "preconditions", "input", "expected", "tolerance", "actual", "result",
                 "evidence_file", "executed_at", "executed_by"]
NOT_RUN, PASS, FAIL, SKIPPED = "NOT_RUN", "PASS", "FAIL", "SKIPPED"
EXECUTED_BY = "uat.run_uat (automated)"
PV_TOL_USD = 1.0                     # PV at par and leg-by-leg agreement, USD
REL_TOL = 1.0e-6                     # relative tolerance for closed-form comparisons
LADDER_REL_TOL = 1.0e-3              # ladder sum versus parallel PV01 (different bump paths)
RUNTIME_BUDGET_S = 5.0               # build the XCCY CRIF, add it to the book, compute SIMM and rank a dispute
HAND_RW_5Y, HAND_RW_XCCY, HAND_RHO_XCCY = 61.0, 21.0, -0.01     # VERIFICATION_LOG row 1 (v2.8+2512, EUR is regular)
HAND_YIELD_USD, HAND_BASIS_USD = 2.0e6, 1.0e6
BIG_BASIS_USD = 1.0e9                # forces any CR-sum leakage to show up
BIG_YIELD_USD = 5.0e8                # pushes EUR yield CR above 1
SCHEDULE_SOURCE_KEY = "12 CFR Part 45"
# ===== END CONFIG =====

SUITE = [
    ("U01", "pricing", "PV is zero when the contract basis equals the market basis and the initial FX equals spot",
     "XCCY booked at market basis and spot", "EUR/USD XCCY 50mm EUR 5y, basis = market basis, initial FX = spot",
     "PV = 0", f"abs {PV_TOL_USD} USD", ""),
    ("U02", "pricing", "Vectorised PV equals an independent leg-by-leg discounting of every cash flow",
     "off-market basis and initial FX", "50mm EUR 5y, basis -15bp, initial FX 1.08", "PV equals independent sum",
     f"abs {PV_TOL_USD} USD", ""),
    ("U03", "sensitivities", "Sum of the IR delta ladder equals the parallel PV01 from a full re-bootstrap",
     "IM-relevant PV (exchange excluded)", "EUR and USD curves", "ladder sum = parallel PV01",
     f"rel {LADDER_REL_TOL}", ""),
    ("U04", "sensitivities", "Basis sensitivity equals notional x annuity x FX x 1bp",
     "contractual basis on the EUR leg", "50mm EUR 5y", "notional * EUR annuity * EURUSD * 1e-4", f"rel {REL_TOL}", ""),
    ("U05", "crif", "CRIF columns, risk types, labels and USD conversion are well formed",
     "XCCY CRIF built on the scenario day", "CRIF of the XCCY alone",
     "CRIF columns plus SigmaMarket; only Risk_IRCurve, Risk_XCcyBasis and Risk_FX; one basis row",
     "exact", EVIDENCE_CRIF),
    ("U06", "simm", "XCcyBasis risk weight is applied with no concentration scaling",
     "EUR yield exposure large enough for CR > 1", "XCCY basis row plus a large EUR yield row",
     f"basis RW = {HAND_RW_XCCY:g}, CR = 1; yield CR > 1", "exact", ""),
    ("U07", "simm", "XCcyBasis is excluded from the concentration-risk sum",
     "same CRIF with and without a huge basis row", "basis row of 1e9 USD added", "yield CR unchanged", "abs 1e-12", ""),
    ("U08", "simm", "Basis-yield correlation -1% hand case for one currency bucket",
     "single EUR 5y yield row and one basis row, CR = 1", f"{HAND_YIELD_USD:g} USD yield, {HAND_BASIS_USD:g} USD basis",
     "K = sqrt(WS_y^2 + WS_b^2 - 0.02 WS_y WS_b)", f"rel {REL_TOL}", ""),
    ("U09", "crif", "Principal exchange exclusion removes exactly the principal FX delta",
     "same trade with exchange eligible and not eligible", "50mm EUR 5y XCCY",
     "FX delta(not eligible) - FX delta(eligible) = 1% * notional * DF_EUR(5y) * EURUSD", f"rel {REL_TOL}", ""),
    ("U10", "schedule", "Schedule IM uses the cross-currency swap row of 12 CFR Part 45 Appendix A",
     "XCCY of 5 years", "schedule_im on the XCCY alone", "rate from schedule_im.csv x USD notional", f"rel {REL_TOL}", ""),
    ("U11", "simm", "Sign symmetry of delta and vega margins with the XCCY added",
     "sample book plus XCCY", "CRIF amounts multiplied by -1", "delta and vega margins unchanged", f"rel {REL_TOL}", ""),
    ("U12", "simm", "Homogeneity: IM scales linearly when CR = 1 and at least linearly otherwise",
     "XCCY alone (CR = 1) and the full book", "CRIF amounts multiplied by 2", "IM(2x) = 2 IM(x); book delta(2x) >= 2 delta(x)",
     f"rel {REL_TOL}", ""),
    ("U13", "simm", "XCCY adds no vega or curvature margin",
     "sample book plus XCCY", "margins of the book with and without the XCCY", "vega and curvature margins unchanged",
     f"rel {REL_TOL}", ""),
    ("U14", "dispute", "A missing XCCY on the counterparty side is detected and ranked first",
     "party A books the XCCY, party B does not", "dispute.reconcile and rank_causes", f"rank 1 = {dispute.CAUSE_MISSING}",
     "exact", EVIDENCE_DISPUTE),
    ("U15", "excel", "Excel workbook rows reconcile to Python for the XCCY",
     "workbook and outputs/excel_reconciliation.json exist", "reconciliation report", "no failed checks", "per report", ""),
    ("U16", "performance", "Onboarding calculation runtime within budget",
     "scenario-day market", "build CRIF, SIMM for book plus XCCY, one dispute ranking", f"runtime <= {RUNTIME_BUDGET_S} s",
     f"{RUNTIME_BUDGET_S} s", ""),
    ("U17", "regression", "Existing portfolio IM is unchanged when the XCCY is not booked",
     "CRIF rows of the book with and without the XCCY", "all sample trades", "identical rows and identical IM",
     "exact rows; IM abs 1e-9", ""),
]


def write_suite(path=None) -> Path:
    """Write the static suite definition (actual, result and execution fields empty, result = NOT_RUN)."""
    path = Path(path) if path is not None else SUITE_DIR / SUITE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for tid, area, desc, pre, inp, exp, tol, ev in SUITE:
        rows.append({"test_id": tid, "area": area, "description": desc, "preconditions": pre, "input": inp, "expected": exp,
                     "tolerance": tol, "actual": "", "result": NOT_RUN, "evidence_file": ev if ev.endswith(".csv") else "",
                     "executed_at": "", "executed_by": ""})
    pd.DataFrame(rows, columns=SUITE_COLUMNS).to_csv(path, index=False)
    return path


# ---------- shared fixtures ----------
class _Fixture:
    def __init__(self, history, out_dir: Path):
        self.out = out_dir
        self.history = history
        self.ctx = dispute.make_context(history)
        self.row = self.ctx.row
        self.p = params_mod.load()
        self.book = sample_portfolio()
        self.spot = float(self.row.fx_spot["EUR"][0])
        self.mkt_basis = float(self.row.xccy_basis["EURUSD"][0])
        self.xccy = new_product_xccy()                                    # exchange eligible (IM-relevant)
        self.xccy_full = new_product_xccy(settle_notional_exchange=EXCHANGE_NONE)
        self.xr = self.xccy.iloc[0]
        self.xr_full = self.xccy_full.iloc[0]
        self.crif_x = build_crif(self.xccy, self.row)
        self.crif_book = build_crif(self.book, self.row)
        self.both = pd.concat([self.book, self.xccy], ignore_index=True)
        self.crif_both = build_crif(self.both, self.row)


def _num(x) -> str:
    return f"{x:.10g}"


def _close(a, b, rel=REL_TOL, absolute=0.0) -> bool:
    return abs(a - b) <= max(absolute, rel * max(abs(a), abs(b)))


# ---------- tests: each returns (passed, actual text, note) ----------
def _u01(f):
    t = new_product_xccy(initial_fx=f.spot, basis=f.mkt_basis).iloc[0]
    pv = float(price_trade(t, f.row, include_exchange=True)[0])
    return abs(pv) <= PV_TOL_USD, f"PV = {_num(pv)} USD at basis {f.mkt_basis * 1e4:.4f}bp, initial FX {f.spot:.6f}"


def _independent_pv(t, row) -> float:
    """Leg-by-leg PV with the final principal exchange, coded separately from curves.py/pricing.py."""
    def df(ccy, tt):
        return float(np.exp(-np.interp(tt, TENOR_YEARS, row.zero[ccy][0]) * tt))
    e, u, m = t[C.CCY], t[C.CCY2], t[C.MATURITY]
    n_e, n_u = t[C.NOTIONAL], t[C.NOTIONAL] * t[C.STRIKE]
    pay = payment_times(0.0, m)
    prev = np.concatenate(([0.0], pay[:-1]))
    leg = {}
    for ccy, n in ((e, n_e), (u, n_u)):
        coupons = sum(df(ccy, ti) * (df(ccy, tp) / df(ccy, ti) - 1.0) for tp, ti in zip(prev, pay))   # forward-rate coupons
        leg[ccy] = coupons * n + n * df(ccy, m)
    basis = (t[BASIS_SPREAD] - float(row.xccy_basis["EURUSD"][0])) * n_e * sum(df(e, ti) * (ti - tp) for tp, ti in zip(prev, pay))
    return t[C.DIRECTION] * ((leg[e] + basis) * float(row.fx_spot[e][0]) - leg[u] * float(row.fx_spot[u][0]))


def _u02(f):
    t = new_product_xccy(initial_fx=1.08, basis=-0.0015).iloc[0]
    a, b = float(price_trade(t, f.row, include_exchange=True)[0]), _independent_pv(t, f.row)
    return abs(a - b) <= PV_TOL_USD, f"vectorised {_num(a)} vs leg-by-leg {_num(b)} USD, difference {_num(a - b)}"


def _u03(f):
    jac = {c: par_to_zero_jacobian(f.row.par[c][0]) for c in ("EUR", "USD")}
    parts, ok = [], True
    for ccy in ("EUR", "USD"):
        ladder = float(sens.ir_delta(f.xr, f.row, ccy, jac[ccy]).sum())
        pv01 = sens.parallel_pv01(f.xr, f.row, ccy)
        ok &= abs(ladder - pv01) <= LADDER_REL_TOL * max(abs(pv01), abs(ladder), 1.0)
        parts.append(f"{ccy} ladder {_num(ladder)} vs PV01 {_num(pv01)}")
    return ok, "; ".join(parts)


def _u04(f):
    ann = float(f.row.curve("EUR").annuity(0.0, f.xr[C.MATURITY])[0])
    exp = f.xr[C.NOTIONAL] * ann * f.spot * 1e-4
    act = sens.xccy_basis_delta(f.xr, f.row)
    return _close(act, exp), f"bump-and-reprice {_num(act)} vs notional*annuity*FX*1bp {_num(exp)} USD per bp"


def _u05(f):
    c = f.crif_x
    cols_ok = list(c.columns) == CRIF_COLUMNS + [simm_mod.SIGMA_MARKET_COL]
    types = set(c[C.RISK_TYPE])
    types_ok = types <= {C.RISK_IRCURVE, C.RISK_XCCYBASIS, C.RISK_FX}
    one_basis = int((c[C.RISK_TYPE] == C.RISK_XCCYBASIS).sum()) == 1
    labels_ok = set(c.loc[c[C.RISK_TYPE] == C.RISK_IRCURVE, C.LABEL1]) <= set(config.IR_TENORS)
    spot = c[C.AMOUNT_CCY].map(lambda x: float(f.row.fx_spot[x][0]))
    usd_ok = bool(np.allclose(c[C.AMOUNT_USD], c[C.AMOUNT] * spot, rtol=1e-9, atol=1e-9))
    header_ok = bool((c[C.PRODUCT_CLASS] == C.RATES_FX).all() and (c[C.PORTFOLIO_ID] == f.xr[C.PORTFOLIO_ID]).all())
    c.to_csv(f.out / EVIDENCE_CRIF, index=False)
    ok = cols_ok and types_ok and one_basis and labels_ok and usd_ok and header_ok
    return ok, (f"{len(c)} rows; columns ok {cols_ok}; risk types {sorted(types)}; one basis row {one_basis}; "
                f"tenor labels ok {labels_ok}; AmountUSD = Amount*FX {usd_ok}; header fields ok {header_ok}")


def _basis_row(amount_usd):
    return pd.DataFrame([{C.CRIF_TRADE_ID: "T_BASIS", C.PORTFOLIO_ID: "NS_A", C.PRODUCT_CLASS: C.RATES_FX,
                          C.RISK_TYPE: C.RISK_XCCYBASIS, C.QUALIFIER: "EUR", C.BUCKET: "", C.LABEL1: "", C.LABEL2: "",
                          C.AMOUNT: amount_usd, C.AMOUNT_CCY: "EUR", C.AMOUNT_USD: amount_usd}])


def _yield_row(amount_usd, tenor="5y"):
    r = _basis_row(amount_usd)
    r[[C.CRIF_TRADE_ID, C.RISK_TYPE, C.LABEL1, C.LABEL2]] = ["T_YIELD", C.RISK_IRCURVE, tenor, "OIS"]
    return r


def _u06(f):
    crif = pd.concat([f.crif_x, _yield_row(BIG_YIELD_USD)], ignore_index=True)
    ws = simm_mod.weighted_sensitivities(crif, f.p)
    b = ws[ws["kind"] == simm_mod.KIND_XCCY].iloc[0]
    y = ws[(ws["kind"] == simm_mod.KIND_YIELD) & (ws["bucket"] == "EUR")]
    ok = b[C.RW] == HAND_RW_XCCY and b[C.CR] == 1.0 and (y[C.CR] > 1.0).all() and _close(b[C.WS], HAND_RW_XCCY * b[C.SENS])
    return ok, f"basis RW {b[C.RW]:g}, CR {b[C.CR]:g}, WS {_num(b[C.WS])}; yield CR {y[C.CR].max():.6g} (>1 so scaling is active)"


def _u07(f):
    base = pd.concat([f.crif_x[f.crif_x[C.RISK_TYPE] != C.RISK_XCCYBASIS], _yield_row(BIG_YIELD_USD)], ignore_index=True)
    with_basis = pd.concat([base, _basis_row(BIG_BASIS_USD)], ignore_index=True)
    def cr(c):
        ws = simm_mod.weighted_sensitivities(c, f.p)
        return float(ws.loc[ws["kind"] == simm_mod.KIND_YIELD, C.CR].iloc[0])
    a, b = cr(base), cr(with_basis)
    return abs(a - b) <= 1e-12, f"yield CR {a:.12g} without basis row, {b:.12g} with a {BIG_BASIS_USD:.0e} USD basis row"


def _u08(f):
    crif = pd.concat([_yield_row(HAND_YIELD_USD), _basis_row(HAND_BASIS_USD)], ignore_index=True)
    ws = simm_mod.weighted_sensitivities(crif, f.p)
    k = simm_mod.ir_bucket_K(ws, f.p)
    wy, wb = HAND_RW_5Y * HAND_YIELD_USD, HAND_RW_XCCY * HAND_BASIS_USD
    hand = float(np.sqrt(wy ** 2 + wb ** 2 + 2.0 * HAND_RHO_XCCY * wy * wb))
    return _close(k, hand), f"engine K {_num(k)} vs hand {_num(hand)} (WS_y {_num(wy)}, WS_b {_num(wb)}, rho {HAND_RHO_XCCY:g})"


def _u09(f):
    d_el = sens.fx_delta(f.xr, f.row, "EUR")
    d_full = sens.fx_delta(f.xr_full, f.row, "EUR")
    df5 = float(f.row.curve("EUR").df(f.xr[C.MATURITY])[0, 0])
    exp = 0.01 * f.xr[C.NOTIONAL] * df5 * f.spot
    return _close(d_full - d_el, exp), (f"FX delta full {_num(d_full)}, exchange excluded {_num(d_el)}; difference "
                                        f"{_num(d_full - d_el)} vs principal {_num(exp)} USD per 1%")


def _u10(f):
    pv = float(price_trade(f.xr, f.row, include_exchange=False)[0])
    res = schedule_im.schedule_im(f.xccy, [pv], fx_spot={c: float(v[0]) for c, v in f.row.fx_spot.items()})
    row = res["by_row"].iloc[0]
    rate = None
    with open(config.PARAM_DIR / "schedule_im.csv", newline="", encoding="utf-8") as fh:        # read the file directly
        for r in csv.DictReader(fh):
            if r["asset_class"] == "Cross-currency swap" and SCHEDULE_SOURCE_KEY in r["source_doc"]:
                lo, hi = float(r["maturity_bucket_low_years"]), r["maturity_bucket_high_years"]
                if lo <= f.xr[C.MATURITY] < (float("inf") if hi == "inf" else float(hi)):
                    rate = float(r["rate_pct"]) / 100.0
    exp_gross = rate * f.xr[C.NOTIONAL] * f.spot
    ok = row["asset_class"] == "Cross-currency swap" and _close(row["rate"], rate) and _close(res["gross"], exp_gross)
    return ok, (f"asset class {row['asset_class']!r}, rate {row['rate']:.4f} (file {rate:.4f}), gross IM {_num(res['gross'])} "
                f"vs {_num(exp_gross)} USD, net {_num(res['net'])}")


def _flip(c):
    out = c.copy()
    out[C.AMOUNT_USD] = -out[C.AMOUNT_USD]
    return out


def _u11(f):
    a, b = simm_mod.simm(f.crif_both, f.p), simm_mod.simm(_flip(f.crif_both), f.p)
    diffs, ok = [], True
    for key in ((C.RC_IR, C.MT_DELTA), (C.RC_FX, C.MT_DELTA), (C.RC_IR, C.MT_VEGA), (C.RC_FX, C.MT_VEGA)):
        x, y = a.by_margin_type[key], b.by_margin_type[key]
        ok &= _close(x, y)
        diffs.append(f"{key[0]} {key[1]} {_num(x)} vs {_num(y)}")
    return ok, "; ".join(diffs)


def _u12(f):
    x1, x2 = simm_mod.simm(f.crif_x, f.p).total, simm_mod.simm(_scale(f.crif_x, 2.0), f.p).total
    d1 = simm_mod.simm(f.crif_both, f.p)
    d2 = simm_mod.simm(_scale(f.crif_both, 2.0), f.p)
    dl1 = d1.by_margin_type[(C.RC_IR, C.MT_DELTA)] + d1.by_margin_type[(C.RC_FX, C.MT_DELTA)]
    dl2 = d2.by_margin_type[(C.RC_IR, C.MT_DELTA)] + d2.by_margin_type[(C.RC_FX, C.MT_DELTA)]
    ok = _close(x2, 2.0 * x1) and dl2 >= 2.0 * dl1 * (1.0 - REL_TOL)
    return ok, f"XCCY alone IM {_num(x1)} -> {_num(x2)} (x{x2 / x1:.8f}); book delta margins {_num(dl1)} -> {_num(dl2)} (x{dl2 / dl1:.6f})"


def _scale(c, k):
    out = c.copy()
    out[C.AMOUNT_USD] = out[C.AMOUNT_USD] * k
    return out


def _u13(f):
    a, b = simm_mod.simm(f.crif_book, f.p), simm_mod.simm(f.crif_both, f.p)
    parts, ok = [], True
    for key in ((C.RC_IR, C.MT_VEGA), (C.RC_FX, C.MT_VEGA), (C.RC_IR, C.MT_CURVATURE), (C.RC_FX, C.MT_CURVATURE)):
        ok &= _close(a.by_margin_type[key], b.by_margin_type[key])
        parts.append(f"{key[0]} {key[1]} {_num(a.by_margin_type[key])} -> {_num(b.by_margin_type[key])}")
    ok &= not f.crif_x[C.RISK_TYPE].isin([C.RISK_IRVOL, C.RISK_FXVOL]).any()
    return ok, "; ".join(parts)


def _u14(f):
    ctx = f.ctx
    a = dispute.PartyView("A", f.both, f.crif_both, f.p)
    kept = f.crif_both[f.crif_both[C.CRIF_TRADE_ID] != f.xr[C.TRADE_ID]]
    b = dispute.PartyView("B", f.book, kept, f.p)
    rep = dispute.reconcile(a, b, ctx, "xccy_missing", with_euler=False)
    rk = dispute.rank_causes(rep)
    rk.to_csv(f.out / EVIDENCE_DISPUTE, index=False)
    top = rk.iloc[0]
    ok = top[dispute.CAUSE_COL] == dispute.CAUSE_MISSING and abs(top[dispute.RESIDUAL]) <= config.DISPUTE_TOL_ABS
    return ok, (f"gap {_num(rep.gap)} USD (IM_A {_num(rep.im_a)}, IM_B {_num(rep.im_b)}); rank 1 = {top[dispute.CAUSE_COL]}, "
                f"residual {_num(top[dispute.RESIDUAL])}; trade-population break {f.xr[C.TRADE_ID]}")


def _u15(f):
    books = [p for p in EXCEL_DIR.glob("*.xlsx") if not p.name.startswith("~$")] if EXCEL_DIR.exists() else []
    if not books:
        return None, "workbook not built yet (no .xlsx under excel/)"
    rep = f.out / EXCEL_RECON_FILE
    if not rep.exists():
        return None, f"workbook {books[0].name} exists but {EXCEL_RECON_FILE} has not been produced"
    data = json.loads(rep.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("all_passed"), bool):
        return data["all_passed"], f"{EXCEL_RECON_FILE}: all_passed = {data['all_passed']} over {data.get('n_cases', '?')} cases"
    fails = _count_failures(data)
    if fails is None:
        return None, f"{EXCEL_RECON_FILE} found but its structure has no recognisable failure count"
    return fails == 0, f"{EXCEL_RECON_FILE}: {fails} failed checks"


def _count_failures(node):
    """Total of integer fields named like n_fail/failed/failures and of list fields named failures; None if absent."""
    found, total = False, 0
    if isinstance(node, dict):
        for k, v in node.items():
            lk = str(k).lower()
            if "fail" in lk and isinstance(v, (int, float)) and not isinstance(v, bool):
                found, total = True, total + int(v)
            elif "fail" in lk and isinstance(v, list):
                found, total = True, total + len(v)
            else:
                sub = _count_failures(v)
                if sub is not None:
                    found, total = True, total + sub
    elif isinstance(node, list):
        for v in node:
            sub = _count_failures(v)
            if sub is not None:
                found, total = True, total + sub
    return total if found else None


def _u16(f):
    t0 = time.perf_counter()
    crif_x = build_crif(f.xccy, f.row)
    crif_all = pd.concat([f.crif_book, crif_x], ignore_index=True)
    im = simm_mod.simm(crif_all, f.p).total
    a = dispute.PartyView("A", f.both, crif_all, f.p)
    b = dispute.PartyView("B", f.book, f.crif_book, f.p)
    dispute.rank_causes(dispute.reconcile(a, b, f.ctx, with_euler=False))
    dt = time.perf_counter() - t0
    return dt <= RUNTIME_BUDGET_S, f"{dt:.3f} s against budget {RUNTIME_BUDGET_S:g} s (book plus XCCY IM {_num(im)} USD)"


def _u17(f):
    ids = set(f.book[C.TRADE_ID])
    sub = f.crif_both[f.crif_both[C.CRIF_TRADE_ID].isin(ids)].reset_index(drop=True)
    same_rows = sub.equals(f.crif_book.reset_index(drop=True)) or (
        list(sub.columns) == list(f.crif_book.columns) and len(sub) == len(f.crif_book)
        and np.allclose(sub.select_dtypes("number").to_numpy(), f.crif_book.select_dtypes("number").to_numpy(), rtol=0, atol=1e-9, equal_nan=True))
    im_a, im_b = simm_mod.simm(f.crif_book, f.p).total, simm_mod.simm(sub, f.p).total
    return same_rows and abs(im_a - im_b) <= 1e-9, f"{len(sub)} existing CRIF rows identical {same_rows}; IM {_num(im_a)} vs {_num(im_b)} USD"


TESTS = {"U01": _u01, "U02": _u02, "U03": _u03, "U04": _u04, "U05": _u05, "U06": _u06, "U07": _u07, "U08": _u08, "U09": _u09,
         "U10": _u10, "U11": _u11, "U12": _u12, "U13": _u13, "U14": _u14, "U15": _u15, "U16": _u16, "U17": _u17}


def run_uat(out_dir=None, suite_path=None, history=None) -> pd.DataFrame:
    """Execute the suite and write uat_results.csv (plus small evidence csv files) to out_dir; returns the results."""
    out = Path(out_dir) if out_dir is not None else config.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    suite_path = Path(suite_path) if suite_path is not None else SUITE_DIR / SUITE_FILE
    if not suite_path.exists():
        write_suite(suite_path)
    suite = pd.read_csv(suite_path, dtype=str, keep_default_na=False)
    fx = _Fixture(history if history is not None else generate_history(), out)
    rows = []
    for _, r in suite.iterrows():
        r = r.to_dict()
        try:
            ok, actual = TESTS[r["test_id"]](fx)
            result = SKIPPED if ok is None else (PASS if ok else FAIL)
        except Exception as exc:                       # an erroring test is a failed test, never a silent skip
            result, actual = FAIL, f"error: {type(exc).__name__}: {exc}"
        r["actual"], r["result"] = actual, result
        r["executed_at"], r["executed_by"] = datetime.now().isoformat(timespec="seconds"), EXECUTED_BY
        if not r["evidence_file"] or not (out / r["evidence_file"]).exists():
            r["evidence_file"] = RESULTS_FILE
        rows.append(r)
    res = pd.DataFrame(rows, columns=SUITE_COLUMNS)
    res.to_csv(out / RESULTS_FILE, index=False)
    return res
