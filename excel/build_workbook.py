"""Builds simm_margin_workbook.xlsx: live SIMM-style RatesFX delta, vega and curvature, schedule IM, backtests and attribution.

Typed numbers live only on Inputs, Parameters, Parameters_Reg, Portfolio, the typed CRIF columns of Sensitivities, the typed
series of Backtest, the typed subset IMs of Attribution and Python_Ref. Every other number is a formula. SYNTHETIC data; SIMM-style,
not ISDA certified. Run via: py -3 -m simm_margin.reconcile_excel (build + LibreOffice recalculation + comparison).
"""
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter as L

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from simm_margin import columns as C  # noqa: E402
from simm_margin import config  # noqa: E402
from simm_margin import reconcile_excel as rx  # noqa: E402

# ===== CONFIG (user inputs) =====
AUTHOR = "Deepak Chaudhary"
SHEETS = ("README", "Inputs", "Parameters", "Parameters_Reg", "Portfolio", "Sensitivities", "SIMM_Delta", "SIMM_Vega",
          "SIMM_Curvature", "SIMM_Total", "Schedule_IM", "Backtest", "Attribution", "Checks", "Conclusions", "Python_Ref")
CALC_SHEETS = ("SIMM_Delta", "SIMM_Vega", "SIMM_Curvature", "SIMM_Total", "Schedule_IM", "Checks")
FIRST = 5                                   # first data row of typed tables
INF_YEARS = 1.0e9                           # stands for an open-ended maturity bucket
HELPER_MAX_X = 100                          # exception counts covered by the binomial helper table
TENOR_ROW_CORR = 8                          # first row of the 12x12 tenor correlation block on SIMM_Delta
DRIVER_BITS = (8, 4, 2, 1)                  # subset code weight of each Shapley driver
BOLD = Font(bold=True)
# ===== END CONFIG =====

NT, NC, NF, NP = len(config.IR_TENORS), len(rx.IR_CCYS), len(rx.FX_CCYS), len(rx.FX_PAIRS)


def ad(c, r):
    return f"${L(c)}${r}"


def X(sh, c, r):
    return f"{sh}!${L(c)}${r}"


def XR(sh, c1, r1, c2, r2):
    return f"{sh}!${L(c1)}${r1}:${L(c2)}${r2}"


def _val(x):
    if x is None or (isinstance(x, float) and x != x):
        return None
    return x.item() if hasattr(x, "item") else x


class Book:
    def __init__(self, inp, ref):
        self.inp, self.ref = inp, ref
        self.wb = openpyxl.Workbook()
        self.wb.remove(self.wb.active)
        self.ws = {n: self.wb.create_sheet(n) for n in SHEETS}
        self.wb.properties.creator = AUTHOR
        self.wb.properties.lastModifiedBy = AUTHOR
        self.checks, self.typed = [], {}
        self.I = {}

    def put(self, sh, r, c, v, bold=False):
        cell = self.ws[sh].cell(row=r, column=c, value=_val(v))
        if bold:
            cell.font = BOLD
        return cell

    def f(self, sh, r, c, expr, bold=False):
        return self.put(sh, r, c, "=" + expr, bold)

    def lab(self, sh, r, c, text):
        return self.put(sh, r, c, text, True)

    def mark_typed(self, sh, rng):
        self.typed.setdefault(sh, []).append(rng)

    def check(self, k, sh, r, c):
        self.checks.append(dict(key=k, sheet=sh, cell=f"{L(c)}{r}"))

    # parameter lookup by key
    def pv(self, rc, name, k1='""', k2='""', name_is_expr=False):
        nm = name if name_is_expr else f'"{name}"'
        return f'INDEX({self.PV},MATCH({self.I["set"]}&"|{rc}|"&{nm}&"|"&{k1}&"|"&{k2},{self.PK},0))'

    def member(self, rc, name, k1, ccy):
        return f'ISNUMBER(MATCH({self.I["set"]}&"|{rc}|{name}|{k1}|"&{ccy},{self.PK},0))'

    def ir_group(self, c):
        return f'IF({self.member("IR", "rw_group_member", "regular", c)},"regular",IF({self.member("IR", "rw_group_member", "low", c)},"low","high"))'

    def ir_ct_group(self, c):
        return (f'IF({self.member("IR", "ct_group_member", "regular_well_traded", c)},"regular_well_traded",'
                f'IF({self.member("IR", "ct_group_member", "regular_less_well_traded", c)},"regular_less_well_traded",'
                f'IF({self.member("IR", "ct_group_member", "low", c)},"low","high")))')

    def fx_group(self, c):
        return f'IF({self.member("FX", "high_vol_member", "high", c)},"high","regular")'

    def fx_cat(self, c):
        return (f'IF({self.member("FX", "ct_category_member", "Category 1", c)},1,'
                f'IF({self.member("FX", "ct_category_member", "Category 2", c)},2,3))')

    # input list cells
    def ccy(self, k):
        return X("Inputs", 2 + k, 13)

    def tenor(self, j):
        return X("Inputs", 2 + j, 14)

    def fxc(self, k):
        return X("Inputs", 2 + k, 16)

    def pair(self, k):
        return X("Inputs", 2 + k, 17)


# ---------------------------------------------------------------- typed sheets
def sheet_inputs(B):
    inp = B.inp
    s = "Inputs"
    B.lab(s, 1, 1, "Inputs (typed). Change a value here and the live sheets recalculate.")
    rows = [("SIMM parameter set", inp["param_set"], "set"), ("Calculation currency", config.CALC_CCY, "calc"),
            ("Exception probability p (1 minus confidence)", inp["p_exc"], "p"),
            ("Basel yellow zone cumulative probability", 0.95, "yellow"), ("Basel red zone cumulative probability", 0.9999, "red"),
            ("USD per million (thresholds are in USD mm)", 1.0e6, "units"), ("Basel window (observations)", rx.LAST_WINDOW, "last_n")]
    for i, (t, v, k) in enumerate(rows):
        B.lab(s, 4 + i, 1, t)
        B.put(s, 4 + i, 2, v)
        B.I[k] = X(s, 2, 4 + i)
    B.lab(s, 12, 1, "Lists used as labels by the live sheets")
    for r, (t, vals) in zip((13, 14, 15, 16, 17), (("IR currencies", rx.IR_CCYS), ("Tenor labels", config.IR_TENORS),
                                                   ("Tenor years", config.IR_TENOR_YEARS), ("FX currencies", rx.FX_CCYS),
                                                   ("FX pairs", rx.FX_PAIRS))):
        B.lab(s, r, 1, t)
        for j, v in enumerate(vals):
            B.put(s, r, 2 + j, v)
    B.lab(s, 19, 1, "Currency")
    B.lab(s, 19, 2, "USD per unit (schedule IM notionals)")
    for k, c in enumerate(rx.IR_CCYS):
        B.put(s, 20 + k, 1, c)
        B.put(s, 20 + k, 2, inp["fx_spot"][c])
    B.I["spot_ccy"], B.I["spot_val"] = XR(s, 1, 20, 1, 19 + NC), XR(s, 2, 20, 2, 19 + NC)
    B.lab(s, 27, 1, "Case")
    B.put(s, 27, 2, inp["case"]["name"])


def sheet_parameters(B):
    s = "Parameters"
    rows = B.inp["param_rows"]
    cols = [C.PARAM_SET, C.RISK_CLASS, C.PARAM_NAME, C.KEY1, C.KEY2, C.PARAM_VALUE, C.UNIT, C.SOURCE_DOC, C.VERSION, C.EFFECTIVE_DATE,
            C.SECTION, C.PARAGRAPH, C.TABLE, C.PDF_PAGE, C.SOURCE_URL, C.RETRIEVED_ON, C.SOURCE_SHA256, C.VERIFICATION_STATUS,
            C.VERIFICATION_METHOD]
    B.lab(s, 1, 1, "SIMM-style parameters (typed from data/parameters/simm_parameters.csv with provenance). Column A is the lookup key.")
    B.lab(s, 4, 1, "key")
    for j, c in enumerate(cols):
        B.lab(s, 4, 2 + j, c)
    for i, r in enumerate(rows):
        rr = FIRST + i
        B.f(s, rr, 1, f'B{rr}&"|"&C{rr}&"|"&D{rr}&"|"&E{rr}&"|"&F{rr}')
        for j, c in enumerate(cols):
            v = r[c]
            if c == C.PARAM_VALUE:
                v = float(v)
            if v != "" and v is not None:
                B.put(s, rr, 2 + j, v)
    last = FIRST + len(rows) - 1
    B.PK, B.PV = XR(s, 1, FIRST, 1, last), XR(s, 7, FIRST, 7, last)
    B.mark_typed(s, f"B{FIRST}:S{last}")


def sheet_parameters_reg(B):
    s = "Parameters_Reg"
    inp = B.inp
    B.lab(s, 1, 1, "Regulatory parameters (typed with provenance): schedule IM rates, weights, BCBS 22 Table 2.")
    for j, h in enumerate(("asset_class", "maturity_low_years", "maturity_high_years (1E9 means open ended)", "rate_pct", "source_doc",
                           "section", "verification_status", "retrieved_on")):
        B.lab(s, 4, 1 + j, h)
    rs = inp["schedule_rows"]
    for i, r in enumerate(rs):
        rr = FIRST + i
        lo = float(r["maturity_bucket_low_years"]) if r["maturity_bucket_low_years"] != "" else 0.0
        hi = r["maturity_bucket_high_years"]
        hi = INF_YEARS if hi in ("", "inf") else float(hi)
        for j, v in enumerate((r["asset_class"], lo, hi, float(r["rate_pct"]), r["source_doc"], r["section"],
                               r["verification_status"], r["retrieved_on"])):
            B.put(s, rr, 1 + j, v)
    last = FIRST + len(rs) - 1
    B.I["sch_asset"], B.I["sch_lo"] = XR(s, 1, FIRST, 1, last), XR(s, 2, FIRST, 2, last)
    B.I["sch_hi"], B.I["sch_rate"] = XR(s, 3, FIRST, 3, last), XR(s, 4, FIRST, 4, last)
    B.mark_typed(s, f"A{FIRST}:H{last}")
    for j, h in enumerate(("constant", "value", "source_doc", "citation", "verification_status")):
        B.lab(s, 4, 10 + j, h)
    for i, r in enumerate(inp["constants"]):
        rr = FIRST + i
        for j, v in enumerate((r["constant"], float(r["value"]), r["source_doc"], r["citation"], r["verification_status"])):
            B.put(s, rr, 10 + j, v)
        B.I[r["constant"]] = X(s, 11, rr)
    B.mark_typed(s, f"J{FIRST}:N{FIRST + len(inp['constants']) - 1}")
    tl = inp["bcbs"]
    for j, h in enumerate(("exceptions", "zone", "plus_factor", "source")):
        B.lab(s, 4, 16 + j, h)
    for i, r in tl.reset_index(drop=True).iterrows():
        rr = FIRST + i
        for j, v in enumerate((int(r["exceptions"]), str(r["zone"]).capitalize(), float(r["plus_factor"]), "BCBS 22 Table 2")):
            B.put(s, rr, 16 + j, v)
    B.I["tl_x"] = XR(s, 16, FIRST, 16, FIRST + len(tl) - 1)
    B.I["tl_plus"] = XR(s, 18, FIRST, 18, FIRST + len(tl) - 1)
    B.I["tl_max"] = int(tl["exceptions"].max())
    B.mark_typed(s, f"P{FIRST}:S{FIRST + len(tl) - 1}")


def sheet_portfolio(B):
    s = "Portfolio"
    port = B.inp["portfolio"]
    cols = [C.TRADE_ID, "counterparty", C.PRODUCT, C.CCY, C.CCY2, C.NOTIONAL, C.DIRECTION, C.START, C.MATURITY, C.EXPIRY, C.STRIKE,
            C.OPT_TYPE, "pv"]
    B.lab(s, 1, 1, "Sample portfolio and PV (typed from Python; SYNTHETIC). PV is in USD.")
    for j, c in enumerate(cols):
        B.lab(s, 4, 1 + j, c)
    for i, r in port.reset_index(drop=True).iterrows():
        for j, c in enumerate(cols):
            B.put(s, FIRST + i, 1 + j, r[c])
    last = FIRST + len(port) - 1
    B.n_trades, B.port_last = len(port), last
    B.mark_typed(s, f"A{FIRST}:M{last}")


def sheet_sensitivities(B):
    s = "Sensitivities"
    crif = B.inp["crif"].reset_index(drop=True)
    cols = [C.CRIF_TRADE_ID, C.RISK_TYPE, C.QUALIFIER, C.BUCKET, C.LABEL1, C.LABEL2, C.AMOUNT, C.AMOUNT_CCY, C.AMOUNT_USD, "SigmaMarket"]
    B.lab(s, 1, 1, "CRIF (typed from Python) with live helper columns, and closed-form Bachelier and Garman-Kohlhagen checks (columns O to P).")
    for j, c in enumerate(cols):
        B.lab(s, 4, 1 + j, c)
    for j, h in enumerate(("fx_vega_per_unit_vol_usd", "sf_of_vertex", "fx_vega_x_sf")):
        B.lab(s, 4, 11 + j, h)
    last = FIRST + len(crif) - 1
    B.crif_last = last
    sfr, sfl = XR("SIMM_Curvature", 4, 15, 4, 14 + NT), XR("SIMM_Curvature", 1, 15, 1, 14 + NT)
    for i, r in crif.iterrows():
        rr = FIRST + i
        for j, c in enumerate(cols):
            B.put(s, rr, 1 + j, r[c])
        B.f(s, rr, 11, f'IF(AND(B{rr}="{C.RISK_FXVOL}",ISNUMBER(J{rr})),I{rr}/J{rr},0)')
        B.f(s, rr, 12, f'IF(OR(B{rr}="{C.RISK_FXVOL}",B{rr}="{C.RISK_IRVOL}"),INDEX({sfr},MATCH(E{rr},{sfl},0)),0)')
        B.f(s, rr, 13, f"K{rr}*L{rr}")
    B.mark_typed(s, f"A{FIRST}:J{last}")
    B.S = {k: XR(s, c, FIRST, c, last) for k, c in (("tid", 1), ("rt", 2), ("q", 3), ("l1", 5), ("usd", 9), ("vpu", 11), ("vsf", 13))}
    # closed-form checks
    o = B.inp["options"]
    sw, fo = o["swaption"], o["fxoption"]
    B.lab(s, 4, 15, "Bachelier swaption (typed inputs, live formulas)")
    names = [("trade_id", sw["trade_id"]), ("notional", sw["notional"]), ("direction", sw["direction"]), ("forward swap rate", sw["forward"]),
             ("annuity", sw["annuity"]), ("strike", sw["strike"]), ("normal vol (bp)", sw["sigma_bp"]), ("expiry (years)", sw["expiry"]),
             ("fx to USD", sw["fx_to_usd"])]
    for i, (t, v) in enumerate(names):
        B.lab(s, 5 + i, 15, t)
        B.put(s, 5 + i, 16, v)
    B.mark_typed(s, "P5:P13")
    P = lambda r: f"$P${r}"
    live = [("normal vol (fraction)", f"{P(11)}/10000"), ("vol * sqrt(T)", f"{P(14)}*SQRT({P(12)})"),
            ("d", f"({P(8)}-{P(10)})/{P(15)}"), ("pdf(d)", f"EXP(-0.5*{P(16)}^2)/SQRT(2*PI())"),
            ("vega per 1bp (USD)", f"{P(6)}*{P(7)}*{P(9)}*SQRT({P(12)})*{P(17)}*0.0001*{P(13)}"),
            ("CRIF vega amount, closed form (USD)", f"{P(18)}*{P(11)}"),
            ("CRIF vega amount, typed CRIF (USD)", f'SUMIFS({B.S["usd"]},{B.S["tid"]},{P(5)},{B.S["rt"]},"{C.RISK_IRVOL}")'),
            ("PV, closed form payer (USD)", f"{P(6)}*{P(7)}*{P(9)}*(({P(8)}-{P(10)})*NORMSDIST({P(16)})+{P(15)}*{P(17)})*{P(13)}"),
            ("PV, typed portfolio (USD)", f'INDEX({XR("Portfolio", 13, FIRST, 13, B.port_last)},MATCH({P(5)},{XR("Portfolio", 1, FIRST, 1, B.port_last)},0))')]
    for i, (t, e) in enumerate(live):
        B.lab(s, 14 + i, 15, t)
        B.f(s, 14 + i, 16, e)
    B.check(rx.key("opt", "bachelier_vega_amount"), s, 19, 16)
    B.check(rx.key("opt", "bachelier_price"), s, 21, 16)
    B.lab(s, 26, 15, "Garman-Kohlhagen FX call (typed inputs, live formulas)")
    gnames = [("trade_id", fo["trade_id"]), ("notional (base)", fo["notional"]), ("direction", fo["direction"]), ("forward", fo["forward"]),
              ("strike", fo["strike"]), ("lognormal vol", fo["sigma"]), ("expiry (years)", fo["expiry"]),
              ("discount factor, quote ccy", fo["df_quote"]), ("discount factor, base ccy", fo["df_base"]),
              ("spot of base in USD", fo["spot_base"]), ("quote ccy to USD", fo["fx_quote_to_usd"])]
    for i, (t, v) in enumerate(gnames):
        B.lab(s, 27 + i, 15, t)
        B.put(s, 27 + i, 16, v)
    B.mark_typed(s, "P27:P37")
    G = lambda r: f"$P${r}"
    glive = [("vol * sqrt(T)", f"{G(32)}*SQRT({G(33)})"), ("d1", f"(LN({G(30)}/{G(31)})+0.5*{G(38)}^2)/{G(38)}"),
             ("d2", f"{G(39)}-{G(38)}"),
             ("PV, closed form (USD)", f"{G(28)}*{G(29)}*{G(34)}*({G(30)}*NORMSDIST({G(39)})-{G(31)}*NORMSDIST({G(40)}))*{G(37)}"),
             ("CRIF vega amount, closed form (USD)",
              f"{G(28)}*{G(29)}*{G(34)}*{G(30)}*EXP(-0.5*{G(39)}^2)/SQRT(2*PI())*SQRT({G(33)})*{G(32)}*{G(37)}"),
             ("CRIF FX delta per 1 percent, closed form (USD)", f"{G(28)}*{G(29)}*{G(35)}*NORMSDIST({G(39)})*{G(36)}*0.01")]
    for i, (t, e) in enumerate(glive):
        B.lab(s, 38 + i, 15, t)
        B.f(s, 38 + i, 16, e)
    B.check(rx.key("opt", "gk_price"), s, 41, 16)
    B.check(rx.key("opt", "gk_vega_amount"), s, 42, 16)
    B.check(rx.key("opt", "gk_delta_amount"), s, 43, 16)


# ---------------------------------------------------------------- SIMM sheets
def corr_cell(i, j):
    return X("SIMM_Delta", 2 + j, TENOR_ROW_CORR + i)


def sheet_delta(B):
    s = "SIMM_Delta"
    S, units = B.S, B.I["units"]
    B.lab(s, 1, 1, "IR and FX delta. All numbers are formulas: risk weights and thresholds by lookup on Parameters, CR, WS, K, S, aggregation.")
    B.lab(s, 4, 1, "Parameter set")
    B.f(s, 4, 2, B.I["set"])
    B.lab(s, 6, 1, "IR tenor correlation (12x12)")
    for j in range(NT):
        B.f(s, 7, 2 + j, B.tenor(j), True)
    for i in range(NT):
        B.f(s, TENOR_ROW_CORR + i, 1, B.tenor(i))
        for j in range(NT):
            B.f(s, TENOR_ROW_CORR + i, 2 + j,
                B.pv("IR", "delta_tenor_corr", f"$A{TENOR_ROW_CORR + i}", f"{L(2 + j)}$7"))
    B.blocks_delta = []
    for k in range(NC):
        r0 = 23 + k * 25
        B.blocks_delta.append(r0)
        for n, t in enumerate(("Currency", "RW group", "Concentration group", "Concentration threshold (USD mm per bp)",
                               "Sum of net sensitivities", "CR", "K", "Sum of WS", "S")):
            B.lab(s, r0 + n, 1, t)
        b = ad(2, r0)
        B.f(s, r0, 2, B.ccy(k))
        B.f(s, r0 + 1, 2, B.ir_group(b))
        B.f(s, r0 + 2, 2, B.ir_ct_group(b))
        B.f(s, r0 + 3, 2, B.pv("IR", "delta_ct", ad(2, r0 + 2)))
        t0, t1 = r0 + 10, r0 + 10 + NT - 1
        B.f(s, r0 + 4, 2, f"SUM(B{t0}:B{t1})")
        B.f(s, r0 + 5, 2, f"MAX(1,SQRT(ABS({ad(2, r0 + 4)})/{units}/{ad(2, r0 + 3)}))")
        B.f(s, r0 + 6, 2, f"SQRT(MAX(0,SUM(E{t0}:{L(4 + NT)}{t1})))")
        B.f(s, r0 + 7, 2, f"SUM(D{t0}:D{t1})")
        B.f(s, r0 + 8, 2, f"MAX(-{ad(2, r0 + 6)},MIN({ad(2, r0 + 6)},{ad(2, r0 + 7)}))")
        for n, t in enumerate(("Tenor", "Net sensitivity", "RW", "WS")):
            B.lab(s, r0 + 9, 1 + n, t)
        for j in range(NT):
            B.f(s, r0 + 9, 5 + j, B.tenor(j), True)
        for i in range(NT):
            rr = t0 + i
            B.f(s, rr, 1, B.tenor(i))
            B.f(s, rr, 2, f'SUMIFS({S["usd"]},{S["rt"]},"{C.RISK_IRCURVE}",{S["q"]},{b},{S["l1"]},$A{rr})')
            B.f(s, rr, 3, B.pv("IR", "delta_rw", ad(2, r0 + 1), f"$A{rr}"))
            B.f(s, rr, 4, f"C{rr}*B{rr}*{ad(2, r0 + 5)}")
            for j in range(NT):
                B.f(s, rr, 5 + j, f"{corr_cell(i, j)}*$D${rr}*$D${t0 + j}")
            B.check(rx.key("ws", "IR", rx.IR_CCYS[k], config.IR_TENORS[i]), s, rr, 4)
        ccy = rx.IR_CCYS[k]
        B.check(rx.key("cr", "IR", C.MT_DELTA, ccy), s, r0 + 5, 2)
        B.check(rx.key("k", "IR", C.MT_DELTA, ccy), s, r0 + 6, 2)
        B.check(rx.key("s", "IR", C.MT_DELTA, ccy), s, r0 + 8, 2)
    ra = 23 + NC * 25
    B.agg_delta = ra
    B.lab(s, ra, 1, "IR delta aggregation across currencies")
    for n, t in enumerate(("Currency", "K", "S", "CR")):
        B.lab(s, ra + 1 + n, 1, t)
    for k in range(NC):
        r0 = B.blocks_delta[k]
        B.f(s, ra + 1, 2 + k, B.ccy(k), True)
        B.f(s, ra + 2, 2 + k, ad(2, r0 + 6))
        B.f(s, ra + 3, 2 + k, ad(2, r0 + 8))
        B.f(s, ra + 4, 2 + k, ad(2, r0 + 5))
    B.lab(s, ra + 5, 1, "Cross-currency terms gamma * min(CR)/max(CR) * S_b * S_c")
    for i in range(NC):
        B.f(s, ra + 6 + i, 1, B.ccy(i))
        for j in range(NC):
            if i != j:
                ci, cj = L(2 + i), L(2 + j)
                B.f(s, ra + 6 + i, 2 + j,
                    f"{ad(2, ra + 12)}*MIN({ci}${ra + 4},{cj}${ra + 4})/MAX({ci}${ra + 4},{cj}${ra + 4})*{ci}${ra + 3}*{cj}${ra + 3}")
    B.lab(s, ra + 11, 1, "gamma")
    B.lab(s, ra + 12, 1, "gamma value")
    B.f(s, ra + 12, 2, B.pv("IR", "gamma"))
    B.lab(s, ra + 13, 1, "Sum of K squared")
    B.f(s, ra + 13, 2, f"SUMPRODUCT(B{ra + 2}:{L(1 + NC)}{ra + 2},B{ra + 2}:{L(1 + NC)}{ra + 2})")
    B.lab(s, ra + 14, 1, "Sum of cross terms")
    B.f(s, ra + 14, 2, f"SUM(B{ra + 6}:{L(1 + NC)}{ra + 5 + NC})")
    B.lab(s, ra + 15, 1, "IR delta margin")
    B.f(s, ra + 15, 2, f"SQRT(MAX(0,B{ra + 13}+B{ra + 14}))")
    B.ir_delta_cell = (s, ra + 15, 2)
    B.check(rx.key("margin", "IR", C.MT_DELTA), s, ra + 15, 2)
    # FX delta
    rf = ra + 19
    B.lab(s, rf, 1, "FX delta (single bucket)")
    B.lab(s, rf + 1, 1, "Calculation currency volatility group")
    B.f(s, rf + 1, 2, B.fx_group(B.I["calc"]))
    B.lab(s, rf + 2, 1, "Correlation table")
    B.f(s, rf + 2, 2, f'IF({ad(2, rf + 1)}="high","delta_corr_calc_high","delta_corr_calc_regular")')
    for n, t in enumerate(("FX currency", "Net sensitivity", "Vol group", "RW", "Category", "Threshold (USD mm per 1 percent)", "CR", "WS")):
        B.lab(s, rf + 3, 1 + n, t)
    t0 = rf + 4
    for j in range(NF):
        B.f(s, rf + 3, 10 + j, B.fxc(j), True)
    for i in range(NF):
        rr = t0 + i
        B.f(s, rr, 1, B.fxc(i))
        B.f(s, rr, 2, f'SUMIFS({S["usd"]},{S["rt"]},"{C.RISK_FX}",{S["q"]},$A{rr})')
        B.f(s, rr, 3, B.fx_group(f"$A{rr}"))
        B.f(s, rr, 4, B.pv("FX", "delta_rw", f"$C{rr}", ad(2, rf + 1)))
        B.f(s, rr, 5, B.fx_cat(f"$A{rr}"))
        B.f(s, rr, 6, B.pv("FX", "delta_ct", f'"Category "&$E{rr}'))
        B.f(s, rr, 7, f"MAX(1,SQRT(ABS(B{rr})/{units}/F{rr}))")
        B.f(s, rr, 8, f"D{rr}*B{rr}*G{rr}")
        for j in range(NF):
            rj = t0 + j
            if i == j:
                B.f(s, rr, 10 + j, f"$H{rr}^2")
            else:
                B.f(s, rr, 10 + j,
                    f'{B.pv("FX", ad(2, rf + 2), f"$C{rr}", f"$C{rj}", True)}*MIN($G{rr},$G{rj})/MAX($G{rr},$G{rj})*$H{rr}*$H{rj}')
        B.check(rx.key("ws", "FX", rx.FX_CCYS[i]), s, rr, 8)
        B.check(rx.key("cr", "FX", C.MT_DELTA, rx.FX_CCYS[i]), s, rr, 7)
    B.lab(s, t0 + NF, 1, "K (FX delta margin)")
    B.f(s, t0 + NF, 2, f"SQRT(MAX(0,SUM(J{t0}:{L(9 + NF)}{t0 + NF - 1})))")
    B.fx_delta_cell = (s, t0 + NF, 2)
    B.check(rx.key("k", "FX", C.MT_DELTA), s, t0 + NF, 2)
    B.check(rx.key("margin", "FX", C.MT_DELTA), s, t0 + NF, 2)


def sheet_vega(B):
    s = "SIMM_Vega"
    S, units = B.S, B.I["units"]
    B.lab(s, 1, 1, "IR vega and FX vega (formulas). IR vega uses the delta tenor correlation with f = 1; FX vega uses sigma_SIMM from RW.")
    B.lab(s, 3, 1, "IR vega risk weight")
    B.f(s, 3, 2, B.pv("IR", "vega_rw"))
    B.lab(s, 4, 1, "IR gamma")
    B.f(s, 4, 2, B.pv("IR", "gamma"))
    B.blocks_vega = []
    for k in range(NC):
        r0 = 8 + k * 24
        B.blocks_vega.append(r0)
        for n, t in enumerate(("Currency", "Concentration group", "Vega threshold (USD mm)", "Sum of net vega*vol", "VCR", "K", "Sum of VR", "S")):
            B.lab(s, r0 + n, 1, t)
        b = ad(2, r0)
        B.f(s, r0, 2, B.ccy(k))
        B.f(s, r0 + 1, 2, B.ir_ct_group(b))
        B.f(s, r0 + 2, 2, B.pv("IR", "vega_ct", ad(2, r0 + 1)))
        t0, t1 = r0 + 9, r0 + 9 + NT - 1
        B.f(s, r0 + 3, 2, f"SUM(B{t0}:B{t1})")
        B.f(s, r0 + 4, 2, f"MAX(1,SQRT(ABS({ad(2, r0 + 3)})/{units}/{ad(2, r0 + 2)}))")
        B.f(s, r0 + 5, 2, f"SQRT(MAX(0,SUM(D{t0}:{L(3 + NT)}{t1})))")
        B.f(s, r0 + 6, 2, f"SUM(C{t0}:C{t1})")
        B.f(s, r0 + 7, 2, f"MAX(-{ad(2, r0 + 5)},MIN({ad(2, r0 + 5)},{ad(2, r0 + 6)}))")
        for n, t in enumerate(("Tenor", "Net vega*vol (USD)", "VR")):
            B.lab(s, r0 + 8, 1 + n, t)
        for j in range(NT):
            B.f(s, r0 + 8, 4 + j, B.tenor(j), True)
        for i in range(NT):
            rr = t0 + i
            B.f(s, rr, 1, B.tenor(i))
            B.f(s, rr, 2, f'SUMIFS({S["usd"]},{S["rt"]},"{C.RISK_IRVOL}",{S["q"]},{b},{S["l1"]},$A{rr})')
            B.f(s, rr, 3, f"$B$3*B{rr}*{ad(2, r0 + 4)}")
            for j in range(NT):
                B.f(s, rr, 4 + j, f"{corr_cell(i, j)}*$C${rr}*$C${t0 + j}")
            B.check(rx.key("vr", "IR", rx.IR_CCYS[k], config.IR_TENORS[i]), s, rr, 3)
        ccy = rx.IR_CCYS[k]
        B.check(rx.key("cr", "IR", C.MT_VEGA, ccy), s, r0 + 4, 2)
        B.check(rx.key("k", "IR", C.MT_VEGA, ccy), s, r0 + 5, 2)
        B.check(rx.key("s", "IR", C.MT_VEGA, ccy), s, r0 + 7, 2)
    ra = 8 + NC * 24
    B.lab(s, ra, 1, "IR vega aggregation across currencies")
    for n, t in enumerate(("Currency", "K", "S", "VCR")):
        B.lab(s, ra + 1 + n, 1, t)
    for k in range(NC):
        r0 = B.blocks_vega[k]
        B.f(s, ra + 1, 2 + k, B.ccy(k), True)
        B.f(s, ra + 2, 2 + k, ad(2, r0 + 5))
        B.f(s, ra + 3, 2 + k, ad(2, r0 + 7))
        B.f(s, ra + 4, 2 + k, ad(2, r0 + 4))
    for i in range(NC):
        B.f(s, ra + 5 + i, 1, B.ccy(i))
        for j in range(NC):
            if i != j:
                ci, cj = L(2 + i), L(2 + j)
                B.f(s, ra + 5 + i, 2 + j,
                    f"$B$4*MIN({ci}${ra + 4},{cj}${ra + 4})/MAX({ci}${ra + 4},{cj}${ra + 4})*{ci}${ra + 3}*{cj}${ra + 3}")
    B.lab(s, ra + 10, 1, "Sum of K squared")
    B.f(s, ra + 10, 2, f"SUMPRODUCT(B{ra + 2}:{L(1 + NC)}{ra + 2},B{ra + 2}:{L(1 + NC)}{ra + 2})")
    B.lab(s, ra + 11, 1, "Sum of cross terms")
    B.f(s, ra + 11, 2, f"SUM(B{ra + 5}:{L(1 + NC)}{ra + 4 + NC})")
    B.lab(s, ra + 12, 1, "IR vega margin")
    B.f(s, ra + 12, 2, f"SQRT(MAX(0,B{ra + 10}+B{ra + 11}))")
    B.ir_vega_cell = (s, ra + 12, 2)
    B.check(rx.key("margin", "IR", C.MT_VEGA), s, ra + 12, 2)
    # FX vega
    rf = ra + 15
    B.lab(s, rf, 1, "FX vega (single bucket, pair-level factors)")
    for n, t in enumerate(("HVR_FX", "VRW_FX", "Vol and curvature correlation", "SIMM sigma days numerator", "SIMM sigma days denominator",
                           "SIMM sigma quantile")):
        B.lab(s, rf + 1 + n, 1, t)
    for n, (nm, k1) in enumerate((("hvr", None), ("vega_rw", None), ("vol_curvature_corr", None))):
        B.f(s, rf + 1 + n, 2, B.pv("FX", nm))
    for n, nm in enumerate(("vega_sigma_days_numerator", "vega_sigma_days_denominator", "vega_sigma_quantile")):
        B.f(s, rf + 4 + n, 2, B.pv("ALL", nm))
    B.fxv_params = dict(hvr=ad(2, rf + 1), vrw=ad(2, rf + 2), rho=ad(2, rf + 3))
    hdr = rf + 8
    for n, t in enumerate(("Pair", "Base", "Quote", "Base group", "Quote group", "RW", "sigma_SIMM", "Net vega per unit vol (USD)",
                           "sigma_SIMM * vega", "VR_ik = HVR * that", "Base category", "Quote category", "Threshold (USD mm)", "VCR", "VR")):
        B.lab(s, hdr, 1 + n, t)
    t0 = hdr + 1
    for j in range(NP):
        B.f(s, hdr, 17 + j, B.pair(j), True)
    for i in range(NP):
        rr = t0 + i
        B.f(s, rr, 1, B.pair(i))
        B.f(s, rr, 2, f"LEFT(A{rr},3)")
        B.f(s, rr, 3, f"RIGHT(A{rr},3)")
        B.f(s, rr, 4, B.fx_group(f"$B{rr}"))
        B.f(s, rr, 5, B.fx_group(f"$C{rr}"))
        B.f(s, rr, 6, B.pv("FX", "delta_rw", f"$D{rr}", f"$E{rr}"))
        B.f(s, rr, 7, f"F{rr}*SQRT({ad(2, rf + 4)}/{ad(2, rf + 5)})/NORMSINV({ad(2, rf + 6)})/100")
        B.f(s, rr, 8, f'SUMIFS({S["vpu"]},{S["rt"]},"{C.RISK_FXVOL}",{S["q"]},$A{rr})')
        B.f(s, rr, 9, f"H{rr}*G{rr}")
        B.f(s, rr, 10, f"{B.fxv_params['hvr']}*I{rr}")
        B.f(s, rr, 11, B.fx_cat(f"$B{rr}"))
        B.f(s, rr, 12, B.fx_cat(f"$C{rr}"))
        B.f(s, rr, 13, B.pv("FX", "vega_ct", f'"Category "&MIN($K{rr},$L{rr})', f'"Category "&MAX($K{rr},$L{rr})'))
        B.f(s, rr, 14, f"MAX(1,SQRT(ABS(J{rr})/{units}/M{rr}))")
        B.f(s, rr, 15, f"{B.fxv_params['vrw']}*J{rr}*N{rr}")
        for j in range(NP):
            rj = t0 + j
            if i == j:
                B.f(s, rr, 17 + j, f"$O{rr}^2")
            else:
                B.f(s, rr, 17 + j, f"{B.fxv_params['rho']}*MIN($N{rr},$N{rj})/MAX($N{rr},$N{rj})*$O{rr}*$O{rj}")
        B.check(rx.key("sigma", rx.FX_PAIRS[i]), s, rr, 7)
        B.check(rx.key("vr", "FX", rx.FX_PAIRS[i]), s, rr, 15)
    B.lab(s, t0 + NP, 1, "K (FX vega margin)")
    B.f(s, t0 + NP, 2, f"SQRT(MAX(0,SUM(Q{t0}:{L(16 + NP)}{t0 + NP - 1})))")
    B.fx_vega_cell = (s, t0 + NP, 2)
    B.sigma_cells = [X(s, 7, t0 + i) for i in range(NP)]
    B.check(rx.key("k", "FX", C.MT_VEGA), s, t0 + NP, 2)
    B.check(rx.key("margin", "FX", C.MT_VEGA), s, t0 + NP, 2)


def sheet_curvature(B):
    s = "SIMM_Curvature"
    S = B.S
    B.lab(s, 1, 1, "Curvature (formulas): CVR = SF(t) * sigma*vega, theta, lambda, rho squared and gamma squared aggregation.")
    for n, t in enumerate(("SF reference days", "SF scale", "Days per year", "Lambda quantile", "Phi inverse of lambda quantile",
                           "FX vol correlation", "IR gamma", "HVR_IR curvature exponent", "HVR_IR", "IR curvature scale HVR^exponent")):
        B.lab(s, 3 + n, 1, t)
    B.f(s, 3, 2, B.pv("ALL", "curvature_sf_ref_days"))
    B.f(s, 4, 2, B.pv("ALL", "curvature_sf_scale"))
    B.f(s, 5, 2, B.pv("ALL", "tenor_days_per_year"))
    B.f(s, 6, 2, B.pv("ALL", "curvature_lambda_quantile"))
    B.f(s, 7, 2, "NORMSINV(B6)")
    B.f(s, 8, 2, B.pv("FX", "vol_curvature_corr"))
    B.f(s, 9, 2, B.pv("IR", "gamma"))
    B.f(s, 10, 2, B.pv("IR", "curvature_hvr_exponent"))
    B.f(s, 11, 2, B.pv("IR", "hvr"))
    B.f(s, 12, 2, "B11^B10")
    for n, t in enumerate(("Tenor", "Years", "Calendar days", "SF")):
        B.lab(s, 14, 1 + n, t)
    for i in range(NT):
        rr = 15 + i
        B.f(s, rr, 1, B.tenor(i))
        B.f(s, rr, 2, X("Inputs", 2 + i, 15))
        B.f(s, rr, 3, "$B$3" if i == 0 else f"B{rr}*$B$5")
        B.f(s, rr, 4, f"$B$4*MIN(1,$B$3/MAX(C{rr},1E-12))")
        B.check(rx.key("sf", config.IR_TENORS[i]), s, rr, 4)
    # IR table
    h = 29
    B.lab(s, h - 1, 1, "IR curvature CVR_k = SF(t) * net vega*vol, by tenor and currency")
    B.lab(s, h, 1, "Tenor")
    for k in range(NC):
        B.f(s, h, 2 + k, B.ccy(k), True)
    for i in range(NT):
        rr = h + 1 + i
        B.f(s, rr, 1, B.tenor(i))
        for k in range(NC):
            B.f(s, rr, 2 + k, f"$D${15 + i}*{X('SIMM_Vega', 2, B.blocks_vega[k] + 9 + i)}")
            B.check(rx.key("c", "IR", rx.IR_CCYS[k], config.IR_TENORS[i]), s, rr, 2 + k)
    tl, br = f"B{h + 1}", f"{L(1 + NC)}{h + NT}"
    B.lab(s, h + 13, 1, "Sum of CVR (all currencies)")
    B.f(s, h + 13, 2, f"SUM({tl}:{br})")
    B.lab(s, h + 14, 1, "Sum of absolute CVR")
    B.f(s, h + 14, 2, f"SUMPRODUCT(ABS({tl}:{br}))")
    B.lab(s, h + 15, 1, "theta")
    B.f(s, h + 15, 2, f"IF(B{h + 14}=0,0,MIN(B{h + 13}/B{h + 14},0))")
    B.lab(s, h + 16, 1, "lambda")
    B.f(s, h + 16, 2, f"($B$7^2-1)*(1+B{h + 15})-B{h + 15}")
    B.check(rx.key("theta", "IR"), s, h + 15, 2)
    B.check(rx.key("lambda", "IR"), s, h + 16, 2)
    # per-currency blocks
    base = h + 20
    kcells, scells = [], []
    for k in range(NC):
        r0 = base + k * 16
        B.lab(s, r0, 1, "Currency")
        B.f(s, r0, 2, B.ccy(k))
        B.lab(s, r0 + 1, 1, "K")
        B.lab(s, r0 + 2, 1, "S")
        B.lab(s, r0 + 3, 1, "Tenor")
        B.lab(s, r0 + 3, 2, "CVR")
        t0, t1 = r0 + 4, r0 + 4 + NT - 1
        B.f(s, r0 + 1, 2, f"SQRT(MAX(0,SUM(C{t0}:{L(2 + NT)}{t1})))")
        B.f(s, r0 + 2, 2, f"MAX(-B{r0 + 1},MIN(B{r0 + 1},SUM(B{t0}:B{t1})))")
        for j in range(NT):
            B.f(s, r0 + 3, 3 + j, B.tenor(j), True)
        for i in range(NT):
            rr = t0 + i
            B.f(s, rr, 1, B.tenor(i))
            B.f(s, rr, 2, f"{L(2 + k)}{h + 1 + i}")
            for j in range(NT):
                B.f(s, rr, 3 + j, f"{corr_cell(i, j)}^2*$B${rr}*$B${t0 + j}")
        kcells.append(ad(2, r0 + 1))
        scells.append(ad(2, r0 + 2))
        B.check(rx.key("k", "IR", C.MT_CURVATURE, rx.IR_CCYS[k]), s, r0 + 1, 2)
        B.check(rx.key("s", "IR", C.MT_CURVATURE, rx.IR_CCYS[k]), s, r0 + 2, 2)
    ra = base + NC * 16
    B.lab(s, ra, 1, "IR curvature aggregation")
    B.lab(s, ra + 1, 1, "Currency")
    B.lab(s, ra + 2, 1, "K")
    B.lab(s, ra + 3, 1, "S")
    for k in range(NC):
        B.f(s, ra + 1, 2 + k, B.ccy(k), True)
        B.f(s, ra + 2, 2 + k, kcells[k])
        B.f(s, ra + 3, 2 + k, scells[k])
    for i in range(NC):
        B.f(s, ra + 4 + i, 1, B.ccy(i))
        for j in range(NC):
            if i != j:
                B.f(s, ra + 4 + i, 2 + j, f"$B$9^2*{L(2 + i)}${ra + 3}*{L(2 + j)}${ra + 3}")
    rng = f"B{ra + 2}:{L(1 + NC)}{ra + 2}"
    B.lab(s, ra + 9, 1, "Aggregate sqrt(sum K squared + cross terms)")
    B.f(s, ra + 9, 2, f"SQRT(MAX(0,SUMPRODUCT({rng},{rng})+SUM(B{ra + 4}:{L(1 + NC)}{ra + 3 + NC})))")
    B.lab(s, ra + 10, 1, "IR curvature margin")
    B.f(s, ra + 10, 2, f"MAX(B{h + 13}+B{h + 16}*B{ra + 9},0)*$B$12")
    B.ir_curv_cell = (s, ra + 10, 2)
    B.check(rx.key("margin", "IR", C.MT_CURVATURE), s, ra + 10, 2)
    # FX curvature
    rf = ra + 14
    B.lab(s, rf, 1, "FX curvature by pair")
    B.lab(s, rf + 1, 1, "Pair")
    B.lab(s, rf + 1, 2, "CVR")
    for j in range(NP):
        B.f(s, rf + 1, 3 + j, B.pair(j), True)
    for i in range(NP):
        rr = rf + 2 + i
        B.f(s, rr, 1, B.pair(i))
        B.f(s, rr, 2, f'SUMIFS({S["vsf"]},{S["rt"]},"{C.RISK_FXVOL}",{S["q"]},$A{rr})*{B.sigma_cells[i]}')
        for j in range(NP):
            if i == j:
                B.f(s, rr, 3 + j, f"$B{rr}^2")
            else:
                B.f(s, rr, 3 + j, f"$B$8^2*$B{rr}*$B{rf + 2 + j}")
        B.check(rx.key("c", "FX", rx.FX_PAIRS[i]), s, rr, 2)
    t0, t1 = rf + 2, rf + 1 + NP
    B.lab(s, rf + 2 + NP, 1, "Sum of CVR")
    B.f(s, rf + 2 + NP, 2, f"SUM(B{t0}:B{t1})")
    B.lab(s, rf + 3 + NP, 1, "Sum of absolute CVR")
    B.f(s, rf + 3 + NP, 2, f"SUMPRODUCT(ABS(B{t0}:B{t1}))")
    B.lab(s, rf + 4 + NP, 1, "theta")
    B.f(s, rf + 4 + NP, 2, f"IF(B{rf + 3 + NP}=0,0,MIN(B{rf + 2 + NP}/B{rf + 3 + NP},0))")
    B.lab(s, rf + 5 + NP, 1, "lambda")
    B.f(s, rf + 5 + NP, 2, f"($B$7^2-1)*(1+B{rf + 4 + NP})-B{rf + 4 + NP}")
    B.lab(s, rf + 6 + NP, 1, "K")
    B.f(s, rf + 6 + NP, 2, f"SQRT(MAX(0,SUM(C{t0}:{L(2 + NP)}{t1})))")
    B.lab(s, rf + 7 + NP, 1, "FX curvature margin")
    B.f(s, rf + 7 + NP, 2, f"MAX(B{rf + 2 + NP}+B{rf + 5 + NP}*B{rf + 6 + NP},0)")
    B.fx_curv_cell = (s, rf + 7 + NP, 2)
    B.check(rx.key("theta", "FX"), s, rf + 4 + NP, 2)
    B.check(rx.key("lambda", "FX"), s, rf + 5 + NP, 2)
    B.check(rx.key("k", "FX", C.MT_CURVATURE), s, rf + 6 + NP, 2)
    B.check(rx.key("margin", "FX", C.MT_CURVATURE), s, rf + 7 + NP, 2)


def sheet_total(B):
    s = "SIMM_Total"
    B.lab(s, 1, 1, "Risk class and RatesFX product class aggregation (formulas).")
    B.lab(s, 3, 1, "Margin type")
    B.lab(s, 3, 2, "IR")
    B.lab(s, 3, 3, "FX")
    cells = [(B.ir_delta_cell, B.fx_delta_cell), (B.ir_vega_cell, B.fx_vega_cell), (B.ir_curv_cell, B.fx_curv_cell)]
    for i, (mt, (a, b)) in enumerate(zip((C.MT_DELTA, C.MT_VEGA, C.MT_CURVATURE), cells)):
        B.lab(s, 4 + i, 1, mt)
        B.f(s, 4 + i, 2, X(*a[:1], a[2], a[1]))
        B.f(s, 4 + i, 3, X(*b[:1], b[2], b[1]))
    B.lab(s, 7, 1, "Risk class IM (delta + vega + curvature)")
    B.f(s, 7, 2, "SUM(B4:B6)")
    B.f(s, 7, 3, "SUM(C4:C6)")
    B.lab(s, 8, 1, "psi (IR, FX)")
    B.f(s, 8, 2, B.pv("ALL", "psi", '"IR"', '"FX"'))
    B.lab(s, 9, 1, "Multiplicative scale (RatesFX)")
    B.f(s, 9, 2, B.pv("ALL", "multiplicative_scale_default", '"RatesFX"'))
    B.lab(s, 10, 1, "RatesFX SIMM-style total IM (USD)")
    B.f(s, 10, 2, "SQRT(MAX(0,B7^2+C7^2+2*B8*B7*C7))*B9")
    B.total_cell = X(s, 2, 10)
    B.check(rx.key("im", "IR"), s, 7, 2)
    B.check(rx.key("im", "FX"), s, 7, 3)
    B.check(rx.key("total"), s, 10, 2)


def sheet_schedule(B):
    s = "Schedule_IM"
    n, I = B.n_trades, B.I
    B.lab(s, 1, 1, "Standardised (schedule) IM: gross = sum of rate * notional, net = 0.4 G + 0.6 NGR G (formulas).")
    for j, t in enumerate(("Trade", "Product", "Asset class", "Duration (years)", "Notional USD", "Rate", "Gross IM", "MTM (USD)")):
        B.lab(s, 6, 1 + j, t)
    t0 = 7
    P = "Portfolio"
    for i in range(n):
        rr, pr = t0 + i, FIRST + i
        B.f(s, rr, 1, X(P, 1, pr))
        B.f(s, rr, 2, X(P, 3, pr))
        B.f(s, rr, 3, f'IF(OR(B{rr}="{C.PRODUCT_IRS}",B{rr}="{C.PRODUCT_SWAPTION}"),"{rx.sched.ASSET_IR}",'
                      f'IF(OR(B{rr}="{C.PRODUCT_FXFWD}",B{rr}="{C.PRODUCT_FXOPT}"),"{rx.sched.ASSET_FX}","{rx.sched.ASSET_XCCY}"))')
        B.f(s, rr, 4, X(P, 9, pr))
        B.f(s, rr, 5, f'ABS({X(P, 6, pr)})*INDEX({I["spot_val"]},MATCH({X(P, 4, pr)},{I["spot_ccy"]},0))')
        B.f(s, rr, 6, f'SUMIFS({I["sch_rate"]},{I["sch_asset"]},C{rr},{I["sch_lo"]},"<="&D{rr},{I["sch_hi"]},">"&D{rr})/100')
        B.f(s, rr, 7, f"E{rr}*F{rr}")
        B.f(s, rr, 8, X(P, 13, pr))
        B.check(rx.key("sched", "rate", B.inp["portfolio"].iloc[i][C.TRADE_ID]), s, rr, 6)
        B.check(rx.key("sched", "gross_im", B.inp["portfolio"].iloc[i][C.TRADE_ID]), s, rr, 7)
    t1 = t0 + n - 1
    lab = ("Gross IM", "Sum of MTM", "Gross replacement cost (sum of positive MTM)", "NGR", "Weight on gross", "Weight on NGR * gross",
           "Net schedule IM", "SIMM-style total IM (USD)", "SIMM over schedule net")
    for i, t in enumerate(lab):
        B.lab(s, t1 + 2 + i, 1, t)
    r = t1 + 2
    B.f(s, r, 2, f"SUM(G{t0}:G{t1})")
    B.f(s, r + 1, 2, f"SUM(H{t0}:H{t1})")
    B.f(s, r + 2, 2, f'SUMIF(H{t0}:H{t1},">0")')
    B.f(s, r + 3, 2, f"IF(B{r + 2}<=0,1,MAX(B{r + 1},0)/B{r + 2})")
    B.f(s, r + 4, 2, I[rx.sched.CONST_GROSS])
    B.f(s, r + 5, 2, I[rx.sched.CONST_NGR])
    B.f(s, r + 6, 2, f"B{r + 4}*B{r}+B{r + 5}*B{r + 3}*B{r}")
    B.f(s, r + 7, 2, B.total_cell)
    B.f(s, r + 8, 2, f"B{r + 7}/B{r + 6}")
    B.check(rx.key("sched", "gross"), s, r, 2)
    B.check(rx.key("sched", "ngr"), s, r + 3, 2)
    B.check(rx.key("sched", "net"), s, r + 6, 2)


def sheet_backtest(B):
    s = "Backtest"
    I = B.I
    B.lab(s, 1, 1, "Backtests. Columns A to I are typed from Python (champion HS IM 1 day, SIMM-style IM 10 day non overlapping); the rest is live.")
    s1, s10 = B.inp["s1d"], B.inp["s10d"]
    for j, t in enumerate(("date", "loss", "var_forecast", "exception")):
        B.lab(s, 4, 1 + j, t)
        B.lab(s, 4, 6 + j, t)
    B.lab(s, 3, 1, "1 day: champion HS IM")
    B.lab(s, 3, 6, "10 day non overlapping: SIMM-style IM")
    for (c0, df) in ((1, s1), (6, s10)):
        for i, r in df.reset_index(drop=True).iterrows():
            rr = FIRST + i
            B.put(s, rr, c0, str(r[C.DATE]))
            B.put(s, rr, c0 + 1, float(r[C.LOSS]))
            B.put(s, rr, c0 + 2, float(r[C.VAR_FORECAST]))
            B.f(s, rr, c0 + 3, f"IF({L(c0 + 1)}{rr}>{L(c0 + 2)}{rr},1,0)")
        B.mark_typed(s, f"{L(c0)}{FIRST}:{L(c0 + 2)}{FIRST + len(df) - 1}")
    spec = {"1d": (12, 1, len(s1)), "10d": (13, 6, len(s10))}
    labels = ["n", "x (exceptions)", "p", "x / n", "log-likelihood under p", "log-likelihood at x/n", "Kupiec LR", "Kupiec p-value",
              "n00", "n01", "n10", "n11", "pi01", "pi11", "pi", "ll alternative (Markov)", "ll null", "LR independence", "p independence",
              "LR conditional coverage", "p conditional coverage", "yellow zone starts at", "red zone starts at", "Basel zone",
              "exceptions in last window", "zone of last window", "plus factor of last window (BCBS 22 Table 2)",
              "yellow start (window)", "red start (window)"]
    for i, t in enumerate(labels):
        B.lab(s, 5 + i, 11, t)
    B.lab(s, 4, 12, "1 day")
    B.lab(s, 4, 13, "10 day")
    hx = FIRST
    B.lab(s, 4, 15, "x")
    for j, t in enumerate(("cum prob n 1d", "cum prob n 10d", "cum prob window")):
        B.lab(s, 4, 16 + j, t)
    for i in range(HELPER_MAX_X + 1):
        rr = hx + i
        B.f(s, rr, 15, "ROW()-5")
        B.f(s, rr, 16, f"IF(O{rr}>{ad(12, 5)},1,BINOMDIST(O{rr},{ad(12, 5)},{ad(12, 7)},TRUE))")
        B.f(s, rr, 17, f"IF(O{rr}>{ad(13, 5)},1,BINOMDIST(O{rr},{ad(13, 5)},{ad(13, 7)},TRUE))")
        B.f(s, rr, 18, f"IF(O{rr}>{I['last_n']},1,BINOMDIST(O{rr},{I['last_n']},{I['p']},TRUE))")
    hl = hx + HELPER_MAX_X
    cum = {"1d": f"$P${hx}:$P${hl}", "10d": f"$Q${hx}:$Q${hl}"}
    for tag, (cl, c0, n) in spec.items():
        col = L(cl)
        e0, e1 = FIRST, FIRST + n - 1
        ex = f"${L(c0 + 3)}${e0}:${L(c0 + 3)}${e1}"
        a_, b_ = f"${L(c0 + 3)}${e0}:${L(c0 + 3)}${e1 - 1}", f"${L(c0 + 3)}${e0 + 1}:${L(c0 + 3)}${e1}"
        R = lambda k: f"{col}{k}"
        B.f(s, 5, cl, f"COUNT(${L(c0 + 1)}${e0}:${L(c0 + 1)}${e1})")
        B.f(s, 6, cl, f"SUM({ex})")
        B.f(s, 7, cl, I["p"])
        B.f(s, 8, cl, f"{R(6)}/{R(5)}")
        B.f(s, 9, cl, f"({R(5)}-{R(6)})*LN(1-{R(7)})+{R(6)}*LN({R(7)})")
        B.f(s, 10, cl, f"IF({R(6)}=0,0,{R(6)}*LN({R(8)}))+IF({R(5)}-{R(6)}=0,0,({R(5)}-{R(6)})*LN(1-{R(8)}))")
        B.f(s, 11, cl, f"MAX(0,-2*{R(9)}+2*{R(10)})")
        B.f(s, 12, cl, f"CHIDIST({R(11)},1)")
        for r_, (u, v) in zip((13, 14, 15, 16), ((0, 0), (0, 1), (1, 0), (1, 1))):
            B.f(s, r_, cl, f"SUMPRODUCT(({a_}={u})*({b_}={v}))")
        B.f(s, 17, cl, f"IF({R(13)}+{R(14)}=0,0,{R(14)}/({R(13)}+{R(14)}))")
        B.f(s, 18, cl, f"IF({R(15)}+{R(16)}=0,0,{R(16)}/({R(15)}+{R(16)}))")
        B.f(s, 19, cl, f"({R(14)}+{R(16)})/({R(13)}+{R(14)}+{R(15)}+{R(16)})")
        B.f(s, 20, cl, f"IF({R(13)}=0,0,{R(13)}*LN(1-{R(17)}))+IF({R(14)}=0,0,{R(14)}*LN({R(17)}))"
                       f"+IF({R(15)}=0,0,{R(15)}*LN(1-{R(18)}))+IF({R(16)}=0,0,{R(16)}*LN({R(18)}))")
        B.f(s, 21, cl, f"IF({R(13)}+{R(15)}=0,0,({R(13)}+{R(15)})*LN(1-{R(19)}))+IF({R(14)}+{R(16)}=0,0,({R(14)}+{R(16)})*LN({R(19)}))")
        B.f(s, 22, cl, f"MAX(0,2*({R(20)}-{R(21)}))")
        B.f(s, 23, cl, f"CHIDIST({R(22)},1)")
        B.f(s, 24, cl, f"{R(11)}+{R(22)}")
        B.f(s, 25, cl, f"CHIDIST({R(24)},2)")
        B.f(s, 26, cl, f'COUNTIF({cum[tag]},"<"&{I["yellow"]})')
        B.f(s, 27, cl, f'COUNTIF({cum[tag]},"<"&{I["red"]})')
        B.f(s, 28, cl, f'IF({R(6)}>={R(27)},"Red",IF({R(6)}>={R(26)},"Yellow","Green"))')
        for k_, row in (("n", 5), ("x", 6), ("LR_uc", 11), ("p_uc", 12), ("n00", 13), ("n01", 14), ("n10", 15), ("n11", 16), ("LR_ind", 22),
                        ("p_ind", 23), ("LR_cc", 24), ("p_cc", 25), ("zone", 28)):
            B.check(rx.key("bt", tag, k_), s, row, cl)
    cl = 12
    wr = f"$P$%d"
    B.f(s, 29, cl, f"SUM(${L(4)}${FIRST + len(s1) - rx.LAST_WINDOW}:${L(4)}${FIRST + len(s1) - 1})")
    win = f"$R${hx}:$R${hl}"
    B.f(s, 32, cl, f'COUNTIF({win},"<"&{I["yellow"]})')
    B.f(s, 33, cl, f'COUNTIF({win},"<"&{I["red"]})')
    B.f(s, 30, cl, f'IF(L29>=L33,"Red",IF(L29>=L32,"Yellow","Green"))')
    B.f(s, 31, cl, f'INDEX({I["tl_plus"]},MATCH(MIN(L29,{I["tl_max"]}),{I["tl_x"]},0))')
    B.check(rx.key("bt", "1d", "x250"), s, 29, cl)
    B.check(rx.key("bt", "1d", "zone250"), s, 30, cl)


def sheet_attribution(B):
    s = "Attribution"
    sub = B.inp["subsets"].reset_index(drop=True)
    drivers = list(rx.attr.DRIVERS)
    B.lab(s, 1, 1, "IM change attribution. Columns A to E are the 16 subset IMs typed from Python; Shapley weights and contributions are live.")
    for j, d in enumerate(drivers):
        B.lab(s, 4, 1 + j, d)
    B.lab(s, 4, 5, "im")
    for j, t in enumerate(("code", "subset size", "Shapley weight")):
        B.lab(s, 4, 6 + j, t)
    for j in range(4):
        B.f(s, 4, 9 + j, f'"contribution of "&{ad(1 + j, 4)}', True)
    for i, r in sub.iterrows():
        rr = FIRST + i
        for j, d in enumerate(drivers):
            B.put(s, rr, 1 + j, int(r[d]))
        B.put(s, rr, 5, float(r["im"]))
        B.f(s, rr, 6, f"A{rr}*8+B{rr}*4+C{rr}*2+D{rr}")
        B.f(s, rr, 7, f"SUM(A{rr}:D{rr})")
        B.f(s, rr, 8, f"IF(G{rr}>3,0,FACT(G{rr})*FACT(3-G{rr})/FACT(4))")
        for j in range(4):
            B.f(s, rr, 9 + j, f"IF({L(1 + j)}{rr}=1,0,$H{rr}*(INDEX($E$5:$E$20,MATCH($F{rr}+{DRIVER_BITS[j]},$F$5:$F$20,0))-$E{rr}))")
    B.mark_typed(s, "A5:E20")
    im = lambda code: f"INDEX($E$5:$E$20,MATCH({code},$F$5:$F$20,0))"
    B.lab(s, 24, 1, "Driver")
    for j, t in enumerate(("Shapley", "Share of dIM", "One at a time", "Sequential bridge")):
        B.lab(s, 24, 2 + j, t)
    bridge = [f"{im(8)}-{im(0)}", f"{im(12)}-{im(8)}", f"{im(14)}-{im(12)}", f"{im(15)}-{im(14)}"]
    for j, d in enumerate(drivers):
        rr = 25 + j
        B.f(s, rr, 1, ad(1 + j, 4))
        B.f(s, rr, 2, f"SUM({L(9 + j)}5:{L(9 + j)}20)")
        B.f(s, rr, 3, f"B{rr}/$B$30")
        B.f(s, rr, 4, f"{im(DRIVER_BITS[j])}-{im(0)}")
        B.f(s, rr, 5, bridge[j])
        B.check(rx.key("attr", "shapley", d), s, rr, 2)
        B.check(rx.key("attr", "oaat", d), s, rr, 4)
        B.check(rx.key("attr", "bridge", d), s, rr, 5)
    B.lab(s, 29, 1, "Sum of drivers")
    B.lab(s, 30, 1, "Total change in IM")
    B.lab(s, 31, 1, "Residual")
    for c in (2, 4, 5):
        B.f(s, 29, c, f"SUM({L(c)}25:{L(c)}28)")
        B.f(s, 30, c, f"{im(15)}-{im(0)}")
        B.f(s, 31, c, f"{L(c)}30-{L(c)}29")
    B.check(rx.key("attr", "dim"), s, 30, 2)
    B.check(rx.key("attr", "shapley_sum"), s, 29, 2)
    B.check(rx.key("attr", "oaat_residual"), s, 31, 4)
    B.lab(s, 34, 1, "Market step split (typed from Python: needs sub-market SIMM runs)")
    scen = B.inp["scenario"].set_index("driver")
    for i, nm in enumerate(("market_rates", "market_fx", "market_vol")):
        B.put(s, 35 + i, 1, nm)
        B.put(s, 35 + i, 2, float(scen.loc[nm, "bridge"]))
    B.mark_typed(s, "B35:B37")
    B.lab(s, 38, 1, "Sum of sub-steps (equals the market step of the bridge)")
    B.f(s, 38, 2, "SUM(B35:B37)")
    B.check(rx.key("attr", "substeps_sum"), s, 38, 2)


def sheet_python_ref_and_checks(B):
    ps, cs = "Python_Ref", "Checks"
    B.lab(ps, 1, 1, "Python reference values (typed). Each row is recomputed through the project modules and compared with the live cell.")
    for j, t in enumerate(("key", "description", "python value", "comparison", "tolerance")):
        B.lab(ps, 4, 1 + j, t)
    B.lab(cs, 1, 1, "Checks: live cell versus Python_Ref. Pass is 1 when the difference is within tolerance.")
    B.lab(cs, 3, 1, "All checks pass (1 = yes)")
    for j, t in enumerate(("check", "workbook value", "python value", "difference", "tolerance", "pass")):
        B.lab(cs, 5, 1 + j, t)
    first = 6
    for i, ck in enumerate(B.checks):
        k = ck["key"]
        kind, tol = rx.tolerance(k)
        r = first + i
        B.put(ps, r, 1, k)
        B.put(ps, r, 2, k.replace("|", " "))
        B.put(ps, r, 3, B.ref[k])
        B.put(ps, r, 4, kind)
        B.put(ps, r, 5, float(tol))
        B.f(cs, r, 1, f"{ps}!B{r}")
        B.f(cs, r, 2, f"{ck['sheet']}!{ck['cell']}")
        B.f(cs, r, 3, f"{ps}!C{r}")
        B.f(cs, r, 4, f'IF({ps}!D{r}="text",IF(B{r}=C{r},0,1),IF({ps}!D{r}="rel",ABS(B{r}-C{r})/MAX(1,ABS(C{r})),ABS(B{r}-C{r})))')
        B.f(cs, r, 5, f"{ps}!E{r}")
        B.f(cs, r, 6, f"IF(D{r}<=E{r},1,0)")
    last = first + len(B.checks) - 1
    B.f(cs, 3, 2, f"IF(COUNTIF(F{first}:F{last},0)=0,1,0)")
    B.mark_typed(ps, f"A{first}:E{last}")
    return first, last


def money(x):
    return f"{x:,.0f}"


def sheet_text(B):
    r, inp = B.ref, B.inp
    ver = inp["param_set"]
    s = "README"
    lines = [
        ("SIMM-style initial margin workbook (RatesFX: IR and FX delta, vega, curvature) with schedule IM, backtests and attribution.", True),
        ("ALL DATA IS SYNTHETIC. This is a SIMM-style implementation for learning and validation practice. It is not ISDA licensed, not ISDA certified and makes no compliance claim. ISDA SIMM is a trademark of ISDA.", False),
        ("How to use: change a value on Inputs (parameter set, exception probability, Basel probabilities) or a typed CRIF amount on Sensitivities, a notional or PV on Portfolio, or a loss or forecast on Backtest, and all live sheets recalculate. Checks!B3 equals 1 when every live cell matches the Python reference within tolerance (the references on Python_Ref belong to the shipped inputs, so Checks will show 0 once you change an input; that is expected).", False),
        ("Sheets: Inputs and Parameters (typed, with provenance), Portfolio, Sensitivities (typed CRIF plus helper columns and closed-form Bachelier and Garman-Kohlhagen checks), SIMM_Delta, SIMM_Vega, SIMM_Curvature, SIMM_Total, Schedule_IM, Backtest, Attribution (live formulas), Checks, Conclusions (static text), Python_Ref (typed Python results), Parameters_Reg (typed schedule rates, weights and BCBS 22 Table 2).", False),
        ("What is replicated in cell formulas: net sensitivities, risk weights by lookup, concentration risk CR, weighted sensitivities, 12x12 tenor correlation grids, K and S per currency, inter-currency aggregation with gamma and g_bc, FX sigma_SIMM, FX delta and vega with concentration scaling, curvature scale factors, theta, lambda, HVR scale, risk class and RatesFX aggregation, schedule IM, Kupiec, Christoffersen, Basel zone, Shapley and bridge.", False),
        ("LIMITS, stated plainly. (1) The CRIF is typed from Python: trade pricing and the bump and reprice sensitivities are not replicated in Excel; only one swaption (Bachelier) and one FX option (Garman-Kohlhagen) are repriced in closed form as spot checks. (2) The sample CRIF has no inflation or cross-currency basis rows and a single OIS sub-curve, so those SIMM branches (inflation risk weight, basis correlation of minus 1 percent, sub-curve phi) are not built into the formulas. (3) The 16 subset IMs of the attribution are typed from Python because each needs a full SIMM run on a different market or portfolio. The rates, FX and vol split of the market step is typed for the same reason. (4) The Backtest series are typed because the VaR models need the full simulated history. (5) The workbook was recalculated and verified in LibreOffice only. It uses functions that Excel 2016 also has, but it was not opened in Excel.", False),
        (f"Shipped inputs: parameter set {ver}, calculation currency {config.CALC_CCY}, p = {inp['p_exc']}.", False),
    ]
    for i, (t, b) in enumerate(lines):
        B.put(s, 1 + 2 * i, 1, t, b)
    s = "Conclusions"
    g = lambda *k: r[rx.key(*k)]
    tot = g("total")
    sh = {d: g("attr", "shapley", d) for d in rx.attr.DRIVERS}
    c = [
        "Conclusions are static text written by the build script from the Python results for the shipped inputs. They do not change when inputs are edited.",
        f"The SIMM-style RatesFX total IM on the synthetic sample portfolio under parameter set {ver} is USD {money(tot)}, made of IR risk class IM USD {money(g('im', 'IR'))} and FX risk class IM USD {money(g('im', 'FX'))}, aggregated with the IR to FX correlation psi.",
        f"By margin type: IR delta {money(g('margin', 'IR', 'Delta'))}, IR vega {money(g('margin', 'IR', 'Vega'))}, IR curvature {money(g('margin', 'IR', 'Curvature'))}, FX delta {money(g('margin', 'FX', 'Delta'))}, FX vega {money(g('margin', 'FX', 'Vega'))}, FX curvature {money(g('margin', 'FX', 'Curvature'))} (USD).",
        f"Schedule IM: gross USD {money(g('sched', 'gross'))}, net to gross ratio {g('sched', 'ngr'):.4f}, net USD {money(g('sched', 'net'))}. The SIMM-style total is {tot / g('sched', 'net'):.1%} of the schedule net amount, so the model based margin is below the standardised floor for this portfolio.",
        f"Backtest, 1 day champion model: {int(g('bt', '1d', 'x'))} exceptions in {int(g('bt', '1d', 'n'))} days, Kupiec LR {g('bt', '1d', 'LR_uc'):.4f} (p {g('bt', '1d', 'p_uc'):.4f}), Christoffersen independence LR {g('bt', '1d', 'LR_ind'):.4f} (p {g('bt', '1d', 'p_ind'):.4f}), conditional coverage p {g('bt', '1d', 'p_cc'):.4f}; Basel zone {g('bt', '1d', 'zone')} over the series and {g('bt', '1d', 'zone250')} over the last {rx.LAST_WINDOW} days ({int(g('bt', '1d', 'x250'))} exceptions).",
        f"Backtest, 10 day non overlapping SIMM-style IM: {int(g('bt', '10d', 'x'))} exceptions in {int(g('bt', '10d', 'n'))} windows, Kupiec p {g('bt', '10d', 'p_uc'):.4f}, zone {g('bt', '10d', 'zone')}. With so few windows the test has low power.",
        f"Attribution of the scenario day IM change of USD {money(g('attr', 'dim'))}: Shapley matured trades {money(sh['matured_trades'])}, new trades {money(sh['new_trades'])}, market move {money(sh['market_move'])}, parameter version {money(sh['parameter_version'])} (USD). The Shapley values add up to the total change exactly; the one at a time method leaves an interaction residual of USD {money(g('attr', 'oaat_residual'))}.",
        "Reconciliation: python/simm_margin/reconcile_excel.py recalculates this workbook in LibreOffice and compares every live cell listed on Checks with the Python engine under three input cases (base, a changed swap with the other parameter set and exception probability, and a concentration stress that activates CR above 1); see python/outputs/excel_reconciliation.json.",
    ]
    for i, t in enumerate(c):
        B.put(s, 1 + 2 * i, 1, t, i == 0)


def build(path, inp, ref) -> dict:
    """Write the workbook for one case. inp = reconcile_excel.apply_case(...), ref = reconcile_excel.reference(inp)."""
    B = Book(inp, ref)
    sheet_inputs(B)
    sheet_parameters(B)
    sheet_parameters_reg(B)
    sheet_portfolio(B)
    # Sensitivities points to SIMM_Curvature rows, which are fixed by layout, so it can be written before it
    sheet_sensitivities(B)
    sheet_delta(B)
    sheet_vega(B)
    sheet_curvature(B)
    sheet_total(B)
    sheet_schedule(B)
    sheet_backtest(B)
    sheet_attribution(B)
    first, last = sheet_python_ref_and_checks(B)
    sheet_text(B)
    for ws in B.ws.values():
        ws.column_dimensions["A"].width = 46
    path = Path(path)
    B.wb.save(path)
    n_formulas = sum(1 for ws in B.wb for row in ws.iter_rows() for c in row if isinstance(c.value, str) and c.value.startswith("="))
    return dict(checks=B.checks, allpass_cell="B3", checks_first=first, checks_last=last, pass_col="F", n_formulas=n_formulas,
                sheets=list(SHEETS), calc_sheets=list(CALC_SHEETS), typed=B.typed, case=inp["case"]["name"],
                n_checks=len(B.checks), data_source=config.DATA_SOURCE)


if __name__ == "__main__":
    inp = rx.apply_case(rx.load_inputs(quick=False), rx.CASES[0])
    print(build(Path(__file__).resolve().parent / "simm_margin_workbook_raw.xlsx", inp, rx.reference(inp))["n_formulas"])
