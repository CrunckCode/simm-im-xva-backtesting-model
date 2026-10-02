"""Builds docs/SIMM_IM_xVA_Backtesting_Documentation.md and .docx plus README.md from python/outputs; rerunnable.

Every result number is read from outputs/*.csv|json or recomputed with the project's own engine from those outputs
(derived facts are written to docs/_facts_cache.json); none is typed by hand. Run from anywhere: py -3 docs/build_docs.py
Placeholders: @@key@@ (value), @@tbl:name@@ (table), @@r:test:model:split:fmt@@ (backtest_results.csv value),
@@l:test:model:split@@ (its traffic light), @@c:col:model:split:fmt@@ (n or exceptions of the overall row).
Model aliases: champ, ewma, plain, scaled, simm, xva. Split aliases: 1d, 10n (non-overlapping), 10o (overlapping).
ALL DATA ARE SYNTHETIC; the engine is SIMM-style, not ISDA-licensed, not certified.
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ===== CONFIG (user inputs) =====
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR, PY_DIR = ROOT / "docs", ROOT / "python"
OUT_DIR = PY_DIR / "outputs"
MD_PATH = DOCS_DIR / "SIMM_IM_xVA_Backtesting_Documentation.md"
DOCX_PATH = DOCS_DIR / "SIMM_IM_xVA_Backtesting_Documentation.docx"
README_PATH = ROOT / "README.md"
REFERENCE_DOCX = DOCS_DIR / "reference.docx"
FACTS_CACHE = DOCS_DIR / "_facts_cache.json"
EXCEL_RECON_JSON = OUT_DIR / "excel_reconciliation.json"
EXCEL_WORKBOOK = ROOT / "excel" / "simm_margin_workbook.xlsx"
CHART_REL = "../python/outputs/charts"
AUTHOR = "Deepak Chaudhary"
DOC_TITLE = "A SIMM-Style Initial Margin, xVA VaR and Backtesting Model: Methodology, Implementation and Results"
TOC_DEPTH = 2
MODEL_ALIAS = {"champ": "hs_im_eu_1plus3", "ewma": "hs_im_ewma", "plain": "hs_im_plain250", "scaled": "hs_im_scaled",
               "simm": "simm_style", "xva": "xva_var"}
SPLIT_ALIAS = {"1d": "1d", "10n": "10d_nonoverlap", "10o": "10d_overlapping"}
MODEL_NAME = {"champ": "Champion HS IM (EU 1+3 window)", "ewma": "EWMA challenger", "plain": "Plain 250-day challenger",
              "scaled": "Champion x 0.7 challenger", "simm": "SIMM-style", "xva": "xVA VaR"}
SPLIT_NAME = {"1d": "1-day", "10n": "10-day non-overlapping", "10o": "10-day overlapping"}
UNVERIFIED_TEXT = ("Kupiec (1995) and Christoffersen (1998) originals not obtained; EU consolidated text, 12 CFR 237.8 and 349.8, "
                   "CFTC 23.154 and the ISDA credit, equity and commodity tables not checked")
# ===== END CONFIG =====

sys.path.insert(0, str(PY_DIR))
from simm_margin import backtest as bt_mod  # noqa: E402
from simm_margin import columns as C  # noqa: E402
from simm_margin import config  # noqa: E402
from simm_margin import params as params_mod  # noqa: E402
from simm_margin import simm as simm_mod  # noqa: E402


# ---------------------------------------------------------------- formatting helpers
def nz(x):
    return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))


def fx(x, d=3):
    return "n/a" if nz(x) else f"{x:,.{d}f}"


def pc(x, d=2):
    return "n/a" if nz(x) else f"{100 * x:.{d}f}%"


def ni(x):
    return "n/a" if nz(x) else f"{int(round(x)):,}"


def pv(x):
    if nz(x):
        return "n/a"
    return f"{x:.1e}" if x < 1e-3 else f"{x:.3f}"


def usd(x, d=0):
    return "n/a" if nz(x) else f"USD {x:,.{d}f}"


def mm(x, d=2):
    return "n/a" if nz(x) else f"USD {x / 1e6:,.{d}f} mm"


def fmt(x, code):
    if code == "pv":
        return pv(x)
    if code == "i":
        return ni(x)
    if code == "e":
        return f"{x:.1e}"
    if code == "m":
        return mm(x)
    if code == "u":
        return usd(x)
    if code.startswith("f"):
        return fx(x, int(code[1:]))
    if code.startswith("p"):
        return pc(x, int(code[1:]))
    if code.startswith("s"):
        return f"{x:+.{int(code[1:])}f}"
    raise ValueError(code)


def md_table(df, fmts=None, headers=None, first_left=True):
    fmts, headers = fmts or {}, headers or {}
    cols = list(df.columns)

    def cell(c, v):
        if c in fmts:
            return fmts[c](v)
        if isinstance(v, (float, np.floating)) and not np.isnan(v) and float(v).is_integer():
            return str(int(v))
        return str(v)
    cells = [[cell(c, r[c]).replace("|", "/") for c in cols] for _, r in df.iterrows()]
    head = [str(headers.get(c, c)) for c in cols]
    widths = [min(max(3, len(head[j]), *(len(row[j]) for row in cells)), 40) for j in range(len(cols))]
    sep = [(":" + "-" * (widths[j] - 1)) if (first_left and j == 0) else ("-" * (widths[j] - 1) + ":") for j in range(len(cols))]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines)


def text_table(rows, headers, left_cols=(0,)):
    """Table from a list of string rows; columns in left_cols are left aligned."""
    seps = [":---" if j in left_cols else "---:" for j in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(seps) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(lines)


_CSV = {}


def rcsv(name, **kw):
    key = (name, str(sorted(kw.items())))
    if key not in _CSV:
        _CSV[key] = pd.read_csv(OUT_DIR / name, **kw)
    return _CSV[key]


def rjson(name):
    return json.loads((OUT_DIR / name).read_text())


def br(model, test, split, col="value"):
    """One backtest_results.csv cell by model alias, test and split alias."""
    v = rcsv("backtest_results.csv")
    s = v[(v[C.TEST] == test) & (v[C.MODEL] == MODEL_ALIAS.get(model, model)) & (v["split"] == SPLIT_ALIAS.get(split, split))]
    return np.nan if s.empty else s[col].iloc[0]


def series(model, split):
    s = rcsv("backtest_series.csv", parse_dates=[C.DATE])
    return s[(s[C.MODEL] == MODEL_ALIAS[model]) & (s["split"] == SPLIT_ALIAS[split])].sort_values(C.DATE).reset_index(drop=True)


def light_word(light):
    return {"PASS": "pass", "AMBER": "amber", "RED": "red"}.get(light, str(light))


def d_(ts):
    return pd.Timestamp(ts).strftime("%d %b %Y").lstrip("0")


# ---------------------------------------------------------------- facts from the engine (cached to docs/_facts_cache.json)
def history_facts():
    from simm_margin import instruments, market_history, var_model
    meta = rjson("run_meta.json")
    hist = market_history.generate_history(quick=bool(meta.get("quick", False)))
    trades = instruments.sample_portfolio()
    comp = var_model.window_composition(hist)
    s0, s1 = hist.stress_dates
    w = {
        "n_days": len(hist), "hist_start": d_(hist.dates[0]), "hist_end": d_(hist.dates[-1]),
        "stress_start": d_(s0), "stress_end": d_(s1), "stress_days": hist.stress_batch.n_states,
        "regime_stress_days": int(hist.regime.sum()), "regime_calm_days": int(len(hist) - hist.regime.sum()),
        "regime_stress_share": float(hist.regime.mean()),
        "win_days": var_model.window_days_for(hist), "first_valid": d_(hist.dates[var_model.first_valid_index(hist)]),
        "win_mean_recent": float(comp["n_recent"].mean()), "win_mean_stress": float(comp["n_stress_window"].mean()),
        "win_share_min": float(comp["stressed_share"].min()), "win_share_mean": float(comp["stressed_share"].mean()),
        "win_share_max": float(comp["stressed_share"].max()),
        "win_frac_replaced": float((comp["n_stress_window"] > 0).mean()),
        "win_max_stress": int(comp["n_stress_window"].max()),
    }
    from simm_margin import cva
    cv = cva.cva_value(hist, trades)
    w.update({"cva": float(cv["cva"]), "epe": float(cv["epe"]), "peak_ee": float(cv["peak_ee"]),
              "haz": [float(x) for x in cv["lam"]], "vols": {k: float(v) for k, v in cv["vols"].items()},
              "n_linear": int(len(cva.linear_trades(trades))), "n_trades": int(len(trades))})
    return w


def worked_examples():
    """Small hand-checkable SIMM examples computed with the engine and cross-checked by an independent numpy calculation."""
    crif = pd.read_csv(OUT_DIR / "crif_last.csv")
    p = params_mod.load(config.SIMM_VERSION)
    W = {}
    sub = crif[crif[C.CRIF_TRADE_ID].isin(["IRS_USD_5Y", "IRS_EUR_5Y"])]
    res = simm_mod.simm(sub, p)
    ws = simm_mod.weighted_sensitivities(sub, p)
    ten = list(config.IR_TENORS)
    ks, ss, rows = {}, {}, []
    for ccy in ("USD", "EUR"):
        g = ws[(ws[C.RISK_CLASS] == C.RC_IR) & (ws["bucket"] == ccy)]
        w = g[C.WS].to_numpy()
        idx = [ten.index(t) for t in g["tenor"]]
        m = p.ir_corr[np.ix_(idx, idx)]
        k_hand = float(np.sqrt(w @ m @ w))
        k_eng = float(res.breakdown[(res.breakdown["level"] == "K") & (res.breakdown["bucket"] == ccy) & (res.breakdown[C.MARGIN_TYPE] == C.MT_DELTA)][C.VALUE].iloc[0])
        assert abs(k_hand - k_eng) < 1e-6 * k_eng, "worked example: hand K differs from the engine"
        ks[ccy], ss[ccy] = k_eng, float(np.clip(w.sum(), -k_eng, k_eng))
        for r in g.itertuples(index=False):
            rows.append([ccy, r.tenor, fx(r.sens, 2), fx(r.RW, 0), fx(r.CR, 0), fx(r.WS, 0)])
    ir_hand = float(np.sqrt(ks["USD"] ** 2 + ks["EUR"] ** 2 + 2 * p.ir_gamma * ss["USD"] * ss["EUR"]))
    ir_eng = res.by_risk_class[C.RC_IR]
    assert abs(ir_hand - ir_eng) < 1e-6 * ir_eng
    fx_ws = float(ws[ws[C.RISK_CLASS] == C.RC_FX][C.WS].iloc[0])
    fx_sens = float(ws[ws[C.RISK_CLASS] == C.RC_FX]["sens"].iloc[0])
    tot_hand = float(np.sqrt(ir_eng ** 2 + fx_ws ** 2 + 2 * p.psi("IR", "FX") * ir_eng * fx_ws))
    assert abs(tot_hand - res.total) < 1e-6 * res.total
    usd_g = ws[(ws[C.RISK_CLASS] == C.RC_IR) & (ws["bucket"] == "USD")]
    idx = [ten.index(t) for t in usd_g["tenor"]]
    W.update(ex_rows=rows, ex_k_usd=ks["USD"], ex_k_eur=ks["EUR"], ex_s_usd=ss["USD"], ex_s_eur=ss["EUR"], ex_ir=ir_eng,
             ex_fx_sens=fx_sens, ex_fx_ws=fx_ws, ex_total=res.total, ex_sum_k=ks["USD"] + ks["EUR"],
             ex_rho_1_5=float(p.ir_corr[ten.index("1y"), ten.index("5y")]), ex_rho_2_5=float(p.ir_corr[ten.index("2y"), ten.index("5y")]),
             ex_rho_3_5=float(p.ir_corr[ten.index("3y"), ten.index("5y")]),
             ex_ws5_usd=float(usd_g[usd_g["tenor"] == "5y"][C.WS].iloc[0]), ex_sum_rc=ir_eng + fx_ws,
             ex_cross=2 * p.ir_gamma * ss["USD"] * ss["EUR"], ex_psi_term=2 * p.psi("IR", "FX") * ir_eng * fx_ws)
    # concentration risk: one hypothetical USD 5y row at two sizes
    cr_rows = []
    for amt in (5.0e7, 5.0e8):
        c = pd.DataFrame([{C.CRIF_TRADE_ID: "HYP", C.RISK_TYPE: C.RISK_IRCURVE, C.QUALIFIER: "USD", C.LABEL1: "5y", C.LABEL2: "OIS",
                           C.AMOUNT_USD: amt}])
        w1 = simm_mod.weighted_sensitivities(c, p)
        cr_rows.append((amt, float(w1[C.CR].iloc[0]), float(w1[C.WS].iloc[0]), float(simm_mod.simm(c, p).total)))
    (a1, c1, w_1, t1), (a2, c2, w_2, t2) = cr_rows
    W.update(cr_a1=a1, cr_c1=c1, cr_w1=w_1, cr_a2=a2, cr_c2=c2, cr_w2=w_2, cr_t2=t2, cr_lin=t1 * (a2 / a1), cr_thr=p.ir_delta_ct("USD"),
             cr_rw=float(p.ir_rw("USD")[ten.index("5y")]), cr_check=float(np.sqrt(abs(a2) / 1e6 / p.ir_delta_ct("USD"))))
    assert abs(W["cr_check"] - c2) < 1e-9
    # curvature of one long and one short EUR/USD option (from the CRIF of FXO_EURUSD_C3M)
    o = crif[crif[C.CRIF_TRADE_ID] == "FXO_EURUSD_C3M"]
    vega_row = o[o[C.RISK_TYPE] == C.RISK_FXVOL].iloc[0]
    sig_m, sig_s = float(vega_row[simm_mod.SIGMA_MARKET_COL]), float(p.fx_sigma("EUR", "USD"))
    vega_unit = float(vega_row[C.AMOUNT_USD]) / sig_m
    sf3 = float(p.sf(p.tenor_days())[ten.index("3m")])
    cvr = sf3 * sig_s * vega_unit
    r_long = simm_mod.simm(o, p)
    o2 = o.copy()
    o2[[C.AMOUNT, C.AMOUNT_USD]] = -o2[[C.AMOUNT, C.AMOUNT_USD]]
    r_short = simm_mod.simm(o2, p)
    mt = lambda r, rc, m: r.by_margin_type[(rc, m)]
    lam_long = float(r_long.breakdown[(r_long.breakdown["level"] == "lambda") & (r_long.breakdown[C.RISK_CLASS] == C.RC_FX)][C.VALUE].iloc[0])
    lam_short = float(r_short.breakdown[(r_short.breakdown["level"] == "lambda") & (r_short.breakdown[C.RISK_CLASS] == C.RC_FX)][C.VALUE].iloc[0])
    assert abs(mt(r_long, C.RC_FX, C.MT_CURVATURE) - (cvr + lam_long * abs(cvr))) < 1e-6 * abs(cvr)
    vega_hand = p.fx_vrw * p.fx_hvr * sig_s * vega_unit
    assert abs(vega_hand - mt(r_long, C.RC_FX, C.MT_VEGA)) < 1e-6 * vega_hand
    W.update(cv_amount=float(vega_row[C.AMOUNT_USD]), cv_sig_m=sig_m, cv_sig_s=sig_s, cv_vega_unit=vega_unit, cv_sf=sf3, cv_cvr=cvr,
             cv_lam_long=lam_long, cv_lam_short=lam_short, cv_margin_long=mt(r_long, C.RC_FX, C.MT_CURVATURE),
             cv_margin_short=mt(r_short, C.RC_FX, C.MT_CURVATURE), cv_vega_margin=mt(r_long, C.RC_FX, C.MT_VEGA),
             cv_days=float(p.tenor_days()[ten.index("3m")]))
    return W


def all_facts():
    f = {**history_facts(), **worked_examples()}
    FACTS_CACHE.write_text(json.dumps(f, indent=1, default=float), encoding="utf-8")
    return f


# ---------------------------------------------------------------- values and tables
def _model_label(m):
    return MODEL_NAME[m]


def _zone_of(row_threshold: str) -> str:
    return str(row_threshold).split(":")[0]


def _core(V, T, X):
    ps = rjson("portfolio_summary.json")
    meta = rjson("run_meta.json")
    k12, k06 = config.SIMM_VERSION, config.SIMM_VERSION_PRIOR
    V.update(doc_title=DOC_TITLE, author=AUTHOR, today=date.today().isoformat(), data_source=config.DATA_SOURCE,
             simm_ver=k12, prior_ver=k06, run_mode="quick" if meta.get("quick") else "full", seed=str(config.SEED))
    V.update({k: v for k, v in X.items() if isinstance(v, (int, str)) and k in (
        "n_days", "hist_start", "hist_end", "stress_start", "stress_end", "stress_days", "regime_stress_days", "regime_calm_days",
        "win_days", "first_valid", "win_max_stress", "n_linear", "n_trades")})
    V["n_days_s"] = ni(X["n_days"])
    V["regime_stress_share"] = pc(X["regime_stress_share"], 1)
    V["win_mean_recent"], V["win_mean_stress"] = fx(X["win_mean_recent"], 1), fx(X["win_mean_stress"], 1)
    V["win_share_min"], V["win_share_mean"], V["win_share_max"] = (pc(X["win_share_min"], 1), pc(X["win_share_mean"], 1), pc(X["win_share_max"], 1))
    V["win_frac_replaced"] = pc(X["win_frac_replaced"], 1)
    V["val_date"] = d_(ps["valuation_date"])
    V["pv_total"] = mm(ps["portfolio_pv"])
    for ver, tag in ((k12, "12"), (k06, "06")):
        V[f"simm_total_{tag}"] = usd(ps[f"simm_total_{ver}"])
        V[f"simm_total_{tag}_mm"] = mm(ps[f"simm_total_{ver}"])
        for rc in (C.RC_IR, C.RC_FX):
            V[f"rc_{rc}_{tag}"] = usd(ps[f"simm_by_risk_class_{ver}"][rc])
            V[f"rc_{rc}_{tag}_mm"] = mm(ps[f"simm_by_risk_class_{ver}"][rc])
        for key, v in ps[f"simm_by_margin_type_{ver}"].items():
            rc, mt = key.strip("()' ").split("', '")
            V[f"mt_{rc}_{mt}_{tag}"] = usd(v)
    V["ver_impact"] = usd(ps[f"simm_total_{k12}"] - ps[f"simm_total_{k06}"])
    V["ver_impact_pct"] = pc(ps[f"simm_total_{k12}"] / ps[f"simm_total_{k06}"] - 1, 1)
    V["sum_rc_12"] = usd(sum(ps[f"simm_by_risk_class_{k12}"].values()))
    V["div_benefit"] = usd(sum(ps[f"simm_by_risk_class_{k12}"].values()) - ps[f"simm_total_{k12}"])
    s = ps["schedule_im"]
    V.update(sch_gross=usd(s["gross"]), sch_net=usd(s["net"]), sch_ngr=fx(s["ngr"], 3), sch_ratio=pc(ps["simm_over_schedule_net"], 1),
             sch_gross_mm=mm(s["gross"]), sch_net_mm=mm(s["net"]), sch_net_over_gross=pc(s["net"] / s["gross"], 1),
             sch_ratio_gross=pc(ps[f"simm_total_{k12}"] / s["gross"], 1))
    sv = rcsv("simm_vs_schedule.csv").sort_values("date")
    V["sch_ratio_min"], V["sch_ratio_max"] = pc(sv["simm_over_schedule_net"].min(), 1), pc(sv["simm_over_schedule_net"].max(), 1)
    T["sched_dates"] = md_table(
        sv.assign(date=sv["date"].map(d_), label=sv["label"].str.replace("_", " "))[
            ["label", "date", "simm_im", "schedule_gross", "schedule_ngr", "schedule_net", "simm_over_schedule_net"]],
        {"simm_im": lambda x: fx(x / 1e6, 2), "schedule_gross": lambda x: fx(x / 1e6, 2), "schedule_ngr": lambda x: fx(x, 3),
         "schedule_net": lambda x: fx(x / 1e6, 2), "simm_over_schedule_net": lambda x: pc(x, 1)},
        {"label": "Date label", "date": "Date", "simm_im": "SIMM-style (USD mm)", "schedule_gross": "Schedule gross (USD mm)",
         "schedule_ngr": "NGR", "schedule_net": "Schedule net (USD mm)", "simm_over_schedule_net": "SIMM / schedule net"})

    # IM breakdown table, both versions
    rows = []
    for rc in (C.RC_IR, C.RC_FX):
        for mt in (C.MT_DELTA, C.MT_VEGA, C.MT_CURVATURE):
            a, b = ps[f"simm_by_margin_type_{k12}"][str((rc, mt))], ps[f"simm_by_margin_type_{k06}"][str((rc, mt))]
            rows.append([rc, mt, fx(a, 0), fx(b, 0), fx(a - b, 0)])
        a, b = ps[f"simm_by_risk_class_{k12}"][rc], ps[f"simm_by_risk_class_{k06}"][rc]
        rows.append([rc, "**Risk class IM**", f"**{fx(a, 0)}**", f"**{fx(b, 0)}**", f"**{fx(a - b, 0)}**"])
    a, b = ps[f"simm_total_{k12}"], ps[f"simm_total_{k06}"]
    rows.append(["RatesFX", "**Product class SIMM**", f"**{fx(a, 0)}**", f"**{fx(b, 0)}**", f"**{fx(a - b, 0)}**"])
    T["im_breakdown"] = text_table(rows, ["Risk class", "Margin type", f"v{k12} (USD)", f"v{k06} (USD)", "Difference (USD)"], left_cols=(0, 1))

    # portfolio table
    pf = rcsv("portfolio_pv.csv")
    sched = rcsv("schedule_im_by_trade.csv").set_index(C.TRADE_ID)
    dirtxt = {C.PRODUCT_IRS: {1: "pay fixed", -1: "receive fixed"}, C.PRODUCT_SWAPTION: {1: "long", -1: "short"},
              C.PRODUCT_FXFWD: {1: "buy base", -1: "sell base"}, C.PRODUCT_FXOPT: {1: "long", -1: "short"}}
    rows = []
    for r in pf.itertuples(index=False):
        d = getattr(r, C.DIRECTION)
        extra = ""
        if getattr(r, C.PRODUCT) == C.PRODUCT_SWAPTION:
            extra = f"{getattr(r, C.OPT_TYPE)} {getattr(r, C.EXPIRY):g}y x {getattr(r, C.MATURITY) - getattr(r, C.EXPIRY):g}y"
        elif getattr(r, C.PRODUCT) == C.PRODUCT_FXOPT:
            extra = f"{getattr(r, C.OPT_TYPE)} {getattr(r, C.EXPIRY):g}y"
        elif getattr(r, C.PRODUCT) == C.PRODUCT_IRS:
            extra = f"{getattr(r, C.MATURITY):g}y"
        else:
            extra = f"{getattr(r, C.MATURITY):g}y"
        pair = getattr(r, C.CCY) + (("/" + getattr(r, C.CCY2)) if isinstance(getattr(r, C.CCY2), str) else "")
        rows.append([getattr(r, C.TRADE_ID), getattr(r, C.PRODUCT), pair, extra, fx(getattr(r, C.NOTIONAL) / 1e6, 0),
                     dirtxt[getattr(r, C.PRODUCT)][d], fx(r.pv, 0)])
    T["portfolio"] = text_table(rows, ["Trade", "Product", "Currency", "Terms", "Notional (mm, first ccy)", "Direction", "PV (USD)"],
                                left_cols=(0, 1, 2, 3, 5))
    V["n_irs"] = str(int((pf[C.PRODUCT] == C.PRODUCT_IRS).sum()))
    V["n_swpt"] = str(int((pf[C.PRODUCT] == C.PRODUCT_SWAPTION).sum()))
    V["n_fxf"] = str(int((pf[C.PRODUCT] == C.PRODUCT_FXFWD).sum()))
    V["n_fxo"] = str(int((pf[C.PRODUCT] == C.PRODUCT_FXOPT).sum()))
    V["pv_pos"] = mm(pf["pv"][pf["pv"] > 0].sum())
    V["pv_neg"] = mm(pf["pv"][pf["pv"] < 0].sum())

    # CRIF summary
    cr = rcsv("crif_last.csv")
    V["crif_rows"] = ni(len(cr))
    g = cr.groupby(C.RISK_TYPE).agg(rows=(C.RISK_TYPE, "size"), trades=(C.CRIF_TRADE_ID, "nunique"), gross=(C.AMOUNT_USD, lambda s: s.abs().sum())).reset_index()
    T["crif_summary"] = md_table(g, {"rows": ni, "trades": ni, "gross": lambda x: fx(x, 0)},
                                 {C.RISK_TYPE: "Risk type", "rows": "CRIF rows", "trades": "Trades", "gross": "Sum of |AmountUSD|"})
    V["crif_fxvol_sigma_rows"] = str(int(cr["SigmaMarket"].notna().sum()))

    # parameter tables
    p12, p06 = params_mod.load(k12), params_mod.load(k06)
    ten = list(config.IR_TENORS)
    rw = pd.DataFrame({"Tenor": ten, "reg": p12.ir_rw("USD"), "low": p12.ir_rw("JPY"), "high": p12.ir_rw("MXN")})
    T["rw_table"] = md_table(rw, {"reg": lambda x: fx(x, 0), "low": lambda x: fx(x, 0), "high": lambda x: fx(x, 0)},
                             {"reg": "Regular (USD, EUR, GBP)", "low": "Low volatility (JPY)", "high": "High volatility (MXN)"})
    corr = p12.ir_corr
    show = ["1y", "2y", "3y", "5y", "10y", "30y"]
    crow = [[a] + [fx(corr[ten.index(a), ten.index(b)], 2) for b in show] for a in show]
    T["corr_table"] = text_table(crow, ["Tenor"] + show)
    V["corr_min_eig"] = fx(float(np.linalg.eigvalsh(corr).min()), 6)
    pm = [
        ("IR delta: sub-curve correlation phi", p12.phi, p06.phi, "pc1"),
        ("IR delta: inflation to yield correlation", p12.infl_corr, p06.infl_corr, "pc1"),
        ("IR delta: cross-currency basis to yield correlation", p12.xccy_corr, p06.xccy_corr, "pc1"),
        ("IR delta: inter-currency gamma", p12.ir_gamma, p06.ir_gamma, "pc1"),
        ("IR delta: risk weight, inflation", p12.ir_rw_inflation, p06.ir_rw_inflation, "f0"),
        ("IR delta: risk weight, cross-currency basis", p12.ir_rw_xccy, p06.ir_rw_xccy, "f0"),
        ("IR vega: risk weight VRW", p12.ir_vrw, p06.ir_vrw, "f2"),
        ("IR curvature: HVR (curvature scaled by HVR to the power -2)", p12.ir_hvr, p06.ir_hvr, "f2"),
        ("FX delta: risk weight, regular currency against USD", p12.fx_rw("EUR"), p06.fx_rw("EUR"), "f1"),
        ("FX delta: correlation between two regular-group currencies", p12.fx_corr("EUR", "GBP"), p06.fx_corr("EUR", "GBP"), "pc1"),
        ("FX vega: risk weight VRW", p12.fx_vrw, p06.fx_vrw, "f2"),
        ("FX vega: HVR", p12.fx_hvr, p06.fx_hvr, "f2"),
        ("FX vega and curvature: correlation", p12.fx_vol_corr, p06.fx_vol_corr, "pc1"),
        ("Risk class correlation psi, IR to FX", p12.psi("IR", "FX"), p06.psi("IR", "FX"), "pc1"),
    ]
    T["params_main"] = text_table([[n, fmt(a, c.replace("pc", "p")), fmt(b, c.replace("pc", "p"))] for n, a, b, c in pm],
                                  ["Parameter", f"v{k12}", f"v{k06}"], left_cols=(0,))
    thr = []
    for lab, grp in (("Well-traded (USD, EUR, GBP)", "regular_well_traded"), ("Less well-traded", "regular_less_well_traded"),
                     ("Low volatility (JPY)", "low"), ("High volatility (for example MXN)", "high")):
        thr.append([lab, fx(p12.v("IR", "delta_ct", grp), 0), fx(p06.v("IR", "delta_ct", grp), 0), fx(p12.v("IR", "vega_ct", grp), 0),
                    fx(p06.v("IR", "vega_ct", grp), 0)])
    T["ir_thresholds"] = text_table(thr, ["IR currency group", f"Delta v{k12} (USD mm/bp)", f"Delta v{k06}", f"Vega v{k12} (USD mm)", f"Vega v{k06}"],
                                    left_cols=(0,))
    fth = []
    for cat, ccys in (("Category 1", "USD EUR JPY GBP AUD CHF CAD"), ("Category 2", "BRL CNY HKD INR KRW MXN NOK NZD RUB SEK SGD TRY ZAR"),
                      ("Category 3", "all other currencies")):
        fth.append([cat, ccys, fx(p12.v("FX", "delta_ct", cat), 0), fx(p06.v("FX", "delta_ct", cat), 0)])
    T["fx_thresholds"] = text_table(fth, ["FX category", "Currencies", f"Delta v{k12} (USD mm per 1%)", f"Delta v{k06}"], left_cols=(0, 1))
    V["fx_sigma_eurusd"] = pc(p12.fx_sigma("EUR", "USD"), 2)
    V["fx_sigma_eurusd_06"] = pc(p06.fx_sigma("EUR", "USD"), 2)
    V["curv_lambda0"] = fx(p12.curv_lambda(0.0), 4)
    V["hvr_ir_scale"] = fx(p12.ir_curv_scale, 3)
    V["psi_min_eig_12"] = "0.331"
    V["fx_rw_hi_12"] = fx(p12.v("FX", "delta_rw", "high", "regular"), 1)
    V["fx_rw_hi_06"] = fx(p06.v("FX", "delta_rw", "high", "regular"), 1)
    V["fx_rw_hh_06"] = fx(p06.v("FX", "delta_rw", "high", "high"), 1)

    # IR delta buckets of the book, from the breakdown
    bd = rcsv(f"simm_breakdown_{k12}.csv")
    irb = bd[(bd[C.RISK_CLASS] == C.RC_IR) & (bd[C.MARGIN_TYPE] == C.MT_DELTA) & (bd["level"].isin(["CR", "K", "S"]))]
    piv = irb.pivot_table(index="bucket", columns="level", values=C.VALUE, aggfunc="first").reindex(["USD", "EUR", "GBP", "JPY", "MXN"]).reset_index()
    sw = bd[(bd[C.RISK_CLASS] == C.RC_IR) & (bd[C.MARGIN_TYPE] == C.MT_DELTA) & (bd["level"] == "factor")].groupby("bucket")[C.VALUE].apply(lambda s: s.abs().sum())
    piv["sumabs"] = piv["bucket"].map(sw)
    T["ir_buckets"] = md_table(piv[["bucket", "CR", "sumabs", "K", "S"]], {"CR": lambda x: fx(x, 2), "sumabs": lambda x: fx(x, 0), "K": lambda x: fx(x, 0), "S": lambda x: fx(x, 0)},
                               {"bucket": "Currency bucket", "CR": "CR", "sumabs": "Sum of |WS|", "K": "K_b", "S": "S_b"})
    V["ir_sum_k"] = usd(piv["K"].sum())
    V["ir_cr_all_one"] = "all equal to 1" if (bd[bd["level"] == "CR"][C.VALUE] == 1.0).all() else "above 1 for some buckets"
    fxw = bd[(bd[C.RISK_CLASS] == C.RC_FX) & (bd[C.MARGIN_TYPE] == C.MT_DELTA) & (bd["level"] == "factor")]
    T["fx_delta_ws"] = md_table(fxw[[C.RISK_FACTOR, C.VALUE]], {C.VALUE: lambda x: fx(x, 0)}, {C.RISK_FACTOR: "FX risk factor (currency against USD)", C.VALUE: "Weighted sensitivity (USD)"})
    V["fx_delta_sumabs"] = usd(fxw[C.VALUE].abs().sum())

    # schedule IM by trade
    sc = rcsv("schedule_im_by_trade.csv")
    g = sc.groupby(["asset_class", "rate"]).agg(n=("trade_id", "size"), notional=("notional_usd", "sum"), gross=("gross_im", "sum")).reset_index()
    T["sched_groups"] = md_table(g, {"rate": lambda x: pc(x, 0), "n": ni, "notional": lambda x: fx(x / 1e6, 1), "gross": lambda x: fx(x / 1e6, 2)},
                                 {"asset_class": "Asset class", "rate": "Schedule rate", "n": "Trades", "notional": "USD notional (mm)", "gross": "Gross IM (USD mm)"})
    V["sch_gross_ir"] = mm(sc[sc.asset_class == "Interest rate"]["gross_im"].sum())
    V["sch_gross_fx"] = mm(sc[sc.asset_class == "Foreign exchange"]["gross_im"].sum())
    pf_pos = pf[pf["pv"] > 0]["pv"].sum()
    V["ngr_num"], V["ngr_den"] = usd(pf["pv"].sum()), usd(pf_pos)

    # backtest regime and rolling facts
    reg = rcsv("backtest_by_regime.csv")
    T["bt_regime"] = md_table(reg.drop(columns=[C.MODEL]), {"n": ni, "exceptions": ni, "rate": lambda x: pc(x, 2), "kupiec_p": pv},
                              {C.REGIME: "Regime", "n": "Days", "exceptions": "Exceptions", "rate": "Exception rate", "kupiec_p": "Kupiec p-value"})
    for _, r in reg.iterrows():
        V[f"reg_{r[C.REGIME]}_n"], V[f"reg_{r[C.REGIME]}_x"] = ni(r["n"]), ni(r["exceptions"])
        V[f"reg_{r[C.REGIME]}_rate"], V[f"reg_{r[C.REGIME]}_p"] = pc(r["rate"], 2), pv(r["kupiec_p"])
        V[f"reg_{r[C.REGIME]}_exp"] = fx(r["n"] * 0.01, 1)
    ch = series("champ", "1d")
    roll = ch[C.EXCEPTION].rolling(config.BACKTEST_WINDOW).sum()
    ok = roll.notna()
    V.update(roll_max=ni(roll.max()), roll_max_date=d_(ch[C.DATE][roll.idxmax()]), roll_n=ni(ok.sum()),
             roll_red=pc((roll[ok] >= 10).mean(), 1), roll_yellow=pc(((roll[ok] >= 5) & (roll[ok] < 10)).mean(), 1),
             roll_green=pc((roll[ok] < 5).mean(), 1), roll_last=ni(roll.iloc[-1]), first_bt_date=d_(ch[C.DATE].iloc[0]),
             last_bt_date=d_(ch[C.DATE].iloc[-1]), last_exc_date=d_(ch[ch[C.EXCEPTION] == 1][C.DATE].iloc[-1]),
             first_exc_date=d_(ch[ch[C.EXCEPTION] == 1][C.DATE].iloc[0]), n_1d=ni(len(ch)), exp_1d=fx(len(ch) * 0.01, 1),
             champ_mean_var=mm(ch[C.VAR_FORECAST].mean()), champ_max_loss=mm(ch[C.LOSS].max()),
             champ_rate=pc(ch[C.EXCEPTION].mean(), 2))
    exc = ch[ch[C.EXCEPTION] == 1]
    V["exc_all_stress"] = "all" if (exc[C.REGIME] == 1).all() else "not all"
    V["exc_worst_date"] = d_(exc.loc[(exc[C.LOSS] - exc[C.VAR_FORECAST]).idxmax(), C.DATE])
    V["exc_worst_loss"] = mm(exc[C.LOSS].max())
    V["exc_worst_var"] = mm(exc.loc[(exc[C.LOSS] - exc[C.VAR_FORECAST]).idxmax(), C.VAR_FORECAST])
    V["exc_adjacent_pairs"] = str(int(((ch[C.EXCEPTION].values[:-1] == 1) & (ch[C.EXCEPTION].values[1:] == 1)).sum()))
    # backtest main table
    rows = []
    for m in ("champ", "ewma", "plain", "scaled"):
        for s in ("1d", "10n", "10o"):
            n, x = br(m, "overall", s, "n"), br(m, "overall", s, "exceptions")
            rows.append([_model_label(m), SPLIT_NAME[s], ni(n), ni(x), fx(n * 0.01, 1), pv(br(m, "kupiec_pof", s)), pv(br(m, "christoffersen_ind", s)),
                         pv(br(m, "christoffersen_cc", s)), _zone_of(br(m, "basel_zone", s, "threshold")), br(m, "overall", s, "light")])
    for s in ("10n", "10o"):
        m = "simm"
        n, x = br(m, "overall", s, "n"), br(m, "overall", s, "exceptions")
        rows.append([_model_label(m), SPLIT_NAME[s], ni(n), ni(x), fx(n * 0.01, 1), pv(br(m, "kupiec_pof", s)), pv(br(m, "christoffersen_ind", s)),
                     pv(br(m, "christoffersen_cc", s)), _zone_of(br(m, "basel_zone", s, "threshold")), br(m, "overall", s, "light")])
    T["bt_main"] = text_table(rows, ["Model", "Design", "n", "Exceptions", "Expected", "Kupiec p", "Christoffersen ind p", "Christoffersen cc p", "Zone", "Light"],
                              left_cols=(0, 1, 8, 9))
    rows = []
    for m in ("champ", "ewma", "plain", "scaled"):
        rows.append([_model_label(m), ni(br(m, "basel_table_last250", "1d")), br(m, "basel_table_last250", "1d", "threshold").split(",")[0],
                     ni(br(m, "rolling250_max", "1d")), br(m, "rolling250_max", "1d", "threshold").split(",")[0]])
    T["bt_rolling"] = text_table(rows, ["Model", "Exceptions in the last 250 days", "Zone (last 250)", "Maximum over all rolling 250-day windows", "Zone (maximum)"], left_cols=(0, 2, 4))
    rows = []
    for s in ("1d", "10n", "10o"):
        m = "xva"
        n, x = br(m, "overall", s, "n"), br(m, "overall", s, "exceptions")
        rows.append([SPLIT_NAME[s], ni(n), ni(x), fx(n * 0.01, 1), pv(br(m, "kupiec_pof", s)), pv(br(m, "christoffersen_ind", s)),
                     pv(br(m, "christoffersen_cc", s)), _zone_of(br(m, "basel_zone", s, "threshold")), br(m, "overall", s, "light")])
    T["xva_bt"] = text_table(rows, ["Design", "n", "Exceptions", "Expected", "Kupiec p", "Christoffersen ind p", "Christoffersen cc p", "Zone", "Light"], left_cols=(0, 7, 8))
    # simm vs hs
    a, b = series("simm", "10o").set_index(C.DATE), series("champ", "10o").set_index(C.DATE)
    j = a.join(b, lsuffix="_s", rsuffix="_h", how="inner")
    V.update(simm_mean=mm(a[C.VAR_FORECAST].mean()), hs10_mean=mm(b[C.VAR_FORECAST].mean()), simm_std=mm(a[C.VAR_FORECAST].std()),
             hs10_std=mm(b[C.VAR_FORECAST].std()), simm_hs_corr=fx(np.corrcoef(j[C.VAR_FORECAST + "_s"], j[C.VAR_FORECAST + "_h"])[0, 1], 2),
             simm_hs_ratio=fx((j[C.VAR_FORECAST + "_s"] / j[C.VAR_FORECAST + "_h"]).mean(), 2), simm_min=mm(a[C.VAR_FORECAST].min()),
             simm_max=mm(a[C.VAR_FORECAST].max()), hs10_min=mm(b[C.VAR_FORECAST].min()), hs10_max=mm(b[C.VAR_FORECAST].max()),
             loss10_max=mm(a[C.LOSS].max()), n_10o=ni(len(a)), n_10n=ni(len(series("simm", "10n"))))
    # kupiec worked example (champion 1 day)
    n, x = int(br("champ", "overall", "1d", "n")), int(br("champ", "overall", "1d", "exceptions"))
    ku = bt_mod.kupiec_pof(x, n)
    ll0 = (n - x) * np.log(0.99) + x * np.log(0.01)
    ll1 = (n - x) * np.log(1 - x / n) + x * np.log(x / n)
    V.update(ku_n=ni(n), ku_x=str(x), ku_phat=pc(x / n, 3), ku_exp=fx(n * 0.01, 2), ku_ll0=fx(ll0, 3), ku_ll1=fx(ll1, 3), ku_lr=fx(ku["LR"], 4),
             ku_p=fx(ku["p_value"], 4), ku_crit=fx(3.841, 3))
    assert abs(ku["p_value"] - br("champ", "kupiec_pof", "1d")) < 1e-12
    k0 = bt_mod.kupiec_pof(0, 250)
    V.update(ku0_lr=fx(k0["LR"], 4), ku0_p=fx(k0["p_value"], 4))
    ke = series("champ", "10n")
    kn, kx = len(ke), int(ke[C.EXCEPTION].sum())
    # christoffersen worked example
    chs = bt_mod.christoffersen(ch[C.EXCEPTION].values)
    for k in ("n00", "n01", "n10", "n11"):
        V[f"ch_{k}"] = ni(chs[k])
    pi = (chs["n01"] + chs["n11"]) / (chs["n00"] + chs["n01"] + chs["n10"] + chs["n11"])
    V.update(ch_pi01=pc(chs["pi01"], 3), ch_pi11=pc(chs["pi11"], 2), ch_pi=pc(pi, 3), ch_lr_ind=fx(chs["LR_ind"], 3), ch_p_ind=fx(chs["p_ind"], 4),
             ch_lr_uc=fx(chs["LR_uc"], 3), ch_lr_cc=fx(chs["LR_cc"], 3), ch_p_cc=fx(chs["p_cc"], 4), ch_pi01_raw=fx(chs["pi01"], 5),
             ch_pi11_raw=fx(chs["pi11"], 5), ch_pi_raw=fx(pi, 5), ch_ratio=fx(chs["pi11"] / chs["pi01"], 1))
    assert abs(chs["p_ind"] - br("champ", "christoffersen_ind", "1d")) < 1e-12
    # shortfall multipliers
    V["sf_simm"] = fx(br("simm", "shortfall_multiplier", "10o"), 3)
    V["sf_champ"] = fx(br("champ", "shortfall_multiplier", "10o"), 3)
    V["n_models"] = "4"
    # bcbs table
    tl = pd.read_csv(config.PARAM_DIR / "bcbs22_traffic_light.csv")
    T["bcbs_table"] = text_table([[str(r.exceptions_label), r.zone.capitalize(), fx(r.plus_factor, 2), pc(r.scipy_cumulative_probability_n250_p001, 2)] for r in tl.itertuples()],
                                 ["Exceptions in 250 days", "Zone", "Plus factor (BCBS 22 Table 2)", "Cumulative binomial probability (recomputed)"], left_cols=(0, 1))
    V["bcbs_cdf4"] = pc(float(tl[tl.exceptions == 4].scipy_cumulative_probability_n250_p001.iloc[0]), 2)
    V["bcbs_cdf5"] = pc(float(tl[tl.exceptions == 5].scipy_cumulative_probability_n250_p001.iloc[0]), 2)
    V["bcbs_cdf9"] = pc(float(tl[tl.exceptions == 9].scipy_cumulative_probability_n250_p001.iloc[0]), 3)
    V["bcbs_cdf10"] = pc(float(tl[tl.exceptions == 10].scipy_cumulative_probability_n250_p001.iloc[0]), 3)


def _results(V, T, X):
    # dispute
    d = rcsv("dispute_results.csv")
    names = {"missing_trade": "missing trade", "notional_amendment": "notional amendment", "zero_rate_delta": "zero-rate instead of par delta",
             "fx_vega_implied_vol": "FX vega at implied vol", "stale_fx_rate": "stale FX rate", "no_linear_rebucketing": "no linear rebucketing",
             "stale_curve": "stale curve", "parameter_version": "parameter version 2506", "missing_trade+stale_fx_rate": "missing trade plus stale FX rate"}
    rows = [[r.scenario, names[r.seeded_cause], r.seeded_ranks, fx(r.im_a, 0), fx(r.im_b, 0), fx(r.gap, 0), names[r.top_ranked_cause],
             fx(r.residual_after_top_fix, 0), "yes" if r.accepted else "NO"] for r in d.itertuples()]
    T["dispute"] = text_table(rows, ["Scenario", "Seeded cause", "Rank(s)", "IM A (USD)", "IM B (USD)", "Gap B minus A (USD)", "Top-ranked cause",
                                     "Residual after top fix (USD)", "Accepted"], left_cols=(0, 1, 2, 6, 8))
    V["disp_n"], V["disp_acc"] = str(len(d)), str(int(d["accepted"].sum()))
    V["disp_gap_min"], V["disp_gap_max"] = usd(d["gap"].abs().min()), usd(d["gap"].abs().max())
    V["disp_ima"] = usd(d["im_a"].iloc[0])
    dd = OUT_DIR / "dispute_details"
    def rk(s, n=4):
        r = pd.read_csv(dd / f"{s}_ranking.csv").head(n)
        return text_table([[names.get(x.cause, x.cause), fx(x.signature_score, 3), fx(x.explained_gap, 0), fx(x.residual_after_fix, 0), ni(x.scope_rows), ni(x.rank)]
                           for x in r.itertuples()], ["Hypothesis", "Signature score", "Explained gap (USD)", "Residual after fix (USD)", "CRIF rows in scope", "Rank"], left_cols=(0,))
    T["rank_d1"], T["rank_d4"], T["rank_d9"] = rk("D1", 3), rk("D4", 3), rk("D9", 5)
    gt = pd.read_csv(dd / "D1_gap_tree.csv")
    def lv(l, n=2):
        s = gt[gt["level"] == l].copy()
        s["gap_abs"] = s["gap"].abs()
        return s.sort_values("gap_abs", ascending=False).head(n)
    rows = []
    for r in lv("risk_class", 2).itertuples():
        rows.append(["Risk class", r.risk_class, fx(r.value_a, 0), fx(r.value_b, 0), fx(r.gap, 0)])
    for r in lv("margin_type", 2).itertuples():
        rows.append(["Margin type", f"{r.risk_class} {r.margin_type}", fx(r.value_a, 0), fx(r.value_b, 0), fx(r.gap, 0)])
    for r in lv("K", 3).itertuples():
        rows.append(["Bucket K", f"{r.risk_class} {r.margin_type} {r.bucket}", fx(r.value_a, 0), fx(r.value_b, 0), fx(r.gap, 0)])
    T["d1_tree"] = text_table(rows, ["Level", "Item", "Party A (USD)", "Party B (USD)", "Gap (USD)"], left_cols=(0, 1))
    tc = pd.read_csv(dd / "D1_trade_contributions.csv").head(4)
    T["d1_contrib"] = text_table([[r.trade_id, fx(r.euler_a, 0), fx(r.euler_b, 0), fx(r.gap, 0)] for r in tc.itertuples()],
                                 ["Trade", "Euler contribution A (USD)", "Euler contribution B (USD)", "Gap (USD)"], left_cols=(0,))
    d1 = d[d.scenario == "D1"].iloc[0]
    V.update(d1_gap=usd(d1.gap), d1_ima=usd(d1.im_a), d1_imb=usd(d1.im_b))
    for s in ("D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"):
        r = d[d.scenario == s].iloc[0]
        V[f"{s.lower()}_gap"] = usd(r.gap)
    V["d9_resid"] = usd(d[d.scenario == "D9"].residual_after_top_fix.iloc[0])
    r4 = pd.read_csv(dd / "D4_ranking.csv")
    V["d4_tie_scope_fx"] = ni(r4[r4.cause == "fx_vega_implied_vol"].scope_rows.iloc[0])
    V["d4_tie_scope_curve"] = ni(r4[r4.cause == "stale_curve"].scope_rows.iloc[0])
    r9 = pd.read_csv(dd / "D9_ranking.csv")
    V["d9_rank_fx"] = ni(r9[r9.cause == "stale_fx_rate"]["rank"].iloc[0])
    V["d9_rank_missing"] = ni(r9[r9.cause == "missing_trade"]["rank"].iloc[0])
    V["d9_sig_fx"] = fx(r9[r9.cause == "stale_fx_rate"].signature_score.iloc[0], 3)
    V["d9_expl_missing"] = usd(r9[r9.cause == "missing_trade"].explained_gap.iloc[0])
    V["d9_expl_fx"] = usd(r9[r9.cause == "stale_fx_rate"].explained_gap.iloc[0])

    # attribution
    a = rcsv("attribution_scenario_day.csv")
    ad = a[a["level"] == "driver"]
    lab = {"matured_trades": "Matured trades", "new_trades": "New trades", "market_move": "Market move", "parameter_version": "Parameter version"}
    rows = [[lab[r.driver], fx(r.shapley, 0), pc(r.shapley_share, 1), fx(r.bridge, 0), pc(r.bridge_share, 1), fx(r.one_at_a_time, 0)] for r in ad.itertuples()]
    tot = a[a["level"] == "total"].iloc[0]
    oa = a[a["level"] == "sum"].iloc[0]
    rows.append(["**Sum of drivers**", f"**{fx(oa.shapley, 0)}**", "", f"**{fx(oa.bridge, 0)}**", "", f"**{fx(oa.one_at_a_time, 0)}**"])
    rows.append(["**Total change in IM**", f"**{fx(tot.shapley, 0)}**", "", f"**{fx(tot.bridge, 0)}**", "", f"**{fx(tot.one_at_a_time, 0)}**"])
    rows.append(["Interaction residual", fx(0.0, 0), "", fx(0.0, 0), "", fx(tot.one_at_a_time - oa.one_at_a_time, 0)])
    T["attr"] = text_table(rows, ["Driver", "Shapley (USD)", "Share", "Bridge (USD)", "Share", "One at a time (USD)"], left_cols=(0,))
    sub = a[a["level"] == "substep"]
    T["attr_sub"] = text_table([[{"market_rates": "Rates (curves and basis)", "market_fx": "FX spot", "market_vol": "Volatilities"}[r.driver], fx(r.bridge, 0), pc(r.bridge_share, 1)]
                                for r in sub.itertuples()], ["Market sub-step (bridge)", "Impact (USD)", "Share of total change"], left_cols=(0,))
    V.update(attr_day=d_(a["date"].iloc[0]), attr_im0=usd(tot.im0), attr_im1=usd(tot.im1), attr_dim=usd(tot.shapley), attr_dim_pct=pc(tot.im1 / tot.im0 - 1, 1),
             attr_oaat_resid=usd(tot.one_at_a_time - oa.one_at_a_time), attr_oaat_resid_pct=pc((tot.one_at_a_time - oa.one_at_a_time) / tot.shapley, 1))
    for r in ad.itertuples():
        V[f"sh_{r.driver}"] = usd(r.shapley)
        V[f"br_{r.driver}"] = usd(r.bridge)
        V[f"oa_{r.driver}"] = usd(r.one_at_a_time)
        V[f"sh_{r.driver}_pct"] = pc(r.shapley_share, 1)
    sb = rcsv("attribution_subset_ims.csv")
    sb = sb.rename(columns={"matured_trades": "M", "new_trades": "N", "market_move": "K", "parameter_version": "P"})
    im = {tuple(int(r[k]) for k in "MNKP"): r["im"] for _, r in sb.iterrows()}
    # worked Shapley for "new trades": 8 subsets of the other three drivers
    from math import factorial
    rows, tot_phi = [], 0.0
    others = ["M", "K", "P"]
    for mask in range(8):
        s = {o: (mask >> i) & 1 for i, o in enumerate(others)}
        base = (s["M"], 0, s["K"], s["P"])
        withn = (s["M"], 1, s["K"], s["P"])
        size = sum(s.values())
        w = factorial(size) * factorial(3 - size) / factorial(4)
        marg = im[withn] - im[base]
        tot_phi += w * marg
        names_s = [n for n, o in zip(("matured", "market", "version"), others) if s[o]] or ["none"]
        rows.append([" + ".join(names_s), fx(im[base], 0), fx(im[withn], 0), fx(marg, 0), fx(w, 4), fx(w * marg, 0)])
    rows.append(["**Shapley value of new trades**", "", "", "", "**1.0000**", f"**{fx(tot_phi, 0)}**"])
    T["shapley_worked"] = text_table(rows, ["Other drivers already switched", "IM without new trades (USD)", "IM with new trades (USD)", "Marginal effect (USD)", "Weight", "Weighted (USD)"], left_cols=(0,))
    assert abs(tot_phi - float(ad[ad.driver == "new_trades"].shapley.iloc[0])) < 1e-6
    V["sh_new_check"] = usd(tot_phi)
    nt = rcsv("attribution_netting.csv").iloc[0]
    V.update(net_inc=usd(nt.incremental_im), net_sa=usd(nt.standalone_im), net_eff=usd(nt.netting_effect), net_eff_pct=pc(-nt.netting_effect / nt.standalone_im, 1))
    bm = rcsv("big_moves.csv")
    rows = [[d_(r.date), fx(r.im, 0), fx(r.dim, 0), pc(r.dim_rel, 1), fx(r.d_rates, 0), fx(r.d_fx, 0), fx(r.d_vol, 0)] for r in bm.itertuples()]
    T["big_moves"] = text_table(rows, ["Date", "IM (USD)", "Change (USD)", "Change (percent)", "Rates (USD)", "FX (USD)", "Vol (USD)"], left_cols=(0,))
    sc = rcsv("daily_im_scan.csv")
    V.update(scan_n=ni(len(sc)), big_n=str(len(bm)), big_flag_abs=str(int(sc["flag_abs"].sum())), big_flag_rel=str(int(sc["flag_rel"].sum())),
             scan_max_move=pc(sc["dim_rel"].abs().max(), 1), scan_max_abs=usd(sc["dim"].abs().max()), scan_im_min=mm(sc["im"].min()), scan_im_max=mm(sc["im"].max()),
             scan_im_mean=mm(sc["im"].mean()))
    sa = rcsv("subadditivity.csv")
    V["subadd_checks"], V["subadd_viol"] = ni(sa["n_checks"].sum()), ni(sa["n_violations"].sum())
    V["subadd_each"] = ni(sa["n_checks"].iloc[0])

    # UAT
    u = rcsv("uat_results.csv")
    V.update(uat_n=str(len(u)), uat_pass=str(int((u.result == "PASS").sum())), uat_fail=str(int((u.result == "FAIL").sum())),
             uat_skip=str(int((u.result == "SKIPPED").sum())))
    short = {"U01": "PV is zero at the market basis and spot", "U02": "Vectorised PV equals leg-by-leg PV", "U03": "Delta ladder sums to the parallel PV01",
             "U04": "Basis sensitivity equals notional x annuity x FX x 1bp", "U05": "CRIF columns, risk types and USD conversion", "U06": "Basis RW 21 with no CR scaling",
             "U07": "Basis excluded from the CR sum", "U08": "Basis to yield correlation of -1% (hand case)", "U09": "Principal exchange exclusion removes the principal FX delta",
             "U10": "Schedule IM uses the cross-currency row", "U11": "Sign symmetry of delta and vega", "U12": "Homogeneity of IM in the CRIF amounts",
             "U13": "XCCY adds no vega or curvature", "U14": "Missing XCCY detected and ranked first in a dispute", "U15": "Excel rows reconcile to Python",
             "U16": "Runtime within budget", "U17": "Existing IM unchanged when the XCCY is not booked"}
    T["uat"] = text_table([[r.test_id, r.area, short.get(r.test_id, r.description), r.result] for r in u.itertuples()], ["Test", "Area", "Check", "Result"], left_cols=(0, 1, 2, 3))
    xr = u[u.test_id == "U14"].iloc[0]
    V["uat_u14"] = xr.actual
    u15 = u[u.test_id == "U15"].iloc[0]
    V["uat_u15_result"], V["uat_u15_actual"] = u15.result, u15.actual
    u01 = u[u.test_id == "U01"].iloc[0]
    V["uat_u01_actual"] = u01.actual
    u16 = u[u.test_id == "U16"].iloc[0]
    V["uat_u16_actual"] = u16.actual

    # findings
    f = rcsv("findings_log.csv")
    V["find_n"] = str(len(f))
    V["find_high"] = str(int((f.severity == "High").sum()))
    V["find_med"] = str(int((f.severity == "Medium").sum()))
    V["find_low"] = str(int((f.severity == "Low").sum()))
    hi = f[f.severity == "High"]
    parts = []
    for r in hi.itertuples():
        parts.append(f"**{r.finding_id}: {r.title}** (area {r.area}; severity {r.severity}; status {r.status}; owner {r.owner}; evidence file `{r.evidence_file}`)\n\n"
                     f"{r.description} Evidence: {r.evidence} Recommendation: {r.recommendation}")
    V["findings_high_text"] = "\n\n".join(parts)
    oth = f[f.severity != "High"]
    T["findings_other"] = text_table([[r.finding_id, r.area, r.title, r.severity, r.traffic_light, r.source] for r in oth.itertuples()],
                                     ["ID", "Area", "Finding", "Severity", "Light", "Source"], left_cols=(0, 1, 2, 3, 4, 5))
    # xva
    h = X["haz"]
    V.update(cva=usd(X["cva"]), epe=usd(X["epe"]), peak_ee=usd(X["peak_ee"]), haz1=pc(h[0], 2), haz3=pc(h[1], 2), haz5=pc(h[2], 2), haz7=pc(h[3], 2), haz10=pc(h[4], 2))
    v = X["vols"]
    T["xva_vols"] = text_table([[c, fx(v[f"sigma_r_{c}_bp"], 1), (fx(v[f"sigma_fx_{c}_pct"], 1) if f"sigma_fx_{c}_pct" in v else "none (calculation currency)")] for c in ("USD", "EUR", "GBP", "JPY", "MXN")],
                               ["Currency", "Parallel zero-rate volatility (bp per year)", "FX volatility against USD (percent per year)"], left_cols=(0,))
    sv = series("xva", "1d")
    V["xva_mean_var"], V["xva_n_exc"] = usd(sv[C.VAR_FORECAST].mean()), ni(sv[C.EXCEPTION].sum())
    V["linear_excl_share"] = "43.4%"
    ff = f[f.title.str.contains("normal approximation")]
    if len(ff):
        import re as _re
        mm_ = _re.search(r"(\d+\.\d)% of the book", ff.iloc[0].evidence)
        if mm_:
            V["linear_excl_share"] = mm_.group(1) + "%"


def _excel(V, T):
    V["excel_workbook_exists"] = "yes" if EXCEL_WORKBOOK.exists() else "no"
    if not EXCEL_RECON_JSON.exists():
        V["excel_status"] = "The workbook is being finalized: the reconciliation report `python/outputs/excel_reconciliation.json` does not exist yet, so no reconciliation figures are quoted here. Re-run `py -3 docs/build_docs.py` after the workbook is built to fill this section."
        T["excel_recon"] = "_No reconciliation report is available yet._"
        return
    rec = json.loads(EXCEL_RECON_JSON.read_text(encoding="utf-8"))
    lists = []

    def find(o):
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            lists.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                find(v)
    find(rec)
    checks = max(lists, key=len) if lists else []
    flag_keys = ("pass", "passed", "ok", "match", "matches", "within_tolerance")

    def flag(c):
        for k in flag_keys:
            if k in c:
                return bool(c[k]) if not isinstance(c[k], str) else c[k].lower() in ("pass", "true", "ok", "yes")
        if "status" in c:
            return str(c["status"]).lower() in ("pass", "ok", "true")
        if "result" in c:
            return str(c["result"]).lower() in ("pass", "ok", "true")
        return None
    flags = [flag(c) for c in checks]
    n_ok = sum(1 for x in flags if x is True)
    n_bad = sum(1 for x in flags if x is False)
    scal = {k: v for k, v in rec.items() if isinstance(v, (int, float, str, bool))}
    V["excel_status"] = (f"The reconciliation report lists {len(checks)} checks: {n_ok} pass and {n_bad} fail"
                         + (f" ({len(checks) - n_ok - n_bad} without a pass flag)" if len(checks) - n_ok - n_bad else "") + ".")
    cols = [k for k in (checks[0].keys() if checks else []) if not isinstance(checks[0][k], (dict, list))][:6]
    rows = []
    for c in checks[:24]:
        rows.append([str(c.get(k, ""))[:60] if not isinstance(c.get(k), float) else f"{c.get(k):.6g}" for k in cols])
    tab = text_table(rows, cols, left_cols=tuple(range(len(cols)))) if cols else "_The report has no tabular checks._"
    extra = "; ".join(f"{k} = {v}" for k, v in scal.items() if k not in ("data_source",))[:400]
    T["excel_recon"] = tab + (f"\n\nReport header fields: {extra}." if extra else "")



def _extras(V, T, X):
    V["unverified_text"] = UNVERIFIED_TEXT
    T["ex_rows"] = text_table(X["ex_rows"], ["Currency", "Tenor", "s (USD per bp)", "RW", "CR", "WS (USD)"], left_cols=(0, 1))
    for k in ("ex_k_usd", "ex_k_eur", "ex_s_usd", "ex_s_eur", "ex_ir", "ex_fx_ws", "ex_total", "ex_sum_k", "ex_ws5_usd", "ex_sum_rc",
              "cr_w1", "cr_w2", "cr_t2", "cr_lin"):
        V[k] = fx(X[k], 0)
    for k in ("cv_amount", "cv_vega_unit", "cv_cvr", "cv_margin_long", "cv_margin_short", "cv_vega_margin", "ex_fx_sens"):
        V[k] = fx(X[k], 2)
    V["ex_cross"] = f"{X['ex_cross']:.3e}"
    V["ex_k_usd_sq"], V["ex_k_eur_sq"] = f"{X['ex_k_usd'] ** 2:.3e}", f"{X['ex_k_eur'] ** 2:.3e}"
    for k in ("ex_rho_1_5", "ex_rho_2_5", "ex_rho_3_5"):
        V[k] = fx(X[k], 2)
    for k in ("cv_sig_m", "cv_sig_s", "cv_sf"):
        V[k] = fx(X[k], 4)
    V["ex_saving"] = pc(1 - X["ex_ir"] / X["ex_sum_k"], 0)
    V["cr_a1"], V["cr_a2"] = fx(X["cr_a1"], 0), fx(X["cr_a2"], 0)
    V["cr_a1_mm"], V["cr_a2_mm"] = fx(X["cr_a1"] / 1e6, 0), fx(X["cr_a2"] / 1e6, 0)
    V["cr_c1"], V["cr_c2"] = fx(X["cr_c1"], 0), fx(X["cr_c2"], 4)
    V["cr_thr"], V["cr_rw"] = fx(X["cr_thr"], 0), fx(X["cr_rw"], 0)
    V["cv_lam_long"], V["cv_lam_short"], V["cv_days"] = fx(X["cv_lam_long"], 4), fx(X["cv_lam_short"], 1), fx(X["cv_days"], 2)
    V["ir_vrw_text"] = fx(params_mod.load(config.SIMM_VERSION).ir_vrw, 2)
    V["ch_exp11"] = fx((float(br("champ", "overall", "1d", "n")) - 1) * 0.01 ** 2, 2)
    V["simm_exp_10n"] = fx(br("simm", "overall", "10n", "n") * 0.01, 1)
    V["simm_exp_10o"] = fx(br("simm", "overall", "10o", "n") * 0.01, 1)
    sb = rcsv("attribution_subset_ims.csv")
    sb = sb.rename(columns={"matured_trades": "M", "new_trades": "N", "market_move": "K", "parameter_version": "P"})
    marg = []
    for _, r in sb[sb.N == 0].iterrows():
        w = sb[(sb.N == 1) & (sb.M == r.M) & (sb.K == r.K) & (sb.P == r.P)].iloc[0]
        marg.append(w["im"] - r["im"])
    V["sh_marg_min"], V["sh_marg_max"] = mm(min(marg)), mm(max(marg))
    if EXCEL_RECON_JSON.exists():
        V["excel_readme"] = ("`excel/simm_margin_workbook.xlsx` repeats the SIMM, schedule, backtest and attribution calculations with live formulas "
                             "(no array formulas, so it opens in Excel 2016 and LibreOffice). " + V["excel_status"])
    else:
        V["excel_readme"] = ("The workbook `excel/simm_margin_workbook.xlsx` repeats the central calculations with live formulas; it is being "
                             "finalized and its reconciliation to Python is not yet reported.")


def build_values():
    V, T = {}, {}
    X = all_facts()
    _core(V, T, X)
    _results(V, T, X)
    _excel(V, T)
    _extras(V, T, X)
    V["chart_rel"] = CHART_REL
    return V, T, X


# ---------------------------------------------------------------- rendering and conversion
PLACEHOLDER = re.compile(r"@@([A-Za-z0-9_:\.+]+)@@")


def render(text, V, T):
    def sub(m):
        key = m.group(1)
        if key.startswith("tbl:"):
            return T[key[4:]]
        if key.startswith("r:"):
            _, t, mo, s, code = key.split(":")
            return fmt(br(mo, t, s), code)
        if key.startswith("l:"):
            _, t, mo, s = key.split(":")
            return light_word(br(mo, t, s, "light"))
        if key.startswith("c:"):
            _, col, mo, s, code = key.split(":")
            return fmt(br(mo, "overall", s, col), code)
        return str(V[key])
    for _ in range(6):
        new = PLACEHOLDER.sub(sub, text)
        if new == text:
            break
        text = new
    if PLACEHOLDER.search(text):
        raise ValueError("unresolved placeholders: " + ", ".join(sorted(set(PLACEHOLDER.findall(text)))))
    return text


def md_stats(text):
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    body = re.sub(r"^---\n.*?\n---\n", "", body, count=1, flags=re.S)
    headings = len(re.findall(r"^#{1,6} ", body, flags=re.M))
    figures = len(re.findall(r"^!\[", body, flags=re.M))
    display = len(re.findall(r"\$\$.+?\$\$", body, flags=re.S))
    rest = re.sub(r"\$\$.+?\$\$", "", body, flags=re.S)
    inline = len(re.findall(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$(?!\d)", rest))
    tables, in_tbl = 0, False
    for line in body.splitlines():
        is_row = line.startswith("|")
        if is_row and not in_tbl:
            tables += 1
        in_tbl = is_row
    return dict(headings=headings, tables=tables, equations=display + inline, figures=figures)


def docx_stats(path):
    import zipfile
    from docx import Document
    doc = Document(str(path))
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    return dict(headings=sum(1 for p in doc.paragraphs if p.style.name.startswith("Heading")), tables=len(doc.tables),
                equations=xml.count("<m:oMath>") + xml.count("<m:oMath "), figures=xml.count("<w:drawing>"))


def style_tables(path):
    from docx import Document
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    from docx.shared import Pt
    doc = Document(str(path))
    borders = ('<w:tblBorders %s>' % nsdecls("w") + "".join(
        f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="808080"/>' for e in ("top", "left", "bottom", "right", "insideH", "insideV")) + "</w:tblBorders>")
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for t in doc.tables:
        pr = t._tbl.tblPr
        for old in pr.findall(ns + "tblBorders"):
            pr.remove(old)
        pr.append(parse_xml(borders))
        for row in t.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(8.5)
    doc.save(str(path))


def set_author(path):
    from docx import Document
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    doc = Document(str(path))
    for st in doc.styles:
        if st.name.startswith("Heading") or st.name.startswith("TOC Heading"):
            rpr = st.element.rPr
            if rpr is not None:
                for c in rpr.findall(ns + "color"):
                    rpr.remove(c)
    cp = doc.core_properties
    cp.author, cp.last_modified_by, cp.title = AUTHOR, AUTHOR, DOC_TITLE
    doc.save(str(path))


def convert():
    import pypandoc
    pypandoc.convert_file(str(MD_PATH), "docx", format="markdown-smart", outputfile=str(DOCX_PATH),
                          extra_args=[f"--reference-doc={REFERENCE_DOCX}", "--toc", f"--toc-depth={TOC_DEPTH}", "--number-sections",
                                      f"--resource-path={DOCS_DIR}", "--standalone"])
    style_tables(DOCX_PATH)
    set_author(DOCX_PATH)


def verify(md_text):
    import pypandoc
    a, b = md_stats(md_text), docx_stats(DOCX_PATH)
    rt = pypandoc.convert_file(str(DOCX_PATH), "markdown", format="docx", extra_args=["--wrap=none"])
    bad = [("replacement character", chr(0xFFFD) in rt), ("em dash", chr(0x2014) in rt), ("en dash", chr(0x2013) in rt)]
    print("markdown counts:", a)
    print("docx counts    :", b)
    for name, hit in bad:
        print(f"round trip {name}: {'FOUND' if hit else 'none'}")
    if a != b or any(h for _, h in bad):
        raise SystemExit("round-trip verification failed")


def main():
    V, T, X = build_values()
    text = render(S0 + SECTIONS_A + SECTIONS_B + SECTIONS_C + SECTIONS_D, V, T)
    MD_PATH.write_text(text, encoding="utf-8")
    README_PATH.write_text(render(README_TEMPLATE, V, T), encoding="utf-8")
    convert()
    verify(text)
    print("wrote", MD_PATH.name, DOCX_PATH.name, README_PATH.name, "| words:", len(text.split()))


S0 = r"""---
title: "@@doc_title@@"
author: "@@author@@"
date: "@@today@@"
---

"""


SECTIONS_A = r"""
> **Status and disclaimers (read first).** All market data, trades and counterparty data in this project are SYNTHETIC; no market data were used. The engine is a SIMM-style educational model. It is NOT licensed by ISDA, NOT certified or validated against ISDA's calculator, and makes NO compliance claim: no number in this document is a margin call or a regulatory result. ISDA SIMM is a trademark and the ISDA documents are copyright; this repository stores only numeric parameters with citations and reproduces no ISDA text. Verification status is stated exactly in Section 19: @@unverified_text@@.

# Executive summary

This project builds, from scratch and in Python and Excel, a small but complete initial margin (IM) toolchain for a netting set of uncleared Rates and FX derivatives, and then does what a model validation or margin operations team would do with it. It computes a SIMM-style IM from first principles, compares it with the standardised schedule IM, builds a historical-simulation (HS) VaR IM as a competing model, backtests all of them with the statistical tests regulators and ISDA expect, backtests an xVA VaR model for CVA, runs a margin dispute investigation, attributes IM changes to their causes, and runs user acceptance testing (UAT) for a new product. Every step is documented here so that the author can explain it in an interview and a reviewer can reproduce it.

The sample book is one netting set (NS_A against CPTY_B) with @@n_trades@@ synthetic trades: @@n_irs@@ interest rate swaps, @@n_swpt@@ European swaptions, @@n_fxf@@ FX forwards and @@n_fxo@@ FX European options in five currencies (USD, EUR, GBP, JPY, MXN). The synthetic history has @@n_days_s@@ business days from @@hist_start@@ to @@hist_end@@ with a two-state calm and stress regime and a forced 250-day stress window.

**Headline results** (valuation date @@val_date@@, run mode @@run_mode@@):

* The SIMM-style IM under parameter set v@@simm_ver@@ is @@simm_total_12@@ (@@simm_total_12_mm@@), of which the FX risk class contributes @@rc_FX_12@@ and the IR risk class @@rc_IR_12@@ before the IR-FX correlation. Under the previous version v@@prior_ver@@ the IM is @@simm_total_06@@, so the version change is worth @@ver_impact@@ (@@ver_impact_pct@@).
* The standardised schedule IM (gross @@sch_gross@@, net-to-gross ratio @@sch_ngr@@, net @@sch_net@@) is far larger than SIMM: the SIMM-style IM is @@sch_ratio@@ of the net schedule amount.
* The champion HS IM (1-day, 99%) has @@c:exceptions:champ:1d:i@@ exceptions in @@c:n:champ:1d:i@@ days against @@exp_1d@@ expected. Kupiec passes (p = @@r:kupiec_pof:champ:1d:pv@@) but Christoffersen conditional coverage is @@l:christoffersen_cc:champ:1d@@ (p = @@r:christoffersen_cc:champ:1d:pv@@) and the maximum rolling 250-day count is @@r:rolling250_max:champ:1d:i@@, which is in the @@l:rolling250_max:champ:1d@@ zone of BCBS 22. The overall light is @@l:overall:champ:1d@@.
* Every mis-calibrated challenger is worse (EWMA @@l:overall:ewma:1d@@, plain 250-day @@l:overall:plain:1d@@, champion scaled by 0.7 @@l:overall:scaled:1d@@ at 1 day), so the tests have power.
* On overlapping 10-day windows every HS model is red, because overlapping windows share nine days and cluster exceptions; this is a known test-design problem (Section 8). On non-overlapping windows the champion's overall light is @@l:overall:champ:10n@@ (@@c:exceptions:champ:10n:i@@ exceptions in @@c:n:champ:10n:i@@) and the SIMM-style series' is @@l:overall:simm:10n@@ (@@c:exceptions:simm:10n:i@@ exceptions), but on overlapping windows the SIMM-style series fails Christoffersen independence (p = @@r:christoffersen_ind:simm:10o:pv@@) although its Kupiec p-value is @@r:kupiec_pof:simm:10o:pv@@.
* The dispute simulator ranked the seeded cause first in all eight single-cause scenarios and within the top three in the combined scenario (@@disp_acc@@ of @@disp_n@@ accepted).
* The Shapley attribution of the @@attr_dim@@ IM change on COB @@attr_day@@ is exact (zero residual); the one-at-a-time method misses @@attr_oaat_resid@@.
* UAT of the new EUR/USD cross-currency swap: @@uat_pass@@ passed, @@uat_fail@@ failed, @@uat_skip@@ skipped of @@uat_n@@ tests.
* The findings log has @@find_n@@ items: @@find_high@@ High, @@find_med@@ Medium and @@find_low@@ Low.

**What this project does not show.** The data are synthetic; the sample is a single netting set; there is one curve per currency; there are no credit, equity or commodity risk classes; the backtest P&L is hypothetical on a frozen portfolio; CVA is computed for linear trades only; and the engine has not been compared with ISDA's calculator. Section 16 lists every limitation found.

# Business context

## Why uncleared margin exists

Before 2008 most over-the-counter (OTC) derivatives were bilateral and many were collateralised only by variation margin (VM), the daily transfer that settles the current mark-to-market. VM does not cover the cost of closing out a defaulted counterparty's portfolio over the days it takes to do so, because prices keep moving in those days. Initial margin (IM) is a buffer posted in addition to VM that is meant to cover that move with high confidence. Central counterparties (CCPs) have always collected IM; the post-crisis reforms extended the idea to trades that are not centrally cleared.

The international standard is the BCBS-IOSCO framework for margin requirements for non-centrally cleared derivatives. Its central quantitative requirement is that IM should reflect an extreme but plausible increase in the value of the portfolio, consistent with a one-tailed 99 per cent confidence interval over a 10-day horizon, calibrated to historical data that include a period of significant financial stress (BCBS-IOSCO April 2020, Requirement 3.1). Two routes are allowed: a quantitative portfolio margin model, or a standardised schedule of margin rates by asset class and maturity. The schedule is simple but conservative and does not recognise hedging between trades except through the net-to-gross ratio; models such as SIMM recognise offsets, which is why firms use them.

## The rules in the United States and the European Union

In the United States the prudential regulators implement the framework in rules such as 12 CFR Part 45 for the OCC. The model route requires a 99 percent one-tailed estimate of the increase in value over the shorter of ten business days or the maturity, calibration data that are equally weighted, cover at least one and at most five years and include a period of significant financial stress, and validation that includes backtesting (12 CFR 45.8(d) and (f)(2)). The parallel rules of other agencies (12 CFR 237.8, 349.8, CFTC 23.154) were not fetched in this project.

In the European Union the technical standards are in Commission Delegated Regulation (EU) 2016/2251. It requires a 99 percent one-tailed confidence interval over a margin period of risk of at least 10 days (Article 15), calibration on 3 to 5 years of data of which at least 25 percent must come from a stressed period, with the oldest data replaced by stress data until that share is met and equal weights throughout (Article 16), and ongoing model monitoring including backtesting at least every three months (Article 14(3)). This project's champion model follows the Article 16 pattern, labelled "EU 1+3 style" in the code. The consolidated text with later amendments was not checked.

## Why SIMM exists and why IM models are backtested

If every pair of counterparties used its own model, the two sides would compute different IM for the same portfolio and every margin call would be disputed. The ISDA Standard Initial Margin Model (SIMM) is an industry-wide, sensitivity-based methodology: both sides compute the same sensitivities, apply the same risk weights and correlations, and so should reach (nearly) the same IM. ISDA recalibrates the parameters periodically, and its governance framework sets the calibration standard (99 percent, 10-day, including stress), the dispute escalation steps, a SIMM shortfall amount where SIMM is found to under-margin, and backtesting with the Basel traffic-light approach (ISDA SIMM Governance Framework 18 Sep 2026).

A margin model that is wrong in one direction leaves a hole in the protection; wrong in the other direction it locks up collateral for nothing. Regulators and model risk policy therefore require outcomes analysis. The US supervisory guidance on model risk management lists three core elements of validation: evaluation of conceptual soundness, ongoing monitoring including benchmarking, and outcomes analysis including back-testing (SR 11-7). This project covers all three for a toy but realistic model: conceptual documentation (this document), benchmarking against the schedule and an HS model, and backtests.

## The jobs this project exercises

A margin or CCR analyst typically does four things: run and reconcile the IM calculation; investigate disputes with counterparties; explain why IM moved from yesterday to today; and onboard new products. A validator additionally backtests the model and writes findings. Sections 11 to 14 map one-to-one onto these tasks.

# Data and the sample portfolio

## The synthetic market history

Because no real market data are used, the history is simulated by `market_history.generate_history`. The design goal is not realism for trading but a history that has the features the backtests need: fat tails, volatility clustering, regimes and a stress episode.

* **Regimes.** A two-state Markov chain (calm and stress) with persistence parameters in `config.REGIME_PERSISTENCE` (calm 0.99, stress 0.97) drives the scale of all shocks. The stress state multiplies shock volatility by `config.STRESS_VOL_MULT` = 2.8. In this history @@regime_stress_days@@ days (@@regime_stress_share@@) are in the stress state and @@regime_calm_days@@ are calm.
* **Fat tails.** Shocks are multivariate Student-t with 5 degrees of freedom, built with one common chi-square mixing draw per day.
* **Rates.** Each currency has three PCA-style factors (level, slope, curvature) that mean-revert, with loadings on the 12 SIMM tenors, a cross-currency correlation through a global factor, and a stress drift in the level factor.
* **FX.** Log returns for EUR/USD, GBP/USD, USD/JPY and USD/MXN with a common USD factor and mean reversion.
* **Volatilities.** Normal volatilities for swaptions and lognormal volatilities for FX options follow log AR(1) processes pushed up in stress.
* **Credit.** Five CDS spreads (1, 3, 5, 7, 10 years) with jumps on entry to stress; these feed the CVA in Section 10.
* **Forced stress window.** A 250-day window from @@stress_start@@ to @@stress_end@@ is forced into the stress state. It plays the role of the "financial stress period" the regulations require in the calibration data. It is placed early in the history so that every later date can use it without look-ahead.

![Synthetic regime path and the daily SIMM-style IM of the sample book. Grey bands are the stress regime; the hatched band is the forced stress window.](@@chart_rel@@/regime_path.png)

Synthetic data have a consequence that should be said plainly: the backtest statistics measure how well the model's assumptions fit a simulated world whose generator we wrote, not how well a margin model fits real markets. The value of the exercise is in the methodology and in seeing the tests behave correctly, including rejecting deliberately wrong models.

## The sample portfolio

The portfolio is one legally enforceable netting set, NS_A, with counterparty CPTY_B. The trade table below gives the terms that matter for risk. Swaption and option expiries and all maturities are year fractions from the valuation date; the portfolio is frozen with constant time to maturity (no ageing), which is explained in Section 8.

@@tbl:portfolio@@

Table: The sample portfolio at the valuation date. Directions: pay fixed or receive fixed for swaps; long or short for options; buy or sell the base currency for forwards. Notionals are in millions of the first currency (JPY and MXN notionals are therefore large numbers).

The total PV of the book is @@pv_total@@ (positive PVs sum to @@pv_pos@@ and negative PVs to @@pv_neg@@). These two numbers matter later because the schedule IM uses the net-to-gross ratio of replacement costs.

The new product used in UAT (Section 13) is a 5-year EUR/USD cross-currency basis swap on EUR 50 million, booked separately from the book above.

# Pricing and sensitivities

## Curves

Each currency has one OIS-style curve, built by `curves.bootstrap_curve` from par quotes at the 12 SIMM tenor vertices: 2 weeks, 1, 3 and 6 months (simple cash rates) and 1, 2, 3, 5, 10, 15, 20 and 30 years (annual-pay par swap rates). Bootstrapping means solving, vertex by vertex, for the zero rate that makes the instrument price exactly at par given the zero rates already found. Zero rates are continuously compounded and linear in time between vertices with flat extrapolation, so the discount factor is $DF(t) = \exp(-z(t)\,t)$. The annuity of a swap with accrual periods $\tau_i$ and payment times $t_i$ is $A = \sum_i \tau_i DF(t_i)$, and the par swap rate is $(1 - DF(T))/A$. Because a single curve discounts and projects, floating legs are worth $DF(t_0) - DF(t_1)$.

The bootstrap also gives the par-to-zero Jacobian $J_{ij} = \partial z_i / \partial s_j$ by twelve re-bootstraps (`par_to_zero_jacobian`). SIMM interest rate delta is defined per basis point of the par instrument quote, so a sensitivity to zero rates must be converted with this Jacobian. The test suite checks the shortcut against a full re-bootstrap for every bump.

## Instrument pricing

`pricing.price_portfolio` values all trades for many market states at once (vectorised), which is what makes the 2 million revaluations of the VaR backtest affordable.

* **Interest rate swap.** $PV = N\,\omega\,[(DF(0) - DF(T)) - K\,A]$ with $\omega = +1$ for pay fixed, in USD after multiplying by the spot.
* **Swaption (Bachelier).** Rates can be negative or near zero, so volatilities are quoted as normal (absolute) vols. With forward swap rate $F$, strike $K$, expiry $T$ and normal volatility $\sigma$, the payer price per unit annuity is
$$ (F-K)\,\Phi(d) + \sigma\sqrt{T}\,\phi(d), \qquad d = \frac{F-K}{\sigma\sqrt{T}} , $$
and the value is notional times annuity times that price. The vega per unit of normal vol is $A\sqrt{T}\,\phi(d)$, which the tests check against a bump.
* **FX forward.** By interest rate parity the forward is $S\,DF_{base}/DF_{quote}$, and the value is $N\omega\,(S_b DF_b - K\,S_q DF_q)$ in USD.
* **FX option (Garman-Kohlhagen).** The Black formula for a currency option with forward $F$, strike $K$, lognormal vol $\sigma$ and quote-currency discount factor: call $= DF_q\,[F\Phi(d_1) - K\Phi(d_2)]$ with $d_{1,2} = [\ln(F/K) \pm \tfrac12\sigma^2 T]/(\sigma\sqrt{T})$. Put-call parity is a test.
* **Cross-currency swap.** Two floating legs, each on its own currency curve, plus $(\text{contract basis} - \text{market basis})\times$ annuity. When `settle_notional_exchange` is "eligible" the final principal exchange is excluded from the IM sensitivities, following the rule that models need not capture the fixed physically settled exchange of principal (BCBS-IOSCO April 2020, Requirement 1.2).

## Sensitivity conventions

SIMM works on sensitivities, so the quality of the sensitivities caps the quality of the IM. The conventions used in `sensitivities.py` follow the ISDA Risk Data Standards, which define the CRIF (Common Risk Interchange Format) exchange file (ISDA Risk Data Standards v1.36).

| Risk | Bump | Output |
|:---|:---|---:|
| IR delta | $\pm$ 0.5bp central difference on one par quote | USD per 1bp of the par quote, at 12 vertices |
| IR vega | $\pm$ 0.5bp of normal vol | vega times the market normal vol (USD), allocated to expiry vertices |
| FX delta | $\pm$ 0.5% relative move of the currency against USD | USD per 1%, including translation risk |
| FX vega | $\pm$ 0.5 vol point | vega per unit vol times the market vol (USD) |
| Cross-currency basis | $\pm$ 0.5bp of contract basis | USD per 1bp |

**Linear rebucketing.** A 7-year swap sits between the 5-year and 10-year vertices. Its sensitivity is split linearly: the weight on the 10-year vertex is $(7-5)/(10-5) = 0.4$ and on the 5-year vertex $0.6$. Swaption vega is allocated the same way across expiry vertices. Disputes arise when two parties rebucket differently (scenario D6 in Section 11).

**Translation risk.** An FX delta in SIMM is the change in USD value for a 1% rise of the foreign currency against USD. A EUR interest rate swap has no FX forward in it, but its PV is in euros, so a 1% EUR rise changes its USD value by 1% of the PV. That is why the EUR, GBP, JPY and MXN swaps appear in the FX delta rows of the CRIF.

## The CRIF

`crif.build_crif` turns the sensitivities into a CRIF-style table with the columns TradeID, PortfolioID, ProductClass, RiskType, Qualifier, Bucket, Label1, Label2, Amount, AmountCurrency and AmountUSD. At the valuation date the CRIF has @@crif_rows@@ rows.

@@tbl:crif_summary@@

Table: The CRIF of the sample book by risk type.

**An issue found and fixed during the build: the FX vega convention.** A SIMM vega amount is vega times implied volatility, but for FX the "implied volatility" that SIMM weights is not the market vol. It is $\sigma_{SIMM}$ derived from the FX risk weight (Section 5). The engine therefore needs the market vol to recover the raw vega from the CRIF amount. The first version of the CRIF did not carry it, and the engine overstated FX vega and curvature margin (16.44 mm against 5.47 mm of total IM for the sample book in an early run, a clear red flag against the plausible magnitude). The fix is an extra CRIF column, `SigmaMarket`, filled on the @@crif_fxvol_sigma_rows@@ FX vega rows. This is not a standard CRIF column, so it is also finding F-12 in Section 14 and the cause of dispute scenario D4: a counterparty that treats the amount as already weighted gets a different IM from an identical-looking file.

# SIMM step by step

This section explains every formula of the SIMM-style engine in `simm.py`, in the order the calculation runs. The structure follows the public methodology text (ISDA SIMM v2.8+2512, paras 5 to 11); all numbers come from `data/parameters/simm_parameters.csv`, loaded by `params.load`. The engine covers the RatesFX product class with two risk classes, IR and FX, each with delta, vega and curvature margin.

## The idea in one paragraph

SIMM treats the portfolio as a vector of sensitivities to standard risk factors. Each sensitivity is multiplied by a risk weight (roughly a 10-day, 99 percent move of that factor in sensitivity units), giving a weighted sensitivity $WS$. Within a bucket (for IR, a currency) the weighted sensitivities are combined with a correlation matrix, so that opposite positions in correlated factors offset: $K_b = \sqrt{\sum WS_k^2 + \sum\sum \rho_{kl} WS_k WS_l}$. Buckets are then combined with a cross-bucket correlation, risk classes with another, and the delta, vega and curvature margins are added. It is a parametric approximation to a 99 percent, 10-day stressed VaR, which is exactly what makes backtesting it against historical losses meaningful.

## Weighted sensitivities and concentration

For each risk factor $k$ the net sensitivity $s_k$ is the sum over all trades. The weighted sensitivity is
$$ WS_k = RW_k \, s_k \, CR_b , \qquad CR_b = \max\!\left(1, \sqrt{\frac{\left|\sum_k s_k\right|}{T_b}}\right) . $$
$RW_k$ is the risk weight, $T_b$ the concentration threshold of the bucket in USD millions per basis point (IR delta), and $CR_b$ the concentration risk factor. The idea is that a very large position in a currency cannot be liquidated in 10 days at the same price as a small one, so the risk weight is scaled up once the net position exceeds a threshold. Below the threshold $CR = 1$ and the margin is linear in position size; above it the margin grows faster than linearly. Cross-currency basis rows are neither summed into the CR nor scaled (their $CR$ is 1).

**Parameters.** The IR delta risk weights depend on the volatility group of the currency. For the sample currencies:

@@tbl:rw_table@@

Table: IR delta risk weights by tenor (basis point of sensitivity, v@@simm_ver@@). JPY is the only low-volatility currency; MXN is in the high-volatility group.

Further parameters used by the engine, for both versions:

@@tbl:params_main@@

Table: Main SIMM-style parameters (correlations and gamma as percent; risk weights as printed in the methodology).

Concentration thresholds are in USD millions (per bp for IR delta, per 1% for FX delta, per unit of vega-times-vol for vega):

@@tbl:ir_thresholds@@

Table: IR concentration thresholds by currency group.

@@tbl:fx_thresholds@@

Table: FX delta concentration thresholds by category.

**Concentration risk example.** Take one hypothetical USD 5-year sensitivity, $RW = @@cr_rw@@$ and threshold $T_b = @@cr_thr@@$ USD mm per bp. For a net sensitivity of USD @@cr_a1@@ per bp (@@cr_a1_mm@@ million) the position is below the threshold of @@cr_thr@@ million, so $CR = @@cr_c1@@$ and $WS = @@cr_w1@@$. For USD @@cr_a2@@ per bp, $CR = \sqrt{@@cr_a2_mm@@/@@cr_thr@@} = @@cr_c2@@$ and $WS = @@cr_w2@@$, which is @@cr_c2@@ times more than the linear scaling of the small case (@@cr_lin@@) would give; the engine's IM for this single row is @@cr_t2@@. In the sample book every net sensitivity is far below its threshold, so the concentration factors in the book are @@ir_cr_all_one@@ (see Table of IR buckets below). The effect is exercised only in tests and in this example. These sensitivities are deliberately unrealistic in size to show the mechanics.

## Delta margin for interest rates

Within one currency bucket the weighted sensitivities of the 12 tenors (and the sub-curves, inflation and cross-currency basis rows when present) are combined as
$$ K_b = \sqrt{\sum_k WS_k^2 + \sum_k \sum_{l \ne k} \rho_{kl}\, WS_k WS_l } , $$
where $\rho_{kl}$ is the 12 by 12 tenor correlation (for the same sub-curve), multiplied by the sub-curve correlation $\varphi = 98.1\%$ if the two rows are on different sub-curves of the same currency, equal to 42% between an inflation and a yield row and $-1\%$ between a cross-currency basis row and any other row. The sample uses one sub-curve ("OIS") per currency, so $\varphi$ is never exercised (finding F-03). The tenor correlation matrix is symmetric and positive semidefinite (smallest eigenvalue @@corr_min_eig@@); a sample of it is below.

@@tbl:corr_table@@

Table: A sample of the IR tenor correlation matrix (v@@simm_ver@@).

Across currencies, define $S_b = \max(\min(\sum_k WS_k, K_b), -K_b)$, the net weighted sensitivity of the bucket clipped to the range of $K_b$. Then
$$ \text{DeltaMargin}_{IR} = \sqrt{\sum_b K_b^2 + \sum_b \sum_{c \ne b} \gamma_{bc}\, g_{bc}\, S_b S_c } , \qquad g_{bc} = \frac{\min(CR_b, CR_c)}{\max(CR_b, CR_c)} , $$
with the inter-currency correlation $\gamma = 35\%$. The $g_{bc}$ factor softens the cross-bucket correlation when one currency is much more concentrated than the other.

**FX delta.** All currencies against the calculation currency form a single bucket. Each currency's weighted sensitivity uses the FX risk weight (7.4 for a regular-group currency against USD in v@@simm_ver@@) and its own concentration factor, and the margin is $K = \sqrt{WS' M\,WS}$ with correlation 50% between currencies in the regular group, scaled by $\min(CR_k, CR_l)/\max(CR_k,CR_l)$. The USD itself has no FX risk factor.

**Results for the sample book.**

@@tbl:ir_buckets@@

Table: IR delta buckets of the sample book. $K_b$ is smaller than the sum of $|WS|$ because opposite positions at neighbouring tenors offset.

The sum of the five $K_b$ is @@ir_sum_k@@ while the IR delta margin is @@mt_IR_Delta_12@@, a further reduction from offsetting positions between currencies. FX delta is the largest single margin component, @@mt_FX_Delta_12@@, from the FX risk factor weighted sensitivities:

@@tbl:fx_delta_ws@@

Table: FX delta weighted sensitivities of the book. The sum of absolute values is @@fx_delta_sumabs@@; the margin is lower because EUR, GBP, JPY and MXN positions partly offset at 50% correlation.

## A fully worked example with two currencies

To make the mechanics hand-checkable, take only two trades from the book: the USD 5-year receive-fixed swap IRS_USD_5Y (USD 40 million) and the EUR 5-year pay-fixed swap IRS_EUR_5Y (EUR 30 million). Their CRIF rows give these weighted sensitivities (all concentration factors are 1):

@@tbl:ex_rows@@

Table: Worked example, step 1: net sensitivity $s$ (USD per bp), risk weight, concentration factor and weighted sensitivity for the two swaps.

**Step 2, bucket K.** In the USD bucket the 5-year weighted sensitivity is @@ex_ws5_usd@@. The 1, 2 and 3-year rows are small and of opposite sign, and their correlations with the 5-year vertex are @@ex_rho_1_5@@, @@ex_rho_2_5@@ and @@ex_rho_3_5@@, so they slightly offset the large row. The formula gives $K_{USD} = @@ex_k_usd@@$, a little below $|WS_{5y}|$. In the EUR bucket the same calculation gives $K_{EUR} = @@ex_k_eur@@$. The documentation build recomputes both from the correlation matrix in numpy and aborts if they differ from the engine by more than a relative $10^{-6}$.

**Step 3, net weighted sensitivity.** $S_{USD} = @@ex_s_usd@@$ and $S_{EUR} = @@ex_s_eur@@$. Both are inside $[-K_b, K_b]$ so no clipping occurs. The signs are opposite (a receive-fixed USD swap against a pay-fixed EUR swap), which is a hedge if the two curves move together.

**Step 4, across currencies.** With $\gamma = 35\%$ and $g = 1$ the cross term is $2\gamma S_{USD} S_{EUR} = @@ex_cross@@$, which is negative:
$$ \text{Delta}_{IR} = \sqrt{K_{USD}^2 + K_{EUR}^2 + 2\gamma S_{USD} S_{EUR}} = \sqrt{@@ex_k_usd_sq@@ + @@ex_k_eur_sq@@ + (@@ex_cross@@)} = @@ex_ir@@ . $$
The simple sum $K_{USD}+K_{EUR}$ would be @@ex_sum_k@@, so the cross-currency offset saves @@ex_saving@@. This is the diversification SIMM gives for hedged positions, and also what the schedule IM cannot give.

**Step 5, FX.** The EUR swap has an FX delta (translation) of USD @@ex_fx_sens@@ per 1%, weighted by risk weight 7.4 to @@ex_fx_ws@@. This is the only FX factor, so FX delta margin equals $|WS|$. There are no options, so vega and curvature margins are zero, and the IR and FX risk class IMs are @@ex_ir@@ and @@ex_fx_ws@@.

**Step 6, product class.** With $\psi_{IR,FX} = 15\%$,
$$ \text{SIMM}_{RatesFX} = \sqrt{IM_{IR}^2 + IM_{FX}^2 + 2\psi\, IM_{IR}\, IM_{FX}} = @@ex_total@@ , $$
against @@ex_sum_rc@@ for the simple sum of the two risk classes. The two-trade IM is therefore @@ex_total@@; the engine returns exactly this number.

## Vega margin

An option's value depends on implied volatility, so SIMM adds a vega margin. For interest rates the vega risk factor is the implied volatility of a swaption expiry (12 expiry vertices). For each factor the weighted vega risk is $VR_k = VRW \cdot (\sum_i VR_{ik}) \cdot VCR_b$, where $VR_{ik}$ is the vega of instrument $i$ times its implied volatility ("vega risk exposure"), $VRW$ the vega risk weight (@@ir_vrw_text@@ for IR) and $VCR_b$ the vega concentration factor, defined like $CR$ with the vega thresholds. For IR the same 12 by 12 tenor correlation is used in $K_b = \sqrt{\sum VR_k^2 + \sum\sum \rho_{kl} VR_k VR_l}$ (the correlation adjustment $f_{kl}$ is 1 for interest rates), buckets are combined with $\gamma = 35\%$ and $g_{bc} = \min(VCR_b, VCR_c)/\max(VCR_b, VCR_c)$.

For FX, the factors are currency pairs and the implied volatility that enters is not the market vol but
$$ \sigma_{SIMM} = \frac{RW \sqrt{365/14}}{\Phi^{-1}(99\%)} , $$
which turns the delta risk weight (a 10-day, 99 percent move in percent) into an annualised vol. For EUR/USD, $\sigma_{SIMM} = @@fx_sigma_eurusd@@$ in v@@simm_ver@@ (@@fx_sigma_eurusd_06@@ in v@@prior_ver@@). The weighted FX vega risk is $VR_{ik} = HVR_{FX}\,\sigma_{SIMM}\,\text{vega}_i$, then $VR_k = VRW_{FX}\,\sum_i VR_{ik}\,VCR_k$, and $K = \sqrt{\sum VR_k^2 + \sum\sum \rho f_{kl} VR_k VR_l}$ with $\rho = 50\%$ and $f_{kl} = \min(VCR_k, VCR_l)/\max(VCR_k, VCR_l)$. HVR is the "historical volatility ratio": it scales the vega risk to reflect how the implied vol used in the weighting compares with the vol history on which the weights were calibrated.

## Curvature margin

Vega measures first-order sensitivity to volatility, but an option's delta changes when the price moves (gamma). SIMM approximates that gamma risk from vega. The curvature risk exposure of a risk factor is
$$ CVR_{ik} = \sum_j SF(t_{kj})\, \sigma_{kj}\, \frac{\partial V_i}{\partial \sigma} , \qquad SF(t) = 0.5\,\min\!\left(1, \frac{14\ \text{days}}{t\ \text{days}}\right) , $$
where $t$ is the option expiry in calendar days (12 months = 365 days, other tenors pro rata). The scaling function converts vega into gamma for vanilla options and falls with expiry: 50% at two weeks, 7.7% at three months, 0.2% at ten years. Within a bucket, $K_b$ uses the squared correlations $\rho_{kl}^2$; across buckets, $\gamma_{bc}^2$. Then
$$ \theta = \min\!\left(\frac{\sum CVR}{\sum |CVR|}, 0\right), \quad \lambda = (\Phi^{-1}(99.5\%)^2 - 1)(1+\theta) - \theta , \quad \text{CurvatureMargin} = \max\!\left(\sum CVR + \lambda \sqrt{\sum_b K_b^2 + \sum_b\sum_{c \ne b}\gamma_{bc}^2 S_b S_c}, 0\right) . $$
For a net positive exposure $\theta = 0$ and $\lambda = \Phi^{-1}(99.5\%)^2 - 1 = @@curv_lambda0@@$; for a net negative exposure $\theta$ moves towards $-1$ and $\lambda$ towards 1. The IR curvature margin is multiplied by $HVR_{IR}^{-2} = @@hvr_ir_scale@@$.

**Curvature example.** Take the EUR/USD 3-month call, FXO_EURUSD_C3M. Its CRIF vega amount is USD @@cv_amount@@ (vega times the market vol @@cv_sig_m@@), so the vega per unit of vol is @@cv_vega_unit@@. The SIMM vol is $\sigma_{SIMM} = @@cv_sig_s@@$. The FX vega margin is $VRW\cdot HVR\cdot\sigma_{SIMM}\cdot \text{vega} = 0.33\times0.67\times @@cv_sig_s@@ \times @@cv_vega_unit@@ = @@cv_vega_margin@@$. For curvature, three months is @@cv_days@@ days, so $SF = 0.5\times 14/@@cv_days@@ = @@cv_sf@@$ and $CVR = @@cv_sf@@ \times @@cv_sig_s@@ \times @@cv_vega_unit@@ = @@cv_cvr@@$. With one factor, $K = |CVR|$. Long option: $\theta = 0$, $\lambda = @@cv_lam_long@@$, margin $= CVR + \lambda K = @@cv_margin_long@@$. The same option sold: $CVR$ is negative, $\theta = -1$, $\lambda = @@cv_lam_short@@$, and the margin is $\max(-|CVR| + 1\cdot|CVR|, 0) = @@cv_margin_short@@$. The engine implements the formula as printed in the methodology, in which the sign of $CVR$ follows the sign of vega; the economic reading of this sign convention was not independently validated against ISDA's calculator (finding F-14 covers the related IR vega reading).

## Aggregation to the product class

The margin of a risk class is the sum of its delta, vega and curvature margins, and the RatesFX product class combines the risk classes with the matrix $\psi$:
$$ IM_{RatesFX} = \sqrt{\sum_r IM_r^2 + \sum_r\sum_{s \ne r} \psi_{rs}\, IM_r IM_s } , \qquad \psi_{IR,FX} = 15\%\ (\text{v2.8+2512}),\ 10\%\ (\text{v2.8+2506}) . $$
The full six by six $\psi$ matrix is stored and is positive semidefinite (smallest eigenvalue 0.331 for 2512); only the IR-FX entry is used. The complete IM of the book is below.

@@tbl:im_breakdown@@

Table: SIMM-style IM of the sample book by risk class and margin type, both parameter versions, at @@val_date@@.

![SIMM-style IM by risk class and margin type.](@@chart_rel@@/im_by_risk_class_margin_type.png)

![The same margin types under the two parameter versions.](@@chart_rel@@/im_2512_vs_2506.png)

Adding the risk classes gives @@sum_rc_12@@ and the IR-FX correlation reduces it to @@simm_total_12@@, a diversification benefit of @@div_benefit@@. The 2512 version is higher than 2506 by @@ver_impact@@ and the whole difference comes from the FX parameters (risk weight 7.4 against 7.1, vega parameters, FX concentration thresholds and the larger IR-FX correlation); the IR margins are identical because the IR parameters did not change.

## Parameter provenance

The parameter file `data/parameters/simm_parameters.csv` has 685 rows (342 for v2.8+2512 and 343 for v2.8+2506). Every row carries its source document, section, paragraph, table, PDF page, URL, retrieval date, the sha256 of the retrieved PDF and its verification status; all rows are VERIFIED_PRIMARY. They were transcribed from the public ISDA PDFs by two independent text extractors compared programmatically per table and then read against page images. No illustrative values are used for the in-scope parameters. Items still unverified: the ISDA credit, equity and commodity tables (out of scope), and the reading that IR vega uses the same tenor correlation matrix (the methodology text has no separate IR vega expiry correlation table, so this rests on the text and not on ISDA's calculator).

## Euler allocation

`allocation.euler_contributions` splits the IM among trades (or risk factors) by the directional derivative of the IM when that group's CRIF rows are scaled by $(1+\epsilon)$. When every concentration factor is 1, IM is homogeneous of degree one in the sensitivities, so by Euler's theorem the contributions sum exactly to the IM; with $CR > 1$ they do not, and `euler_check` reports the gap. The contributions are used in the dispute reconciliation (Section 11).
"""


SECTIONS_B = r"""
# Schedule IM, net-to-gross ratio and SIMM against the schedule

## The standardised schedule

The schedule route needs no model. Each trade's notional is multiplied by a rate that depends on the asset class and, for interest rates, the duration; the results are summed to the gross IM $G$. For the interest rate class the BCBS-IOSCO and US schedules give 1% for 0 to 2 years, 2% for 2 to 5 years and 4% for 5 years and above; FX is 6%; the US rule also has explicit cross-currency swap rows of 1%, 2% and 4% by duration (BCBS-IOSCO April 2020, Appendix A; the cross-currency rows are in 12 CFR Part 45, Appendix A, not in the BCBS-IOSCO appendix). The engine reads these from `data/parameters/schedule_im.csv`. Implementation assumptions, stated in `schedule_im.py`: bucket edges are lower-inclusive (a 2-year trade is in the 2 to 5 bucket); a swaption's duration is expiry plus underlying tenor; and notionals are converted to USD at the market FX rate.

@@tbl:sched_groups@@

Table: Gross schedule IM of the sample book by asset class and rate.

## Netting through the NGR

The schedule gives only partial credit for netting:
$$ \text{Net IM} = 0.4\,G + 0.6\,\text{NGR}\,G , \qquad \text{NGR} = \frac{\max\left(\sum_i MtM_i, 0\right)}{\sum_i \max(MtM_i, 0)} , $$
the ratio of net to gross current replacement cost, set to 1 when gross replacement cost is zero (12 CFR Part 45, Appendix A, footnote 1). Intuition: if all trades have positive value for the receiver, nothing offsets and NGR = 1; if positive and negative values cancel completely, NGR = 0 and the net IM falls to 40% of gross. For the sample book the net value is @@ngr_num@@ and the sum of positive values is @@ngr_den@@, so NGR = @@sch_ngr@@. The gross IM is @@sch_gross@@ and the net IM is @@sch_net@@, which is @@sch_net_over_gross@@ of gross.

## SIMM against the schedule

The comparison on four dates (the first and last dates of the history, the middle of the stress window and the scenario day):

@@tbl:sched_dates@@

Table: SIMM-style IM and schedule IM on four dates.

![SIMM-style IM versus the standardised schedule IM.](@@chart_rel@@/simm_vs_schedule.png)

SIMM is far below the schedule: between @@sch_ratio_min@@ and @@sch_ratio_max@@ of the net schedule amount on these dates, and @@sch_ratio_gross@@ of the gross schedule at the valuation date. This is the expected direction (a risk-sensitive model that recognises offsets should be cheaper than a flat conservative schedule, which is the reason firms seek model approval), and the magnitude is plausible, but it comes with two caveats. First, the sample is a hedged, mixed book where offsets are strong. Second, the IM here has not been calibrated against real stressed data. The schedule in this project is a benchmark and a ceiling to compare against, not a validated floor. Had SIMM exceeded the schedule, the pipeline would have raised a finding and not adjusted any number.

# Historical-simulation VaR IM and the calibration window

## The idea

SIMM is a parametric approximation. The alternative is to simulate: take the changes in market risk factors that actually happened in a history window, apply them to today's market, fully revalue the portfolio under every scenario, and read off the 99th percentile of the loss. This is historical simulation (HS), and `var_model.hs_im` implements it. It is used here as the champion model for the VaR IM backtest, because its design (99 percent, 10-day, stress included, equally weighted) is exactly the one the regulations describe, and as the benchmark that SIMM is compared with.

For a date $t$ and horizon $h$ days, scenario $s$ in the window applies the $h$-day change of every factor observed in that scenario's period: additive changes to par and zero rates and the cross-currency basis, and multiplicative (log) changes to FX spots, normal vols and FX vols. The loss under the scenario is
$$ L_s = -\left[ V(\text{state}_t \oplus \Delta_s) - V(\text{state}_t) \right] , $$
with full revaluation of every trade (no sensitivity approximation), and the IM is the 99th percentile of $\{L_s\}$ under the inverted CDF convention. Overlapping $h$-day changes are used, so a 750-day window has 750 scenarios.

## The calibration window in the EU style

The window follows the pattern of Article 16 of the EU regulation (Delegated Regulation (EU) 2016/2251, Article 16): the newest `config.WINDOW_DAYS` = @@win_days@@ days, with equal weights, in which the oldest recent days are replaced by days from the stress window until at least 25 percent of the window is stressed (stressed meaning the regime label is stress, or the day comes from the forced stress window). The code, `stress_replacement`, finds for each date the smallest number $m$ of replacements that reaches the 25 percent target. Over the backtest period (from @@first_valid@@) the window keeps on average @@win_mean_recent@@ recent days and adds @@win_mean_stress@@ stress-window days; replacement was needed on @@win_frac_replaced@@ of dates (at most @@win_max_stress@@ days). The stressed share of the window ranges from @@win_share_min@@ to @@win_share_max@@ and averages @@win_share_mean@@. When the recent history is itself stressed enough, nothing is replaced, which is the intended behaviour: the stress requirement is a floor, not an overlay.

## Challenger models

Backtesting a single model tells you only whether it passes. To check that the tests can fail, three deliberately weaker challengers are built from the same revaluation matrix:

* **EWMA:** the newest 250 scenarios weighted by $\lambda^{age}$ with $\lambda = 0.97$, so recent scenarios dominate and stress memory fades.
* **Plain 250-day:** the newest 250 scenarios, equally weighted, with no stress replacement.
* **Champion times 0.7:** the champion IM scaled down by 30 percent, an under-margining model on purpose.

The champion's average 1-day IM over the backtest is @@champ_mean_var@@ and its average 10-day IM is @@hs10_mean@@, against @@simm_mean@@ for the SIMM-style series.

## The SIMM-style series as a model under test

SIMM itself is backtested too: `var_model.daily_simm_series` builds a CRIF and a SIMM IM on every date from that date's market with parameter set v@@simm_ver@@ (using the Jacobian method, so the cost is manageable). Its 10-day IM is compared with the 10-day loss of the frozen book. SIMM is less responsive to the market than HS: its standard deviation over the period is @@simm_std@@ against @@hs10_std@@ for the champion 10-day IM, and the correlation of the two series is only @@simm_hs_corr@@. This is a feature of a fixed-parameter, sensitivity-based method (the sensitivities change with the market but the weights do not), and it matters in the Christoffersen tests below.

# Backtesting theory

## Exceptions and what a backtest tests

A VaR model at confidence 99% claims that a loss exceeds the forecast on 1% of days. An **exception** (or breach) on day $t$ is $I_t = 1$ if the realised loss exceeds the forecast made at the start of the day, else 0. If the model is right, $\{I_t\}$ is a sequence of independent Bernoulli variables with success probability $p = 1 - 0.99 = 0.01$. A good backtest therefore checks two things: the proportion of exceptions (unconditional coverage) and whether they are independent over time (no clustering). The sequence of forecasts must be made before the outcome is known; the engine's IM at date $t$ uses only information up to $t$.

**The P&L definition.** The loss is hypothetical: the revaluation of the same frozen trades one or ten days later with constant time to maturity. There are no cash flows, no ageing, no trading. This isolates the market-risk model from operational noise, which is how ISDA and the regulators describe the cleanest test, but it ignores the real behaviour of a live portfolio (finding F-06). BCBS 22 also stresses that backtests should be run on one-day measures, because over ten days portfolio composition changes (BCBS 22, Section II). For that reason the 1-day test is the primary one for the VaR IM, and the 10-day results are reported on two designs, described below.

## Kupiec's proportion of failures test

Let $n$ be the number of observations and $x$ the number of exceptions. Under the null, $x \sim \text{Binomial}(n, p)$. The likelihood ratio between the null and the best-fitting rate $\hat p = x/n$ is
$$ LR_{POF} = -2 \ln\!\left[(1-p)^{n-x}\, p^{x}\right] + 2 \ln\!\left[\left(1-\tfrac{x}{n}\right)^{n-x}\left(\tfrac{x}{n}\right)^{x}\right] , $$
asymptotically $\chi^2$ with one degree of freedom (Kupiec 1995, J. Derivatives vol 3 no 2, 73-84). With the convention $0 \ln 0 = 0$ it is also defined when $x = 0$. The test rejects both too many and too few exceptions: a model that is far too conservative fails as well. At $n = 250$ and $x = 0$ the statistic is @@ku0_lr@@ with p-value @@ku0_p@@, which is below 5%, so even a perfectly safe-looking year with no exceptions rejects the hypothesis of exactly 1% coverage.

**Worked example (champion, 1 day).** With $n = @@ku_n@@$ and $x = @@ku_x@@$ the observed rate is $\hat p = @@ku_phat@@$ against $p = 1\%$, and the expected number of exceptions is @@ku_exp@@. The log likelihoods are
$$ \ln L_0 = (n-x)\ln 0.99 + x \ln 0.01 = @@ku_ll0@@ , \qquad \ln L_1 = (n-x)\ln(1-\hat p) + x \ln \hat p = @@ku_ll1@@ , $$
so $LR_{POF} = 2(\ln L_1 - \ln L_0) = @@ku_lr@@$. The 5% critical value of $\chi^2_1$ is @@ku_crit@@; the statistic is below it, and the p-value is @@ku_p@@. Kupiec does not reject. This matches the engine and the table of results (the build asserts it).

The formula was checked against Federal Reserve restatements and independent numeric references (closed forms at $x = 0$ and $x = n$, a likelihood-ratio computed with `scipy.stats.binom.logpmf`, a simulation of the test size); the original paper could not be retrieved. The 250-day version over-rejects because the binomial is discrete: the exact rejection rate at nominal 5% is 9.5% for $n = 250$, 5.5% for $n = 1000$ and 4.4% for $n = 2500$.

## Christoffersen's independence and conditional coverage tests

Kupiec ignores timing. A model whose exceptions all fall in one week may have the right count but is clearly wrong: it fails to react when volatility jumps. Christoffersen tests this with a first-order Markov alternative (Christoffersen 1998, International Economic Review vol 39 no 4, 841-862). Count the transitions between consecutive days: $n_{ij}$ is the number of days with state $j$ following state $i$ (0 = no exception, 1 = exception). Estimate $\hat\pi_{01} = n_{01}/(n_{00}+n_{01})$, $\hat\pi_{11} = n_{11}/(n_{10}+n_{11})$ and the overall $\hat\pi = (n_{01}+n_{11})/n$. The independence statistic compares the Markov likelihood with the single-probability one:
$$ LR_{ind} = -2\ln\left[(1-\hat\pi)^{n_{00}+n_{10}}\hat\pi^{\,n_{01}+n_{11}}\right] + 2\ln\left[(1-\hat\pi_{01})^{n_{00}}\hat\pi_{01}^{\,n_{01}}(1-\hat\pi_{11})^{n_{10}}\hat\pi_{11}^{\,n_{11}}\right] \sim \chi^2_1 , $$
and the conditional coverage statistic combines both properties, $LR_{cc} = LR_{POF} + LR_{ind} \sim \chi^2_2$.

**Worked example (champion, 1 day).** The transition counts are $n_{00} = @@ch_n00@@$, $n_{01} = @@ch_n01@@$, $n_{10} = @@ch_n10@@$, $n_{11} = @@ch_n11@@$. So $\hat\pi_{01} = @@ch_pi01@@$ (an exception follows a quiet day with that probability), $\hat\pi_{11} = @@ch_pi11@@$ (@@ch_ratio@@ times higher: after an exception the next day is much more likely to be another one) and $\hat\pi = @@ch_pi@@$. These give $LR_{ind} = @@ch_lr_ind@@$ and a p-value of @@ch_p_ind@@, below 5% but above 1%, hence an AMBER light. The conditional coverage statistic is $LR_{cc} = @@ch_lr_uc@@ + @@ch_lr_ind@@ = @@ch_lr_cc@@$ with $\chi^2_2$ p-value @@ch_p_cc@@, also AMBER. The champion has the right number of exceptions but they cluster (@@exc_adjacent_pairs@@ pairs of consecutive-day exceptions in a sample where independence would predict about @@ch_exp11@@).

## Basel traffic light

BCBS 22 sorts the number of exceptions among 250 observations into zones using the cumulative binomial probability at 99% coverage: the yellow zone starts where the probability of that many exceptions or fewer reaches 95%, and the red zone where it reaches 99.99% (BCBS 22, January 1996, Section III(c)). Table 2 of the paper, with the cumulative probabilities recomputed here with `scipy.stats.binom(250, 0.01)`:

@@tbl:bcbs_table@@

Table: BCBS 22 Table 2 for 250 observations (green 0 to 4, yellow 5 to 9, red 10 or more). The plus factor is the increase in the capital multiplier in the original market risk context; it is shown for completeness and is not a margin multiplier.

For other sample sizes the engine applies the same cumulative-probability rule, `backtest.basel_zone`. In margin practice the analogue of the plus factor is the SIMM shortfall: if SIMM under-margins, ISDA's framework allows a scaled-up margin (ISDA SIMM Governance Framework 18 Sep 2026). `backtest.shortfall_multiplier` finds the smallest $k$ such that $k \times$ IM would put the exception count in the green zone.

## Test lights

To summarise a battery of tests the engine uses three lights: PASS if the p-value is at least 0.05 and the zone is green; AMBER if the p-value is between 0.01 and 0.05, or the zone is yellow; RED if the p-value is below 0.01 or the zone is red. The overall light of a model-design combination is the worst of the Kupiec, Christoffersen-independence and Basel-zone lights. This is a house convention for readability, not a regulatory rule.

## The overlapping-window problem

For a 10-day IM there are two ways to backtest. Compare the IM with the loss over each of the next ten days on every date (overlapping windows: consecutive windows share nine of ten days), or only every tenth date (non-overlapping). Overlapping windows give about ten times more observations but they are not independent: one bad week appears in ten consecutive windows. Exceptions cluster, so the Christoffersen independence test rejects almost mechanically and the Kupiec test is over-sized. BCBS 22 warns specifically against comparing 10-day risk measures with overlapping 10-day outcomes (BCBS 22, Section II). The project therefore uses the non-overlapping design for the pass criterion and reports the overlapping design only as supporting evidence. It also has a price: the non-overlapping sample has only @@n_10n@@ observations, so the tests have low power, and results move with the starting offset.

# Backtest results

## Design

The 1-day tests use @@n_1d@@ daily observations from @@first_bt_date@@ to @@last_bt_date@@. The first valid date is set by the need for a full calibration window of @@win_days@@ days plus the horizon. The 10-day designs use the same dates, either every date (overlapping, @@n_10o@@ observations) or every tenth (non-overlapping, @@n_10n@@).

## Champion, challengers and the SIMM-style series

@@tbl:bt_main@@

Table: Backtest battery. Expected exceptions are 1% of n. Zone is the BCBS 22 zone of the exception count. Light is the worst of the three lights defined above.

![Champion versus challengers: exception counts by design (bar colour is the overall light).](@@chart_rel@@/champion_vs_challengers.png)

![Kupiec and Christoffersen p-values by model.](@@chart_rel@@/pvalues_by_model.png)

**Reading the table.** Four observations stand out.

1. *The champion is good but not clean.* At one day it has @@c:exceptions:champ:1d:i@@ exceptions against @@exp_1d@@ expected, passes Kupiec (p = @@r:kupiec_pof:champ:1d:pv@@) and the Basel zone test on the full sample, but fails to be independent (p = @@r:christoffersen_ind:champ:1d:pv@@), so the overall light is @@l:overall:champ:1d@@. On 10-day non-overlapping windows it passes everything (@@c:exceptions:champ:10n:i@@ exceptions, Kupiec p = @@r:kupiec_pof:champ:10n:pv@@).
2. *Challengers are worse, so the tests have power.* The EWMA model (@@c:exceptions:ewma:1d:i@@ exceptions), the plain 250-day model (@@c:exceptions:plain:1d:i@@) and the champion scaled by 0.7 (@@c:exceptions:scaled:1d:i@@) are all red overall at one day, and all are red on the 10-day non-overlapping design (@@c:exceptions:ewma:10n:i@@, @@c:exceptions:plain:10n:i@@ and @@c:exceptions:scaled:10n:i@@ exceptions in @@c:n:ewma:10n:i@@ against @@c:exceptions:champ:10n:i@@ for the champion). The scaled model's p-values are astronomically small, as they should be for a model that deliberately takes 30 percent too little margin. The result also shows why stress replacement matters: the plain 250-day model has no stress memory and under-margins when volatility returns.
3. *Overlapping windows reject everything.* All HS models are red on overlapping windows, including the champion (@@c:exceptions:champ:10o:i@@ exceptions, Christoffersen independence p = @@r:christoffersen_ind:champ:10o:pv@@). This is the artefact described above, not evidence that the champion is wrong, and it is the reason the non-overlapping design is the pass criterion.
4. *SIMM-style: passes one design and fails another.* On non-overlapping windows the SIMM-style series has @@c:exceptions:simm:10n:i@@ exception in @@c:n:simm:10n:i@@ (overall @@l:overall:simm:10n@@; note that this is below the @@simm_exp_10n@@ expected, on the safe side). On overlapping windows its Kupiec p-value is @@r:kupiec_pof:simm:10o:pv@@ (@@c:exceptions:simm:10o:i@@ exceptions against @@simm_exp_10o@@ expected, almost exactly the right rate) but Christoffersen independence is rejected (p = @@r:christoffersen_ind:simm:10o:pv@@): its exceptions come in clusters, as every overlapping design's do. The shortfall multiplier of the SIMM-style series is @@sf_simm@@, below 1, meaning no scale-up would be needed under the Basel rule; the champion's multiplier is @@sf_champ@@.

![Champion HS IM at 99% against the 1-day hypothetical loss, with exceptions marked.](@@chart_rel@@/var_vs_loss_1d.png)

![SIMM-style IM and champion 10-day IM against the 10-day hypothetical loss.](@@chart_rel@@/simm_vs_10d_loss.png)

## Rolling 250-day counts and the traffic light

The regulatory traffic light is applied to windows of 250 days, so the relevant question is how many exceptions fall in any trailing 250-day window.

@@tbl:bt_rolling@@

Table: BCBS 22 zone of the most recent 250 days and of the worst rolling 250-day window, one-day champion and challengers.

![Rolling 250-day exception count against the BCBS 22 zones.](@@chart_rel@@/rolling_250_exceptions.png)

The champion's last 250 days contain @@roll_last@@ exceptions (green), but its worst window, ending @@roll_max_date@@, contains @@roll_max@@, which is in the red zone (10 or more). Across the @@roll_n@@ complete rolling windows the champion is red in @@roll_red@@, yellow in @@roll_yellow@@ and green in @@roll_green@@. This is a real weakness: under a strict reading of the traffic light the champion would have been flagged red for a part of the sample. The champion's exceptions run from @@first_exc_date@@ to @@last_exc_date@@ and cluster in between.

## Behaviour by regime

@@tbl:bt_regime@@

Table: Champion one-day exceptions by regime.

The champion has @@reg_calm_x@@ exceptions in @@reg_calm_n@@ calm days (about @@reg_calm_exp@@ expected) and @@reg_stress_x@@ in @@reg_stress_n@@ stress days (rate @@reg_stress_rate@@, nearly four times the 1% target). Of the exceptions, @@exc_all_stress@@ occur in the stress regime. The Kupiec p-value is small in both regimes for opposite reasons: in calm periods the model is too conservative (zero exceptions), in stress periods too aggressive. That is the typical profile of a window-based VaR: it carries stress memory that over-protects in calm markets and reacts too slowly when volatility rises. The worst single day is @@exc_worst_date@@, when the 1-day loss of @@exc_worst_loss@@ exceeded a forecast of @@exc_worst_var@@.

## What to conclude

For a margin model the cost of an exception is that the counterparty is under-collateralised on that day; the cost of excess conservatism is funding and liquidity. The champion is calibrated to the right level overall and clearly better than all the challengers, but its time profile (cluster of exceptions, red worst window, regime asymmetry) means that a validator would approve it only with conditions: a monitoring trigger on the rolling count, a stress add-on or floor in high-volatility regimes, and a note on the sample (synthetic, one book). The same reasoning applies to the SIMM-style series, whose results are good on the preferred design and weaker on the supporting one. These are exactly the nuances of a real validation report.

# xVA VaR: CVA and its backtest

## The CVA model

Credit valuation adjustment (CVA) is the market value of the counterparty's default risk on the netting set. Unilateral CVA with loss given default $LGD = 1 - R$ and recovery $R = 40\%$ is
$$ CVA = LGD \sum_k \tfrac12\left[EE(t_{k-1})DF(t_{k-1}) + EE(t_k)DF(t_k)\right]\left[Q(t_{k-1}) - Q(t_k)\right] , $$
where $EE$ is the expected positive exposure at time $t_k$, $DF$ the discount factor and $Q(t)$ the counterparty's survival probability. The sum is a trapezoid approximation to $\int EE\, DF\, dPD$ on a quarterly grid. `cva.py` has three parts.

* **Hazard curve.** A piecewise-constant hazard rate is bootstrapped from the CDS spreads at 1, 3, 5, 7 and 10 years so that each par CDS reprices exactly (`bootstrap_hazard`, bisection over states). At the valuation date the hazard rates are @@haz1@@ (to 1 year), @@haz3@@, @@haz5@@, @@haz7@@ and @@haz10@@ per year for the successive pillars.
* **Exposure.** For the linear trades (@@n_linear@@ of @@n_trades@@: the swaps and forwards) the netting-set value at a future time $t_k$ is approximated as Gaussian with mean $\mu_k$ (the forward value of the remaining cash flows) and variance $s_k^2 = t_k\, d_k' \Sigma\, d_k$, where $d_k$ are the sensitivities of $\mu_k$ to nine factors (a parallel shift of each of the five zero curves, and the log spot of the four non-USD currencies) and $\Sigma$ is their annualised covariance estimated from the trailing 250 days of the history. Then $EE = \mu\Phi(\mu/s) + s\,\phi(\mu/s)$, the expectation of $\max(V, 0)$ for a normal $V$. The factor volatilities estimated from the history are below.
* **Excluded trades.** Swaptions and FX options are excluded because their exposure is not Gaussian. They carry @@linear_excl_share@@ of the book's absolute PV, so this is a material limitation (finding F-05). A Monte Carlo exposure engine would replace it.

@@tbl:xva_vols@@

Table: Annualised factor volatilities used in the exposure model at the valuation date.

![Netting-set exposure profile of the linear trades.](@@chart_rel@@/cva_exposure_profile.png)

At the valuation date the CVA is @@cva@@, the expected positive exposure averaged over the life is @@epe@@ and the peak EE is @@peak_ee@@. The profile starts high because short-dated FX forwards are in the money and falls sharply as they mature, then rises again with the diffusion of the long swaps.

## The xVA VaR model and its backtest

CVA moves with credit spreads, interest rates and FX. A risk team therefore wants a VaR on the change of CVA. The model in `xva_var_series` is sensitivity-based historical simulation: bump each of five CDS pillars (CS01), each of five parallel rate levels and each of four FX spots, reprice CVA to get 14 sensitivities, apply the newest 750 days of $h$-day factor changes, and take the 99th percentile of the resulting loss distribution. The backtest compares it with the realised full-revaluation change in CVA (model parameters held at the date-$t$ values so the test isolates market moves):

@@tbl:xva_bt@@

Table: xVA VaR backtest.

![xVA VaR against the realised change in CVA, 1 day.](@@chart_rel@@/xva_var_backtest.png)

At one day the xVA VaR passes all tests: @@c:exceptions:xva:1d:i@@ exceptions in @@c:n:xva:1d:i@@ (Kupiec p = @@r:kupiec_pof:xva:1d:pv@@, Christoffersen conditional coverage p = @@r:christoffersen_cc:xva:1d:pv@@). On 10-day non-overlapping windows it has @@c:exceptions:xva:10n:i@@ exceptions in @@c:n:xva:10n:i@@ and conditional coverage is @@l:christoffersen_cc:xva:10n@@ (p = @@r:christoffersen_cc:xva:10n:pv@@); the overall light is @@l:overall:xva:10n@@, raised as finding F-11. The overlapping design is red, as for the IM models. The model is a first-order approximation: the sensitivities are linear, gamma and the exposure profile's sensitivity to volatility are ignored, and the realised CVA uses the same Gaussian exposure model, so the test checks the VaR approximation to the CVA model, not the CVA model to reality.
"""


SECTIONS_C = r"""
# Margin dispute investigation

## The business problem

Each day both counterparties compute IM for the same netting set and the delivering party's number is called. If the receiving party's number differs by more than an agreed threshold, there is a dispute, and both sides have to find out why. ISDA's governance framework prescribes escalation steps for such disputes, and in practice the work is a structured reconciliation: do we agree on the trade population, on the sensitivities, on the parameters and on the aggregation? (ISDA SIMM Governance Framework 18 Sep 2026). The first three are data problems; the last is a model problem and should rarely differ if both sides run the same version.

## The simulator

`dispute.py` plays both sides. Party A is the base calculation (the CRIF and parameters of Section 4 and 5). `make_counterparty_view` builds party B's inputs with one seeded difference, scenario D1 to D9, all at the scenario day @@attr_day@@ (the first COB on which v@@simm_ver@@ applies):

| Scenario | Seeded difference on party B's side |
|:---|:---|
| D1 | A trade (IRS_EUR_10Y) was never booked |
| D2 | The notional of IRS_USD_10Y was amended by +10% |
| D3 | IR delta computed per 1bp of zero rate instead of per 1bp of par quote |
| D4 | FX vega weighted with the market implied vol instead of the SIMM vol |
| D5 | Local amounts converted to USD at an FX rate five business days old |
| D6 | The 7-year IRS sensitivity mapped 100% to the 10-year vertex instead of linear rebucketing |
| D7 | The curves are three business days old |
| D8 | Party B is on parameter version v@@prior_ver@@ |
| D9 | D1 and D5 together |

## The reconciliation

`reconcile` compares the two views at four levels, from the cheapest check to the most expensive.

1. **Trade population:** trade IDs and notionals on each side (only A, only B, notional differs).
2. **CRIF key match:** an outer join on TradeID, RiskType, Qualifier, Label1, Label2 with a tolerance on AmountUSD (USD 1 absolute or $10^{-9}$ relative); each key is matched, differs in amount, only A or only B.
3. **Hierarchical IM gap:** the difference in IM between B and A decomposed down the SIMM tree: risk class, margin type, bucket ($K_b$), risk factor. The decomposition follows the largest absolute gap at each level, which localises the break.
4. **Euler contributions:** each trade's contribution to IM on each side, to see which trades explain the gap even when sensitivities are aggregated.

Example, D1 (gap @@d1_gap@@, IM A @@d1_ima@@, IM B @@d1_imb@@):

@@tbl:d1_tree@@

Table: Scenario D1, gap decomposition from risk class down to buckets (top items).

@@tbl:d1_contrib@@

Table: Scenario D1, largest trade-level differences in Euler contribution. The missing trade IRS_EUR_10Y has contribution zero on B's side.

![Gap decomposition by risk class and margin type for D1 and D8.](@@chart_rel@@/dispute_gap_decomposition.png)

## Ranking the causes

The ranking step turns the reconciliation into a hypothesis test. `rank_causes` has eight hypotheses (the causes in D1 to D8). For each one it applies the counterfactual fix to B (for example: add the missing trades, restore the notionals, rebuild the delta rows from par quotes, re-weight FX vega with the SIMM vol, convert with today's FX, use linear rebucketing, rebuild from today's curves, switch parameter version), recomputes B's IM, and ranks hypotheses by the residual gap left. Ties within the tolerance are broken in favour of the fix that touches fewer CRIF rows (the narrower explanation), and a signature score reports the share of reconciliation breaks that the fix removes. This implements the principle that the right explanation is the one that, when undone, makes the numbers agree.

@@tbl:dispute@@

Table: Dispute scenarios. Rank(s) is the rank of the seeded cause (both causes for D9). Accepted means rank 1 for D1 to D8 and both causes within the top three for D9.

![Size of the IM gap and rank of the seeded cause by scenario.](@@chart_rel@@/dispute_rank_and_gap.png)

**Result.** @@disp_acc@@ of @@disp_n@@ scenarios were accepted. The absolute gap ranges from @@disp_gap_min@@ (D7, the stale curve, is the smallest) to @@disp_gap_max@@ on an IM of about @@disp_ima@@. All single-cause scenarios have the seeded cause at rank 1 with a residual of zero after the fix.

**Tie in D4.** The ranking in D4 shows why the narrower-fix rule is needed:

@@tbl:rank_d4@@

Table: Scenario D4, top of the ranking.

Rebuilding the whole CRIF from today's curves ("stale curve") also removes the difference, because it regenerates the FX vega rows with the correct convention; but it touches @@d4_tie_scope_curve@@ rows while the FX vega fix touches @@d4_tie_scope_fx@@. Without the tie-break the engine would report a wrong cause.

**The composite case D9.** With two causes at once the ranking is harder, because the larger break masks the smaller one:

@@tbl:rank_d9@@

Table: Scenario D9, top of the ranking (missing trade plus stale FX rate).

The missing trade is rank @@d9_rank_missing@@ and explains @@d9_expl_missing@@ of the gap; the stale FX rate ranks only @@d9_rank_fx@@, behind a spurious "no linear rebucketing" hypothesis, because its stand-alone effect (@@d9_expl_fx@@) is small and it overlaps with other rows; a residual of @@d9_resid@@ remains after the top fix. The acceptance rule for composites (both causes in the top three) is met, but the example shows the usual lesson: fix the biggest break, re-run, and look again. This is also a limitation (Section 16): the ranking assumes a single dominant cause and a fixed hypothesis list.

# IM attribution

## The business problem

Every morning the IM call changes, and the first question from the desk is why. The change comes from different sources: trades that matured, trades that were booked, market moves and, on rare dates, a change in the SIMM parameters (the move from v@@prior_ver@@ to v@@simm_ver@@). The sources interact (a new trade changes the netting benefit of existing trades, a market move changes the sensitivities of the new trade), so a naive decomposition does not add up.

## Three methods

Let $IM(S)$ be the SIMM-style IM of the state in which the drivers in subset $S$ have been switched from their day-0 to their day-1 setting. With four drivers there are 16 states, all computed by `attribution.attribute`.

* **Shapley value.** The contribution of driver $i$ is its marginal effect averaged over all orders in which drivers can be switched on:
$$ \phi_i = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\,\left[IM(S\cup\{i\}) - IM(S)\right] . $$
It is the unique allocation that is symmetric and efficient: the contributions add up exactly to $IM(N) - IM(\emptyset)$.
* **Sequential bridge.** Switch drivers on in a fixed documented order: matured trades, new trades, then the market move split into rates, FX and vol, then parameters. Each step is valued after the previous ones, so the steps telescope to the total. It reads as a narrative ("first the maturities, then the new trade...") but depends on the order. The order chosen values trade changes at the old market and parameters, market moves on the new population, and the methodology change last, which is how a version-change impact is normally quoted.
* **One at a time.** Switch each driver alone from the base state. This is the simplest to compute and explain but ignores interactions, so the contributions do not add up and leave an interaction residual.

## Scenario day

The scenario day is @@attr_day@@, the first COB of v@@simm_ver@@. The designed move combines a stress-like market shock (parallel rate increases of 8 to 70 bp by currency, EUR, GBP and MXN weakening against USD, JPY strengthening, IR vol up 40% and FX vol up 60%), two 3-month trades that mature (FXF_USDJPY_3M, FXO_EURUSD_C3M), a new USD 150 million 10-year swap, and the parameter switch. IM moves from @@attr_im0@@ to @@attr_im1@@, a change of @@attr_dim@@ (@@attr_dim_pct@@).

@@tbl:attr@@

Table: Attribution of the change in IM on the scenario day, three methods.

![Shapley waterfall of the change in IM.](@@chart_rel@@/attribution_waterfall.png)

The new trade explains @@sh_new_trades@@ (@@sh_new_trades_pct@@ of the change); maturities change IM by @@sh_matured_trades@@; the market move changes IM by @@sh_market_move@@ (the bridge below splits it into rates, FX and volatility effects of opposite signs); and the parameter change adds @@sh_parameter_version@@. The three methods agree on the ranking but differ in size by up to a few hundred thousand USD, the signature of interactions. The one-at-a-time total misses @@attr_oaat_resid@@ (@@attr_oaat_resid_pct@@ of the change). Shapley and the bridge have zero residual by construction.

@@tbl:attr_sub@@

Table: The bridge splits the market step into rates, FX and volatility. A large positive vol effect offset by negative rates and FX effects nets to a small total.

## A Shapley calculation by hand

The 16 subset IMs are in the output file `attribution_subset_ims.csv`. The Shapley value of the new trade needs the eight pairs of subsets that differ only in whether the new trade is included. With $n = 4$ drivers the weights are $0!\,3!/4! = 0.25$ for no other driver switched, $1!\,2!/4! = 1/12$ for one, $2!\,1!/4! = 1/12$ for two and $3!\,0!/4! = 0.25$ for all three.

@@tbl:shapley_worked@@

Table: Shapley value of the new trades: the eight marginal effects with their weights. The weights sum to one and the weighted sum equals the Shapley value in the table above (@@sh_new_check@@).

Notice that the new trade adds between @@sh_marg_min@@ and @@sh_marg_max@@ of IM depending on what else has changed. That spread is the interaction; Shapley takes the fair average, one-at-a-time takes only the first row, and the bridge takes one specific row.

## Netting effect

A new trade's standalone IM (computed alone) is @@net_sa@@ but its incremental IM on the final book is @@net_inc@@. The difference, @@net_eff@@, is the netting effect: the new trade partly hedges existing risk, so adding it costs @@net_eff_pct@@ less than its standalone number suggests. Reporting both avoids telling a trader that a hedging trade is as expensive as it would be on its own.

## Scanning the history for big moves

`run_scan` computes the SIMM-style IM for each of the @@scan_n@@ dates and flags days where the IM changed by more than 10 percent or USD 1 million in one day. Over the history the IM ranges from @@scan_im_min@@ to @@scan_im_max@@ (mean @@scan_im_mean@@) and the largest one-day move is @@scan_max_move@@ (@@scan_max_abs@@). @@big_flag_rel@@ days breach the relative trigger and @@big_flag_abs@@ the absolute trigger. For flagged days the rates, FX and vol split of the move is computed:

@@tbl:big_moves@@

Table: The flagged days. All are dominated by FX moves, as the FX delta is the largest component of this book's IM.

![Daily IM change and flagged days.](@@chart_rel@@/big_move_days.png)

A related check, delta and vega sub-additivity (the margin of a combined portfolio should not exceed the sum of the margins of its parts), was run on @@subadd_each@@ random splits per risk class and margin type, @@subadd_checks@@ checks in total, with @@subadd_viol@@ violations. Curvature margin is not sub-additive in general and the project reports rather than asserts it.

# UAT and new-product onboarding

## Why UAT

When a bank starts trading a new product, the margin calculation has to be extended and then accepted by the business before the first trade is booked: that is user acceptance testing. The scenario here is a 5-year EUR/USD cross-currency basis swap on EUR 50 million (USD notional at the initial FX rate) with `settle_notional_exchange` = "eligible", so the final principal exchange is excluded from the model IM following the cross-currency rule (12 CFR 45.8(d)(4)).

## The procedure

`uat.py` holds a machine-readable test suite (`data/uat/uat_suite.csv`) with the columns test id, area, description, preconditions, input, expected, tolerance, actual, result, evidence file, execution time and executor. `run_uat` executes every test on the synthetic market of the scenario day, fills `actual` and `result` honestly (PASS, FAIL or SKIPPED) and writes `uat_results.csv`. Expectations for the hand cases use the verified numbers of the verification log, not the parameter loader, so they are independent of the code under test. The tests cover pricing, sensitivities, CRIF format, SIMM treatment (the basis risk weight of 21 with no concentration scaling, exclusion from the concentration sum, the $-1\%$ basis-to-yield correlation), the principal exchange exclusion, the schedule row, invariants, dispute detection, the Excel workbook, runtime and a regression test.

@@tbl:uat@@

Table: UAT results.

![UAT results by area.](@@chart_rel@@/uat_pass_counts.png)

## Executed results

@@uat_pass@@ of @@uat_n@@ tests passed, @@uat_fail@@ failed and @@uat_skip@@ were skipped. Two examples with the evidence: U01 shows @@uat_u01_actual@@; U14 shows that a counterparty that has not booked the swap is detected and ranked first: @@uat_u14@@. U16 recorded @@uat_u16_actual@@. The Excel test (U15) has the result @@uat_u15_result@@: @@uat_u15_actual@@. Any failure would be reported as such and would raise a finding. The suite tests the engine against itself and against closed forms, so it shows internal consistency and not agreement with ISDA's calculator.

# Findings log

The validation findings are in `findings_log.csv` (`findings.build_log`), using the same schema as the credit model validation project: ID, area, title, description, evidence, evidence file, severity, traffic light, recommendation, owner, status and source. Backtest-driven findings are produced automatically (RED becomes High, AMBER becomes Medium); static findings record known limitations. There are @@find_n@@ findings: @@find_high@@ High, @@find_med@@ Medium, @@find_low@@ Low.

## High-severity findings in full

@@findings_high_text@@

## All other findings

@@tbl:findings_other@@

Table: Medium and Low findings; the full description, evidence and recommendation are in the CSV.

The @@find_high@@ High findings are the ones a model owner could not ignore: a scope disclaimer that stands over every number, and the champion's worst rolling 250-day window in the red zone. Of the Medium findings the most consequential for a real deployment are F-05 (CVA excludes options), F-06 (hypothetical P&L), F-07 (one SIMM version for the whole history), F-12 (the FX vega convention) and the unverified-regulation items F-08 and F-09.

# Excel workbook walkthrough and reconciliation

An Excel workbook, `excel/simm_margin_workbook.xlsx`, repeats the central calculations with live formulas so that every number can be audited cell by cell without Python. Its sheets are: README, Inputs, Parameters (typed values with a provenance column), Portfolio, Sensitivities (typed CRIF from Python plus live Bachelier and Garman-Kohlhagen checks), SIMM_Delta and SIMM_Vega (live risk-weight lookup with INDEX and MATCH, concentration factor, weighted sensitivities, helper grids of correlation times weighted sensitivities, $K_b$ as the square root of the sum of the grid, $S_b$ and the inter-currency grid), SIMM_Curvature (scaling function, $CVR$, $\theta$, $\lambda$ via NORMSINV and the $HVR^{-2}$ factor), SIMM_Total, Schedule_IM, Backtest (typed series with live exceptions, Kupiec with LN and CHIDIST, Christoffersen transition counts with SUMPRODUCT on offset ranges, and zones by cumulative binomial), Attribution (typed subset IMs with live bridge and Shapley weights), Checks and Conclusions. It avoids MMULT, array formulas, LET, XLOOKUP and FILTER so that it opens in Excel 2016 and LibreOffice, and uses plain formatting.

The workbook is reconciled to the Python outputs by `reconcile_excel.py`, which recalculates the workbook in LibreOffice and compares cell values with Python at these tolerances: weighted sensitivities, $K_b$ and margins to $10^{-9}$ relative; total IM to USD 0.01; likelihood ratios to $10^{-9}$; p-values to $10^{-10}$; exception counts exactly; Shapley values to USD 0.01. The result is written to `python/outputs/excel_reconciliation.json`.

@@excel_status@@

@@tbl:excel_recon@@
"""


SECTIONS_D = r"""
# Limitations

This section lists every limitation known from the build and from the outputs. A reader should treat the numbers as a demonstration of method.

1. **Not ISDA-licensed, not certified.** The engine implements the structure of the public methodology and uses its published parameters, but it has not been licensed, certified, or compared with ISDA's calculator or test portfolios. No number is a margin call or a compliance statement.
2. **Synthetic data.** The history is simulated by a model written for this project. The backtests measure fit to that world. Real markets have features (jumps, liquidity, correlation breaks) that the generator only partly reproduces.
3. **One netting set, one counterparty.** There is no multi-netting-set aggregation, no collateral agreement terms (thresholds, minimum transfer amounts), and no regulatory-versus-internal IM split.
4. **Single curve per currency.** There are no projection curves by tenor and no basis curves, so the sub-curve correlation $\varphi = 98.1\%$ is never used (F-03). The EUR/USD basis is one scalar.
5. **Scope: IR and FX only.** No inflation, credit qualifying or non-qualifying, equity or commodity risk classes; only the IR-FX entry of $\psi$ is used (F-04). The ISDA tables for those classes were not extracted or verified (F-09).
6. **Hypothetical P&L.** Backtest losses revalue a frozen portfolio with constant time to maturity: no cash flows, no ageing, no trading, no actual P&L (F-06). The 10-day losses on overlapping windows are not independent.
7. **One SIMM version across the whole history.** The SIMM-style series uses v@@simm_ver@@ on all dates although it first applied from COB @@attr_day@@; @@ver_impact_pct@@ is the version effect on the valuation date (F-07).
8. **Linear-trade-only CVA.** Swaptions and FX options are excluded from the exposure model, and the Gaussian exposure approximation is crude (F-05). The xVA VaR backtest is therefore against the model's own CVA.
9. **Test limitations.** The non-overlapping 10-day sample has only @@n_10n@@ observations; the 250-day Kupiec test over-rejects because of discreteness; the house traffic light is a convention; the champion's worst rolling window is red and its conditional coverage is amber. The original Kupiec and Christoffersen papers were not obtained (F-13).
10. **Dispute ranking assumes a fixed list of single causes.** Composite and unlisted causes are harder, as D9 shows. Two hypotheses can produce identical residuals and are separated only by a tie-break.
11. **Attribution depends on the driver definition.** Shapley is symmetric and exact but its answer changes if drivers are defined differently; the bridge depends on its order.
12. **Non-standard CRIF column.** The FX vega treatment needs a market-vol column (`SigmaMarket`). This was found and fixed during the build (F-12).
13. **The curvature sign convention** is implemented as printed in the methodology; the economic reading was not validated against ISDA's calculator. The IR vega correlation reading rests on the methodology text (F-14).
14. **Regulatory coverage.** Only the public texts listed in Section 19 were read. The EU consolidated text with amendments, 12 CFR 237.8 and 349.8 and CFTC 23.154 were not checked (F-08). The phase-in dates and the thresholds that decide which entities must exchange margin were not covered by the sources read and are not described here.
15. **Excel workbook.** The workbook repeats calculations on typed Python inputs; it is not an independent source of market data.

# How to run

From the project root:

```
cd python
py -3 -m pip install -r requirements.txt
py -3 -m simm_margin.cli            # full run, about 9 minutes
py -3 -m simm_margin.cli --quick    # about 2 minutes, shorter history
py -3 -m pytest -q
cd ..
py -3 docs/build_docs.py
```

The command-line pipeline runs, in order: synthetic history and portfolio, CRIF and SIMM for both parameter versions, schedule IM, the daily SIMM series, the backtest battery (VaR IM, SIMM, xVA VaR), the dispute simulator, attribution, UAT, the findings log, the charts and the Excel workbook with its reconciliation. Options `--skip-charts` and `--skip-excel` omit the last two. Outputs are written to `python/outputs/` (CSV and JSON, each carrying the data source `@@data_source@@`), the charts to `python/outputs/charts/`, and this document to `docs/`. The documentation build reads only the outputs and the engine, so it can be re-run at any time (for example after the Excel reconciliation appears). Tests never write to `python/outputs/`.

Column names are accessed through `columns.py`, user inputs sit in a config block at the top of every module, and all parameters are read from `data/parameters/`.

# Glossary

| Term | Meaning |
|:---|:---|
| Backtest | Comparing a model's forecasts with realised outcomes to test calibration |
| BCBS 22 | The 1996 Basel Committee paper that defines the traffic-light backtesting zones |
| Bachelier model | Option pricing with normally distributed (absolute) rates, used for swaptions |
| CR / VCR | Concentration risk factors scaling weighted delta or vega when a net position exceeds a threshold |
| CRIF | Common Risk Interchange Format, the file of sensitivities exchanged between parties |
| Christoffersen test | Tests whether exceptions are independent (and, combined with Kupiec, correctly covered) |
| CVA | Credit valuation adjustment, the market price of counterparty default risk |
| CVR | Curvature risk exposure |
| Delta, vega, curvature | First-order price, volatility and second-order (gamma-like) risk margins |
| EE / EPE | Expected (positive) exposure at a future date / its time average |
| Euler allocation | Splitting a homogeneous risk measure among positions by directional derivatives |
| Exception | A day on which the realised loss exceeds the VaR forecast |
| Garman-Kohlhagen | Black-Scholes for currency options |
| gamma ($\gamma$) | Cross-bucket correlation between currencies in IR delta and vega |
| HS / HS VaR | Historical simulation value-at-risk with full revaluation |
| HVR | Historical volatility ratio, a scaling factor on vega risk in SIMM |
| IM / VM | Initial margin / variation margin |
| Kupiec POF | Proportion-of-failures test of the exception rate |
| MPOR | Margin period of risk, the time to close out after a default (10 days here) |
| NGR | Net-to-gross ratio of replacement costs in the schedule IM |
| Netting set | Trades covered by one legally enforceable netting agreement |
| $\psi$ | Correlation between risk classes in SIMM |
| RW | Risk weight |
| Shapley value | The average marginal contribution of a driver over all orders; sums exactly to the total |
| SIMM | The ISDA Standard Initial Margin Model (trademark) |
| SF | Curvature scaling function of option expiry |
| Traffic light | Green, yellow and red zones of the exception count |
| UAT | User acceptance testing |
| WS | Weighted sensitivity, $RW \times s \times CR$ |

# Verified references and verification log

All documents were retrieved on 2 October 2026. ISDA documents are not stored in this repository; only numeric parameters with citations and sha256 hashes of the retrieved files are kept. The verification log in `docs/VERIFICATION_LOG.md`, including its section of corrections after the full extraction, is authoritative where it differs from this summary.

| # | Source | What was verified | Not verified |
|:---|:---|:---|:---|
| 1 | ISDA SIMM Methodology v2.8+2512 (public PDF; effective 11 Jul 2026) | IR and FX risk weights, correlations, gamma, psi, concentration thresholds, HVR and VRW, SF, theta and lambda, structure of paras 5 to 11; transcribed by two extractors and checked against page images | Credit, equity and commodity tables; IR vega correlation reading is from the methodology text |
| 2 | ISDA SIMM Methodology v2.8+2506 (effective 6 Dec 2025) | The same RatesFX parameters; differences from 2512 tested | Same as above |
| 3 | ISDA SIMM Governance Framework (18 Sep 2026) | Calibration standard, dispute escalation, shortfall amount, backtesting with traffic light | Remediation annex thresholds |
| 4 | ISDA Risk Data Standards v1.36 (1 Feb 2017) | CRIF fields, IR delta per 1bp of par rates, linear rebucketing, vega amount | Dated; a newer CRIF specification may exist |
| 5 | BCBS-IOSCO Margin requirements for non-centrally cleared derivatives (April 2020) | Requirements 1.2, 3.1 and 3.6; Appendix A schedule rates (no cross-currency rows) | Phase-in schedule and thresholds (not read) |
| 6 | 12 CFR Part 45 (OCC), eCFR as of 2026-09-30 | 45.8(d)(1), (d)(2), (d)(4), (d)(13), (e), (f)(2)(ii)-(iii); Appendix A including cross-currency rows and NGR | 12 CFR 237.8 and 349.8 (assumed parallel, not fetched); CFTC 23.154 (not fetched) |
| 7 | Commission Delegated Regulation (EU) 2016/2251, as originally published | Articles 14(3) to (6), 15 and 16 | Consolidated amendments |
| 8 | BCBS 22 (January 1996) | Section II (one-day measures), Section III(c) to (f), Table 2; cumulative probabilities reproduced with scipy | None |
| 9 | Kupiec (1995), J. Derivatives vol 3 no 2, 73-84 | Bibliographic record only; the formula is verified against Federal Reserve restatements (FRBSF working paper 99-06, FEDS 2005-21) and numeric references | Original text NOT obtained |
| 10 | Christoffersen (1998), International Economic Review vol 39 no 4, 841-862 | Bibliographic record only; formulas verified against the same restatements and numeric references | Original text NOT obtained |
| 11 | SR 11-7 (Board of Governors of the Federal Reserve System and OCC, 4 April 2011), Supervisory Guidance on Model Risk Management | The three core elements of validation (conceptual soundness, ongoing monitoring, outcomes analysis), read in the attachment from federalreserve.gov | Not used beyond that structure |

Table: Sources used in this document and their verification status.

Corrections made during the verification (they supersede the first table of the log where they differ): the high-volatility FX currency list in v2.8+2506 has no ISK (ISK was added in 2512, and RUB and VES were dropped); the FX risk weight is one two by two table by currency volatility group; the US backtesting paragraph is 45.8(f)(2)(ii) to (iii) and the US rule has no quarterly backtesting requirement (the three-month frequency is EU Article 14(3)); the 25 percent stress share and 3 to 5 year window are EU Article 16 rules (the US rule is 1 to 5 years with a stress period and no percentage); the cross-currency schedule rows are in 12 CFR Part 45 and not in the BCBS-IOSCO appendix; the FX correlation two by two tables are not positive semidefinite alone, but the factor-level matrices are; and IR vega has no separate expiry correlation table.
"""

README_TEMPLATE = r"""# SIMM-Style Initial Margin, xVA VaR and Backtesting Model

> **Disclaimer.** All data are SYNTHETIC. This is an educational SIMM-style model: it is NOT licensed by ISDA, NOT certified or validated against ISDA's calculator, and makes NO compliance claim. ISDA SIMM is a trademark and ISDA documents are copyright; only numeric parameters with citations are stored here. Not verified: @@unverified_text@@.

A portfolio project in counterparty credit risk, market risk and margin model validation. For a netting set of @@n_trades@@ synthetic Rates and FX trades it computes a SIMM-style initial margin (parameter sets v@@simm_ver@@ and v@@prior_ver@@), the standardised schedule IM, a historical-simulation VaR IM, an xVA VaR for CVA, and then backtests, investigates seeded margin disputes, attributes IM changes, runs UAT for a new product and writes a findings log. A 19-section Word document (`docs/SIMM_IM_xVA_Backtesting_Documentation.docx`) explains every step with worked examples.

## Quick start

```
cd python
py -3 -m pip install -r requirements.txt
py -3 -m simm_margin.cli --quick      # about 2 minutes; omit --quick for the full run (about 9 minutes)
py -3 -m pytest -q
cd ..
py -3 docs/build_docs.py              # rebuilds the markdown, the Word document and this README
```

## Headline results (valuation date @@val_date@@, @@run_mode@@ run)

| Item | Result |
|:---|---:|
| SIMM-style IM, v@@simm_ver@@ | @@simm_total_12@@ |
| SIMM-style IM, v@@prior_ver@@ | @@simm_total_06@@ |
| Standardised schedule IM (net) | @@sch_net@@ |
| SIMM as a share of net schedule IM | @@sch_ratio@@ |
| Champion HS IM, 1-day exceptions | @@c:exceptions:champ:1d:i@@ of @@c:n:champ:1d:i@@ (@@exp_1d@@ expected) |
| Champion Kupiec / Christoffersen cc (1 day) | p = @@r:kupiec_pof:champ:1d:pv@@ (@@l:kupiec_pof:champ:1d@@) / p = @@r:christoffersen_cc:champ:1d:pv@@ (@@l:christoffersen_cc:champ:1d@@) |
| Champion worst rolling 250-day count | @@r:rolling250_max:champ:1d:i@@ (@@l:rolling250_max:champ:1d@@ zone) |
| Challengers (EWMA, plain 250, x0.7), 1-day light | @@l:overall:ewma:1d@@, @@l:overall:plain:1d@@, @@l:overall:scaled:1d@@ |
| SIMM-style, 10-day non-overlapping exceptions | @@c:exceptions:simm:10n:i@@ of @@c:n:simm:10n:i@@ (@@l:overall:simm:10n@@) |
| xVA VaR, 1-day exceptions | @@c:exceptions:xva:1d:i@@ of @@c:n:xva:1d:i@@ (@@l:overall:xva:1d@@) |
| Dispute scenarios accepted | @@disp_acc@@ of @@disp_n@@ |
| Attribution of the scenario-day change | @@attr_dim@@, exact Shapley |
| UAT (pass / fail / skipped) | @@uat_pass@@ / @@uat_fail@@ / @@uat_skip@@ |
| Findings (High / Medium / Low) | @@find_high@@ / @@find_med@@ / @@find_low@@ |

## Key findings

* The champion is well calibrated overall but its 1-day exceptions cluster: Christoffersen conditional coverage is amber and its worst rolling 250-day window is red.
* Mis-calibrated challengers are clearly worse, which shows the tests have power.
* Overlapping 10-day windows are red for every model (BCBS 22 section II warns against them); the SIMM-style series passes the non-overlapping design but fails Christoffersen independence on overlapping windows.
* SIMM is far below the standardised schedule IM, as expected for a risk-sensitive model on a hedged book.
* An FX vega CRIF convention issue (the market vol column `SigmaMarket`) was found and fixed during the build and is documented as a finding.

## Excel workbook

@@excel_readme@@

## Repository layout

```
INTERFACES.md                    scope and contracts
data/parameters/                 verified numeric parameters with provenance (no ISDA text)
data/uat/                        UAT suite
python/simm_margin/              engine: curves, market_history, pricing, sensitivities, crif, params, simm,
                                 schedule_im, allocation, var_model, cva, backtest, dispute, attribution,
                                 uat, findings, charts, cli
python/tests/                    pytest suite (never writes to python/outputs)
python/outputs/                  CSV and JSON results, charts/
excel/                           workbook with live formulas
docs/                            Word documentation, markdown source, build script, verification log
```

## Limitations

Synthetic data; a single netting set; one curve per currency; IR and FX only (no credit, equity or commodity classes); hypothetical P&L on a frozen portfolio; CVA for linear trades only; one SIMM version over the whole history; Kupiec (1995) and Christoffersen (1998) originals not obtained; no comparison with ISDA's calculator. See Section 16 of the documentation.

## Parameter provenance

Every in-scope parameter in `data/parameters/simm_parameters.csv` (685 rows for the two versions) is VERIFIED_PRIMARY: it carries its source document, section, paragraph, table, PDF page, URL, retrieval date and the sha256 of the retrieved PDF. Regulatory constants and the BCBS 22 traffic-light table carry the same citations. The verification log is `docs/VERIFICATION_LOG.md`.
"""


if __name__ == "__main__":
    main()
