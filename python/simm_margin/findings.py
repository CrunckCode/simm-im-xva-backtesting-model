"""Findings log: static review findings on the model plus findings derived from the result files.

SIMM-style, not ISDA-licensed or certified. ALL DATA IS SYNTHETIC. Schema and mapping style adapted from the sibling credit
validation project. Severity: High = material error in model output, a Red test on the champion, or a scope limit that
blocks production use; Medium = Amber or a design weakness with bounded impact; Low = documentation or hygiene.
Automatic sources: backtest_results.csv (RED -> High, AMBER -> Medium), SIMM versus schedule ordering, curvature
sub-additivity, dispute misrankings, UAT fails. Evidence strings quote numbers read from the files at build time.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import columns as C
from . import config
from . import params as params_mod
from . import schedule_im
from .crif import SUBCURVE, build_crif
from .cva import linear_trades
from .instruments import sample_portfolio
from .market_history import generate_history
from .pricing import price_portfolio
from .simm import simm

# ===== CONFIG (user inputs) =====
LOG_FILE = "findings_log.csv"
LOG_COLUMNS = (C.FINDING_ID, C.AREA, C.TITLE, C.DESCRIPTION, C.EVIDENCE, C.EVIDENCE_FILE, C.SEVERITY, C.TRAFFIC_LIGHT,
               C.RECOMMENDATION, C.OWNER, C.STATUS, C.SOURCE)
SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
LIGHT_TO_SEVERITY = {"RED": "High", "AMBER": "Medium"}
SEVERITY_TO_LIGHT = {"High": "Red", "Medium": "Amber", "Low": "Green"}
BACKTEST_FILE, SERIES_FILE = "backtest_results.csv", "backtest_series.csv"
DISPUTE_FILE, UAT_FILE, ATTR_FILE = "dispute_results.csv", "uat_results.csv", "attribution_methods_comparison.csv"
SCHEDULE_FILE, SUBADD_FILE = "simm_vs_schedule.csv", "subadditivity.csv"
VERIFICATION_LOG = "docs/VERIFICATION_LOG.md"
BACKTEST_DOC = "docs/sources/backtest_formula_verification.md"
CHAMPION_MODEL, SIMM_MODEL, XVA_MODEL = "hs_im_eu_1plus3", "simm_style", "xva_var"
AUTO_MODELS = (CHAMPION_MODEL, SIMM_MODEL, XVA_MODEL)            # challengers are comparators, not validated models
AUTO_TESTS = ("overall", "basel_table_last250", "rolling250_max")
OVERLAP_SPLIT, NONOVERLAP_SPLIT, ONE_DAY_SPLIT = "10d_overlapping", "10d_nonoverlap", "1d"
SHORTFALL_TEST = "shortfall_multiplier"
SUBADD_SPLITS, SUBADD_SEED, SUBADD_TOL = 60, 7, 1.0
OWNERS = ("Model Developer", "Model Owner", "Validation")
# ===== END CONFIG =====

BT_TEST, BT_MODEL, BT_SPLIT, BT_VALUE, BT_THRESHOLD, BT_LIGHT, BT_N, BT_EXC = (
    C.TEST, C.MODEL, "split", "value", "threshold", C.LIGHT, "n", "exceptions")


def _f(area, title, description, evidence, evidence_file, severity, recommendation, owner, source="review") -> dict:
    return {C.AREA: area, C.TITLE: title, C.DESCRIPTION: description, C.EVIDENCE: evidence, C.EVIDENCE_FILE: evidence_file,
            C.SEVERITY: severity, C.TRAFFIC_LIGHT: SEVERITY_TO_LIGHT[severity], C.RECOMMENDATION: recommendation,
            C.OWNER: owner, C.STATUS: "Open", C.SOURCE: source}


def _read(out_dir: Path, name: str):
    p = out_dir / name
    return pd.read_csv(p) if p.exists() else None


def _file(name: str, out_dir: Path, fallback: str = VERIFICATION_LOG) -> str:
    """Evidence file name as written in the log: an output file if it exists, else a project file, else the fallback."""
    if (out_dir / name).exists() or (config.ROOT / name).exists():
        return name
    return fallback


def _usd(x) -> str:
    return f"USD {x:,.0f}"


# ---------- helper tables written to out_dir when no earlier stage recorded them ----------
def simm_vs_schedule_table(history=None, trades=None, out_dir=None) -> pd.DataFrame:
    """SIMM-style IM, schedule gross and net IM for the sample book on four dates (market FX passed to the schedule)."""
    history = history if history is not None else generate_history()
    trades = trades if trades is not None else sample_portfolio()
    p = params_mod.load()
    i_scn = int(history.index_of([config.SCENARIO_DAY])[0])
    i_str = int(history.stress_slice[0] + (history.stress_slice[1] - history.stress_slice[0]) // 2) if history.stress_slice else i_scn
    picks = [("stress_window_mid", i_str), ("scenario_day", i_scn), ("last_date", len(history) - 1), ("first_date", 0)]
    rows = []
    for label, i in picks:
        row = history.market.row(i)
        spot = {c: float(v[0]) for c, v in row.fx_spot.items()}
        im = simm(build_crif(trades, row), p).total
        s = schedule_im.schedule_im(trades, price_portfolio(trades, row)[0], fx_spot=spot)
        rows.append({"label": label, C.DATE: history.dates[i].date().isoformat(), "simm_im": im, "schedule_gross": s["gross"],
                     "schedule_ngr": s["ngr"], "schedule_net": s["net"], "simm_over_schedule_net": im / s["net"],
                     "data_source": config.DATA_SOURCE})
    df = pd.DataFrame(rows)
    if out_dir is not None:
        df.to_csv(Path(out_dir) / SCHEDULE_FILE, index=False)
    return df


def subadditivity_table(history=None, trades=None, out_dir=None) -> pd.DataFrame:
    """Random bipartitions of the book: how often does margin(A+B) exceed margin(A) + margin(B) by more than SUBADD_TOL USD."""
    history = history if history is not None else generate_history()
    trades = trades if trades is not None else sample_portfolio()
    p = params_mod.load()
    crif = build_crif(trades, history.market.row(int(history.index_of([config.SCENARIO_DAY])[0])))
    ids = list(trades[C.TRADE_ID])
    whole = simm(crif, p).by_margin_type
    rng = np.random.default_rng(SUBADD_SEED)
    stats = {k: {"n": 0, "viol": 0, "worst": 0.0} for k in whole}
    for _ in range(SUBADD_SPLITS):
        mask = rng.random(len(ids)) < 0.5
        if mask.all() or not mask.any():
            continue
        a_ids = set(np.array(ids)[mask])
        in_a = crif[C.CRIF_TRADE_ID].isin(a_ids)
        pa, pb = simm(crif[in_a], p).by_margin_type, simm(crif[~in_a], p).by_margin_type
        for k, v in whole.items():
            excess = v - (pa[k] + pb[k])
            stats[k]["n"] += 1
            stats[k]["viol"] += int(excess > SUBADD_TOL)
            stats[k]["worst"] = max(stats[k]["worst"], excess)
    df = pd.DataFrame([{C.RISK_CLASS: k[0], C.MARGIN_TYPE: k[1], "n_checks": s["n"], "n_violations": s["viol"],
                        "max_excess_usd": s["worst"], "data_source": config.DATA_SOURCE} for k, s in stats.items()])
    if out_dir is not None:
        df.to_csv(Path(out_dir) / SUBADD_FILE, index=False)
    return df


# ---------- automatic findings ----------
def backtest_findings(res: pd.DataFrame, out_dir: Path) -> list:
    """One finding per (test, model) whose worst non-overlapping light is RED (High) or AMBER (Medium)."""
    if res is None or res.empty:
        return []
    r = res[res[BT_MODEL].isin(AUTO_MODELS) & res[BT_TEST].isin(AUTO_TESTS)]
    short = res[(res[BT_TEST] == SHORTFALL_TEST)].set_index(BT_MODEL)["value"].to_dict()
    out = []
    for (test, model), g in r.groupby([BT_TEST, BT_MODEL], sort=False):
        g = g[g[BT_SPLIT] != OVERLAP_SPLIT]               # overlapping windows are over-sized; see the overlap finding
        bad = g[g[BT_LIGHT].isin(LIGHT_TO_SEVERITY)]
        if bad.empty:
            continue
        sev = "High" if (bad[BT_LIGHT] == "RED").any() else "Medium"
        detail = "; ".join(f"{s}: {int(x)} exceptions in {int(n)} observations ({li})"
                           for s, x, n, li in zip(bad[BT_SPLIT], bad[BT_EXC], bad[BT_N], bad[BT_LIGHT]))
        is_xva = model == XVA_MODEL
        sf = short.get(model)
        extra = f" Shortfall multiplier (smallest k so that k x IM is Green) is {sf:.3f}." if (sf is not None and not is_xva) else ""
        what = {"overall": "worst of Kupiec POF, Christoffersen independence and Basel zone",
                "basel_table_last250": "Basel traffic light on the last 250 observations",
                "rolling250_max": "highest rolling 250-day exception count"}[test]
        out.append(_f("xVA VaR backtest" if is_xva else "IM backtest", f"{test} {sorted(set(bad[BT_LIGHT]))[0]} for {model}",
                      f"Backtest {what} returned {'/'.join(sorted(set(bad[BT_LIGHT])))} for model {model}.",
                      f"{detail}. Expected exceptions at 99% are 1% of observations.{extra}", _file(BACKTEST_FILE, out_dir), sev,
                      "Recalibrate (longer stress share or a floor on the multiplier) and re-run the battery; report the shortfall "
                      "multiplier to the model owner." if not is_xva else
                      "Review the sensitivity set and the CS01 ladder, add second-order terms, and re-run the 10-day backtest.",
                      "Model Developer", source="auto"))
    return out


def schedule_findings(tab: pd.DataFrame, out_dir: Path) -> list:
    if tab is None or tab.empty:
        return []
    bad = tab[tab["simm_im"] > tab["schedule_net"]]
    if bad.empty:
        return []
    worst = bad.loc[(bad["simm_im"] / bad["schedule_net"]).idxmax()]
    lines = "; ".join(f"{r.label} {r.date}: SIMM {_usd(r.simm_im)} vs schedule net {_usd(r.schedule_net)} (ratio {r.simm_over_schedule_net:.2f})"
                      for r in bad.itertuples())
    return [_f("IM comparison", "SIMM-style IM exceeds the schedule IM on the sample book",
               f"The schedule IM is normally the upper bound that SIMM stays below for a portfolio of this kind; SIMM is above it on "
               f"{len(bad)} of {len(tab)} test dates (worst ratio {worst.simm_over_schedule_net:.2f}).",
               lines + ". The numbers are reported, not adjusted.", _file(SCHEDULE_FILE, out_dir), "Medium",
               "Check notional-driven drivers (FX delta and curvature with synthetic vols), the swaption duration assumption in the "
               "schedule, and the FX conversion; document why SIMM exceeds the standardised amount.", "Model Developer", source="auto")]


def subadditivity_findings(tab: pd.DataFrame, out_dir: Path) -> list:
    if tab is None or tab.empty:
        return []
    out = []
    for r in tab[tab["n_violations"] > 0].itertuples():
        is_curv = r.margin_type == C.MT_CURVATURE
        out.append(_f("SIMM properties", f"{r.risk_class} {r.margin_type} margin is not sub-additive on random book splits",
                      "Margin of a combined book exceeds the sum of the margins of its two halves in some splits"
                      + (", which is a known property of the curvature margin (the max with zero and the lambda term)." if is_curv
                         else ", which is unexpected for a norm-based margin and points to concentration scaling or clipping effects."),
                      f"{int(r.n_violations)} of {int(r.n_checks)} random splits violate sub-additivity by more than USD {SUBADD_TOL:g}; "
                      f"largest excess {_usd(r.max_excess_usd)}.", _file(SUBADD_FILE, out_dir),
                      "Low" if is_curv else "Medium",
                      "Do not assume IM(A+B) <= IM(A) + IM(B) in netting-set or compression logic; test the property per margin type.",
                      "Model Owner", source="auto"))
    return out


def dispute_findings(df: pd.DataFrame, out_dir: Path) -> list:
    if df is None or df.empty:
        return []
    out = []
    for r in df[~df["accepted"].astype(bool)].itertuples():
        out.append(_f("Dispute workflow", f"Dispute {r.scenario}: seeded cause {r.seeded_cause} is not ranked as required",
                      "The cause ranking did not place the seeded cause first (or both causes in the top three for the composite).",
                      f"Seeded cause {r.seeded_cause} has rank {r.seeded_ranks}; top-ranked cause is {r.top_ranked_cause} with residual "
                      f"{_usd(r.residual_after_top_fix)} after its fix; IM A {_usd(r.im_a)}, IM B {_usd(r.im_b)}, gap {_usd(r.gap)}.",
                      _file(DISPUTE_FILE, out_dir), "High",
                      "Add a distinguishing signature for the confused hypotheses or narrow the scope of the broader fix.",
                      "Model Developer", source="auto"))
    return out


def uat_findings(df: pd.DataFrame, out_dir: Path) -> list:
    if df is None or df.empty:
        return []
    out = []
    for r in df[df["result"] == "FAIL"].itertuples():
        sev = "Medium" if r.area == "performance" else "High"
        out.append(_f("New product UAT", f"UAT {r.test_id} failed: {r.description}",
                      f"Executed acceptance test {r.test_id} ({r.area}) did not meet its expected result: {r.expected}.",
                      f"Actual: {r.actual}. Tolerance: {r.tolerance}.", _file(UAT_FILE, out_dir), sev,
                      "Fix the defect, re-run the suite and keep the failed run in the evidence trail.", "Model Developer", source="auto"))
    return out


def attribution_findings(df: pd.DataFrame, out_dir: Path) -> list:
    if df is None or df.empty:
        return []
    res = df[df["level"] == "residual"].iloc[0]
    tot = df[df["level"] == "total"].iloc[0]
    oaat, dim = float(res["one_at_a_time"]), float(tot["shapley"])
    if abs(dim) <= 0 or abs(oaat) < 0.01 * abs(dim):
        return []
    return [_f("Attribution", "One-at-a-time attribution leaves a large interaction residual",
               "Changing one driver at a time ignores interactions between drivers, so the pieces do not add up to the IM change; "
               "Shapley (exact) and the sequential bridge both add up to zero residual.",
               f"On the scenario day the total IM change is {_usd(dim)}; one-at-a-time drivers miss {_usd(oaat)} "
               f"({oaat / dim:.1%}); Shapley and bridge residuals are {_usd(float(res['shapley']))} and {_usd(float(res['bridge']))}.",
               _file(ATTR_FILE, out_dir), "Low", "Report Shapley as the headline attribution and the bridge as the narrative.",
               "Model Owner", source="auto")]


def dispute_d4_finding(df: pd.DataFrame, out_dir: Path) -> list:
    if df is None or "D4" not in set(df["scenario"]):
        return []
    r = df[df["scenario"] == "D4"].iloc[0]
    return [_f("CRIF design", "FX vega IM depends on a non-standard CRIF column (SigmaMarket)",
               "FX vega amounts in the CRIF are vega times the market implied vol. The engine needs the market vol (column SigmaMarket) to "
               "turn them back into vega before weighting with the SIMM vol; a counterparty that treats the amount as already weighted "
               "gets a different IM from an identical-looking CRIF.",
               f"Scenario D4 (implied vol instead of the SIMM vol) moves IM from {_usd(r.im_a)} to {_usd(r.im_b)}, a gap of {_usd(r.gap)} "
               f"with no CRIF amount differing.", _file(DISPUTE_FILE, out_dir), "Medium",
               "Document the FX vega convention in the exchange specification and add the vol column to the CRIF reconciliation.",
               "Model Developer", source="auto")]


def schedule_fx_default_finding(history, trades, out_dir: Path) -> list:
    """schedule_im defaults to the synthetic base FX levels, not the valuation date market."""
    row = history.market.row(int(history.index_of([config.SCENARIO_DAY])[0]))
    mtm = price_portfolio(trades, row)[0]
    spot = {c: float(v[0]) for c, v in row.fx_spot.items()}
    g_default = schedule_im.schedule_im(trades, mtm)["gross"]
    g_market = schedule_im.schedule_im(trades, mtm, fx_spot=spot)["gross"]
    if abs(g_default - g_market) <= 1.0:
        return []
    return [_f("Schedule IM", "schedule_im converts notionals at fixed base FX levels unless a market rate is passed",
               "The default FX argument of schedule_im is the synthetic base level of each pair, so a caller that omits fx_spot gets a "
               "schedule amount on stale rates.",
               f"Sample book on {config.SCENARIO_DAY.date()}: gross schedule IM {_usd(g_default)} with default FX versus "
               f"{_usd(g_market)} with the market FX ({g_market / g_default - 1:+.2%}).", _file(SCHEDULE_FILE, out_dir, "python/simm_margin/schedule_im.py"),
               "Low", "Make fx_spot a required argument or default it to the valuation-date market.", "Model Developer", source="auto")]


# ---------- static review findings ----------
def static_findings(history, trades, out_dir: Path, bt: pd.DataFrame) -> list:
    p12, p06 = params_mod.load(config.SIMM_VERSION), params_mod.load(config.SIMM_VERSION_PRIOR)
    i = int(history.index_of([config.SCENARIO_DAY])[0])
    row = history.market.row(i)
    n_days = len(pd.bdate_range(config.HIST_START, config.HIST_END))
    n_before = int((pd.bdate_range(config.HIST_START, config.HIST_END) < config.SCENARIO_DAY).sum())
    lin = linear_trades(trades)
    pv = np.abs(price_portfolio(trades, row)[0])
    opt_share = float(pv[~trades[C.TRADE_ID].isin(lin[C.TRADE_ID]).to_numpy()].sum() / pv.sum())
    n_opt = len(trades) - len(lin)
    basis_bp = float(row.xccy_basis["EURUSD"][0]) * 1e4
    series = _read(out_dir, SERIES_FILE)
    if series is not None:
        s1 = series[(series[C.MODEL] == CHAMPION_MODEL) & (series["split"] == ONE_DAY_SPLIT)]
        hyp_ev = (f"The 1-day champion series has {len(s1)} observations from {s1[C.DATE].min()} to {s1[C.DATE].max()}, all computed on the "
                  f"{len(trades)}-trade book with constant times to maturity.") if len(s1) else "Series file has no 1-day champion rows."
    else:
        hyp_ev = f"The book has {len(trades)} trades with constant times to maturity (no ageing)."
    overlap = ""
    if bt is not None and not bt.empty:
        o = bt[(bt[BT_TEST] == "overall") & (bt[BT_MODEL] == CHAMPION_MODEL)].set_index(BT_SPLIT)
        if OVERLAP_SPLIT in o.index and NONOVERLAP_SPLIT in o.index:
            overlap = (f"Champion: {int(o.loc[OVERLAP_SPLIT, BT_EXC])} exceptions in {int(o.loc[OVERLAP_SPLIT, BT_N])} overlapping 10-day windows "
                       f"({o.loc[OVERLAP_SPLIT, BT_LIGHT]}) versus {int(o.loc[NONOVERLAP_SPLIT, BT_EXC])} in {int(o.loc[NONOVERLAP_SPLIT, BT_N])} "
                       f"non-overlapping windows ({o.loc[NONOVERLAP_SPLIT, BT_LIGHT]}).")
    overlap = overlap or "Overlapping windows share nine of ten days, so exceptions cluster."
    rows = [
        _f("Scope", "All data is synthetic and the engine is not ISDA-licensed or certified",
           "Market history, trades and counterparty data are simulated; the SIMM-style engine uses published parameter values but is not ISDA "
           "licensed, not certified by ISDA and not validated against ISDA's calculator, so no number here is a margin call or a compliance claim.",
           f"Data source recorded on every output: '{config.DATA_SOURCE}'; history of {n_days} business days "
           f"({config.HIST_START.date()} to {config.HIST_END.date()}), {len(trades)} trades.",
           _file(VERIFICATION_LOG, out_dir), "High",
           "Treat outputs as a methodology demonstration; obtain the ISDA licence and the calculator test cases before any production use.",
           "Model Owner"),
        _f("Market model", "Single OIS-style curve per currency; no multi-curve or tenor basis",
           "Each currency has one curve used for discounting and projection, so tenor-basis and OIS-versus-IBOR spreads, and the SIMM sub-curve "
           "correlation, are never exercised.",
           f"All IR delta rows carry the sub-curve label '{SUBCURVE}'; the sub-curve correlation parameter phi = {p12.phi:.3f} is never used. "
           f"The EUR/USD cross-currency basis is a single scalar ({basis_bp:.1f}bp on {config.SCENARIO_DAY.date()}).",
           "data/parameters/simm_parameters.csv", "Medium",
           "Add projection curves per tenor and a basis curve; extend the CRIF with sub-curve labels.", "Model Developer"),
        _f("Scope", "No inflation or credit-risk-class exposure; RatesFX only",
           "The engine supports the IR and FX risk classes only and rejects other CRIF risk types, so inflation, credit qualifying and "
           "non-qualifying, equity and commodity exposures cannot be margined.",
           f"Risk-class correlation IR-FX is {p12.psi('IR', 'FX'):.2f} in {config.SIMM_VERSION} and {p06.psi('IR', 'FX'):.2f} in {config.SIMM_VERSION_PRIOR}; "
           f"all other psi entries are unused. Inflation rows are parsed but have no pricing or sensitivity source.",
           "data/parameters/simm_parameters.csv", "Medium",
           "Extend the portfolio and the parameter extraction before using the engine for mixed-asset netting sets.", "Model Developer"),
        _f("xVA", "CVA uses a normal approximation of expected exposure for linear trades only",
           "EE = mu Phi(mu/s) + s phi(mu/s) assumes a Gaussian netting-set value; swaptions and FX options are excluded from the CVA because "
           "their exposure is not Gaussian.",
           f"{n_opt} of {len(trades)} trades are excluded from CVA; they carry {opt_share:.1%} of the book's gross absolute PV on {config.SCENARIO_DAY.date()}.",
           "python/simm_margin/cva.py", "Medium",
           "Replace with Monte Carlo exposure simulation including option trades, then back-test the xVA VaR on the full book.", "Model Developer"),
        _f("Backtesting", "Hypothetical P&L on a frozen portfolio",
           "Losses are the revaluation of the same trades one and ten days later with constant times to maturity, so there are no cash flows, "
           "no ageing and no trading; this is hypothetical P&L net of cash flows, not actual P&L.",
           hyp_ev, _file(SERIES_FILE, out_dir, "python/simm_margin/var_model.py"), "Medium",
           "Add an actual-P&L backtest and ageing of trades when real positions become available.", "Model Owner"),
        _f("Backtesting", "One SIMM version is used for the whole historical backtest",
           f"The SIMM-style series uses {config.SIMM_VERSION} for every date although it first applies from COB {config.SCENARIO_DAY.date()}; "
           f"earlier dates would have used earlier versions with different weights and thresholds.",
           f"{n_before} of {n_days} history days ({n_before / n_days:.1%}) precede {config.SCENARIO_DAY.date()}; the 2506 versus 2512 impact on the scenario day is in the attribution outputs.",
           _file("attribution_scenario_day.csv", out_dir), "Medium",
           "Run the backtest with the version in force on each date or report the version-change impact separately.", "Model Developer"),
        _f("Verification", "Kupiec (1995) and Christoffersen (1998) verified only against secondary sources and numeric references",
           "The original papers could not be retrieved; the formulas were cross-checked against Federal Reserve restatements and independent "
           "numeric references (closed forms, chi-square size simulation).",
           "Retrieval attempts returned HTTP 404 and 403 (FRASER, Federal Reserve, journal); restatements read: FRBSF WP 99-06 and FEDS 2005-21.",
           BACKTEST_DOC, "Low", "Obtain the originals through a library and confirm the statistics line by line.", "Validation"),
        _f("Verification", "EU consolidated amendments, 12 CFR 237.8, 12 CFR 349.8 and CFTC 23.154 were not checked",
           "Regulatory requirements were read from Delegated Regulation 2016/2251 as published and 12 CFR 45.8; later EU amendments and the "
           "parallel US rules of other agencies were assumed equivalent and not fetched.",
           "VERIFICATION_LOG rows 6 and 7: 12 CFR 237.8 and 349.8 assumed parallel, not fetched; CFTC 23.154 not fetched; EU consolidated amendments not checked.",
           VERIFICATION_LOG, "Medium", "Fetch and compare the consolidated texts before quoting any requirement as current.", "Validation"),
        _f("Verification", "IR vega correlation rests on the methodology text, not on ISDA's calculator",
           "The methodology has no separate IR vega expiry correlation table; the engine applies the 12x12 tenor correlation with f = 1 as "
           "described in the text, which was not reconciled to ISDA's reference calculator.",
           "VERIFICATION_LOG corrections 7 and 8: tenor matrix minimum eigenvalue 0.005015 in both versions, positive semidefinite; reading from text.",
           VERIFICATION_LOG, "Low", "Reconcile IR vega and curvature results to the ISDA calculator test portfolios when licensed.", "Validation"),
        _f("Verification", "ISDA credit, equity and commodity parameter tables not covered",
           "Only the RatesFX parameters were extracted and cross-checked; tables for the other product classes were not read.",
           "VERIFICATION_LOG row 1 'Not verified' and correction 9: credit, equity and commodity tables not checked (out of scope).",
           VERIFICATION_LOG, "Medium", "Extract and verify the remaining tables before extending the product scope.", "Validation"),
        _f("Backtesting", "Backtest windows overlap for the 10-day horizon",
           "Consecutive 10-day windows share nine days, so exceptions cluster and the Kupiec and Christoffersen p-values are over-sized; BCBS 22 "
           "section II warns against comparing 10-day measures with overlapping outcomes. The log uses the non-overlapping design for severity.",
           overlap, _file(BACKTEST_FILE, out_dir), "Low",
           "Keep the non-overlapping design as the pass criterion and report overlapping results only as supporting evidence.", "Model Owner"),
    ]
    return rows


# ---------- build ----------
def build_log(out_dir=None, history=None, trades=None) -> pd.DataFrame:
    """Merge static and automatic findings, sort by severity, assign F-nn ids and write findings_log.csv to out_dir."""
    out_dir = Path(out_dir) if out_dir is not None else config.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    history = history if history is not None else generate_history()
    trades = trades if trades is not None else sample_portfolio()
    bt = _read(out_dir, BACKTEST_FILE)
    sched = _read(out_dir, SCHEDULE_FILE)
    if sched is None:
        sched = simm_vs_schedule_table(history, trades, out_dir)
    subadd = _read(out_dir, SUBADD_FILE)
    if subadd is None:
        subadd = subadditivity_table(history, trades, out_dir)
    disp = _read(out_dir, DISPUTE_FILE)
    rows = (static_findings(history, trades, out_dir, bt) + backtest_findings(bt, out_dir) + schedule_findings(sched, out_dir)
            + subadditivity_findings(subadd, out_dir) + dispute_findings(disp, out_dir) + dispute_d4_finding(disp, out_dir)
            + uat_findings(_read(out_dir, UAT_FILE), out_dir) + attribution_findings(_read(out_dir, ATTR_FILE), out_dir)
            + schedule_fx_default_finding(history, trades, out_dir))
    df = pd.DataFrame(rows)
    df["_s"] = df[C.SEVERITY].map(SEVERITY_ORDER)
    df = df.sort_values("_s", kind="stable").drop(columns="_s").reset_index(drop=True)
    width = max(2, len(str(len(df))))
    df.insert(0, C.FINDING_ID, [f"F-{k + 1:0{width}d}" for k in range(len(df))])
    df = df[list(LOG_COLUMNS)]
    df.to_csv(out_dir / LOG_FILE, index=False)
    return df
