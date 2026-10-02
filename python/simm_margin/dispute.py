"""Margin dispute workflow: seeded counterparty differences, reconciliation and cause ranking.

SIMM-style, not ISDA-licensed or certified. ALL TRADES AND MARKET DATA ARE SYNTHETIC.
Party A is the base calculation (primary parameter set). make_counterparty_view builds party B's inputs for a seeded
difference (D1..D9). reconcile compares the two views (trade population, CRIF keys within a tolerance, hierarchical IM
gap risk class -> margin type -> bucket -> risk factor, Euler trade contributions). rank_causes tests eight
hypotheses: each one normalises B along one dimension to the reference convention, recomputes B's IM and is ranked by the
residual gap that is left (ties within the tolerance go to the hypothesis that touches fewer CRIF rows).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import columns as C
from . import config
from . import params as params_mod
from .allocation import euler_contributions
from .crif import CRIF_COLUMNS, IR_BUCKET, MIN_ABS_USD, SUBCURVE, build_crif
from .market_history import MarketBatch, MarketHistory
from .sensitivities import fx_vega, rebucket_weights, trade_currencies, zero_pillar_deltas
from .simm import SIGMA_MARKET_COL, simm

# ===== CONFIG (user inputs) =====
SCENARIOS = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9")
D1_MISSING_TRADE = "IRS_EUR_10Y"            # trade party B never booked
D2_AMENDED_TRADE, D2_NOTIONAL_FACTOR = "IRS_USD_10Y", 1.10     # notional amended on B's side only
STALE_FX_DAYS = 5                           # D5: B converts local amounts at the FX rate of this many business days earlier
STALE_CURVE_DAYS = 3                        # D7: B's curves are this many business days old
CAUSE_MISSING, CAUSE_NOTIONAL, CAUSE_ZERO = "missing_trade", "notional_amendment", "zero_rate_delta"
CAUSE_FXVEGA, CAUSE_FXRATE, CAUSE_REBUCKET = "fx_vega_implied_vol", "stale_fx_rate", "no_linear_rebucketing"
CAUSE_CURVE, CAUSE_PARAMS = "stale_curve", "parameter_version"
SEEDED = {"D1": (CAUSE_MISSING,), "D2": (CAUSE_NOTIONAL,), "D3": (CAUSE_ZERO,), "D4": (CAUSE_FXVEGA,),
          "D5": (CAUSE_FXRATE,), "D6": (CAUSE_REBUCKET,), "D7": (CAUSE_CURVE,), "D8": (CAUSE_PARAMS,),
          "D9": (CAUSE_MISSING, CAUSE_FXRATE)}
COMPOSITE_TOP_N = 3                         # acceptance for D9: both causes within the top N
KEY_COLUMNS = [C.CRIF_TRADE_ID, C.RISK_TYPE, C.QUALIFIER, C.LABEL1, C.LABEL2]
SIGMA_TOL = 1.0e-12
RESULTS_FILE, DETAILS_DIR = "dispute_results.csv", "dispute_details"
RESULT_COLUMNS = ["scenario", "seeded_cause", "rank_of_seeded_cause", "seeded_ranks", "im_a", "im_b", "gap",
                  "top_ranked_cause", "residual_after_top_fix", "accepted", "data_source"]
TRADE_COL, IN_A, IN_B, NOTIONAL_A, NOTIONAL_B, STATUS_COL = "trade_id", "in_a", "in_b", "notional_a", "notional_b", "status"
CAUSE_COL, SIGNATURE, EXPLAINED, RESIDUAL, SCOPE, RANK = ("cause", "signature_score", "explained_gap",
                                                          "residual_after_fix", "scope_rows", "rank")
VALUE_A, VALUE_B, GAP = "value_a", "value_b", "gap"
# ===== END CONFIG =====

_GAP_LEVELS = ("risk_class", "margin_type", "K", "factor")      # simm breakdown levels used in the gap hierarchy
_BUCKET = "bucket"


@dataclass
class PartyView:
    """One party's inputs: trade table, CRIF (with SigmaMarket) and parameter set."""
    label: str
    trades: pd.DataFrame
    crif: pd.DataFrame
    params: object


@dataclass
class Context:
    """Market information the counterfactual fixes may use (valuation-date row and optionally the history)."""
    row: MarketBatch
    history: MarketHistory = None
    index: int = None
    _cache: dict = field(default_factory=dict, repr=False)

    def spot(self, ccy: str, lag: int = 0) -> float:
        row = self.row if lag == 0 else self.history.market.row(self.index - lag)
        return float(row.fx_spot[ccy][0])

    def fresh_crif(self, trades: pd.DataFrame) -> pd.DataFrame:
        key = tuple(trades[C.TRADE_ID])
        if key not in self._cache:
            self._cache[key] = build_crif(trades, self.row)
        return self._cache[key]


def make_context(market, history: MarketHistory = None, date=None) -> Context:
    """market: a MarketHistory (row taken at `date`, default config.SCENARIO_DAY) or a single-state MarketBatch."""
    if isinstance(market, MarketHistory):
        history = market
    if history is not None:
        idx = int(history.index_of([config.SCENARIO_DAY if date is None else date])[0])
        if isinstance(market, MarketBatch):
            return Context(market, history, idx)
        return Context(history.market.row(idx), history, idx)
    if not isinstance(market, MarketBatch) or market.n_states != 1:
        raise ValueError("market must be a MarketHistory or a single-state MarketBatch")
    return Context(market)


def _need_history(ctx: Context, what: str):
    if ctx.history is None or ctx.index is None:
        raise ValueError(f"{what} needs the market history")


# ---------- building blocks shared by the seeded views and the fixes ----------
def _rec(trade, risk_type, qualifier, bucket, label1, label2, amount, ccy, usd) -> dict:
    return {C.CRIF_TRADE_ID: trade[C.TRADE_ID], C.PORTFOLIO_ID: trade[C.PORTFOLIO_ID], C.PRODUCT_CLASS: C.RATES_FX,
            C.RISK_TYPE: risk_type, C.QUALIFIER: qualifier, C.BUCKET: bucket, C.LABEL1: label1, C.LABEL2: label2,
            C.AMOUNT: amount, C.AMOUNT_CCY: ccy, C.AMOUNT_USD: usd}


def zero_delta_rows(trades: pd.DataFrame, row: MarketBatch) -> pd.DataFrame:
    """IR delta rows from 1bp bumps of the zero rates instead of the par quotes (the D3 convention)."""
    rows = []
    for _, t in trades.iterrows():
        for ccy in trade_currencies(t):
            spot = float(row.fx_spot[ccy][0])
            for tenor, usd in zip(config.IR_TENORS, zero_pillar_deltas(t, row, ccy)):
                if abs(usd) > MIN_ABS_USD:
                    rows.append(_rec(t, C.RISK_IRCURVE, ccy, IR_BUCKET[ccy], tenor, SUBCURVE, usd / spot, ccy, usd))
    return pd.DataFrame(rows, columns=CRIF_COLUMNS)


def _swap_rows(crif: pd.DataFrame, new_rows: pd.DataFrame, remove_mask) -> pd.DataFrame:
    new_rows = new_rows.reindex(columns=crif.columns)
    return pd.concat([crif[~np.asarray(remove_mask)], new_rows], ignore_index=True)


def _off_vertex(years: float):
    """(lower tenor label, upper tenor label) when `years` lies strictly between two IR vertices, else None."""
    parts = rebucket_weights(years)
    if len(parts) < 2:
        return None
    return config.IR_TENORS[parts[0][0]], config.IR_TENORS[parts[1][0]]


def _map_up_targets(trades: pd.DataFrame) -> list:
    """[(trade_id, risk_type, lower label, upper label)] for off-vertex IRS maturities and swaption expiries."""
    out = []
    for _, t in trades.iterrows():
        if t[C.PRODUCT] == C.PRODUCT_IRS:
            br = _off_vertex(float(t[C.MATURITY]))
            if br:
                out.append((t[C.TRADE_ID], C.RISK_IRCURVE) + br)
        elif t[C.PRODUCT] == C.PRODUCT_SWAPTION:
            br = _off_vertex(float(t[C.EXPIRY]))
            if br:
                out.append((t[C.TRADE_ID], C.RISK_IRVOL) + br)
    return out


# ---------- seeded counterparty views ----------
def _d1(t, c, p, ctx):
    return t[t[C.TRADE_ID] != D1_MISSING_TRADE].reset_index(drop=True), c[c[C.CRIF_TRADE_ID] != D1_MISSING_TRADE].reset_index(drop=True), p


def _d2(t, c, p, ctx):
    t = t.copy()
    t.loc[t[C.TRADE_ID] == D2_AMENDED_TRADE, C.NOTIONAL] *= D2_NOTIONAL_FACTOR
    c = c.copy()
    m = c[C.CRIF_TRADE_ID] == D2_AMENDED_TRADE
    c.loc[m, [C.AMOUNT, C.AMOUNT_USD]] = c.loc[m, [C.AMOUNT, C.AMOUNT_USD]] * D2_NOTIONAL_FACTOR
    return t, c, p


def _d3(t, c, p, ctx):
    return t, _swap_rows(c, zero_delta_rows(t, ctx.row), c[C.RISK_TYPE] == C.RISK_IRCURVE), p


def _d4(t, c, p, ctx):
    c = c.copy()
    m = c[C.RISK_TYPE] == C.RISK_FXVOL
    # B weights vega by the market implied vol itself: with SigmaMarket = sigma_SIMM the engine leaves Amount unscaled
    c.loc[m, SIGMA_MARKET_COL] = [p.fx_sigma(q[:3], q[3:]) for q in c.loc[m, C.QUALIFIER]]
    return t, c, p


def _d5(t, c, p, ctx):
    _need_history(ctx, "D5")
    c = c.copy()
    m = (c[C.AMOUNT_CCY] != config.CALC_CCY).to_numpy()
    stale = c.loc[m, C.AMOUNT_CCY].map(lambda x: ctx.spot(x, STALE_FX_DAYS))
    c.loc[m, C.AMOUNT_USD] = c.loc[m, C.AMOUNT] * stale
    return t, c, p


def _d6(t, c, p, ctx):
    c = c.copy()
    for tid, rtype, lo, hi in _map_up_targets(t):
        m = (c[C.CRIF_TRADE_ID] == tid) & (c[C.RISK_TYPE] == rtype) & (c[C.LABEL1] == lo)
        c.loc[m, C.LABEL1] = hi                # sensitivity linear rebucketing would split goes 100% to the upper vertex
    return t, c, p


def _d7(t, c, p, ctx):
    _need_history(ctx, "D7")
    stale = ctx.history.market.row(ctx.index - STALE_CURVE_DAYS)
    b = ctx.row.copy_with()
    for ccy in b.par:
        b.par[ccy], b.zero[ccy] = stale.par[ccy], stale.zero[ccy]
    b.xccy_basis = {k: v for k, v in stale.xccy_basis.items()}
    return t, build_crif(t, b), p


def _d8(t, c, p, ctx):
    return t, c, params_mod.load(config.SIMM_VERSION_PRIOR)


_APPLY = {"D1": _d1, "D2": _d2, "D3": _d3, "D4": _d4, "D5": _d5, "D6": _d6, "D7": _d7, "D8": _d8}
_COMPOSITE = {"D9": ("D1", "D5")}


def make_counterparty_view(trades: pd.DataFrame, market, crif: pd.DataFrame, scenario: str, history: MarketHistory = None,
                           date=None, ctx: Context = None):
    """(trades_B, crif_B, params_B) for a seeded scenario. market: MarketHistory (valuation date = `date`, default
    config.SCENARIO_DAY) or a single-state MarketBatch (D5 and D7 then need `history`)."""
    key = scenario.upper()
    if key not in SCENARIOS:
        raise ValueError(f"unknown dispute scenario {scenario!r}; use one of {SCENARIOS}")
    ctx = ctx or make_context(market, history, date)
    t, c, p = trades.copy(), crif.copy(), params_mod.load(config.SIMM_VERSION)
    for s in _COMPOSITE.get(key, (key,)):
        t, c, p = _APPLY[s](t, c, p, ctx)
    return t.reset_index(drop=True), c.reset_index(drop=True), p


# ---------- reconciliation ----------
@dataclass
class ReconReport:
    scenario: str
    a: PartyView
    b: PartyView
    context: Context
    im_a: float
    im_b: float
    population: pd.DataFrame
    crif_diff: pd.DataFrame
    sigma_diffs: int
    param_mismatch: bool
    gap_tree: pd.DataFrame
    gap_path: list
    contributions: pd.DataFrame
    evidence_units: int

    @property
    def gap(self) -> float:
        return self.im_b - self.im_a

    def crif_counts(self) -> dict:
        return self.crif_diff[STATUS_COL].value_counts().to_dict() if len(self.crif_diff) else {}


def _population(ta: pd.DataFrame, tb: pd.DataFrame) -> pd.DataFrame:
    a = ta.set_index(C.TRADE_ID)[C.NOTIONAL].rename(NOTIONAL_A)
    b = tb.set_index(C.TRADE_ID)[C.NOTIONAL].rename(NOTIONAL_B)
    d = pd.concat([a, b], axis=1).rename_axis(TRADE_COL).reset_index()
    d[IN_A], d[IN_B] = d[NOTIONAL_A].notna(), d[NOTIONAL_B].notna()
    same = np.isclose(d[NOTIONAL_A], d[NOTIONAL_B], rtol=config.DISPUTE_TOL_REL, atol=0.0)
    d[STATUS_COL] = np.where(~d[IN_B], "only_a", np.where(~d[IN_A], "only_b", np.where(same, "both", "notional_differs")))
    return d


def _keyed(crif: pd.DataFrame) -> pd.DataFrame:
    k = crif.copy()
    for col in KEY_COLUMNS:
        k[col] = k[col].fillna("").astype(str)
    return k.groupby(KEY_COLUMNS, sort=False)[[C.AMOUNT, C.AMOUNT_USD]].sum().reset_index()


def crif_key_match(ca: pd.DataFrame, cb: pd.DataFrame, tol_abs: float = config.DISPUTE_TOL_ABS,
                   tol_rel: float = config.DISPUTE_TOL_REL) -> pd.DataFrame:
    """Outer join on the CRIF key; status matched, amount_differs, only_a or only_b (AmountUSD within tol_abs/tol_rel)."""
    m = _keyed(ca).merge(_keyed(cb), on=KEY_COLUMNS, how="outer", suffixes=("_a", "_b"), indicator=True)
    ua, ub = m[C.AMOUNT_USD + "_a"].fillna(0.0), m[C.AMOUNT_USD + "_b"].fillna(0.0)
    ok = (ua - ub).abs() <= np.maximum(tol_abs, tol_rel * np.maximum(ua.abs(), ub.abs()))
    m[STATUS_COL] = np.where(m["_merge"] == "left_only", "only_a", np.where(m["_merge"] == "right_only", "only_b",
                                                                              np.where(ok, "matched", "amount_differs")))
    m["usd_diff"] = ub - ua
    return m.drop(columns="_merge")


def _sigma_diffs(ca: pd.DataFrame, cb: pd.DataFrame) -> int:
    if SIGMA_MARKET_COL not in ca.columns or SIGMA_MARKET_COL not in cb.columns:
        return 0
    f = lambda c: c[c[C.RISK_TYPE] == C.RISK_FXVOL].groupby(C.CRIF_TRADE_ID)[SIGMA_MARKET_COL].first()
    j = pd.concat([f(ca).rename("a"), f(cb).rename("b")], axis=1).dropna()
    return int(((j["a"] - j["b"]).abs() > SIGMA_TOL).sum())


def _gap_tree(ra, rb) -> pd.DataFrame:
    cols = [C.LEVEL, C.RISK_CLASS, C.MARGIN_TYPE, _BUCKET, C.RISK_FACTOR]
    fa, fb = ra.breakdown, rb.breakdown
    fa, fb = fa[fa[C.LEVEL].isin(_GAP_LEVELS)], fb[fb[C.LEVEL].isin(_GAP_LEVELS)]
    t = fa.merge(fb, on=cols, how="outer", suffixes=("_a", "_b"))
    t[VALUE_A], t[VALUE_B] = t[C.VALUE + "_a"].fillna(0.0), t[C.VALUE + "_b"].fillna(0.0)
    t[GAP] = t[VALUE_B] - t[VALUE_A]
    t["level_order"] = t[C.LEVEL].map({lv: i for i, lv in enumerate(_GAP_LEVELS)})
    return t[cols + [VALUE_A, VALUE_B, GAP, "level_order"]].sort_values(["level_order", GAP], key=lambda s: s if s.name == "level_order" else -s.abs()).reset_index(drop=True)


def _gap_path(tree: pd.DataFrame) -> list:
    """Follow the largest absolute gap down risk class -> margin type -> bucket -> risk factor."""
    def top(df):
        return df.loc[df[GAP].abs().idxmax()] if len(df) and df[GAP].abs().max() > 0 else None
    path = []
    rc = top(tree[tree[C.LEVEL] == "risk_class"])
    if rc is None:
        return path
    path.append(("risk_class", rc[C.RISK_CLASS], rc[GAP]))
    mt = top(tree[(tree[C.LEVEL] == "margin_type") & (tree[C.RISK_CLASS] == rc[C.RISK_CLASS])])
    if mt is None:
        return path
    path.append(("margin_type", mt[C.MARGIN_TYPE], mt[GAP]))
    sel = (tree[C.RISK_CLASS] == rc[C.RISK_CLASS]) & (tree[C.MARGIN_TYPE] == mt[C.MARGIN_TYPE])
    bk = top(tree[sel & (tree[C.LEVEL] == "K")])
    if bk is None:
        return path
    path.append(("bucket", bk[_BUCKET], bk[GAP]))
    fc = top(tree[sel & (tree[C.LEVEL] == "factor") & (tree[_BUCKET] == bk[_BUCKET])])
    if fc is not None:
        path.append(("risk_factor", fc[C.RISK_FACTOR], fc[GAP]))
    return path


def _contributions(a: PartyView, b: PartyView) -> pd.DataFrame:
    ca = euler_contributions(a.crif, a.params).rename("euler_a")
    cb = euler_contributions(b.crif, b.params).rename("euler_b")
    d = pd.concat([ca, cb], axis=1).fillna(0.0)
    d[GAP] = d["euler_b"] - d["euler_a"]
    return d.rename_axis(TRADE_COL).reset_index().sort_values(GAP, key=lambda s: -s.abs()).reset_index(drop=True)


def _units(pop: pd.DataFrame, diff: pd.DataFrame, sigma: int, param_mismatch: bool) -> int:
    return int((pop[STATUS_COL] != "both").sum() + (diff[STATUS_COL] != "matched").sum() + sigma + int(param_mismatch))


def reconcile(a: PartyView, b: PartyView, context: Context = None, scenario: str = "", with_euler: bool = True) -> ReconReport:
    """Compare two party views. context is carried so that rank_causes can apply counterfactual fixes."""
    ra, rb = simm(a.crif, a.params), simm(b.crif, b.params)
    pop = _population(a.trades, b.trades)
    diff = crif_key_match(a.crif, b.crif)
    sig = _sigma_diffs(a.crif, b.crif)
    pm = a.params.version != b.params.version
    tree = _gap_tree(ra, rb)
    contrib = _contributions(a, b) if with_euler else pd.DataFrame(columns=[TRADE_COL, "euler_a", "euler_b", GAP])
    return ReconReport(scenario, a, b, context, ra.total, rb.total, pop, diff, sig, pm, tree, _gap_path(tree), contrib,
                       _units(pop, diff, sig, pm))


# ---------- counterfactual fixes (normalise B to the reference convention along one dimension) ----------
def _fix_missing(a, b, ctx):
    pop = _population(a.trades, b.trades)
    only_a, only_b = set(pop.loc[pop[STATUS_COL] == "only_a", TRADE_COL]), set(pop.loc[pop[STATUS_COL] == "only_b", TRADE_COL])
    rm = b.crif[C.CRIF_TRADE_ID].isin(only_b).to_numpy()
    add = a.crif[a.crif[C.CRIF_TRADE_ID].isin(only_a)]
    trades = pd.concat([b.trades[~b.trades[C.TRADE_ID].isin(only_b)], a.trades[a.trades[C.TRADE_ID].isin(only_a)]], ignore_index=True)
    return PartyView(b.label, trades, _swap_rows(b.crif, add, rm), b.params), int(rm.sum() + len(add))


def _fix_notional(a, b, ctx):
    pop = _population(a.trades, b.trades)
    bad = pop[pop[STATUS_COL] == "notional_differs"]
    trades, crif = b.trades.copy(), b.crif.copy()
    scope = 0
    for _, r in bad.iterrows():
        k = r[NOTIONAL_A] / r[NOTIONAL_B]
        trades.loc[trades[C.TRADE_ID] == r[TRADE_COL], C.NOTIONAL] = r[NOTIONAL_A]
        m = crif[C.CRIF_TRADE_ID] == r[TRADE_COL]
        crif.loc[m, [C.AMOUNT, C.AMOUNT_USD]] = crif.loc[m, [C.AMOUNT, C.AMOUNT_USD]] * k
        scope += int(m.sum())
    return PartyView(b.label, trades, crif, b.params), scope


def _fix_zero_rate(a, b, ctx):
    fresh = ctx.fresh_crif(b.trades)
    rm = (b.crif[C.RISK_TYPE] == C.RISK_IRCURVE).to_numpy()
    return PartyView(b.label, b.trades, _swap_rows(b.crif, fresh[fresh[C.RISK_TYPE] == C.RISK_IRCURVE], rm), b.params), int(rm.sum())


def _fix_fx_vega(a, b, ctx):
    crif = b.crif.copy()
    m = (crif[C.RISK_TYPE] == C.RISK_FXVOL).to_numpy()
    sig = {t[C.TRADE_ID]: fx_vega(t, ctx.row)[1] for _, t in b.trades.iterrows() if t[C.PRODUCT] == C.PRODUCT_FXOPT}
    if SIGMA_MARKET_COL not in crif.columns:
        crif[SIGMA_MARKET_COL] = np.nan
    crif.loc[m, SIGMA_MARKET_COL] = crif.loc[m, C.CRIF_TRADE_ID].map(sig)
    return PartyView(b.label, b.trades, crif, b.params), int(m.sum())


def _fix_fx_rate(a, b, ctx):
    crif = b.crif.copy()
    m = (crif[C.AMOUNT_CCY] != config.CALC_CCY).to_numpy()
    crif.loc[m, C.AMOUNT_USD] = crif.loc[m, C.AMOUNT] * crif.loc[m, C.AMOUNT_CCY].map(ctx.spot)
    return PartyView(b.label, b.trades, crif, b.params), int(m.sum())


def _fix_rebucket(a, b, ctx):
    fresh = ctx.fresh_crif(b.trades)
    rm = np.zeros(len(b.crif), dtype=bool)
    add = []
    for tid, rtype, _, _ in _map_up_targets(b.trades):
        rm |= ((b.crif[C.CRIF_TRADE_ID] == tid) & (b.crif[C.RISK_TYPE] == rtype)).to_numpy()
        add.append(fresh[(fresh[C.CRIF_TRADE_ID] == tid) & (fresh[C.RISK_TYPE] == rtype)])
    new = pd.concat(add, ignore_index=True) if add else b.crif.iloc[0:0]
    return PartyView(b.label, b.trades, _swap_rows(b.crif, new, rm), b.params), int(rm.sum())


def _fix_curve(a, b, ctx):
    fresh = ctx.fresh_crif(b.trades)
    return PartyView(b.label, b.trades, fresh.reindex(columns=b.crif.columns), b.params), len(b.crif)


def _fix_params(a, b, ctx):
    return PartyView(b.label, b.trades, b.crif, a.params), int(a.params.version != b.params.version)


HYPOTHESES = {CAUSE_MISSING: _fix_missing, CAUSE_NOTIONAL: _fix_notional, CAUSE_ZERO: _fix_zero_rate,
              CAUSE_FXVEGA: _fix_fx_vega, CAUSE_FXRATE: _fix_fx_rate, CAUSE_REBUCKET: _fix_rebucket,
              CAUSE_CURVE: _fix_curve, CAUSE_PARAMS: _fix_params}


def rank_causes(report: ReconReport) -> pd.DataFrame:
    """Rank the hypotheses by the |gap| left after applying each one's fix to B (ties within the tolerance: fewer rows
    in scope first). Columns cause, signature_score (share of reconciliation breaks the fix removes), explained_gap,
    residual_after_fix (both absolute USD), scope_rows, rank."""
    if report.context is None:
        raise ValueError("rank_causes needs the reconcile(..., context=...) market context")
    a, b, ctx = report.a, report.b, report.context
    gap0 = abs(report.gap)
    rows = []
    for cause, fix in HYPOTHESES.items():
        fixed, scope = fix(a, b, ctx)
        res = abs(simm(fixed.crif, fixed.params).total - report.im_a)
        pop = _population(a.trades, fixed.trades)
        diff = crif_key_match(a.crif, fixed.crif)
        units = _units(pop, diff, _sigma_diffs(a.crif, fixed.crif), a.params.version != fixed.params.version)
        score = (report.evidence_units - units) / report.evidence_units if report.evidence_units > 0 else 0.0
        rows.append({CAUSE_COL: cause, SIGNATURE: float(score), EXPLAINED: gap0 - res, RESIDUAL: res, SCOPE: scope})
    tie = max(config.DISPUTE_TOL_ABS, config.DISPUTE_TOL_REL * abs(report.im_a))
    rows.sort(key=lambda r: r[RESIDUAL])
    ordered, i = [], 0
    while i < len(rows):                       # cluster residuals within the tolerance, then prefer the narrower fix
        j = i
        while j < len(rows) and rows[j][RESIDUAL] - rows[i][RESIDUAL] <= tie:
            j += 1
        ordered += sorted(rows[i:j], key=lambda r: (r[SCOPE], -r[SIGNATURE]))
        i = j
    out = pd.DataFrame(ordered)
    out[RANK] = np.arange(1, len(out) + 1)
    return out[[CAUSE_COL, SIGNATURE, EXPLAINED, RESIDUAL, SCOPE, RANK]]


# ---------- driver ----------
def run_scenario(scenario: str, trades, crif, ctx: Context, with_euler: bool = True):
    """(ReconReport, ranking) for one seeded scenario."""
    tb, cb, pb = make_counterparty_view(trades, ctx.row, crif, scenario, history=ctx.history, ctx=ctx)
    a = PartyView("A", trades, crif, params_mod.load(config.SIMM_VERSION))
    rep = reconcile(a, PartyView("B", tb, cb, pb), ctx, scenario, with_euler)
    return rep, rank_causes(rep)


def _accepted(scenario: str, ranks: list) -> bool:
    return max(ranks) <= COMPOSITE_TOP_N if scenario == "D9" else ranks == [1]


def run_disputes(history: MarketHistory = None, trades: pd.DataFrame = None, out_dir=None, scenarios=SCENARIOS) -> pd.DataFrame:
    """Run the scenarios at config.SCENARIO_DAY; write dispute_results.csv and dispute_details/<scenario>_*.csv."""
    from .instruments import sample_portfolio
    from .market_history import generate_history
    history = history if history is not None else generate_history()
    trades = trades if trades is not None else sample_portfolio()
    out = Path(out_dir) if out_dir is not None else config.OUT_DIR
    det = out / DETAILS_DIR
    det.mkdir(parents=True, exist_ok=True)
    ctx = make_context(history)
    crif = ctx.fresh_crif(trades)
    rows = []
    for s in scenarios:
        rep, rk = run_scenario(s, trades, crif, ctx)
        ranks = [int(rk.loc[rk[CAUSE_COL] == c, RANK].iloc[0]) for c in SEEDED[s]]
        top = rk.iloc[0]
        rows.append({"scenario": s, "seeded_cause": "+".join(SEEDED[s]), "rank_of_seeded_cause": max(ranks),
                     "seeded_ranks": "+".join(str(r) for r in ranks), "im_a": rep.im_a, "im_b": rep.im_b, "gap": rep.gap,
                     "top_ranked_cause": top[CAUSE_COL], "residual_after_top_fix": float(top[RESIDUAL]),
                     "accepted": _accepted(s, ranks), "data_source": config.DATA_SOURCE})
        rk.to_csv(det / f"{s}_ranking.csv", index=False)
        rep.gap_tree.drop(columns="level_order").head(40).to_csv(det / f"{s}_gap_tree.csv", index=False)
        rep.contributions.to_csv(det / f"{s}_trade_contributions.csv", index=False)
    res = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    res.to_csv(out / RESULTS_FILE, index=False)
    return res
