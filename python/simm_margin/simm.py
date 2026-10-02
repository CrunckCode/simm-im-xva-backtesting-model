"""SIMM-style initial margin for the RatesFX product class (risk classes IR and FX). SYNTHETIC inputs; not ISDA-licensed.

Formulas follow the structure of the public methodology text (paras 5-11, Sections D, I, J, K) and are described in
plain maths here; all numbers come from params.SimmParams.
  Delta (IR):  WS = RW * s * CR_b, CR_b = max(1, sqrt(|sum s| / T_b)) over yield and inflation rows only (basis rows
    are neither summed nor scaled). K_b = sqrt(WS' M WS) with M = phi * tenor corr for yields, 0.42 inflation-yield,
    -0.01 basis-anything. Margin = sqrt(sum K_b^2 + sum_b sum_{c!=b} gamma g_bc S_b S_c), S_b = clip(sum WS, +-K_b),
    g_bc = min(CR_b, CR_c) / max(CR_b, CR_c).
  Delta (FX): single bucket, CR_k = max(1, sqrt(|s_k| / T_cat)), WS = RW * s * CR, K = sqrt(WS' M WS) with
    M_kl = rho_kl * min(CR)/max(CR); margin = K.
  Vega (IR): VR_k = VRW * (net USD vega*vol) * VCR_b, VCR_b = max(1, sqrt(|sum VR_ik| / VT_b)); K_b uses the tenor
    matrix (f = 1) and 0.42 between inflation vol and IR vol; across currencies gamma and g_bc = min(VCR)/max(VCR).
  Vega (FX): VR_ik = HVR * sigma_SIMM * vega per pair, sigma_SIMM = RW*sqrt(365/14)/Phi^-1(99%); VR_k = VRW * VR_ik *
    VCR_k; K = sqrt(sum VR^2 + sum rho f VR VR) with rho = 0.5, f = min(VCR)/max(VCR); margin = K.
  Curvature: CVR = SF(t) * sigma * vega with SF(t) = 0.5 * min(1, 14/t_days); K_b from rho^2; theta = min(sum CVR /
    sum |CVR|, 0), lambda = (Phi^-1(99.5%)^2 - 1)(1 + theta) - theta; margin = max(sum CVR + lambda *
    sqrt(sum K_b^2 + sum gamma^2 S_b S_c), 0), IR margin times HVR_IR^-2. IR factors are (currency, expiry vertex);
    FX factors are currency pairs (all expiries netted with SF weights).
  Risk class IM = delta + vega + curvature; RatesFX = sqrt(IR^2 + FX^2 + 2 psi IR FX).
CRIF conventions: AmountUSD is used for everything; IR vol Amount is vega*normal vol (USD); FX vol Amount is vega per
vol point (USD) unless the column SigmaMarket is present, in which case Amount is vega*SigmaMarket (as crif.py emits).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import columns as C
from . import config
from .params import SimmParams, load

# ===== CONFIG (user inputs) =====
RISK_INFLATION, RISK_INFLATIONVOL = "Risk_Inflation", "Risk_InflationVol"   # not in columns.py (frozen)
SIGMA_MARKET_COL = "SigmaMarket"          # optional CRIF column: market vol (fraction) used to weight FX vega rows
KIND_YIELD, KIND_INFL, KIND_XCCY = "yield", "inflation", "xccy_basis"
FX_BUCKET = "FX"
USD_MM = 1.0e6                            # thresholds are in USD millions
DEFAULT_SUBCURVE = "OIS"
BUCKET_COL = "bucket"                     # breakdown column name (columns.py is frozen and only has the CRIF "Bucket")
LEVEL_FACTOR, LEVEL_K, LEVEL_S, LEVEL_MARGIN, LEVEL_RC, LEVEL_PRODUCT, LEVEL_TOTAL = (
    "factor", "K", "S", "margin_type", "risk_class", "product_class", "total")
LEVEL_CR = "CR"
# ===== END CONFIG =====

BREAKDOWN_COLUMNS = [C.LEVEL, C.RISK_CLASS, C.MARGIN_TYPE, BUCKET_COL, C.RISK_FACTOR, C.VALUE]
_IR_TYPES = {C.RISK_IRCURVE, C.RISK_XCCYBASIS, RISK_INFLATION, C.RISK_IRVOL, RISK_INFLATIONVOL}
_ALL_TYPES = _IR_TYPES | {C.RISK_FX, C.RISK_FXVOL}


@dataclass
class SimmResult:
    """total (USD), breakdown (columns level, risk_class, margin_type, bucket, risk_factor, value) and risk class IMs."""
    total: float
    breakdown: pd.DataFrame
    by_risk_class: dict = field(default_factory=dict)
    by_margin_type: dict = field(default_factory=dict)       # {(risk_class, margin_type): value}


def _row(level, rc, mt, bucket, factor, value) -> dict:
    return {C.LEVEL: level, C.RISK_CLASS: rc, C.MARGIN_TYPE: mt, BUCKET_COL: bucket, C.RISK_FACTOR: factor, C.VALUE: float(value)}


def _sqrt0(x: float) -> float:
    return float(np.sqrt(max(x, 0.0)))


def _clean(crif: pd.DataFrame) -> pd.DataFrame:
    """Validate the CRIF and normalise text columns; unknown risk types or product classes raise."""
    need = {C.RISK_TYPE, C.QUALIFIER, C.LABEL1, C.LABEL2, C.AMOUNT_USD}
    missing = need - set(crif.columns)
    if missing:
        raise KeyError(f"CRIF is missing columns {sorted(missing)}")
    if len(crif) == 0:
        return crif
    bad = set(crif[C.RISK_TYPE]) - _ALL_TYPES
    if bad:
        raise ValueError(f"unsupported risk types {sorted(bad)} (RatesFX scope only)")
    if C.PRODUCT_CLASS in crif.columns and set(crif[C.PRODUCT_CLASS].dropna()) - {C.RATES_FX}:
        raise ValueError("only the RatesFX product class is supported")
    out = crif.copy()
    for col in (C.QUALIFIER, C.LABEL1, C.LABEL2):
        out[col] = out[col].fillna("").astype(str)
    return out


def _sel(crif, *types):
    return crif[crif[C.RISK_TYPE].isin(types)] if len(crif) else crif


def _vertex(label: str) -> int:
    try:
        return config.IR_TENORS.index(label)
    except ValueError:
        raise ValueError(f"tenor label {label!r} is not one of {config.IR_TENORS}")


def _pair(qualifier: str) -> tuple:
    if len(qualifier) != 6:
        raise ValueError(f"FX pair qualifier {qualifier!r} must be two three-letter codes")
    return qualifier[:3], qualifier[3:]


# == delta
def weighted_sensitivities(crif: pd.DataFrame, p: SimmParams) -> pd.DataFrame:
    """Net delta sensitivities per risk factor with RW, CR and WS = RW*s*CR (columns risk_class, bucket, kind, subcurve,
    tenor, risk_factor, sens, RW, CR, WS). IR rows are bucketed by currency, FX rows form one bucket."""
    crif = _clean(crif)
    rows = []
    ir = _sel(crif, C.RISK_IRCURVE, RISK_INFLATION, C.RISK_XCCYBASIS)
    if len(ir):
        kind = ir[C.RISK_TYPE].map({C.RISK_IRCURVE: KIND_YIELD, RISK_INFLATION: KIND_INFL, C.RISK_XCCYBASIS: KIND_XCCY})
        sub = np.where(kind == KIND_YIELD, ir[C.LABEL2].replace("", DEFAULT_SUBCURVE), "")
        ten = np.where(kind == KIND_YIELD, ir[C.LABEL1], "")
        net = (pd.DataFrame({"ccy": ir[C.QUALIFIER].values, "kind": kind.values, "subcurve": sub, "tenor": ten,
                             "s": ir[C.AMOUNT_USD].values})
               .groupby(["ccy", "kind", "subcurve", "tenor"], sort=False)["s"].sum().reset_index())
        for ccy, g in net.groupby("ccy", sort=False):
            crsum = g.loc[g["kind"] != KIND_XCCY, "s"].sum()
            cr = max(1.0, np.sqrt(abs(crsum) / USD_MM / p.ir_delta_ct(ccy)))
            rw_t = p.ir_rw(ccy)
            for r in g.itertuples(index=False):
                if r.kind == KIND_YIELD:
                    rw, crf = rw_t[_vertex(r.tenor)], cr
                elif r.kind == KIND_INFL:
                    rw, crf = p.ir_rw_inflation, cr
                else:
                    rw, crf = p.ir_rw_xccy, 1.0
                name = "|".join(x for x in (r.subcurve, r.tenor) if x) or r.kind
                rows.append((C.RC_IR, ccy, r.kind, r.subcurve, r.tenor, name, r.s, rw, crf, rw * r.s * crf))
    fx = _sel(crif, C.RISK_FX)
    if len(fx):
        net = fx.groupby(C.QUALIFIER, sort=False)[C.AMOUNT_USD].sum()
        for ccy, s in net.items():
            if ccy == p.calc_ccy:
                continue                                  # no FX risk factor for the calculation currency
            rw = p.fx_rw(ccy)
            crf = max(1.0, np.sqrt(abs(s) / USD_MM / p.fx_delta_ct(ccy)))
            rows.append((C.RC_FX, FX_BUCKET, "fx", "", "", ccy, s, rw, crf, rw * s * crf))
    return pd.DataFrame(rows, columns=[C.RISK_CLASS, BUCKET_COL, "kind", "subcurve", "tenor", C.RISK_FACTOR, C.SENS, C.RW, C.CR, C.WS])


def ir_corr_matrix(kinds, subcurves, tenors, p: SimmParams) -> np.ndarray:
    """Correlation matrix of IR delta factors in one currency: phi*rho for yields, 0.42 inflation-yield, -0.01 basis."""
    n = len(kinds)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ki, kj = kinds[i], kinds[j]
            if KIND_XCCY in (ki, kj):
                c = p.xccy_corr
            elif KIND_INFL in (ki, kj):
                c = p.infl_corr
            else:
                phi = 1.0 if subcurves[i] == subcurves[j] else p.phi
                c = phi * p.ir_corr[_vertex(tenors[i]), _vertex(tenors[j])]
            m[i, j] = m[j, i] = c
    return m


def ir_bucket_K(ws_bucket: pd.DataFrame, p: SimmParams) -> float:
    """K for one currency from rows with columns kind, subcurve, tenor, WS."""
    if len(ws_bucket) == 0:
        return 0.0
    w = ws_bucket[C.WS].to_numpy(dtype=float)
    m = ir_corr_matrix(ws_bucket["kind"].tolist(), ws_bucket["subcurve"].tolist(), ws_bucket["tenor"].tolist(), p)
    return _sqrt0(w @ m @ w)


def _cross_bucket(ks, ss, gam_mat) -> float:
    """sqrt(sum K^2 + sum_{b!=c} gamma_bc S_b S_c) with a ready gamma matrix (diagonal ignored)."""
    ks, ss = np.asarray(ks, dtype=float), np.asarray(ss, dtype=float)
    g = np.array(gam_mat, dtype=float)
    np.fill_diagonal(g, 0.0)
    return _sqrt0(float((ks ** 2).sum() + ss @ g @ ss))


def _clip(x, k):
    return max(min(x, k), -k)


def delta_margin_ir(ws: pd.DataFrame, p: SimmParams):
    """(IR delta margin, breakdown rows) from weighted_sensitivities output."""
    d = ws[ws[C.RISK_CLASS] == C.RC_IR]
    rows, ks, ss, crs = [], [], [], []
    for ccy, g in d.groupby(BUCKET_COL, sort=False):
        k = ir_bucket_K(g, p)
        s = _clip(g[C.WS].sum(), k)
        cr = g.loc[g["kind"] != KIND_XCCY, C.CR].max() if (g["kind"] != KIND_XCCY).any() else 1.0
        ks.append(k), ss.append(s), crs.append(cr)
        for r in g.itertuples(index=False):
            rows.append(_row(LEVEL_FACTOR, C.RC_IR, C.MT_DELTA, ccy, getattr(r, "risk_factor"), r.WS))
        rows.append(_row(LEVEL_CR, C.RC_IR, C.MT_DELTA, ccy, "", cr))
        rows.append(_row(LEVEL_K, C.RC_IR, C.MT_DELTA, ccy, "", k))
        rows.append(_row(LEVEL_S, C.RC_IR, C.MT_DELTA, ccy, "", s))
    if not ks:
        return 0.0, rows
    crs = np.array(crs)
    g_bc = np.minimum.outer(crs, crs) / np.maximum.outer(crs, crs)
    return _cross_bucket(ks, ss, p.ir_gamma * g_bc), rows


def delta_margin_fx(ws: pd.DataFrame, p: SimmParams):
    """(FX delta margin, breakdown rows): one bucket, correlation rho_kl * min(CR)/max(CR)."""
    d = ws[ws[C.RISK_CLASS] == C.RC_FX]
    rows = [_row(LEVEL_FACTOR, C.RC_FX, C.MT_DELTA, FX_BUCKET, r.risk_factor, r.WS) for r in d.itertuples(index=False)]
    if len(d) == 0:
        return 0.0, rows
    ccys, w, cr = d[C.RISK_FACTOR].tolist(), d[C.WS].to_numpy(dtype=float), d[C.CR].to_numpy(dtype=float)
    m = p.fx_corr_matrix(ccys) * (np.minimum.outer(cr, cr) / np.maximum.outer(cr, cr))
    np.fill_diagonal(m, 1.0)
    k = _sqrt0(w @ m @ w)
    rows.append(_row(LEVEL_K, C.RC_FX, C.MT_DELTA, FX_BUCKET, "", k))
    return k, rows


# == vega and curvature inputs
def _ir_vol_factors(crif: pd.DataFrame, p: SimmParams = None):
    """Net IR/inflation vol exposures VR_ik (USD): list of (ccy, kind, tenor, amount); with p, rows are first weighted
    by SF(t_vertex) (curvature). Inflation vol is one factor per currency (tenor key empty)."""
    v = _sel(crif, C.RISK_IRVOL, RISK_INFLATIONVOL)
    if len(v) == 0:
        return []
    amt = v[C.AMOUNT_USD].to_numpy(dtype=float)
    if p is not None:
        sf = p.sf(p.tenor_days())
        amt = amt * np.array([sf[_vertex(t)] for t in v[C.LABEL1]])
    kind = v[C.RISK_TYPE].map({C.RISK_IRVOL: KIND_YIELD, RISK_INFLATIONVOL: KIND_INFL})
    ten = np.where(kind == KIND_YIELD, v[C.LABEL1], "")
    net = (pd.DataFrame({"ccy": v[C.QUALIFIER].values, "kind": kind.values, "tenor": ten, "a": amt})
           .groupby(["ccy", "kind", "tenor"], sort=False)["a"].sum().reset_index())
    return list(net.itertuples(index=False, name=None))


def _fx_vol_by_pair(crif: pd.DataFrame, p: SimmParams, sf_weighted: bool) -> dict:
    """Net FX vega per pair as sigma_SIMM * vega (USD), optionally with SF(t_vertex) weights (curvature)."""
    v = _sel(crif, C.RISK_FXVOL)
    out = {}
    if len(v) == 0:
        return out
    sf = p.sf(p.tenor_days())
    has_sig = SIGMA_MARKET_COL in v.columns
    for r in v.itertuples(index=False):
        a, q = getattr(r, C.AMOUNT_USD), getattr(r, C.QUALIFIER)
        sig_simm = p.fx_sigma(*_pair(q))                       # fraction
        if has_sig and not pd.isna(getattr(r, SIGMA_MARKET_COL)):
            val = a / getattr(r, SIGMA_MARKET_COL) * sig_simm   # vega per unit vol times sigma_SIMM
        else:
            val = a * sig_simm * 100.0                          # vega per vol point times sigma_SIMM in vol points
        if sf_weighted:
            val *= sf[_vertex(getattr(r, C.LABEL1))]
        out[q] = out.get(q, 0.0) + val
    return out


def vega_margin_ir(crif: pd.DataFrame, p: SimmParams):
    """(IR vega margin, breakdown rows)."""
    fac = _ir_vol_factors(_clean(crif))
    rows = []
    if not fac:
        return 0.0, rows
    by_ccy = {}
    for ccy, kind, ten, a in fac:
        by_ccy.setdefault(ccy, []).append((kind, ten, a))
    ks, ss, vcrs = [], [], []
    for ccy, items in by_ccy.items():
        raw = np.array([a for _, _, a in items])
        vcr = max(1.0, np.sqrt(abs(raw.sum()) / USD_MM / p.ir_vega_ct(ccy)))
        vr = p.ir_vrw * raw * vcr
        n = len(items)
        m = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                (ki, ti, _), (kj, tj, _) = items[i], items[j]
                m[i, j] = m[j, i] = (p.infl_vol_corr if KIND_INFL in (ki, kj) else p.ir_corr[_vertex(ti), _vertex(tj)])
        k = _sqrt0(vr @ m @ vr)
        s = _clip(vr.sum(), k)
        ks.append(k), ss.append(s), vcrs.append(vcr)
        for (kind, ten, _), x in zip(items, vr):
            rows.append(_row(LEVEL_FACTOR, C.RC_IR, C.MT_VEGA, ccy, ten or kind, x))
        rows += [_row(LEVEL_CR, C.RC_IR, C.MT_VEGA, ccy, "", vcr), _row(LEVEL_K, C.RC_IR, C.MT_VEGA, ccy, "", k),
                 _row(LEVEL_S, C.RC_IR, C.MT_VEGA, ccy, "", s)]
    vcrs = np.array(vcrs)
    g_bc = np.minimum.outer(vcrs, vcrs) / np.maximum.outer(vcrs, vcrs)
    return _cross_bucket(ks, ss, p.ir_gamma * g_bc), rows


def vega_margin_fx(crif: pd.DataFrame, p: SimmParams):
    """(FX vega margin, breakdown rows): one bucket, pair-level factors, rho 0.5, f = min(VCR)/max(VCR)."""
    vr_ik = {q: p.fx_hvr * x for q, x in _fx_vol_by_pair(_clean(crif), p, False).items()}
    if not vr_ik:
        return 0.0, []
    pairs = list(vr_ik)
    raw = np.array([vr_ik[q] for q in pairs])
    vcr = np.array([max(1.0, np.sqrt(abs(x) / USD_MM / p.fx_vega_ct(*_pair(q)))) for q, x in zip(pairs, raw)])
    vr = p.fx_vrw * raw * vcr
    f = np.minimum.outer(vcr, vcr) / np.maximum.outer(vcr, vcr)
    m = p.fx_vol_corr * f
    np.fill_diagonal(m, 1.0)
    k = _sqrt0(vr @ m @ vr)
    rows = [_row(LEVEL_FACTOR, C.RC_FX, C.MT_VEGA, FX_BUCKET, q, x) for q, x in zip(pairs, vr)]
    rows.append(_row(LEVEL_K, C.RC_FX, C.MT_VEGA, FX_BUCKET, "", k))
    return k, rows


def _curvature_theta_lambda(cvr_all, p):
    tot_abs = float(np.abs(cvr_all).sum())
    theta = min(float(np.sum(cvr_all)) / tot_abs, 0.0) if tot_abs > 0 else 0.0
    return theta, p.curv_lambda(theta)


def curvature_margin(crif: pd.DataFrame, p: SimmParams, risk_class: str):
    """(curvature margin, breakdown rows) for IR or FX; IR already includes the HVR_IR^-2 scale."""
    crif = _clean(crif)
    rows = []
    if risk_class == C.RC_IR:
        by_ccy = {}
        for ccy, kind, ten, a in _ir_vol_factors(crif, p):
            by_ccy.setdefault(ccy, []).append((kind, ten, a))
        if not by_ccy:
            return 0.0, rows
        allv = np.array([x for items in by_ccy.values() for _, _, x in items])
        theta, lam = _curvature_theta_lambda(allv, p)
        ks, ss = [], []
        for ccy, items in by_ccy.items():
            c = np.array([x for _, _, x in items])
            n = len(items)
            m = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    (ki, ti, _), (kj, tj, _) = items[i], items[j]
                    r = p.infl_vol_corr if KIND_INFL in (ki, kj) else p.ir_corr[_vertex(ti), _vertex(tj)]
                    m[i, j] = m[j, i] = r * r
            k = _sqrt0(c @ m @ c)
            ks.append(k), ss.append(_clip(c.sum(), k))
            for (kind, ten, _), x in zip(items, c):
                rows.append(_row(LEVEL_FACTOR, C.RC_IR, C.MT_CURVATURE, ccy, ten or kind, x))
            rows += [_row(LEVEL_K, C.RC_IR, C.MT_CURVATURE, ccy, "", k), _row(LEVEL_S, C.RC_IR, C.MT_CURVATURE, ccy, "", ss[-1])]
        g2 = np.full((len(ks), len(ks)), p.ir_gamma ** 2)
        agg = _cross_bucket(ks, ss, g2)
        margin = max(float(allv.sum()) + lam * agg, 0.0) * p.ir_curv_scale
        rows.append(_row("theta", C.RC_IR, C.MT_CURVATURE, "", "", theta))
        rows.append(_row("lambda", C.RC_IR, C.MT_CURVATURE, "", "", lam))
        return margin, rows
    pairs = _fx_vol_by_pair(crif, p, True)
    if not pairs:
        return 0.0, rows
    names = list(pairs)
    c = np.array([pairs[q] for q in names])
    theta, lam = _curvature_theta_lambda(c, p)
    r2 = p.fx_vol_corr ** 2
    m = np.full((len(c), len(c)), r2)
    np.fill_diagonal(m, 1.0)
    k = _sqrt0(c @ m @ c)
    margin = max(float(c.sum()) + lam * k, 0.0)
    rows = [_row(LEVEL_FACTOR, C.RC_FX, C.MT_CURVATURE, FX_BUCKET, q, x) for q, x in zip(names, c)]
    rows += [_row(LEVEL_K, C.RC_FX, C.MT_CURVATURE, FX_BUCKET, "", k), _row("theta", C.RC_FX, C.MT_CURVATURE, "", "", theta),
             _row("lambda", C.RC_FX, C.MT_CURVATURE, "", "", lam)]
    return margin, rows


# == aggregation
def risk_class_im(delta: float, vega: float, curvature: float) -> float:
    """IM of one risk class: the sum of its delta, vega and curvature margins."""
    return delta + vega + curvature


def product_class_simm(im_by_class: dict, p: SimmParams, psi_override: float = None) -> float:
    """RatesFX product class SIMM = sqrt(sum IM_r^2 + sum_{r!=s} psi_rs IM_r IM_s) over the risk classes present.
    psi_override replaces the IR-FX entry (used to test psi = 0)."""
    names = [c for c in im_by_class]
    v = np.array([im_by_class[c] for c in names], dtype=float)
    m = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = {names[i], names[j]}
            m[i, j] = m[j, i] = psi_override if (psi_override is not None and pair == {C.RC_IR, C.RC_FX}) else p.psi(names[i], names[j])
    return _sqrt0(v @ m @ v)


def simm(crif: pd.DataFrame, p: SimmParams = None, psi_override: float = None) -> SimmResult:
    """SIMM-style total IM (USD) and breakdown for a CRIF DataFrame (see module docstring for conventions)."""
    p = p or load()
    crif = _clean(crif) if len(crif) else crif
    if len(crif) == 0:
        crif = pd.DataFrame({k: [] for k in (C.RISK_TYPE, C.QUALIFIER, C.LABEL1, C.LABEL2, C.AMOUNT_USD)})
    ws = weighted_sensitivities(crif, p)
    d_ir, r1 = delta_margin_ir(ws, p)
    d_fx, r2 = delta_margin_fx(ws, p)
    v_ir, r3 = vega_margin_ir(crif, p)
    v_fx, r4 = vega_margin_fx(crif, p)
    c_ir, r5 = curvature_margin(crif, p, C.RC_IR)
    c_fx, r6 = curvature_margin(crif, p, C.RC_FX)
    margins = {(C.RC_IR, C.MT_DELTA): d_ir, (C.RC_IR, C.MT_VEGA): v_ir, (C.RC_IR, C.MT_CURVATURE): c_ir,
               (C.RC_FX, C.MT_DELTA): d_fx, (C.RC_FX, C.MT_VEGA): v_fx, (C.RC_FX, C.MT_CURVATURE): c_fx}
    rows = r1 + r2 + r3 + r4 + r5 + r6
    ims = {}
    for rc in (C.RC_IR, C.RC_FX):
        for mt in (C.MT_DELTA, C.MT_VEGA, C.MT_CURVATURE):
            rows.append(_row(LEVEL_MARGIN, rc, mt, "", "", margins[(rc, mt)]))
        ims[rc] = risk_class_im(*(margins[(rc, mt)] for mt in (C.MT_DELTA, C.MT_VEGA, C.MT_CURVATURE)))
        rows.append(_row(LEVEL_RC, rc, "", "", "", ims[rc]))
    total = product_class_simm(ims, p, psi_override) * p.multiplicative_scale
    rows.append(_row(LEVEL_PRODUCT, "", "", "", C.RATES_FX, total))
    rows.append(_row(LEVEL_TOTAL, "", "", "", "", total))
    return SimmResult(total=float(total), breakdown=pd.DataFrame(rows, columns=BREAKDOWN_COLUMNS), by_risk_class=ims,
                      by_margin_type=margins)


def attach_fx_sigma(crif: pd.DataFrame, trades: pd.DataFrame, batch_row) -> pd.DataFrame:
    """Add the SigmaMarket column (market vol of the trade's expiry, fraction) to Risk_FXVol rows so that the engine can
    turn crif.py's vega*sigma amounts back into vega. Needed until crif.build_crif emits the column itself."""
    from .sensitivities import fx_vega              # local import keeps simm.py free of pricing at import time
    by_id = trades.set_index(C.TRADE_ID)
    out = crif.copy()
    sig = {}
    out[SIGMA_MARKET_COL] = np.nan
    for idx, r in out[out[C.RISK_TYPE] == C.RISK_FXVOL].iterrows():
        tid = r[C.CRIF_TRADE_ID]
        if tid not in sig:
            sig[tid] = fx_vega(by_id.loc[tid], batch_row)[1]
        out.at[idx, SIGMA_MARKET_COL] = sig[tid]
    return out
