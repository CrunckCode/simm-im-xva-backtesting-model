"""Plain matplotlib charts (Agg, 150 dpi) built from the output CSVs; a chart is skipped when its input is missing.

ALL DATA ARE SYNTHETIC. SIMM-style engine, not ISDA-licensed, not certified. Entry point: make_all_charts().
"""
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import columns as C
from . import config

log = logging.getLogger(__name__)

# ===== CONFIG (user inputs) =====
DPI = 150
FIG_WIDE, FIG_MID, FIG_TALL = (10, 4.5), (8, 4.5), (9, 7)
GREEN, YELLOW, RED = "#9ccc9c", "#f2dc7a", "#e8a0a0"
INK, MID, LIGHT, ACCENT, ACCENT2 = "black", "dimgray", "darkgray", "#1f4e79", "#b03a2e"
SPLIT = "split"
CHAMPION_MODEL = "hs_im_eu_1plus3"
MODEL_LABELS = {"hs_im_eu_1plus3": "Champion (EU 1+3 window)", "hs_im_ewma": "EWMA challenger",
                "hs_im_plain250": "Plain 250-day challenger", "hs_im_scaled": "Champion x 0.7 challenger",
                "simm_style": "SIMM-style", "xva_var": "xVA VaR"}
IM_MODEL_ORDER = ("hs_im_eu_1plus3", "hs_im_ewma", "hs_im_plain250", "hs_im_scaled")
SPLIT_1D, SPLIT_NONOVERLAP, SPLIT_OVERLAP = "1d", "10d_nonoverlap", "10d_overlapping"
BCBS_YELLOW_FROM, BCBS_RED_FROM = 5, 10
P_PASS, P_AMBER = 0.05, 0.01
MARGIN_ORDER = ("Delta", "Vega", "Curvature")
FILE_SUMMARY, FILE_SCHEDULE, FILE_SERIES, FILE_RESULTS = ("portfolio_summary.json", "simm_vs_schedule.csv",
                                                          "backtest_series.csv", "backtest_results.csv")
FILE_SCAN, FILE_DISPUTE, FILE_ATTR, FILE_UAT, FILE_CRIF = ("daily_im_scan.csv", "dispute_results.csv",
                                                           "attribution_scenario_day.csv", "uat_results.csv", "crif_last.csv")
FILE_BIG, FILE_CVA_PROFILE, FILE_META = "big_moves.csv", "cva_profile.csv", "run_meta.json"
DETAIL_DIR = "dispute_details"
DISPUTE_DETAIL_SCENARIOS = ("D1", "D8")
WATERFALL_DRIVER_LABELS = {"matured_trades": "Matured trades", "new_trades": "New trades", "market_move": "Market move",
                           "parameter_version": "Parameter version"}
# ===== END CONFIG =====

plt.rcParams.update({"figure.dpi": DPI, "savefig.dpi": DPI, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

DATA_NOTE = "Synthetic data; SIMM-style, not ISDA-licensed or certified"


def _save(fig, path, bottom=0.02):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.995, 0.005, DATA_NOTE, ha="right", va="bottom", fontsize=6.5, color=MID)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _csv(out: Path, name: str, **kw):
    p = out / name
    return pd.read_csv(p, **kw) if p.exists() else None


def _json(out: Path, name: str):
    p = out / name
    return json.loads(p.read_text()) if p.exists() else None


def _usd_m(ax, axis="y"):
    fmt = plt.FuncFormatter(lambda v, _: f"{v / 1e6:,.1f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _series(df, model, split):
    s = df[(df[C.MODEL] == model) & (df[SPLIT] == split)].copy()
    s[C.DATE] = pd.to_datetime(s[C.DATE])
    return s.sort_values(C.DATE).reset_index(drop=True)


def _stress_window():
    """First and last date of the forced stress window of the synthetic history."""
    from .market_history import STRESS_WINDOW_START
    dates = pd.bdate_range(config.HIST_START, config.HIST_END)
    i0 = int(np.clip(dates.searchsorted(STRESS_WINDOW_START), 0, max(len(dates) - config.STRESS_WINDOW_DAYS, 0)))
    i1 = min(i0 + config.STRESS_WINDOW_DAYS, len(dates))
    return dates[i0], dates[i1 - 1]


def _shade_regime(ax, dates, regime, color=LIGHT, alpha=0.25):
    reg = np.asarray(regime, dtype=int)
    start = None
    for i, r in enumerate(reg):
        if r == C.REGIME_STRESS and start is None:
            start = dates[i]
        if r != C.REGIME_STRESS and start is not None:
            ax.axvspan(start, dates[i], color=color, alpha=alpha, lw=0)
            start = None
    if start is not None:
        ax.axvspan(start, dates[-1], color=color, alpha=alpha, lw=0)


# ---------------------------------------------------------------- individual charts
def chart_im_breakdown(out, cdir):
    s = _json(out, FILE_SUMMARY)
    if not s:
        return None
    key = f"simm_by_margin_type_{config.SIMM_VERSION}"
    mt = s.get(key)
    if not mt:
        return None
    vals = {(k.strip("()' ").split("', '")[0], k.strip("()' ").split("', '")[1]): v for k, v in mt.items()}
    fig, ax = plt.subplots(figsize=FIG_MID)
    x = np.arange(2)
    bottom = np.zeros(2)
    greys = (ACCENT, "#6f93b8", "#bcd0e3")
    for m, col in zip(MARGIN_ORDER, greys):
        h = np.array([vals.get((rc, m), 0.0) for rc in (C.RC_IR, C.RC_FX)])
        ax.bar(x, h, bottom=bottom, color=col, edgecolor="white", label=m)
        for xi, hi, bi in zip(x, h, bottom):
            if hi > 0.04 * max(1.0, bottom.max() + hi):
                ax.text(xi, bi + hi / 2, f"{hi / 1e6:.2f}", ha="center", va="center", fontsize=8)
        bottom += h
    ax.set_xticks(x, [f"{rc} class IM" for rc in (C.RC_IR, C.RC_FX)])
    total = s.get(f"simm_total_{config.SIMM_VERSION}")
    ax.set_title(f"SIMM-style IM by risk class and margin type, v{config.SIMM_VERSION}, valuation date "
                 f"{s.get('valuation_date', '')}" + (f"; product total USD {total / 1e6:.2f} mm" if total else ""), fontsize=9)
    ax.set_ylabel("USD million (sum of margins before IR-FX aggregation)")
    _usd_m(ax)
    ax.legend(title="Margin type")
    return _save(fig, cdir / "im_by_risk_class_margin_type.png")


def chart_version_compare(out, cdir):
    s = _json(out, FILE_SUMMARY)
    if not s:
        return None
    k1, k0 = f"simm_by_margin_type_{config.SIMM_VERSION}", f"simm_by_margin_type_{config.SIMM_VERSION_PRIOR}"
    if k1 not in s or k0 not in s:
        return None
    labels = list(s[k1])
    nice = [l.strip("()' ").replace("', '", " ") for l in labels]
    new = [s[k1][l] for l in labels]
    old = [s[k0].get(l, 0.0) for l in labels]
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, old, 0.4, color=LIGHT, label=f"v{config.SIMM_VERSION_PRIOR}")
    ax.bar(x + 0.2, new, 0.4, color=ACCENT, label=f"v{config.SIMM_VERSION}")
    ax.set_xticks(x, nice)
    t1, t0 = s[f"simm_total_{config.SIMM_VERSION}"], s[f"simm_total_{config.SIMM_VERSION_PRIOR}"]
    ax.set_title(f"Margin by type under the two parameter versions (product totals: USD {t0 / 1e6:.2f} mm vs USD {t1 / 1e6:.2f} mm)",
                 fontsize=9)
    ax.set_ylabel("USD million")
    _usd_m(ax)
    ax.legend()
    return _save(fig, cdir / "im_2512_vs_2506.png")


def chart_simm_vs_schedule(out, cdir):
    d = _csv(out, FILE_SCHEDULE)
    if d is None or d.empty:
        return None
    d = d.sort_values("date")
    fig, ax = plt.subplots(figsize=FIG_MID)
    x = np.arange(len(d))
    ax.bar(x - 0.27, d["simm_im"], 0.27, color=ACCENT, label="SIMM-style IM")
    ax.bar(x, d["schedule_net"], 0.27, color=MID, label="Schedule IM, net (NGR applied)")
    ax.bar(x + 0.27, d["schedule_gross"], 0.27, color=LIGHT, label="Schedule IM, gross")
    for xi, r in zip(x, d["simm_over_schedule_net"]):
        ax.text(xi - 0.27, d["simm_im"].iloc[xi] + 3e5, f"{r:.0%} of net", ha="center", fontsize=7.5)
    ax.set_xticks(x, [f"{l}\n{dt}" for l, dt in zip(d["label"], d["date"])], fontsize=8)
    ax.set_ylabel("USD million")
    _usd_m(ax)
    ax.set_title("SIMM-style IM versus the standardised schedule IM on four dates", fontsize=9.5)
    ax.set_ylim(0, float(d["schedule_gross"].max()) * 1.2)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    return _save(fig, cdir / "simm_vs_schedule.png")


def chart_regime_path(out, cdir):
    d = _csv(out, FILE_SCAN, parse_dates=[C.DATE])
    if d is None or d.empty:
        return None
    s0, s1 = _stress_window()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=FIG_WIDE, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    _shade_regime(a1, d[C.DATE].values, d[C.REGIME].values)
    a1.axvspan(s0, s1, facecolor="none", edgecolor=ACCENT2, hatch="///", alpha=0.6, lw=0, label="forced stress window")
    a1.plot(d[C.DATE], d["im"], color=INK, lw=0.9, label="SIMM-style IM (daily)")
    a1.set_ylabel("USD million")
    _usd_m(a1)
    a1.legend(loc="upper right", fontsize=8)
    a1.set_title("Synthetic history: two-state regime path (grey = stress regime) and the SIMM-style IM of the sample book", fontsize=9.5)
    a2.step(d[C.DATE], d[C.REGIME], where="post", color=INK, lw=0.9)
    a2.set_yticks([0, 1], ["calm", "stress"])
    a2.set_ylim(-0.2, 1.2)
    return _save(fig, cdir / "regime_path.png")


def chart_var_vs_loss(out, cdir):
    s = _csv(out, FILE_SERIES)
    if s is None:
        return None
    c = _series(s, CHAMPION_MODEL, SPLIT_1D)
    if c.empty:
        return None
    exc = c[c[C.EXCEPTION] == 1]
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(c[C.DATE], c[C.LOSS], color=LIGHT, lw=0.5, label="1-day hypothetical loss")
    ax.plot(c[C.DATE], c[C.VAR_FORECAST], color=ACCENT, lw=1.1, label="99% 1-day HS IM (champion)")
    ax.scatter(exc[C.DATE], exc[C.LOSS], color=ACCENT2, s=18, zorder=3, label=f"exceptions ({len(exc)} of {len(c)})")
    ax.set_ylabel("USD million")
    _usd_m(ax)
    ax.set_title("Champion HS IM at 99% versus the realised 1-day hypothetical loss", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, cdir / "var_vs_loss_1d.png")


def chart_rolling(out, cdir):
    s = _csv(out, FILE_SERIES)
    if s is None:
        return None
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ymax = 14
    for model in IM_MODEL_ORDER:
        c = _series(s, model, SPLIT_1D)
        if c.empty:
            continue
        r = c[C.EXCEPTION].rolling(config.BACKTEST_WINDOW).sum()
        ymax = max(ymax, float(np.nanmax(r.values)) + 2) if r.notna().any() else ymax
        ax.plot(c[C.DATE], r, lw=2.0 if model == CHAMPION_MODEL else 0.9, color=INK if model == CHAMPION_MODEL else None,
                label=MODEL_LABELS[model])
    ax.axhspan(0, BCBS_YELLOW_FROM - 0.5, color=GREEN, alpha=0.45, lw=0)
    ax.axhspan(BCBS_YELLOW_FROM - 0.5, BCBS_RED_FROM - 0.5, color=YELLOW, alpha=0.45, lw=0)
    ax.axhspan(BCBS_RED_FROM - 0.5, ymax, color=RED, alpha=0.45, lw=0)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(f"exceptions in the last {config.BACKTEST_WINDOW} days")
    ax.set_title("Rolling 250-day exception count against the BCBS 22 zones (green 0-4, yellow 5-9, red 10+)", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, cdir / "rolling_250_exceptions.png")


def chart_pvalues(out, cdir):
    r = _csv(out, FILE_RESULTS)
    if r is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE, sharey=True)
    tests = (("kupiec_pof", "Kupiec POF", "#6f93b8"), ("christoffersen_ind", "Christoffersen independence", LIGHT),
             ("christoffersen_cc", "Christoffersen conditional coverage", INK))
    for ax, split, title in zip(axes, (SPLIT_1D, SPLIT_NONOVERLAP), ("1-day horizon", "10-day horizon, non-overlapping windows")):
        models = [m for m in IM_MODEL_ORDER if ((r[C.MODEL] == m) & (r[SPLIT] == split)).any()]
        x = np.arange(len(models))
        for k, (t, lab, col) in enumerate(tests):
            vals = []
            for m in models:
                sel = r[(r[C.TEST] == t) & (r[C.MODEL] == m) & (r[SPLIT] == split)]
                vals.append(max(float(sel["value"].iloc[0]), 1e-30) if len(sel) else np.nan)
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=col, label=lab)
        ax.set_yscale("log")
        ax.axhline(P_PASS, color=ACCENT2, ls="--", lw=1)
        ax.axhline(P_AMBER, color=ACCENT2, ls=":", lw=1)
        ax.set_xticks(x, [MODEL_LABELS[m].replace(" challenger", "").replace(" (EU 1+3 window)", "") for m in models], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(1e-22, 3)
    axes[0].set_ylabel("p-value (log scale; floor 1e-30)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab + ["dashed line p = 0.05, dotted line p = 0.01"], loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Backtest p-values by model: low values mean rejection", fontsize=10)
    return _save(fig, cdir / "pvalues_by_model.png", bottom=0.1)


def chart_exceptions_by_model(out, cdir):
    r = _csv(out, FILE_RESULTS)
    if r is None:
        return None
    rows = r[r[C.TEST] == "overall"]
    if rows.empty:
        return None
    splits = (SPLIT_1D, SPLIT_NONOVERLAP, SPLIT_OVERLAP)
    models = [m for m in (*IM_MODEL_ORDER, "simm_style") if (rows[C.MODEL] == m).any()]
    fig, axes = plt.subplots(1, 3, figsize=FIG_WIDE)
    for ax, split in zip(axes, splits):
        sub = rows[rows[SPLIT] == split]
        ms = [m for m in models if (sub[C.MODEL] == m).any()]
        ex = [int(sub[sub[C.MODEL] == m]["exceptions"].iloc[0]) for m in ms]
        n = int(sub["n"].iloc[0])
        cols = [{"PASS": GREEN, "AMBER": YELLOW, "RED": RED}[sub[sub[C.MODEL] == m][C.LIGHT].iloc[0]] for m in ms]
        ax.bar(range(len(ms)), ex, color=cols, edgecolor=INK, lw=0.6)
        ax.axhline(n * (1 - config.CONF), color=INK, ls="--", lw=1, label=f"expected {n * (1 - config.CONF):.1f}")
        ax.set_xticks(range(len(ms)), [MODEL_LABELS[m].split(" (")[0].replace(" challenger", "").replace("Champion x 0.7", "Champ x0.7")
                                       for m in ms], rotation=35, ha="right", fontsize=7.5)
        for i, v in enumerate(ex):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{split} (n = {n})", fontsize=9)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("exceptions (bar colour = overall light)")
    fig.suptitle("Champion versus challengers: exception counts by test design", fontsize=10)
    return _save(fig, cdir / "champion_vs_challengers.png")


def chart_simm_vs_loss(out, cdir):
    s = _csv(out, FILE_SERIES)
    if s is None:
        return None
    a = _series(s, "simm_style", SPLIT_NONOVERLAP)
    b = _series(s, CHAMPION_MODEL, SPLIT_OVERLAP)
    full = _series(s, "simm_style", SPLIT_OVERLAP)
    if full.empty:
        return None
    exc = full[full[C.EXCEPTION] == 1]
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(full[C.DATE], full[C.LOSS], color=LIGHT, lw=0.5, label="10-day hypothetical loss")
    ax.plot(full[C.DATE], full[C.VAR_FORECAST], color=ACCENT, lw=1.2, label="SIMM-style IM (10-day, 99%)")
    if not b.empty:
        ax.plot(b[C.DATE], b[C.VAR_FORECAST], color=MID, lw=0.9, ls="--", label="Champion HS IM (10-day)")
    ax.scatter(exc[C.DATE], exc[C.LOSS], color=ACCENT2, s=14, zorder=3, label=f"SIMM exceptions, overlapping windows ({len(exc)})")
    if not a.empty:
        e0 = a[a[C.EXCEPTION] == 1]
        ax.scatter(e0[C.DATE], e0[C.LOSS], facecolors="none", edgecolors=INK, s=70, zorder=4, label=f"non-overlapping design ({len(e0)})")
    ax.set_ylabel("USD million")
    _usd_m(ax)
    ax.set_title("SIMM-style IM versus the 10-day hypothetical loss of the frozen book", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, cdir / "simm_vs_10d_loss.png")


def chart_dispute(out, cdir):
    d = _csv(out, FILE_DISPUTE)
    if d is None or d.empty:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_WIDE, gridspec_kw={"width_ratios": [3, 2]})
    cols = [ACCENT2 if g < 0 else ACCENT for g in d["gap"]]
    a1.bar(d["scenario"], d["gap"], color=cols)
    a1.axhline(0, color=INK, lw=0.7)
    a1.set_ylabel("IM gap, party B minus party A (USD thousand)")
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}"))
    a1.set_title("Seeded dispute scenarios: size of the IM gap", fontsize=9)
    a2.bar(d["scenario"], d["rank_of_seeded_cause"], color=[GREEN if a else RED for a in d["accepted"]], edgecolor=INK, lw=0.6)
    a2.axhline(1, color=INK, ls="--", lw=0.8)
    a2.set_ylim(0, max(4, d["rank_of_seeded_cause"].max() + 1))
    a2.set_yticks(range(0, int(max(4, d["rank_of_seeded_cause"].max() + 1)) + 1))
    a2.set_ylabel("rank of the seeded cause (1 = found first)")
    a2.set_title("Rank of the seeded cause (D9: worst of the two)", fontsize=9)
    return _save(fig, cdir / "dispute_rank_and_gap.png")


def chart_dispute_tree(out, cdir):
    fig, axes = plt.subplots(1, len(DISPUTE_DETAIL_SCENARIOS), figsize=FIG_WIDE)
    axes = np.atleast_1d(axes)
    done = 0
    for ax, sc in zip(axes, DISPUTE_DETAIL_SCENARIOS):
        t = _csv(out / DETAIL_DIR, f"{sc}_gap_tree.csv")
        if t is None:
            ax.set_visible(False)
            continue
        m = t[t[C.LEVEL] == C.MARGIN_TYPE].copy()
        m["lab"] = m[C.RISK_CLASS] + " " + m[C.MARGIN_TYPE]
        m = m.sort_values("gap")
        ax.barh(m["lab"], m["gap"], color=[ACCENT2 if g < 0 else ACCENT for g in m["gap"]])
        ax.axvline(0, color=INK, lw=0.7)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}"))
        ax.set_xlabel("margin gap B minus A (USD thousand)")
        ax.set_title(f"{sc}: gap by risk class and margin type", fontsize=9)
        done += 1
    if not done:
        plt.close(fig)
        return None
    return _save(fig, cdir / "dispute_gap_decomposition.png")


def chart_waterfall(out, cdir):
    a = _csv(out, FILE_ATTR)
    if a is None:
        return None
    dr = a[a["level"] == "driver"]
    if dr.empty:
        return None
    im0, im1 = float(dr["im0"].iloc[0]), float(dr["im1"].iloc[0])
    day = str(dr[C.DATE].iloc[0])
    labels = ["IM before"] + [WATERFALL_DRIVER_LABELS.get(d, d) for d in dr["driver"]] + ["IM after"]
    steps = list(dr["shapley"])
    fig, ax = plt.subplots(figsize=FIG_MID)
    ax.bar(0, im0, color=MID)
    run = im0
    for i, v in enumerate(steps, start=1):
        ax.bar(i, v, bottom=run, color=ACCENT if v >= 0 else ACCENT2)
        ax.text(i, max(run, run + v) + 1.5e5, f"{v / 1e6:+.2f}", ha="center", fontsize=8)
        run += v
    ax.bar(len(steps) + 1, im1, color=MID)
    ax.text(0, im0 + 1.5e5, f"{im0 / 1e6:.2f}", ha="center", fontsize=8)
    ax.text(len(steps) + 1, im1 + 1.5e5, f"{im1 / 1e6:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("USD million")
    _usd_m(ax)
    ax.set_title(f"Shapley attribution of the change in IM, COB {day} (exact: drivers sum to the change)", fontsize=9.5)
    return _save(fig, cdir / "attribution_waterfall.png")


def chart_big_moves(out, cdir):
    d = _csv(out, FILE_SCAN, parse_dates=[C.DATE])
    if d is None or "dim_rel" not in d.columns:
        return None
    d = d.dropna(subset=["dim_rel"])
    fl = d[d["flagged"].astype(bool)]
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(d[C.DATE], d["dim_rel"] * 100, color=LIGHT, lw=0.6)
    ax.scatter(fl[C.DATE], fl["dim_rel"] * 100, color=ACCENT2, s=20, zorder=3, label=f"flagged days ({len(fl)})")
    for v in (config.BIG_MOVE_REL * 100, -config.BIG_MOVE_REL * 100):
        ax.axhline(v, color=ACCENT2, ls="--", lw=0.8)
    ax.set_ylabel("daily change in SIMM-style IM (percent)")
    ax.set_title("Daily IM changes of the frozen book; flagged when |change| exceeds 10% or USD 1 mm", fontsize=9.5)
    ax.legend(fontsize=8)
    return _save(fig, cdir / "big_move_days.png")


def chart_uat(out, cdir):
    u = _csv(out, FILE_UAT)
    if u is None or u.empty:
        return None
    ct = u.groupby(["area", "result"]).size().unstack(fill_value=0)
    order = [c for c in ("PASS", "FAIL", "SKIPPED") if c in ct.columns]
    colors = {"PASS": GREEN, "FAIL": RED, "SKIPPED": YELLOW}
    fig, ax = plt.subplots(figsize=FIG_MID)
    left = np.zeros(len(ct))
    for c in order:
        ax.barh(ct.index, ct[c], left=left, color=colors[c], edgecolor=INK, lw=0.6, label=f"{c} ({int(ct[c].sum())})")
        left += ct[c].values
    ax.set_xlabel("number of UAT tests")
    ax.set_title(f"New-product UAT (EUR/USD cross-currency swap): {len(u)} tests by area and result", fontsize=9.5)
    ax.legend(fontsize=8)
    return _save(fig, cdir / "uat_pass_counts.png")


def chart_xva(out, cdir):
    s = _csv(out, FILE_SERIES)
    if s is None:
        return None
    c = _series(s, "xva_var", SPLIT_1D)
    if c.empty:
        return None
    exc = c[c[C.EXCEPTION] == 1]
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(c[C.DATE], c[C.LOSS], color=LIGHT, lw=0.5, label="realised 1-day change in CVA (loss positive)")
    ax.plot(c[C.DATE], c[C.VAR_FORECAST], color=ACCENT, lw=1.1, label="99% sensitivity-based xVA VaR")
    ax.scatter(exc[C.DATE], exc[C.LOSS], color=ACCENT2, s=18, zorder=3, label=f"exceptions ({len(exc)} of {len(c)})")
    ax.set_ylabel("USD thousand")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}"))
    ax.set_title("xVA VaR backtest: sensitivity-based 1-day VaR against the full-revaluation change in CVA", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, cdir / "xva_var_backtest.png")


def chart_cva_profile(out, cdir):
    p = _csv(out, FILE_CVA_PROFILE)
    if p is None or p.empty:
        return None
    fig, ax = plt.subplots(figsize=FIG_MID)
    ax.plot(p["t"], p["ee"], color=ACCENT, lw=1.5, label="expected exposure EE(t), normal approximation")
    ax.plot(p["t"], p["mu"].clip(lower=0), color=MID, lw=1.0, ls="--", label="forward value of the netting set, floored at zero")
    ax.set_xlabel("years from valuation date")
    ax.set_ylabel("USD million")
    _usd_m(ax)
    ax.set_title("Netting-set exposure profile of the linear trades used for CVA (frozen portfolio)", fontsize=9.5)
    ax.legend(fontsize=8)
    return _save(fig, cdir / "cva_exposure_profile.png")


def chart_delta_ladder(out, cdir):
    d = _csv(out, FILE_CRIF)
    if d is None:
        return None
    ir = d[d[C.RISK_TYPE] == C.RISK_IRCURVE]
    if ir.empty:
        return None
    piv = ir.pivot_table(index=C.LABEL1, columns=C.QUALIFIER, values=C.AMOUNT_USD, aggfunc="sum").reindex(list(config.IR_TENORS))
    piv = piv.dropna(how="all").fillna(0.0)
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    w = 0.8 / len(piv.columns)
    for k, ccy in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + k * w, piv[ccy], w, label=ccy)
    ax.set_xticks(np.arange(len(piv)) + 0.4 - w / 2, piv.index)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_ylabel("USD per 1bp of par quote")
    ax.set_title("Net IR delta ladder of the sample book by currency (CRIF, valuation date)", fontsize=9.5)
    ax.legend(ncol=5, fontsize=8)
    return _save(fig, cdir / "ir_delta_ladder.png")


# ---------------------------------------------------------------- exposure profile (needs the model, not only CSVs)
def write_cva_profile(out: Path) -> Path:
    """Compute the EE profile at the last date of the history and store it as cva_profile.csv (lazy heavy imports)."""
    from . import cva, instruments, market_history
    meta = _json(out, FILE_META) or {}
    hist = market_history.generate_history(quick=bool(meta.get("quick", False)))
    trades = instruments.sample_portfolio()
    ti = len(hist) - 1
    res = cva.cva_state(trades, hist.market.row(ti), cva.rolling_factor_cov(hist)[ti])
    prof = res["profile"]
    df = pd.DataFrame({"t": prof["grid"], "ee": prof["ee"][0], "mu": prof["mu"][0], "s": prof["s"][0], "df": prof["df"][0]})
    df["data_source"] = config.DATA_SOURCE
    path = out / FILE_CVA_PROFILE
    df.to_csv(path, index=False)
    return path


CHARTS = (chart_im_breakdown, chart_version_compare, chart_simm_vs_schedule, chart_regime_path, chart_var_vs_loss,
          chart_rolling, chart_pvalues, chart_exceptions_by_model, chart_simm_vs_loss, chart_dispute, chart_dispute_tree,
          chart_waterfall, chart_big_moves, chart_uat, chart_xva, chart_cva_profile, chart_delta_ladder)


def make_all_charts(out_dir=None, chart_dir=None, build_profile: bool = True) -> dict:
    """Write every chart whose input exists; returns {chart function name: path}. Failures are logged and skipped."""
    out = Path(out_dir) if out_dir is not None else config.OUT_DIR
    cdir = Path(chart_dir) if chart_dir is not None else out / "charts"
    cdir.mkdir(parents=True, exist_ok=True)
    if build_profile and not (out / FILE_CVA_PROFILE).exists() and (out / FILE_SERIES).exists():
        try:
            write_cva_profile(out)
        except Exception as exc:  # the profile needs the model; the other charts do not
            log.warning("cva profile not written: %s", exc)
    written = {}
    for fn in CHARTS:
        try:
            path = fn(out, cdir)
        except Exception as exc:
            log.warning("chart %s skipped: %s", fn.__name__, exc)
            plt.close("all")
            continue
        if path is None:
            log.info("chart %s skipped (input missing)", fn.__name__)
        else:
            written[fn.__name__] = path
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for k, v in make_all_charts().items():
        print(k, v)
