"""End-to-end pipeline: py -3 -m simm_margin.cli [--quick] [--skip-excel] [--skip-charts]. SYNTHETIC data."""
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from . import (attribution, backtest, config, crif as crif_mod, dispute, findings, instruments, market_history, params,
               pricing, schedule_im, simm, uat, var_model)
from . import columns as C

# ===== CONFIG (user inputs) =====
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
# ===== END CONFIG =====


def portfolio_snapshot(history, trades, out):
    """Valuation-date PV, CRIF, SIMM (both versions) and schedule IM for the sample portfolio."""
    row = history.market.row(len(history) - 1)
    pv = pricing.price_portfolio(trades, row)[0]
    trades_pv = trades.assign(pv=pv)
    trades_pv.to_csv(out / "portfolio_pv.csv", index=False)
    cr = crif_mod.build_crif(trades, row)   # emits SigmaMarket on FX vega rows
    cr.to_csv(out / "crif_last.csv", index=False)
    summary = {"data_source": config.DATA_SOURCE, "valuation_date": str(history.dates[-1].date()),
               "portfolio_pv": float(pv.sum())}
    for ver in (config.SIMM_VERSION, config.SIMM_VERSION_PRIOR):
        res = simm.simm(cr, params.load(ver))
        res.breakdown.to_csv(out / f"simm_breakdown_{ver}.csv", index=False)
        summary[f"simm_total_{ver}"] = float(res.total)
        summary[f"simm_by_risk_class_{ver}"] = {k: float(v) for k, v in res.by_risk_class.items()}
        summary[f"simm_by_margin_type_{ver}"] = {str(k): float(v) for k, v in res.by_margin_type.items()}
    sched = schedule_im.schedule_im(trades, pv)
    summary["schedule_im"] = {k: (float(v) if np.isscalar(v) else None) for k, v in sched.items() if k != "by_row"}
    summary["simm_over_schedule_net"] = summary[f"simm_total_{config.SIMM_VERSION}"] / sched["net"]
    pd.DataFrame(sched["by_row"]).to_csv(out / "schedule_im_by_trade.csv", index=False)
    (out / "portfolio_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    t0 = time.time()
    step = lambda msg: logging.info("[%5.0fs] %s", time.time() - t0, msg)
    out = config.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    step("generating synthetic history and portfolio")
    history = market_history.generate_history(quick=quick)
    trades = instruments.sample_portfolio()
    (out / "run_meta.json").write_text(json.dumps(dict(
        data_source=config.DATA_SOURCE, quick=quick, n_days=len(history), simm_version=config.SIMM_VERSION), indent=1))

    step("portfolio snapshot: PV, CRIF, SIMM, schedule IM")
    portfolio_snapshot(history, trades, out)

    step("daily SIMM series")
    first = var_model.first_valid_index(history)
    dates = history.dates[first:-config.HORIZON_DAYS]
    simm_series = var_model.daily_simm_series(history, trades, dates=dates)
    simm_series.to_csv(out / "simm_daily_series.csv", header=["simm_im"])

    step("backtesting battery (VaR IM, SIMM, xVA VaR)")
    backtest.run_battery(history, trades, out_dir=out, simm_series=simm_series, quick=quick)

    step("dispute simulator")
    dispute.run_disputes(history, trades, out_dir=out)
    step("IM attribution")
    attribution.run_scenario_day(history, trades, out_dir=out)
    attribution.run_scan(history, trades, out_dir=out)
    step("UAT for the new cross-currency swap")
    uat.run_uat(out_dir=out, history=history)
    step("findings log")
    findings.build_log(out_dir=out, history=history, trades=trades)

    if "--skip-charts" not in argv:
        step("charts")
        from . import charts
        charts.make_all_charts()
    if "--skip-excel" not in argv:
        step("excel workbook and reconciliation")
        from . import reconcile_excel
        reconcile_excel.build_and_reconcile(quick=quick)
        step("re-running UAT now that the workbook exists")
        uat.run_uat(out_dir=out, history=history)
    step("done")


if __name__ == "__main__":
    main()
