"""Dispute workflow: seeded views, reconciliation, cause ranking. SYNTHETIC data; SIMM-style, not ISDA-certified."""
import pandas as pd
import pytest

from simm_margin import columns as C
from simm_margin import config
from simm_margin import dispute as D
from simm_margin import params as P
from simm_margin.instruments import sample_portfolio
from simm_margin.market_history import generate_history

# ===== CONFIG (user inputs) =====
SINGLE_SCENARIOS = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def setup():
    h = generate_history(quick=True)
    ctx = D.make_context(h)
    trades = sample_portfolio()
    return h, ctx, trades, ctx.fresh_crif(trades)


@pytest.mark.parametrize("scenario", SINGLE_SCENARIOS)
def test_seeded_cause_ranks_first(setup, scenario):
    _, ctx, trades, crif = setup
    rep, rk = D.run_scenario(scenario, trades, crif, ctx, with_euler=False)
    assert abs(rep.gap) > config.DISPUTE_TOL_ABS                       # the seeded difference is visible in IM
    assert rk.iloc[0][D.CAUSE_COL] == D.SEEDED[scenario][0]
    assert rk.iloc[0][D.RESIDUAL] <= config.DISPUTE_TOL_ABS            # its fix closes the gap
    assert list(rk.columns[:4]) == [D.CAUSE_COL, D.SIGNATURE, D.EXPLAINED, D.RESIDUAL]


def test_composite_both_causes_in_top_three(setup):
    _, ctx, trades, crif = setup
    _, rk = D.run_scenario("D9", trades, crif, ctx, with_euler=False)
    ranks = rk.set_index(D.CAUSE_COL)[D.RANK]
    assert all(ranks[c] <= D.COMPOSITE_TOP_N for c in D.SEEDED["D9"])
    assert ranks[D.CAUSE_MISSING] == 1


def test_recon_structure_for_missing_trade(setup):
    _, ctx, trades, crif = setup
    rep, _ = D.run_scenario("D1", trades, crif, ctx)
    pop = rep.population.set_index(D.TRADE_COL)
    assert pop.loc[D.D1_MISSING_TRADE, D.STATUS_COL] == "only_a"
    assert (rep.population[D.STATUS_COL] != "both").sum() == 1
    assert set(rep.crif_diff.loc[rep.crif_diff[D.STATUS_COL] == "only_a", C.CRIF_TRADE_ID]) == {D.D1_MISSING_TRADE}
    assert rep.gap_path[0][0] == "risk_class" and rep.gap_path[0][2] != 0
    row = rep.contributions.set_index(D.TRADE_COL).loc[D.D1_MISSING_TRADE]
    assert row["euler_b"] == 0 and row["euler_a"] != 0
    assert rep.evidence_units > 0


def test_param_version_view_has_identical_crif(setup):
    _, ctx, trades, crif = setup
    _, cb, pb = D.make_counterparty_view(trades, ctx.row, crif, "D8", history=ctx.history)
    assert pb.version == config.SIMM_VERSION_PRIOR
    pd.testing.assert_frame_equal(cb.reset_index(drop=True), crif.reset_index(drop=True))


def test_fx_vega_view_changes_only_sigma_column(setup):
    _, ctx, trades, crif = setup
    _, cb, _ = D.make_counterparty_view(trades, ctx.row, crif, "D4", history=ctx.history)
    assert (cb[C.AMOUNT_USD] == crif[C.AMOUNT_USD]).all()
    assert not (cb["SigmaMarket"].dropna() == crif["SigmaMarket"].dropna()).all()


def test_crif_key_match_tolerance(setup):
    _, _, _, crif = setup
    same = D.crif_key_match(crif, crif)
    assert (same[D.STATUS_COL] == "matched").all()
    nudged = crif.copy()
    nudged.loc[nudged.index[0], C.AMOUNT_USD] += 0.5 * config.DISPUTE_TOL_ABS
    nudged.loc[nudged.index[1], C.AMOUNT_USD] += 10 * config.DISPUTE_TOL_ABS + 1e-3 * abs(nudged.loc[nudged.index[1], C.AMOUNT_USD])
    st = D.crif_key_match(crif, nudged)[D.STATUS_COL].value_counts().to_dict()
    assert st.get("amount_differs") == 1 and st.get("matched") == len(crif) - 1


def test_identical_views_have_no_gap_and_fixes_are_noops(setup):
    _, ctx, trades, crif = setup
    a = D.PartyView("A", trades, crif, P.load())
    rep = D.reconcile(a, D.PartyView("B", trades.copy(), crif.copy(), P.load()), ctx, with_euler=False)
    assert rep.gap == 0 and rep.evidence_units == 0
    rk = D.rank_causes(rep)
    assert (rk[D.RESIDUAL] <= 1e-6).all() and (rk[D.EXPLAINED].abs() <= 1e-6).all()


def test_errors(setup):
    h, ctx, trades, crif = setup
    with pytest.raises(ValueError):
        D.make_counterparty_view(trades, ctx.row, crif, "D42", history=h)
    with pytest.raises(ValueError):
        D.make_counterparty_view(trades, ctx.row, crif, "D5")           # needs the history
    rep = D.reconcile(D.PartyView("A", trades, crif, P.load()), D.PartyView("B", trades, crif, P.load()), None, with_euler=False)
    with pytest.raises(ValueError):
        D.rank_causes(rep)


def test_run_disputes_writes_files_to_given_dir_only(setup, tmp_path):
    h, _, trades, _ = setup
    before = set(config.OUT_DIR.glob("dispute*")) if config.OUT_DIR.exists() else set()
    res = D.run_disputes(h, trades, tmp_path, scenarios=("D1", "D9"))
    assert list(res.columns) == D.RESULT_COLUMNS and res["accepted"].all()
    assert (tmp_path / D.RESULTS_FILE).exists()
    for s in ("D1", "D9"):
        for part in ("ranking", "gap_tree", "trade_contributions"):
            assert (tmp_path / D.DETAILS_DIR / f"{s}_{part}.csv").exists()
    after = set(config.OUT_DIR.glob("dispute*")) if config.OUT_DIR.exists() else set()
    assert before == after
