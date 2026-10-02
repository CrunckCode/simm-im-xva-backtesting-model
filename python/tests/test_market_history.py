"""History generator tests (synthetic data): shape, reproducibility, regimes, quoting."""
import numpy as np
import pandas as pd
import pytest

from simm_margin import config
from simm_margin.curves import CURRENCIES
from simm_margin.market_history import HistoryConfig, generate_history

# ===== CONFIG (user inputs) =====
STRESS_SHARE_RANGE = (0.20, 0.50)
MIN_PERSISTENCE = 0.90
# ===== END CONFIG =====


@pytest.fixture(scope="module")
def hist():
    return generate_history()


def test_shapes_and_calendar(hist):
    n = len(hist)
    assert 2080 <= n <= 2095 and hist.market.n_states == n
    assert (hist.dates.dayofweek < 5).all() and hist.dates.is_monotonic_increasing
    for c in CURRENCIES:
        assert hist.par[c].shape == (n, 12) and hist.zero[c].shape == (n, 12) and hist.fx_spot[c].shape == (n,)
        assert hist.ir_nvol[c].shape == (n, 4, 2)
    for p in ("EURUSD", "GBPUSD", "USDJPY", "USDMXN"):
        assert hist.fx_vol[p].shape == (n, 4)
    assert hist.cds.shape == (n, 5)
    assert all(np.isfinite(hist.zero[c]).all() for c in CURRENCIES)


def test_seed_reproducibility():
    a, b = generate_history(), generate_history()
    assert np.array_equal(a.par["USD"], b.par["USD"]) and np.array_equal(a.regime, b.regime)
    c = generate_history(HistoryConfig(seed=7))
    assert not np.array_equal(a.par["USD"], c.par["USD"])


def test_regime_fractions_and_persistence(hist):
    r = hist.regime
    assert STRESS_SHARE_RANGE[0] < r.mean() < STRESS_SHARE_RANGE[1]
    stay = ((r[1:] == 1) & (r[:-1] == 1)).sum() / (r[:-1] == 1).sum()
    assert stay > MIN_PERSISTENCE


def test_forced_stress_window(hist):
    i0, i1 = hist.stress_slice
    assert i1 - i0 == config.STRESS_WINDOW_DAYS and (hist.regime[i0:i1] == 1).all()
    assert hist.stress_batch.n_states == config.STRESS_WINDOW_DAYS


def test_stress_has_higher_vol_and_spreads(hist):
    s = hist.regime == 1
    d = np.diff(hist.par["USD"][:, 7])
    assert d[s[1:]].std() > 1.8 * d[~s[1:]].std()
    assert hist.cds[s, 2].mean() > 1.5 * hist.cds[~s, 2].mean()
    assert hist.ir_nvol["USD"][s].mean() > hist.ir_nvol["USD"][~s].mean()
    assert hist.fx_vol["EURUSD"][s].mean() > hist.fx_vol["EURUSD"][~s].mean()


def test_fx_quoting_and_cross_currency_correlation(hist):
    b = hist.market
    assert 80 < b.pair_spot("USDJPY").mean() < 250 and 0.7 < b.pair_spot("EURUSD").mean() < 1.7
    assert np.allclose(b.pair_spot("USDJPY") * b.fx_spot["JPY"], 1.0)
    assert (hist.fx_spot["USD"] == 1).all()
    lvl = {c: np.diff(hist.par[c][:, 7]) for c in CURRENCIES}
    assert np.corrcoef(lvl["USD"], lvl["EUR"])[0, 1] > 0.3
    assert np.corrcoef(lvl["USD"], lvl["EUR"])[0, 1] > np.corrcoef(lvl["USD"], lvl["JPY"])[0, 1]


def test_batch_by_dates_and_quick_mode(hist):
    d = hist.dates[[10, 500, 1000]]
    b = hist.batch(d)
    assert b.n_states == 3 and np.array_equal(b.par["EUR"][1], hist.par["EUR"][500])
    with pytest.raises(KeyError):
        hist.batch([pd.Timestamp("2020-01-04")])        # a Saturday
    q = generate_history(quick=True)
    assert q.dates[0] >= config.HIST_QUICK_START and len(q) < 600
    assert np.array_equal(q.par["USD"], hist.par["USD"][-len(q):])
    assert q.stress_batch.n_states == config.STRESS_WINDOW_DAYS


def test_regime_path_saved_to_tmp_dir(tmp_path):
    h = generate_history(HistoryConfig(save_dir=tmp_path))
    out = pd.read_csv(tmp_path / "regime_path.csv")
    assert len(out) == len(h) and "data_source" in out.columns
