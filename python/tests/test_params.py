"""params.load: typed accessors, group mapping, matrix checks and version differences."""
import numpy as np
import pytest
from scipy.stats import norm

from simm_margin import columns as C
from simm_margin import config
from simm_margin import params as P

# ===== CONFIG (user inputs) =====
V12, V06 = "2.8+2512", "2.8+2506"
REG_5Y_RW, LOW_5Y_RW, HIGH_5Y_RW = 61.0, 25.0, 88.0
TENOR_5Y = config.IR_TENORS.index("5y")
# ===== END CONFIG =====


def test_unknown_version_raises_and_known_load():
    with pytest.raises(ValueError):
        P.load("2.8+1999")
    assert set(P.available_versions()) == {V12, V06}
    assert P.load().version == config.SIMM_VERSION and P.load(V06).version == V06


def test_matrices_symmetric_psd_both_versions():
    for v in (V12, V06):
        p = P.load(v)
        for m in (p.ir_corr, p.psi_matrix):
            assert np.allclose(m, m.T) and np.linalg.eigvalsh(m).min() > 0
        m = p.fx_corr_matrix(["USD", "EUR", "MXN", "ARS", "ISK", "GBP"])
        assert np.allclose(m, m.T) and np.linalg.eigvalsh(m).min() >= -1e-9
        assert p.ir_corr[0, 1] == 0.74 and p.psi("IR", "FX") == p.psi("FX", "IR")


def test_check_matrix_rejects_bad_matrices():
    with pytest.raises(ValueError):
        P.check_matrix(np.array([[1, 0.5], [0.4, 1]]), "asym")
    with pytest.raises(ValueError):
        P.check_matrix(np.array([[1, 2.0], [2.0, 1]]), "notpsd")
    with pytest.raises(ValueError):
        P.check_matrix(np.array([[0.9, 0.0], [0.0, 1]]), "diag")


def test_currency_groups_for_scope_currencies():
    p = P.load(V12)
    assert [p.ir_rw_group(c) for c in ("USD", "EUR", "GBP", "JPY", "MXN")] == ["regular", "regular", "regular", "low", "high"]
    assert p.ir_rw("USD")[TENOR_5Y] == REG_5Y_RW and p.ir_rw("JPY")[TENOR_5Y] == LOW_5Y_RW and p.ir_rw("MXN")[TENOR_5Y] == HIGH_5Y_RW
    assert [p.ir_ct_group(c) for c in ("USD", "EUR", "GBP", "JPY", "MXN", "CHF")] == [
        "regular_well_traded"] * 3 + ["low", "high", "regular_less_well_traded"]
    assert (p.ir_delta_ct("USD"), p.ir_delta_ct("JPY"), p.ir_delta_ct("MXN")) == (220, 370, 71)
    assert (p.ir_vega_ct("EUR"), p.ir_vega_ct("MXN")) == (3800, 160)
    assert [p.fx_category(c) for c in ("USD", "MXN", "ARS")] == ["Category 1", "Category 2", "Category 3"]
    assert p.fx_group("MXN") == "regular" and p.fx_group("ARS") == "high"


def test_scalars_and_fx_tables_by_version():
    a, b = P.load(V12), P.load(V06)
    assert (a.phi, a.infl_corr, a.xccy_corr, a.ir_gamma, a.ir_hvr, a.ir_vrw) == (0.981, 0.42, -0.01, 0.35, 0.74, 0.20)
    assert (a.ir_rw_inflation, a.ir_rw_xccy) == (51, 21)
    assert (a.fx_rw("EUR"), a.fx_rw("ARS"), a.fx_rw("ARS", "ARS")) == (7.4, 31.7, 31.7)
    assert (b.fx_rw("EUR"), b.fx_rw("ARS"), b.fx_rw("ARS", "ARS")) == (7.1, 18.0, 30.6)
    assert (a.fx_hvr, a.fx_vrw, b.fx_hvr, b.fx_vrw) == (0.67, 0.33, 0.68, 0.34)
    assert (a.fx_corr("EUR", "MXN"), a.fx_corr("EUR", "ARS"), a.fx_corr("ARS", "NGN")) == (0.5, 0.12, 0.03)
    assert (b.fx_corr("EUR", "ARS"), b.fx_corr("ARS", "NGN")) == (0.20, 0.08)
    assert (a.fx_delta_ct("EUR"), b.fx_delta_ct("EUR")) == (2100, 3100)
    assert a.fx_vega_ct("MXN", "EUR") == a.fx_vega_ct("EUR", "MXN") == 1500
    assert (a.psi("IR", "FX"), b.psi("IR", "FX")) == (0.15, 0.10)
    assert a.fx_vol_corr == 0.5 and a.multiplicative_scale == 1.0


def test_derived_formulas_match_definitions():
    p = P.load(V12)
    assert p.fx_sigma("EUR", "USD") == pytest.approx(7.4 * np.sqrt(365 / 14) / norm.ppf(0.99) / 100)
    assert p.ir_curv_scale == pytest.approx(0.74 ** -2)
    assert p.curv_lambda(0.0) == pytest.approx(norm.ppf(0.995) ** 2 - 1) and p.curv_lambda(-1.0) == pytest.approx(1.0)
    days = p.tenor_days()
    assert days[0] == 14 and days[4] == 365 and days[-1] == 365 * 30
    assert p.sf(days)[:3] == pytest.approx([0.5, 0.5 * 14 / (365 / 12), 0.5 * 14 / (365 / 4)])
