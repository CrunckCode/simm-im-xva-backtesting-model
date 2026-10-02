"""Curve bootstrap, Jacobian and Nelson-Siegel level tests (synthetic curves)."""
import numpy as np
import pytest

from simm_margin.curves import (CURRENCIES, TENOR_YEARS, base_par, bootstrap_curve, par_rates_of,
                                par_to_zero_jacobian, interp_weights)

# ===== CONFIG (user inputs) =====
PAR_TOL = 1e-12
JAC_TOL = 1e-5      # bump-size nonlinearity of the 30y zero at 1bp vs 0.2bp
# ===== END CONFIG =====


@pytest.mark.parametrize("ccy", CURRENCIES)
def test_bootstrap_reprices_par_quotes(ccy):
    par = base_par(ccy)
    curve = bootstrap_curve(par)
    assert np.abs(par_rates_of(curve)[0] - par).max() < PAR_TOL


def test_swap_pv_zero_at_par_for_every_swap_vertex():
    par = base_par("USD")
    curve = bootstrap_curve(par)
    for k in range(5, 12):
        T = TENOR_YEARS[k]
        pv = 1.0 - curve.df(T)[0, 0] - par[k] * curve.annuity(0.0, T)[0]
        assert abs(pv) < 1e-12


def test_vectorised_bootstrap_matches_row_by_row():
    rng = np.random.default_rng(1)
    par = base_par("EUR")[None] + 1e-4 * rng.normal(0, 30, (7, 12))
    z = bootstrap_curve(par).zero
    for i in range(7):
        assert np.allclose(z[i], bootstrap_curve(par[i]).zero[0], atol=1e-14)


def test_interpolation_linear_flat_extrapolation_and_df_at_zero():
    c = bootstrap_curve(base_par("USD"))
    assert c.df(0.0)[0, 0] == 1.0
    assert np.isclose(c.zero_rate(0.001)[0, 0], c.zero[0, 0])
    assert np.isclose(c.zero_rate(45.0)[0, 0], c.zero[0, -1])
    mid = 0.5 * (TENOR_YEARS[7] + TENOR_YEARS[8])
    assert np.isclose(c.zero_rate(mid)[0, 0], 0.5 * (c.zero[0, 7] + c.zero[0, 8]))
    i, w = interp_weights(7.0)
    assert i[0] == 7 and np.isclose(w[0], 0.4)


def test_jacobian_matches_independent_bump_and_is_lower_triangular():
    par = base_par("GBP")
    J = par_to_zero_jacobian(par)
    J2 = par_to_zero_jacobian(par, bump=2e-5)           # different bump width
    assert np.abs(J - J2).max() < JAC_TOL
    assert np.abs(np.triu(J, 1)).max() < 1e-10            # zero_i depends on par_j only for j <= i
    # direct forward bump of 5y par
    p2 = par.copy(); p2[7] += 1e-6
    dz = (bootstrap_curve(p2).zero[0] - bootstrap_curve(par).zero[0]) / 1e-6
    assert np.abs(dz - J[:, 7]).max() < 1e-4


def test_nelson_siegel_levels_differ_by_currency():
    p5 = {c: base_par(c)[7] for c in CURRENCIES}
    assert p5["MXN"] > p5["USD"] > p5["EUR"] > p5["JPY"]
    assert p5["GBP"] > p5["EUR"]
    assert all(0 < base_par(c)[0] < base_par(c)[-1] for c in CURRENCIES)
