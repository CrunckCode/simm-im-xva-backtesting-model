"""Parameter files: provenance completeness, headline values (both SIMM versions), matrices, regulatory tables."""
import csv
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from simm_margin import columns as C
from simm_margin import parameter_io as pio
from simm_margin.config import PARAM_DIR, ROOT, IR_TENORS

# ===== CONFIG (user inputs) =====
V12, V06 = "2.8+2512", "2.8+2506"
SHA_BY_SET = {
    V12: "8484d3de7dcb5280b6ad6458d051c50f8ccc3370bc279ae1c47a7bccb8064636",
    V06: "6d39aa13282428baffc684274d8b4d921933d9913ae640a6ed2e0280b18e47cf",
}
PSI_CLASSES = ["IR", "CreditQ", "CreditNonQ", "Equity", "Commodity", "FX"]
GRP = ["regular", "high"]
EIG_TOL = 1e-9
EXPAND_MAX = 6   # factors per FX volatility group in the expanded-matrix PSD check
# columns of simm_parameters.csv that may be empty (single-key parameters)
OPTIONAL_COLUMNS = {C.KEY1, C.KEY2}
BANNED_DASHES = ("—", "–", "--")
TEXT_FILES = [
    PARAM_DIR / "PROVENANCE.md",
    *sorted((ROOT / "docs" / "sources").glob("*.txt")),
    PARAM_DIR / "simm_parameters.csv", PARAM_DIR / "schedule_im.csv",
    PARAM_DIR / "bcbs22_traffic_light.csv", PARAM_DIR / "regulatory_constants.csv",
]
# ===== END CONFIG =====

REGULAR_RW = {V12: [107, 101, 90, 69, 68, 69, 66, 61, 60, 58, 58, 66], V06: [107, 101, 90, 69, 68, 69, 66, 61, 60, 58, 58, 66]}
LOW_RW = [15, 18, 12, 11, 15, 21, 23, 25, 29, 27, 26, 28]
HIGH_RW = [167, 102, 79, 82, 90, 93, 92, 88, 88, 98, 101, 96]
CATS = ["Category 1", "Category 2", "Category 3"]
CT_GROUPS = ["high", "regular_well_traded", "regular_less_well_traded", "low"]
VEGA_PAIRS = [("Category 1", "Category 1"), ("Category 1", "Category 2"), ("Category 1", "Category 3"),
              ("Category 2", "Category 2"), ("Category 2", "Category 3"), ("Category 3", "Category 3")]

# expected values per version: (risk_class, param_name, key1, key2) -> value
EXPECT = {
    V12: {
        (C.RC_IR, "delta_rw_inflation", "any", ""): 51, (C.RC_IR, "delta_rw_xccy_basis", "any", ""): 21,
        (C.RC_IR, "hvr", "", ""): 0.74, (C.RC_IR, "vega_rw", "", ""): 0.20,
        (C.RC_IR, "subcurve_phi", "", ""): 0.981, (C.RC_IR, "inflation_yield_corr", "", ""): 0.42,
        (C.RC_IR, "xccy_basis_yield_corr", "", ""): -0.01, (C.RC_IR, "gamma", "", ""): 0.35,
        (C.RC_FX, "delta_rw", "regular", "regular"): 7.4, (C.RC_FX, "delta_rw", "regular", "high"): 31.7,
        (C.RC_FX, "delta_rw", "high", "regular"): 31.7, (C.RC_FX, "delta_rw", "high", "high"): 31.7,
        (C.RC_FX, "hvr", "", ""): 0.67, (C.RC_FX, "vega_rw", "", ""): 0.33, (C.RC_FX, "vol_curvature_corr", "", ""): 0.50,
        (C.RC_FX, "delta_corr_calc_regular", "regular", "regular"): 0.50, (C.RC_FX, "delta_corr_calc_regular", "regular", "high"): 0.12,
        (C.RC_FX, "delta_corr_calc_regular", "high", "high"): 0.03,
        (C.RC_FX, "delta_corr_calc_high", "regular", "regular"): 0.97, (C.RC_FX, "delta_corr_calc_high", "regular", "high"): 0.70,
        (C.RC_FX, "delta_corr_calc_high", "high", "high"): 0.50,
        ("ALL", "psi", "IR", "FX"): 0.15, ("ALL", "psi", "FX", "IR"): 0.15,
    },
    V06: {
        (C.RC_IR, "delta_rw_inflation", "any", ""): 51, (C.RC_IR, "delta_rw_xccy_basis", "any", ""): 21,
        (C.RC_IR, "hvr", "", ""): 0.74, (C.RC_IR, "vega_rw", "", ""): 0.20,
        (C.RC_IR, "subcurve_phi", "", ""): 0.981, (C.RC_IR, "inflation_yield_corr", "", ""): 0.42,
        (C.RC_IR, "xccy_basis_yield_corr", "", ""): -0.01, (C.RC_IR, "gamma", "", ""): 0.35,
        (C.RC_FX, "delta_rw", "regular", "regular"): 7.1, (C.RC_FX, "delta_rw", "regular", "high"): 18.0,
        (C.RC_FX, "delta_rw", "high", "regular"): 18.0, (C.RC_FX, "delta_rw", "high", "high"): 30.6,
        (C.RC_FX, "hvr", "", ""): 0.68, (C.RC_FX, "vega_rw", "", ""): 0.34, (C.RC_FX, "vol_curvature_corr", "", ""): 0.50,
        (C.RC_FX, "delta_corr_calc_regular", "regular", "regular"): 0.50, (C.RC_FX, "delta_corr_calc_regular", "regular", "high"): 0.20,
        (C.RC_FX, "delta_corr_calc_regular", "high", "high"): 0.08,
        (C.RC_FX, "delta_corr_calc_high", "regular", "regular"): 0.92, (C.RC_FX, "delta_corr_calc_high", "regular", "high"): 0.68,
        (C.RC_FX, "delta_corr_calc_high", "high", "high"): 0.50,
        ("ALL", "psi", "IR", "FX"): 0.10, ("ALL", "psi", "FX", "IR"): 0.10,
    },
}
CT_IR_DELTA = {V12: [71, 220, 110, 370], V06: [51, 210, 100, 230]}
CT_IR_VEGA = {V12: [160, 3800, 520, 1100], V06: [110, 4400, 480, 860]}
CT_FX_DELTA = {V12: [2100, 710, 120], V06: [3100, 950, 160]}
CT_FX_VEGA = {V12: [2900, 1500, 840, 760, 490, 310], V06: [2800, 1400, 740, 670, 440, 270]}
HIGH_VOL_FX = {
    V12: {"ARS", "EGP", "ETB", "GHS", "ISK", "LBP", "NGN", "SCR", "ZMW"},
    V06: {"ARS", "EGP", "ETB", "GHS", "LBP", "NGN", "RUB", "SCR", "VES", "ZMW"},
}


@pytest.fixture(scope="module")
def rows():
    return pio.read_csv(pio.SIMM_PARAMS_FILE)


def val(rows, ps, rc, pn, k1="", k2=""):
    hits = [r for r in rows if (r[C.PARAM_SET], r[C.RISK_CLASS], r[C.PARAM_NAME], r[C.KEY1], r[C.KEY2]) == (ps, rc, pn, k1, k2)]
    assert len(hits) == 1, (ps, rc, pn, k1, k2, len(hits))
    return float(hits[0][C.PARAM_VALUE])


def test_columns_exact(rows):
    with open(PARAM_DIR / pio.SIMM_PARAMS_FILE, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == [C.PARAM_SET, C.RISK_CLASS, C.PARAM_NAME, C.KEY1, C.KEY2, C.PARAM_VALUE, C.UNIT, C.SOURCE_DOC, C.VERSION,
                      C.EFFECTIVE_DATE, C.SECTION, C.PARAGRAPH, C.TABLE, C.PDF_PAGE, C.SOURCE_URL, C.RETRIEVED_ON,
                      C.SOURCE_SHA256, C.VERIFICATION_STATUS, C.VERIFICATION_METHOD]


def test_every_row_has_complete_provenance(rows):
    assert rows
    for r in rows:
        for col, v in r.items():
            if col not in OPTIONAL_COLUMNS:
                assert v.strip() != "", (col, r)
        assert r[C.VERIFICATION_STATUS] == C.VERIFIED_PRIMARY
        assert r[C.PARAM_SET] == r[C.VERSION] and r[C.PARAM_SET] in SHA_BY_SET
        assert r[C.SOURCE_SHA256] == SHA_BY_SET[r[C.PARAM_SET]]
        assert 1 <= int(r[C.PDF_PAGE]) <= 30
        assert r[C.SOURCE_URL].startswith("https://www.isda.org/a/")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", r[C.RETRIEVED_ON]) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", r[C.EFFECTIVE_DATE])
        float(r[C.PARAM_VALUE])


def test_effective_dates(rows):
    eff = {r[C.PARAM_SET]: r[C.EFFECTIVE_DATE] for r in rows}
    assert eff == {V12: "2026-07-11", V06: "2025-12-06"}


@pytest.mark.parametrize("ps", [V12, V06])
def test_ir_risk_weight_tables(rows, ps):
    for grp, expected in (("regular", REGULAR_RW[ps]), ("low", LOW_RW), ("high", HIGH_RW)):
        got = [val(rows, ps, C.RC_IR, "delta_rw", grp, t) for t in IR_TENORS]
        assert got == expected, (ps, grp)


@pytest.mark.parametrize("ps", [V12, V06])
def test_headline_scalars(rows, ps):
    for (rc, pn, k1, k2), exp in EXPECT[ps].items():
        assert val(rows, ps, rc, pn, k1, k2) == pytest.approx(exp, abs=1e-12), (ps, rc, pn, k1, k2)


@pytest.mark.parametrize("ps", [V12, V06])
def test_concentration_thresholds(rows, ps):
    assert [val(rows, ps, C.RC_IR, "delta_ct", g) for g in CT_GROUPS] == CT_IR_DELTA[ps]
    assert [val(rows, ps, C.RC_IR, "vega_ct", g) for g in CT_GROUPS] == CT_IR_VEGA[ps]
    assert [val(rows, ps, C.RC_FX, "delta_ct", c) for c in CATS] == CT_FX_DELTA[ps]
    assert [val(rows, ps, C.RC_FX, "vega_ct", a, b) for a, b in VEGA_PAIRS] == CT_FX_VEGA[ps]


@pytest.mark.parametrize("ps", [V12, V06])
def test_currency_lists(rows, ps):
    def members(rc, pn, k1):
        return {r[C.KEY2] for r in rows if (r[C.PARAM_SET], r[C.RISK_CLASS], r[C.PARAM_NAME], r[C.KEY1]) == (ps, rc, pn, k1)}
    assert members(C.RC_FX, "high_vol_member", "high") == HIGH_VOL_FX[ps]
    assert members(C.RC_IR, "rw_group_member", "low") == {"JPY"}
    assert members(C.RC_IR, "ct_group_member", "regular_well_traded") == {"USD", "EUR", "GBP"}
    assert members(C.RC_IR, "rw_group_member", "regular") == {"USD", "EUR", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "DKK", "HKD", "KRW", "SGD", "TWD"}
    assert members(C.RC_IR, "ct_group_member", "regular_less_well_traded") == {"AUD", "CAD", "CHF", "DKK", "HKD", "KRW", "NOK", "NZD", "SEK", "SGD", "TWD"}
    # scope currencies: JPY low, MXN high-vol IR (in no regular/low list), FX Category 2 for MXN
    assert "MXN" not in members(C.RC_IR, "rw_group_member", "regular") | members(C.RC_IR, "rw_group_member", "low")
    assert "MXN" in members(C.RC_FX, "ct_category_member", "Category 2")
    assert {"USD", "EUR", "GBP", "JPY"} <= members(C.RC_FX, "ct_category_member", "Category 1")
    assert "MXN" not in HIGH_VOL_FX[ps]


@pytest.mark.parametrize("ps", [V12, V06])
def test_matrices_symmetric_unit_diagonal_psd(ps):
    ir = pio.ir_tenor_corr(ps)
    psi = pio.square_matrix(ps, "ALL", "psi", PSI_CLASSES)
    for m in (ir, psi):
        assert np.allclose(m, m.T, atol=0, rtol=0)
        assert np.allclose(np.diag(m), 1.0)
        assert np.linalg.eigvalsh(m).min() >= -EIG_TOL
    assert np.linalg.eigvalsh(ir).min() == pytest.approx(0.005, abs=5e-4)
    assert ir[IR_TENORS.index("2w"), IR_TENORS.index("1m")] == 0.74 and ir[IR_TENORS.index("10y"), IR_TENORS.index("30y")] == 0.95


@pytest.mark.parametrize("ps", [V12, V06])
def test_fx_correlation_tables_symmetric_and_expanded_matrices_psd(ps):
    # the 2x2 tables are pairwise values for two distinct factors (diagonal is not 1, table alone is not PSD);
    # the factor-level matrix (unit diagonal, table values off-diagonal) must be PSD for any mix of groups
    for pn in ("delta_corr_calc_regular", "delta_corr_calc_high"):
        t = pio.square_matrix(ps, C.RC_FX, pn, GRP)
        assert np.allclose(t, t.T) and ((t > 0) & (t < 1)).all()
        for n_reg in range(0, EXPAND_MAX + 1):
            for n_high in range(0, EXPAND_MAX + 1):
                g = [0] * n_reg + [1] * n_high
                if len(g) < 2:
                    continue
                m = np.array([[1.0 if i == j else t[g[i], g[j]] for j in range(len(g))] for i in range(len(g))])
                assert np.linalg.eigvalsh(m).min() >= -EIG_TOL, (ps, pn, n_reg, n_high)


def test_ir_tenor_matrix_identical_across_versions():
    assert np.array_equal(pio.ir_tenor_corr(V12), pio.ir_tenor_corr(V06))


def test_sf_example_table_matches_formula(rows):
    days = {"2w": 14, "1m": 365 / 12, "3m": 365 / 4, "6m": 365 / 2, "12m": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650}
    for ps in (V12, V06):
        sf0 = val(rows, ps, "ALL", "curvature_sf_scale")
        ref = val(rows, ps, "ALL", "curvature_sf_ref_days")
        for k, d in days.items():
            assert sf0 * min(1.0, ref / d) == pytest.approx(val(rows, ps, "ALL", "curvature_sf_example", k), abs=5e-4)
        assert val(rows, ps, "ALL", "curvature_lambda_quantile") == 0.995
        assert val(rows, ps, C.RC_IR, "curvature_hvr_exponent") == -2
        assert val(rows, ps, "ALL", "vega_sigma_days_numerator") == 365 and val(rows, ps, "ALL", "vega_sigma_days_denominator") == 14


def test_v2512_vs_v2506_differences_are_exactly_the_listed_ones(rows):
    def table(ps):
        return {(r[C.RISK_CLASS], r[C.PARAM_NAME], r[C.KEY1], r[C.KEY2]): float(r[C.PARAM_VALUE]) for r in rows if r[C.PARAM_SET] == ps}
    a, b = table(V12), table(V06)
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    assert only_a == {(C.RC_FX, "high_vol_member", "high", "ISK")}
    assert only_b == {(C.RC_FX, "high_vol_member", "high", "RUB"), (C.RC_FX, "high_vol_member", "high", "VES")}
    changed = {k for k in set(a) & set(b) if a[k] != b[k] and k[0] != "ALL"}
    expected_changed_names = {
        (C.RC_IR, "delta_ct"), (C.RC_IR, "vega_ct"),
        (C.RC_FX, "delta_rw"), (C.RC_FX, "hvr"), (C.RC_FX, "vega_rw"),
        (C.RC_FX, "delta_corr_calc_regular"), (C.RC_FX, "delta_corr_calc_high"),
        (C.RC_FX, "delta_ct"), (C.RC_FX, "vega_ct"),
    }
    assert {(k[0], k[1]) for k in changed} == expected_changed_names
    count = {n: sum(1 for k in changed if (k[0], k[1]) == n) for n in expected_changed_names}
    assert count[(C.RC_IR, "delta_ct")] == 4 and count[(C.RC_IR, "vega_ct")] == 4
    assert count[(C.RC_FX, "delta_rw")] == 4 and count[(C.RC_FX, "hvr")] == 1 and count[(C.RC_FX, "vega_rw")] == 1
    assert count[(C.RC_FX, "delta_corr_calc_regular")] == 3   # regular/regular 50% unchanged
    assert count[(C.RC_FX, "delta_corr_calc_high")] == 3      # high/high 50% unchanged
    assert count[(C.RC_FX, "delta_ct")] == 3 and count[(C.RC_FX, "vega_ct")] == 6
    # in-scope psi entry changes; everything else in ALL that the engine uses is constant
    assert a[("ALL", "psi", "IR", "FX")] != b[("ALL", "psi", "IR", "FX")]
    for k in set(a) & set(b):
        if k[0] == "ALL" and k[1] != "psi":
            assert a[k] == b[k], k


# ---------------- regulatory files ----------------
def test_bcbs22_traffic_light():
    tl = pio.read_csv(pio.TRAFFIC_LIGHT_FILE)
    assert [int(r["exceptions"]) for r in tl] == list(range(11))
    zone = {int(r["exceptions"]): r["zone"] for r in tl}
    assert [k for k, z in zone.items() if z == "green"] == [0, 1, 2, 3, 4]
    assert [k for k, z in zone.items() if z == "yellow"] == [5, 6, 7, 8, 9]
    assert zone[10] == "red"
    plus = [float(r["plus_factor"]) for r in tl]
    assert plus == [0, 0, 0, 0, 0, 0.40, 0.50, 0.65, 0.75, 0.85, 1.00]
    for r in tl:
        k = int(r["exceptions"])
        cdf = float(binom.cdf(k, 250, 0.01))
        assert float(r["scipy_cumulative_probability_n250_p001"]) == pytest.approx(cdf, abs=1e-9)
        assert round(cdf * 100, 2) == pytest.approx(float(r["doc_cumulative_probability_pct"]), abs=1e-9)   # matches the document
        assert r["verification_status"] == C.VERIFIED_PRIMARY and r["doc_matches_scipy_rounded"] == "true"
        assert int(r["pdf_page"]) == 15 and len(r["source_sha256"]) == 64
    cdf = {k: float(binom.cdf(k, 250, 0.01)) for k in range(11)}
    assert round(cdf[4], 4) == 0.8922 and round(cdf[5], 4) == 0.9588
    assert round(cdf[9], 5) == 0.99975 and round(cdf[10], 5) == 0.99995
    # zone rule: yellow from cdf >= 95%, red from cdf >= 99.99%
    assert cdf[4] < 0.95 <= cdf[5] and cdf[9] < 0.9999 <= cdf[10]


def test_schedule_rates():
    sc = pio.read_csv(pio.SCHEDULE_FILE)
    assert set(sc[0].keys()) == {"asset_class", "maturity_bucket_low_years", "maturity_bucket_high_years", "rate_pct",
                                 "source_doc", "section", "retrieved_on", "verification_status"}
    assert {r["verification_status"] for r in sc} == {C.VERIFIED_PRIMARY}

    def rates(doc_key, asset):
        out = {}
        for r in sc:
            if doc_key in r["source_doc"] and r["asset_class"] == asset:
                out[(r["maturity_bucket_low_years"], r["maturity_bucket_high_years"])] = float(r["rate_pct"])
        return out
    buckets = {("0", "2"), ("2", "5"), ("5", "inf")}
    for key in ("BCBS-IOSCO", "12 CFR Part 45"):
        assert rates(key, "Interest rate") == {("0", "2"): 1.0, ("2", "5"): 2.0, ("5", "inf"): 4.0}
        assert rates(key, "Credit") == {("0", "2"): 2.0, ("2", "5"): 5.0, ("5", "inf"): 10.0}
        assert rates(key, "Foreign exchange") == {("", ""): 6.0}
        assert rates(key, "Equity") == {("", ""): 15.0} and rates(key, "Commodity") == {("", ""): 15.0} and rates(key, "Other") == {("", ""): 15.0}
    assert rates("12 CFR Part 45", "Cross-currency swap") == {("0", "2"): 1.0, ("2", "5"): 2.0, ("5", "inf"): 4.0}
    assert rates("BCBS-IOSCO", "Cross-currency swap") == {}   # no separate rows in the BCBS-IOSCO table
    assert set(rates("12 CFR Part 45", "Cross-currency swap")) == buckets


def test_regulatory_constants():
    rc = pio.read_csv(pio.REG_CONSTANTS_FILE)
    assert {r["verification_status"] for r in rc} == {C.VERIFIED_PRIMARY}

    def get(name, jur):
        hits = [r for r in rc if r["constant"] == name and r["jurisdiction"] == jur]
        assert len(hits) == 1, (name, jur)
        return hits[0]["value"]
    for jur in ("BCBS-IOSCO", "EU", "US"):
        assert float(get("confidence_level", jur)) == 0.99 and float(get("mpor_days", jur)) == 10
        assert get("equally_weighted", jur) == "true"
    assert float(get("stress_share_min", "EU")) == 0.25
    assert (float(get("calibration_min_years", "EU")), float(get("calibration_max_years", "EU"))) == (3, 5)
    assert (float(get("calibration_min_years", "US")), float(get("calibration_max_years", "US"))) == (1, 5)
    assert float(get("backtest_min_frequency_months", "EU")) == 3
    assert get("backtest_min_frequency_months", "US") == "NOT_SPECIFIED"
    assert float(get("backtest_observations", "BCBS 22")) == 250
    assert float(get("schedule_gross_weight", "BCBS-IOSCO")) == 0.4 and float(get("schedule_ngr_weight", "US")) == 0.6
    assert float(get("schedule_ngr_when_gross_rc_zero", "US")) == 1.0
    for r in rc:
        assert r["citation"] and r["source_url"] and len(r["source_sha256"]) == 64 and r["retrieved_on"] == "2026-10-02"


def test_no_dash_substitutes_and_sources_headers():
    for p in TEXT_FILES:
        assert p.exists(), p
        txt = p.read_text(encoding="utf-8")
        for bad in BANNED_DASHES:
            assert bad not in txt, (p.name, repr(bad))
    for p in sorted((ROOT / "docs" / "sources").glob("*.txt")):
        head = p.read_text(encoding="utf-8").splitlines()
        assert head[0].startswith("Source URL: http") and head[1].startswith("Retrieved: 2026-10-02"), p.name


def test_no_isda_pdfs_in_repo():
    assert not list(PARAM_DIR.rglob("*.pdf")) and not list((ROOT / "docs" / "sources").rglob("*.pdf"))
