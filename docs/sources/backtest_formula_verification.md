# Backtest formula verification (retrieved 2026-10-02)

Status: the ORIGINAL texts of Kupiec (1995) and Christoffersen (1998) were NOT obtained. The formulas were checked against Federal Reserve publications that restate them, and against independent numeric references in `python/tests/test_backtest.py`.

## Attempts for the originals
- Kupiec, "Techniques for verifying the accuracy of risk measurement models", FEDS 95-24 / J. Derivatives 3(2):73-84. `federalreserve.gov/pubs/feds/1995/199524/199524pap.pdf` and `...199524abs.html`: HTTP 404. `fraser.stlouisfed.org/files/docs/historical/fedgfe/fedgfe_199524.pdf`: HTTP 403 (AccessDenied). `fedinprint.org/item/fedgfe/34596/original`: no connection. `pm-research.com` PDF (journal copy): HTTP 403. The RePEc and EconPapers pages are bibliographic only.
- Christoffersen, "Evaluating interval forecasts", International Economic Review 39(4):841-862: paywalled, not attempted beyond the bibliographic record.

## Federal Reserve restatements that were read (curl -L -A "Mozilla/5.0", markitdown)
1. Lopez, "Regulatory Evaluation of Value-at-Risk Models", Federal Reserve Bank of San Francisco Working Paper 99-06 (draft 30 June 1999), https://www.frbsf.org/wp-content/uploads/wpjl99-06.pdf, section on backtesting of exception series (text pages 7 to 9 of the PDF): LR_uc(alpha) = 2 log[((1 - x/T)/(1 - alpha))^(T-x) * ((x/T)/alpha)^x], chi-square(1); LR_ind = 2 (log L_A - log L_0) with L_A = (1-pi01)^T00 pi01^T01 (1-pi11)^T10 pi11^T11, pi01 = T01/(T00+T01), pi11 = T11/(T10+T11), L_0 = (1-pi)^(T00+T10) pi^(T01+T11), pi = (T01+T11)/T, chi-square(1); LR_cc = LR_uc + LR_ind, chi-square(2).
2. Campbell, "A Review of Backtesting and Backtesting Procedures", FEDS 2005-21, https://www.federalreserve.gov/pubs/feds/2005/200521/200521pap.pdf, section 3.1: POF = 2 log[((1 - alpha_hat)/(1 - alpha))^(T - I) * (alpha_hat/alpha)^I]. Footnote 3 notes the statistic is undefined when I = 0 because log 0 is undefined; the code uses 0 ln 0 = 0 instead.

Both agree with `backtest.kupiec_pof` and `backtest.christoffersen`: LR = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x].

## Numeric references used in the tests
- Kupiec: closed forms at x = 0 and x = n, LR = 0 at x = p n, scipy `binom.logpmf` likelihood ratio, hand p-value 2*(1 - Phi(sqrt(5.0252))) = 0.02498 for n = 250, x = 0.
- Christoffersen: hand transition counts (n00, n01, n10, n11 = 4, 2, 2, 1), an independent plain-python LR with 0 ln 0 = 0, zero-cell cases (n11 = 0, no exceptions), LR_cc = LR_uc + LR_ind, scipy chi-square p-values.
- Size: 2,000 simulated series. The exact finite-sample rejection rate of the Kupiec test is computed by enumerating the binomial distribution and the simulated rate agrees within four standard errors. It is 9.5% at n = 250, 5.5% at n = 1,000 and 4.4% at n = 2,500 for a nominal 5%, so the 250-day test over-rejects because of discreteness.
- BCBS 22 Table 2: zones at 4/5 and 9/10 exceptions for n = 250, cumulative binomial probabilities 0.8922, 0.9588, 0.99975, 0.99995, plus factors read from `data/parameters/bcbs22_traffic_light.csv` (VERIFIED_PRIMARY in the verification log).
