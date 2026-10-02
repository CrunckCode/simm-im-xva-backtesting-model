# SIMM-Style Initial Margin, xVA VaR and Backtesting Model

> **Disclaimer.** All data are SYNTHETIC. This is an educational SIMM-style model: it is NOT licensed by ISDA, NOT certified or validated against ISDA's calculator, and makes NO compliance claim. ISDA SIMM is a trademark and ISDA documents are copyright; only numeric parameters with citations are stored here. Not verified: Kupiec (1995) and Christoffersen (1998) originals not obtained; EU consolidated text, 12 CFR 237.8 and 349.8, CFTC 23.154 and the ISDA credit, equity and commodity tables not checked.

A portfolio project in counterparty credit risk, market risk and margin model validation. For a netting set of 20 synthetic Rates and FX trades it computes a SIMM-style initial margin (parameter sets v2.8+2512 and v2.8+2506), the standardised schedule IM, a historical-simulation VaR IM, an xVA VaR for CVA, and then backtests, investigates seeded margin disputes, attributes IM changes, runs UAT for a new product and writes a findings log. A 19-section Word document (`docs/SIMM_IM_xVA_Backtesting_Documentation.docx`) explains every step with worked examples.

## Quick start

```
cd python
py -3 -m pip install -r requirements.txt
py -3 -m simm_margin.cli --quick      # about 2 minutes; omit --quick for the full run (about 9 minutes)
py -3 -m pytest -q
cd ..
py -3 docs/build_docs.py              # rebuilds the markdown, the Word document and this README
```

## Headline results (valuation date 30 Sep 2026, full run)

| Item | Result |
|:---|---:|
| SIMM-style IM, v2.8+2512 | USD 5,465,781 |
| SIMM-style IM, v2.8+2506 | USD 5,216,561 |
| Standardised schedule IM (net) | USD 19,099,501 |
| SIMM as a share of net schedule IM | 28.6% |
| Champion HS IM, 1-day exceptions | 18 of 1,327 (13.3 expected) |
| Champion Kupiec / Christoffersen cc (1 day) | p = 0.216 (pass) / p = 0.034 (amber) |
| Champion worst rolling 250-day count | 12 (red zone) |
| Challengers (EWMA, plain 250, x0.7), 1-day light | red, red, red |
| SIMM-style, 10-day non-overlapping exceptions | 1 of 132 (pass) |
| xVA VaR, 1-day exceptions | 15 of 1,327 (pass) |
| Dispute scenarios accepted | 9 of 9 |
| Attribution of the scenario-day change | USD 4,040,470, exact Shapley |
| UAT (pass / fail / skipped) | 17 / 0 / 0 |
| Findings (High / Medium / Low) | 2 / 10 / 5 |

## Key findings

* The champion is well calibrated overall but its 1-day exceptions cluster: Christoffersen conditional coverage is amber and its worst rolling 250-day window is red.
* Mis-calibrated challengers are clearly worse, which shows the tests have power.
* Overlapping 10-day windows are red for every model (BCBS 22 section II warns against them); the SIMM-style series passes the non-overlapping design but fails Christoffersen independence on overlapping windows.
* SIMM is far below the standardised schedule IM, as expected for a risk-sensitive model on a hedged book.
* An FX vega CRIF convention issue (the market vol column `SigmaMarket`) was found and fixed during the build and is documented as a finding.

## Excel workbook

`excel/simm_margin_workbook.xlsx` repeats the SIMM, schedule, backtest and attribution calculations with live formulas (no array formulas, so it opens in Excel 2016 and LibreOffice). The reconciliation report lists 3 checks: 3 pass and 0 fail.

## Repository layout

```
INTERFACES.md                    scope and contracts
data/parameters/                 verified numeric parameters with provenance (no ISDA text)
data/uat/                        UAT suite
python/simm_margin/              engine: curves, market_history, pricing, sensitivities, crif, params, simm,
                                 schedule_im, allocation, var_model, cva, backtest, dispute, attribution,
                                 uat, findings, charts, cli
python/tests/                    pytest suite (never writes to python/outputs)
python/outputs/                  CSV and JSON results, charts/
excel/                           workbook with live formulas
docs/                            Word documentation, markdown source, build script, verification log
```

## Limitations

Synthetic data; a single netting set; one curve per currency; IR and FX only (no credit, equity or commodity classes); hypothetical P&L on a frozen portfolio; CVA for linear trades only; one SIMM version over the whole history; Kupiec (1995) and Christoffersen (1998) originals not obtained; no comparison with ISDA's calculator. See Section 16 of the documentation.

## Parameter provenance

Every in-scope parameter in `data/parameters/simm_parameters.csv` (685 rows for the two versions) is VERIFIED_PRIMARY: it carries its source document, section, paragraph, table, PDF page, URL, retrieval date and the sha256 of the retrieved PDF. Regulatory constants and the BCBS 22 traffic-light table carry the same citations. The verification log is `docs/VERIFICATION_LOG.md`.
