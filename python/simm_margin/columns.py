"""Column-name constants. All other modules import names from here; no string-literal column names elsewhere."""

# trades
TRADE_ID, PORTFOLIO_ID, PRODUCT, CCY, CCY2 = "trade_id", "portfolio_id", "product", "ccy", "ccy2"
NOTIONAL, DIRECTION, START, MATURITY, EXPIRY = "notional", "direction", "start", "maturity", "expiry"
STRIKE, VOL_TYPE, OPT_TYPE, EXCHANGE_FLAG = "strike", "vol_type", "opt_type", "settle_notional_exchange"
PRODUCT_IRS, PRODUCT_SWAPTION, PRODUCT_FXFWD, PRODUCT_FXOPT, PRODUCT_XCCY = "IRS", "SWAPTION", "FXFWD", "FXOPT", "XCCY"

# CRIF
PRODUCT_CLASS, RISK_TYPE, QUALIFIER, BUCKET = "ProductClass", "RiskType", "Qualifier", "Bucket"
LABEL1, LABEL2, AMOUNT, AMOUNT_CCY, AMOUNT_USD = "Label1", "Label2", "Amount", "AmountCurrency", "AmountUSD"
CRIF_TRADE_ID = "TradeID"
RISK_IRCURVE, RISK_XCCYBASIS, RISK_IRVOL, RISK_FX, RISK_FXVOL = "Risk_IRCurve", "Risk_XCcyBasis", "Risk_IRVol", "Risk_FX", "Risk_FXVol"
RATES_FX = "RatesFX"

# SIMM calculation tables
RW, CR, WS, SENS, K_B, S_B = "RW", "CR", "WS", "sens", "K_b", "S_b"
LEVEL, RISK_CLASS, MARGIN_TYPE, RISK_FACTOR, VALUE = "level", "risk_class", "margin_type", "risk_factor", "value"
RC_IR, RC_FX = "IR", "FX"
MT_DELTA, MT_VEGA, MT_CURVATURE = "Delta", "Vega", "Curvature"

# parameters (long format)
PARAM_SET, PARAM_NAME, KEY1, KEY2, PARAM_VALUE, UNIT = "param_set", "param_name", "key1", "key2", "value", "unit"
SOURCE_DOC, VERSION, EFFECTIVE_DATE, SECTION, PARAGRAPH = "source_doc", "version", "effective_date", "section", "paragraph"
TABLE, PDF_PAGE, SOURCE_URL, RETRIEVED_ON, SOURCE_SHA256 = "table", "pdf_page", "source_url", "retrieved_on", "source_sha256"
VERIFICATION_STATUS, VERIFICATION_METHOD = "verification_status", "verification_method"
VERIFIED_PRIMARY, ILLUSTRATIVE = "VERIFIED_PRIMARY", "ILLUSTRATIVE"

# backtest
DATE, LOSS, VAR_FORECAST, EXCEPTION, REGIME, MODEL, TEST, LIGHT = "date", "loss", "var_forecast", "exception", "regime", "model", "test", "light"
REGIME_CALM, REGIME_STRESS = 0, 1

# findings (same schema as the sibling validation project)
FINDING_ID, AREA, TITLE, DESCRIPTION, EVIDENCE, EVIDENCE_FILE = "finding_id", "area", "title", "description", "evidence", "evidence_file"
SEVERITY, TRAFFIC_LIGHT, RECOMMENDATION, OWNER, STATUS, SOURCE = "severity", "traffic_light", "recommendation", "owner", "status", "source"
