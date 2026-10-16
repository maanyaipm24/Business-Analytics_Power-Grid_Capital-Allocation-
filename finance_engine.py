"""
Finance engine for the POWERGRID Capital-Allocation Decision App.

Pure pandas/numpy so every number in the app can be traced and tested.
All money in ₹ crore unless stated. FY2026 = year ended 31 March 2026.

Contents
  1. Loading & validation          load_bundled, validate_upload, quality_report
  2. Ratio engine                  build_ratios (profitability, returns, leverage,
                                   coverage, efficiency, cash flow, per-share, valuation)
  3. DuPont                        dupont, roe_bridge, pbt_bridge
  4. Cost of capital               estimate_beta, rolling_beta, wacc, eva_history
  5. Forecasting                   fit_revenue_model, backtest_revenue
  6. Projection & scenarios        project (vectorised), run_policy_grid,
                                   monte_carlo, max_sustainable_payout, valuation
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data" if (_HERE / "data").exists() else _HERE
SHARES_NOW = 930.06   # crore shares


# =============================================================================
# 1. LOADING & VALIDATION
# =============================================================================
REQUIRED_FIN = ["fy", "revenue", "other_income", "employee_exp", "finance_costs", "depreciation",
                "other_exp", "pbt", "total_tax", "pat", "ppe", "cwip", "total_assets",
                "equity_capital", "other_equity", "nc_borrowings", "c_borrowings"]
OPTIONAL_FIN = {"jv_share": 0.0, "exceptional_items": 0.0, "current_tax": np.nan, "deferred_tax": np.nan,
                "rda_net": 0.0, "intangibles": 0.0, "trade_receivables": np.nan, "cash": 0.0,
                "other_bank_balances": 0.0, "current_maturities_ltd": 0.0, "cfo": np.nan,
                "capex_cf": np.nan, "dividends_paid_cf": np.nan, "payout_pct_declared": np.nan,
                "eps_reported": np.nan, "rda_debit": 0.0, "dtl": np.nan, "cfi": np.nan, "cff": np.nan,
                "finance_costs_paid": np.nan, "inv_jv": 0.0, "investments": 0.0,
                "source": "", "secondary_fields": "", "basis_note": ""}


def load_bundled():
    return dict(
        fin=pd.read_csv(DATA / "powergrid_financials.csv"),
        seg=pd.read_csv(DATA / "segments.csv"),
        px=pd.read_csv(DATA / "market_monthly.csv"),
        peers=pd.read_csv(DATA / "peers.csv"),
        xchk=pd.read_csv(DATA / "screener_crosscheck.csv"),
        assum=pd.read_csv(DATA / "assumptions.csv"),
    )


def validate_upload(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Check an uploaded statement file against the template; fill optional columns."""
    msgs = []
    miss = [c for c in REQUIRED_FIN if c not in df.columns]
    if miss:
        return None, [f"Missing required columns: {', '.join(miss)}"]
    df = df.copy()
    for c, d in OPTIONAL_FIN.items():
        if c not in df.columns:
            df[c] = d
            msgs.append(f"Optional column '{c}' not supplied – treated as {d if d != '' else 'blank'}.")
    num = [c for c in df.columns if c not in ("source", "secondary_fields", "basis_note")]
    bad = df[num].apply(pd.to_numeric, errors="coerce").isna() & df[num].notna()
    if bad.any().any():
        msgs.append(f"Non-numeric values found in: {', '.join(bad.columns[bad.any()])} – set to blank.")
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("fy").reset_index(drop=True)
    return df, msgs


def quality_report(fin: pd.DataFrame, xchk: pd.DataFrame | None = None) -> pd.DataFrame:
    """Integrity checks on the statement data. Returns one row per check."""
    f = fin.copy()
    rows = []
    total_income = f.revenue + f.other_income
    total_exp = f.employee_exp + f.finance_costs + f.depreciation + f.other_exp
    pbt_calc = total_income - total_exp + f.jv_share.fillna(0) + f.exceptional_items.fillna(0)
    d = (pbt_calc - f.pbt).abs()
    rows.append(("P&L arithmetic: income − expenses ± JV ± exceptional = PBT",
                 int((d > 1).sum()), f"max gap ₹{d.max():.1f} Cr", "Pass" if (d <= 1).all() else "Investigate"))
    d = (f.pbt - f.total_tax + f.rda_net.fillna(0) - f.pat).abs()
    rows.append(("PAT = PBT − tax ± regulatory deferral movement", int((d > 1).sum()),
                 f"max gap ₹{d.max():.1f} Cr", "Pass" if (d <= 1).all() else "Investigate"))
    dup = int(f.fy.duplicated().sum())
    rows.append(("Duplicate fiscal years", dup, "", "Pass" if dup == 0 else "Kept latest filing"))
    years = set(f.fy)
    gaps = [y for y in range(int(f.fy.min()), int(f.fy.max()) + 1) if y not in years]
    rows.append(("Missing years inside the range", len(gaps), ", ".join(map(str, gaps)) or "none",
                 "Pass" if not gaps else "Ratios needing prior year left blank"))
    key = ["revenue", "pat", "total_assets", "nc_borrowings", "equity_capital", "other_equity"]
    nmiss = int(f[key].isna().sum().sum())
    rows.append(("Missing values in key fields", nmiss, "", "Pass" if nmiss == 0 else "Excluded from affected ratios"))
    opt_miss = {c: int(f[c].isna().sum()) for c in ["cfo", "capex_cf", "dividends_paid_cf", "trade_receivables"]
                if c in f}
    rows.append(("Missing values in cash-flow fields", sum(opt_miss.values()),
                 ", ".join(f"{k}: {v}" for k, v in opt_miss.items() if v),
                 "Capex derived from balance sheet (Δ net block + Δ CWIP + depreciation) where cash flow is absent"))
    neg = int((f[["revenue", "total_assets", "ppe"]] < 0).sum().sum())
    rows.append(("Negative values where impossible", neg, "", "Pass" if neg == 0 else "Investigate"))
    # outliers: YoY change in revenue / PAT beyond 3 robust z
    for col in ["revenue", "pat", "other_exp", "total_tax"]:
        g = f[col].pct_change()
        med, mad = g.median(), (g - g.median()).abs().median()
        z = (g - med) / (1.4826 * mad) if mad else g * 0
        out = f.loc[z.abs() > 3, "fy"].tolist()
        rows.append((f"Outlier YoY change in {col} (robust z > 3)", len(out),
                     ", ".join(f"FY{int(y)}" for y in out) or "none",
                     "Explained in notes; one-offs normalised in 'core PAT'" if out else "Pass"))
    if "secondary_fields" in f:
        sec = f.loc[f.secondary_fields.fillna("") != "", "fy"].tolist()
        rows.append(("Fields from secondary source", len(sec), ", ".join(f"FY{int(y)}" for y in sec),
                     "Flagged; used only where primary filing lacks the line"))
    if "basis_note" in f:
        b = f.loc[f.basis_note.fillna("") != "", ["fy", "basis_note"]]
        rows.append(("Restatements / basis changes", len(b),
                     " | ".join(f"FY{int(r.fy)}: {r.basis_note}" for r in b.itertuples()),
                     "Latest restated figures used; FY17 balance-sheet ratios marked not fully comparable"))
    if xchk is not None and len(xchk):
        m = f[["fy", "revenue", "pat"]].merge(xchk, on="fy", how="inner")
        dr = ((m.revenue - m.sales).abs() / m.revenue * 100)
        dp = ((m.pat - m.net_profit).abs() / m.pat * 100)
        rows.append(("Cross-check vs Screener.in: revenue (max % gap)", int((dr > 0.5).sum()),
                     f"{dr.max():.2f}%", "Pass" if dr.max() < 0.5 else "Differences explained"))
        rows.append(("Cross-check vs Screener.in: PAT (max % gap)", int((dp > 0.5).sum()),
                     f"{dp.max():.2f}%", "Pass" if dp.max() < 0.5 else "Differences explained"))
        ex = m.merge(f[["fy", "employee_exp", "other_exp"]], on="fy")
        diff = ex.expenses - (ex.employee_exp + ex.other_exp)
        big = ex.loc[diff.abs() > 1000, "fy"].tolist()
        rows.append(("Cross-check: Screener 'expenses' vs filing opex", len(big),
                     ", ".join(f"FY{int(y)}" for y in big) or "none",
                     "Screener folds regulatory-deferral/tax items into expenses in these years – filing figures used"))
    return pd.DataFrame(rows, columns=["Check", "Issues", "Detail", "Treatment / result"])


# =============================================================================
# 2. RATIO ENGINE
# =============================================================================
def _avg(s: pd.Series) -> pd.Series:
    """Average of opening and closing balance; first year uses closing balance."""
    return ((s + s.shift(1)) / 2).fillna(s)


def build_ratios(fin: pd.DataFrame, px: pd.DataFrame | None = None, shares_now: float = SHARES_NOW) -> pd.DataFrame:
    f = fin.sort_values("fy").reset_index(drop=True).copy()
    z = lambda c: f[c].fillna(0) if c in f else 0
    f["equity"] = f.equity_capital + f.other_equity
    f["debt"] = f.nc_borrowings + f.c_borrowings + z("current_maturities_ltd")
    f["cash_all"] = z("cash") + z("other_bank_balances")
    f["net_debt"] = f.debt - f.cash_all
    f["net_block"] = f.ppe + z("intangibles")
    f["capital_employed"] = f.equity + f.debt
    f["ebitda"] = f.revenue - f.employee_exp - f.other_exp
    f["ebit"] = f.pbt + f.finance_costs                       # incl. other income, JV, exceptional
    f["ebit_core"] = f.ebit - z("exceptional_items")
    f["opex"] = f.employee_exp + f.other_exp
    # economic ("core") profit: deferred tax & RDA movements are largely recoverable from
    # beneficiaries under CERC tariffs, so core PAT = PBT before exceptional items − current tax.
    ctax = f.current_tax if "current_tax" in f and f.current_tax.notna().all() else f.total_tax
    f["core_pat"] = f.pbt - z("exceptional_items") - ctax
    f["cash_tax_rate"] = ctax / (f.pbt - z("exceptional_items"))
    f["eff_tax_rate"] = (f.pbt - f.pat) / f.pbt            # all-in incl. RDA movement

    A = {c: _avg(f[c]) for c in ["total_assets", "equity", "debt", "capital_employed", "net_block"]}
    if "trade_receivables" in f:
        A["trade_receivables"] = _avg(f.trade_receivables)
    r = pd.DataFrame({"fy": f.fy})
    # --- growth & profitability
    r["revenue"] = f.revenue
    r["revenue_growth_%"] = f.revenue.pct_change() * 100
    r["ebitda"] = f.ebitda
    r["ebitda_margin_%"] = f.ebitda / f.revenue * 100
    r["ebit"] = f.ebit
    r["pbt"] = f.pbt
    r["pat"] = f.pat
    r["core_pat"] = f.core_pat
    r["pat_growth_%"] = f.pat.pct_change() * 100
    r["core_pat_growth_%"] = f.core_pat.pct_change() * 100
    r["net_margin_%"] = f.pat / f.revenue * 100
    r["opex_to_revenue_%"] = f.opex / f.revenue * 100
    r["other_income_share_of_pbt_%"] = f.other_income / f.pbt * 100
    # --- returns
    r["roe_%"] = f.pat / A["equity"] * 100
    r["core_roe_%"] = f.core_pat / A["equity"] * 100
    r["roa_%"] = f.pat / A["total_assets"] * 100
    r["roce_pretax_%"] = f.ebit_core / A["capital_employed"] * 100
    r["roic_posttax_%"] = f.ebit_core * (1 - f.cash_tax_rate) / A["capital_employed"] * 100
    r["roic_ex_cwip_%"] = f.ebit_core * (1 - f.cash_tax_rate) / (A["capital_employed"] - _avg(f.cwip)) * 100
    # --- efficiency
    r["asset_turnover_x"] = f.revenue / A["total_assets"]
    r["fixed_asset_turnover_x"] = f.revenue / A["net_block"]
    r["cwip_share_%"] = f.cwip / (f.net_block + f.cwip) * 100
    if "trade_receivables" in f:
        r["receivable_days"] = A["trade_receivables"] / f.revenue * 365
    # --- leverage & coverage
    r["debt"] = f.debt
    r["net_debt"] = f.net_debt
    r["equity"] = f.equity
    r["debt_equity_x"] = f.debt / f.equity
    r["net_debt_equity_x"] = f.net_debt / f.equity
    r["net_debt_ebitda_x"] = f.net_debt / f.ebitda
    r["debt_to_capital_%"] = f.debt / f.capital_employed * 100
    r["interest_coverage_x"] = f.ebit_core / f.finance_costs
    r["ebitda_interest_x"] = f.ebitda / f.finance_costs
    r["equity_multiplier_x"] = A["total_assets"] / A["equity"]
    r["cost_of_debt_pl_%"] = f.finance_costs / A["debt"] * 100
    if "finance_costs_paid" in f:
        r["cost_of_debt_paid_%"] = f.finance_costs_paid / A["debt"] * 100
    # --- cash flow
    capex_derived = (f.net_block + f.cwip).diff() + f.depreciation
    r["capex_derived"] = capex_derived
    r["capex"] = f.capex_cf.fillna(capex_derived) if "capex_cf" in f else capex_derived
    r["capitalised"] = f.net_block.diff() + f.depreciation
    if "cfo" in f:
        r["cfo"] = f.cfo
        r["fcf"] = f.cfo - r.capex
        r["cfo_to_debt_%"] = f.cfo / f.debt * 100
        r["cash_conversion_%"] = f.cfo / f.ebitda * 100
    # --- per share (adjusted to today's share count: bonus 1:3 in 2021 and 2023)
    r["eps_adj"] = f.pat / shares_now
    payout = f.payout_pct_declared if "payout_pct_declared" in f else pd.Series(np.nan, index=f.index)
    r["payout_%"] = payout
    r["dps_adj"] = r.eps_adj * payout / 100
    r["dividends_declared"] = f.pat * payout / 100
    r["bvps_adj"] = f.equity / shares_now
    r["retention_%"] = 100 - payout
    r["sustainable_growth_%"] = r["roe_%"] * (1 - payout / 100)
    # --- valuation at fiscal year-end (March close, split-adjusted)
    if px is not None and len(px):
        mar = px[px.month.str.endswith("-03")].copy()
        mar["fy"] = mar.month.str[:4].astype(int)
        r = r.merge(mar[["fy", "pg_close"]].rename(columns={"pg_close": "price_mar"}), on="fy", how="left")
        r["mcap"] = r.price_mar * shares_now
        r["pe_x"] = r.price_mar / r.eps_adj
        r["pb_x"] = r.price_mar / r.bvps_adj
        r["div_yield_%"] = r.dps_adj / r.price_mar * 100
        r["ev_ebitda_x"] = (r.mcap + r.net_debt) / r.ebitda
    r["_cash_tax_rate"] = f.cash_tax_rate.values
    r["_total_assets"] = f.total_assets.values
    r["_net_block"] = f.net_block.values
    r["_cwip"] = f.cwip.values
    return r


RATIO_GROUPS = {
    "Profitability": ["ebitda_margin_%", "net_margin_%", "opex_to_revenue_%", "revenue_growth_%",
                      "pat_growth_%", "core_pat_growth_%", "other_income_share_of_pbt_%"],
    "Returns": ["roe_%", "core_roe_%", "roa_%", "roce_pretax_%", "roic_posttax_%", "roic_ex_cwip_%"],
    "Leverage & solvency": ["debt_equity_x", "net_debt_equity_x", "net_debt_ebitda_x", "debt_to_capital_%",
                            "interest_coverage_x", "ebitda_interest_x", "cost_of_debt_pl_%"],
    "Efficiency & asset build": ["asset_turnover_x", "fixed_asset_turnover_x", "cwip_share_%", "receivable_days"],
    "Cash flow": ["cfo", "capex", "fcf", "cfo_to_debt_%", "cash_conversion_%"],
    "Shareholder & valuation": ["eps_adj", "dps_adj", "payout_%", "bvps_adj", "sustainable_growth_%",
                                "pe_x", "pb_x", "div_yield_%", "ev_ebitda_x"],
}

RATIO_DEFS = {
    "ebitda_margin_%": "(Revenue − employee − other expenses) ÷ revenue from operations",
    "net_margin_%": "PAT ÷ revenue from operations",
    "opex_to_revenue_%": "(Employee + other expenses) ÷ revenue",
    "roe_%": "PAT ÷ average shareholders' equity",
    "core_roe_%": "Core PAT (PBT before exceptional items − current tax) ÷ average equity",
    "roa_%": "PAT ÷ average total assets",
    "roce_pretax_%": "EBIT before exceptional items ÷ average (equity + debt)",
    "roic_posttax_%": "EBIT × (1 − cash tax rate) ÷ average (equity + debt)",
    "roic_ex_cwip_%": "Post-tax ROIC on capital that is already earning (excludes capital work-in-progress)",
    "debt_equity_x": "Total borrowings (incl. current maturities) ÷ equity",
    "net_debt_ebitda_x": "(Borrowings − cash & bank balances) ÷ EBITDA",
    "interest_coverage_x": "EBIT before exceptional items ÷ finance costs",
    "asset_turnover_x": "Revenue ÷ average total assets",
    "fixed_asset_turnover_x": "Revenue ÷ average net block (PPE + intangibles)",
    "cwip_share_%": "Capital work-in-progress ÷ (net block + CWIP) – assets not yet earning tariff",
    "receivable_days": "Average trade receivables ÷ revenue × 365",
    "cfo_to_debt_%": "Cash from operations ÷ total borrowings",
    "cash_conversion_%": "Cash from operations ÷ EBITDA",
    "payout_%": "Dividend declared for the year ÷ PAT",
    "sustainable_growth_%": "ROE × retention ratio – equity growth possible without new shares",
    "pe_x": "March-end price ÷ EPS (bonus-adjusted)",
    "ev_ebitda_x": "(Market cap + net debt) ÷ EBITDA",
}


# =============================================================================
# 3. DUPONT
# =============================================================================
def dupont(fin: pd.DataFrame) -> pd.DataFrame:
    f = fin.sort_values("fy").reset_index(drop=True)
    eq = f.equity_capital + f.other_equity
    ta, aeq = _avg(f.total_assets), _avg(eq)
    ebit = f.pbt + f.finance_costs
    d = pd.DataFrame({"fy": f.fy})
    d["tax_burden"] = f.pat / f.pbt                    # incl. regulatory-deferral movement
    d["interest_burden"] = f.pbt / ebit
    d["ebit_margin"] = ebit / f.revenue
    d["asset_turnover"] = f.revenue / ta
    d["equity_multiplier"] = ta / aeq
    d["net_margin"] = f.pat / f.revenue
    d["roe"] = f.pat / aeq
    d["roe_check"] = d.tax_burden * d.interest_burden * d.ebit_margin * d.asset_turnover * d.equity_multiplier
    return d


def roe_bridge(dp: pd.DataFrame, y0: int, y1: int, five_step: bool = True) -> pd.DataFrame:
    """Additive attribution of ROE change to DuPont factors (log-share method, exact)."""
    a, b = dp.set_index("fy").loc[y0], dp.set_index("fy").loc[y1]
    facs = (["tax_burden", "interest_burden", "ebit_margin", "asset_turnover", "equity_multiplier"]
            if five_step else ["net_margin", "asset_turnover", "equity_multiplier"])
    dl = np.log(b.roe / a.roe)
    total = (b.roe - a.roe) * 100
    out = []
    for f_ in facs:
        share = np.log(b[f_] / a[f_]) / dl if dl != 0 else 0
        out.append(dict(factor=f_, value_from=a[f_], value_to=b[f_], contribution_pp=share * total))
    return pd.DataFrame(out)


def pbt_bridge(fin: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    f = fin.set_index("fy")
    a, b = f.loc[y0], f.loc[y1]
    items = [("Revenue from operations", b.revenue - a.revenue),
             ("Other income", b.other_income - a.other_income),
             ("Employee cost", -(b.employee_exp - a.employee_exp)),
             ("Other expenses", -(b.other_exp - a.other_exp)),
             ("Depreciation", -(b.depreciation - a.depreciation)),
             ("Finance costs", -(b.finance_costs - a.finance_costs)),
             ("JV share", (b.jv_share - a.jv_share)),
             ("Exceptional items", (b.exceptional_items - a.exceptional_items))]
    pbt_items = pd.DataFrame(items, columns=["item", "impact"])
    tax_items = pd.DataFrame([("Tax expense", -(b.total_tax - a.total_tax)),
                              ("Regulatory deferral movement", b.rda_net - a.rda_net)],
                             columns=["item", "impact"])
    return pbt_items, tax_items, dict(pbt0=a.pbt, pbt1=b.pbt, pat0=a.pat, pat1=b.pat)


# =============================================================================
# 4. COST OF CAPITAL
# =============================================================================
def _returns(px: pd.DataFrame) -> pd.DataFrame:
    p = px[px.get("partial_month", 0) == 0].copy() if "partial_month" in px else px.copy()
    p = p.sort_values("month")
    r = pd.DataFrame({"month": p.month,
                      "stock": np.log(p.pg_adj_close).diff(),
                      "market": np.log(p.nifty_close).diff()}).dropna()
    return r


def estimate_beta(px: pd.DataFrame, months: int = 60) -> dict:
    r = _returns(px).tail(months)
    x, y = r.market.values, r.stock.values
    b = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    r2 = 1 - resid.var() / y.var()
    return dict(beta=b, alpha_monthly=a, se=se, r2=r2, n=len(r), adj_beta=0.67 * b + 0.33,
                ci_low=b - 1.96 * se, ci_high=b + 1.96 * se, data=r)


def rolling_beta(px: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    r = _returns(px).reset_index(drop=True)
    cov = r.stock.rolling(window).cov(r.market)
    var = r.market.rolling(window).var()
    return pd.DataFrame({"month": r.month, "beta": cov / var}).dropna()


@dataclass
class CapitalInputs:
    rf: float = 7.39            # % 10-year G-sec (nominal, contains sovereign default risk)
    default_spread: float = 1.87  # % India sovereign default spread (Damodaran) removed for a riskless rate
    erp: float = 7.08           # % India total ERP (mature market 4.23% + country risk 2.85%)
    beta: float = 0.88
    kd_spread: float = 0.45     # % over G-sec for new AAA PSU debt
    tax: float = 25.17          # %
    price: float = 250.0
    shares: float = SHARES_NOW
    debt: float = 148009.0
    cash: float = 8920.0
    book_equity: float = 100494.0


def wacc(ci: CapitalInputs) -> dict:
    rf_riskless = ci.rf - ci.default_spread     # avoid double-counting country risk
    ke = rf_riskless + ci.beta * ci.erp
    kd = ci.rf + ci.kd_spread                    # company actually borrows at G-sec + spread
    kd_post = kd * (1 - ci.tax / 100)
    E = ci.price * ci.shares
    D = ci.debt
    we, wd = E / (E + D), D / (E + D)
    return dict(rf_riskless=rf_riskless, ke=ke, kd_pre=kd, kd_post=kd_post, E=E, D=D, we=we, wd=wd,
                wacc=we * ke + wd * kd_post,
                wacc_target=0.30 * ke + 0.70 * kd_post,        # CERC normative 70:30 – hurdle for new regulated projects
                wacc_book=(1 - D / (D + ci.book_equity)) * ke + D / (D + ci.book_equity) * kd_post)


def wacc_sensitivity(ci: CapitalInputs, betas, erps) -> pd.DataFrame:
    rows = {}
    for b in betas:
        rows[f"β {b:.2f}"] = [wacc(CapitalInputs(**{**asdict(ci), "beta": b, "erp": e}))["wacc"] for e in erps]
    return pd.DataFrame(rows, index=[f"ERP {e:.1f}%" for e in erps]).T


def eva_history(ratios: pd.DataFrame, fin: pd.DataFrame, wacc_pct: float, ke_pct: float) -> pd.DataFrame:
    f = fin.sort_values("fy").reset_index(drop=True)
    ce = _avg(f.equity_capital + f.other_equity + f.nc_borrowings + f.c_borrowings
              + f.get("current_maturities_ltd", 0).fillna(0))
    eq = _avg(f.equity_capital + f.other_equity)
    out = pd.DataFrame({"fy": f.fy})
    out["roic_%"] = ratios["roic_posttax_%"].values
    out["roic_ex_cwip_%"] = ratios["roic_ex_cwip_%"].values
    out["wacc_%"] = wacc_pct
    out["capital_employed"] = ce.values
    out["eva"] = (out["roic_%"] - wacc_pct) / 100 * out.capital_employed
    out["roe_%"] = ratios["roe_%"].values
    out["core_roe_%"] = ratios["core_roe_%"].values
    out["ke_%"] = ke_pct
    out["equity_spread_pp"] = out["core_roe_%"] - ke_pct
    out["equity_value_added"] = out.equity_spread_pp / 100 * eq.values
    return out


# =============================================================================
# 5. FORECASTING: revenue driven by capitalisation (regulated asset logic)
# =============================================================================
def revenue_design(r: pd.DataFrame) -> pd.DataFrame:
    """Δrevenue_t = y · capitalisation_{t−1} − e · revenue_{t−1}
    Regulated logic: an asset starts earning tariff once commissioned (full-year effect the
    following year); tariff on existing assets erodes as their loans are repaid and they age."""
    d = pd.DataFrame({"fy": r.fy, "revenue": r.revenue, "cap": r.capitalised})
    d["rev_lag"] = d.revenue.shift(1)
    d["cap_lag"] = d.cap.shift(1)
    d["d_rev"] = d.revenue.diff()
    return d.dropna()


def fit_revenue_model(r: pd.DataFrame, upto: int | None = None) -> dict:
    d = revenue_design(r)
    if upto:
        d = d[d.fy <= upto]
    X = np.column_stack([d.cap_lag, -d.rev_lag])
    coef, *_ = np.linalg.lstsq(X, d.d_rev, rcond=None)
    fitted = X @ coef
    return dict(yield_new=coef[0], erosion=coef[1], n=len(d), fy=d.fy.tolist(),
                r2=1 - ((d.d_rev - fitted) ** 2).sum() / ((d.d_rev - d.d_rev.mean()) ** 2).sum())


def backtest_revenue(r: pd.DataFrame, first_test: int = 2022) -> pd.DataFrame:
    """One-year-ahead forecasts with an expanding window (what a CFO does each budget
    cycle): for each test year, fit only on earlier years, forecast, compare with actual.
    Baselines: flat (no change), last year's growth repeated, 3-year CAGR."""
    rr = r.set_index("fy")
    rows = []
    for y in [y for y in rr.index if y >= first_test]:
        m = fit_revenue_model(r[r.fy < y])
        prev, prev2, prev4 = rr.revenue.get(y - 1), rr.revenue.get(y - 2), rr.revenue.get(y - 4)
        rows.append(dict(
            fy=y, actual=rr.revenue[y],
            driver_model=prev + m["yield_new"] * rr.capitalised[y - 1] - m["erosion"] * prev,
            naive_flat=prev, naive_last_growth=prev * prev / prev2,
            naive_cagr3=prev * (prev / prev4) ** (1 / 3) if prev4 else np.nan))
    bt = pd.DataFrame(rows)
    models = ["driver_model", "naive_flat", "naive_last_growth", "naive_cagr3"]
    for c in models:
        bt[f"{c}_err_%"] = (bt[c] / bt.actual - 1) * 100
    bt.attrs["mape"] = {c: bt[f"{c}_err_%"].abs().mean() for c in models}
    bt.attrs["model"] = fit_revenue_model(r)
    return bt


# =============================================================================
# 6. PROJECTION ENGINE (vectorised: every input may be scalar or numpy array)
# =============================================================================
@dataclass
class Scenario:
    name: str = "Expected"
    capex: tuple = (37000, 40000, 40000, 38000, 35000)      # FY27..FY31
    cap_rate: float = 0.37         # share of (opening CWIP + capex) capitalised each year
    yield_new: float = 0.166       # revenue per ₹ of assets capitalised last year (fitted FY18-26)
    erosion: float = 0.015         # annual erosion of revenue on existing assets (fitted)
    opex_pct: float = 15.0         # (employee + other expenses) % of revenue
    other_income: float = 1200.0
    dep_existing_decay: float = 1.0  # % p.a. decline in depreciation on the FY26 asset base
    dep_rate_new: float = 5.28     # % straight-line on newly capitalised assets (CERC rate, first 12 years)
    kd_new: float = 7.84           # % pre-tax on new borrowing
    kd_old: float = 6.84           # % paid on existing borrowing
    repay_rate: float = 12.0       # % of existing debt repaid each year
    idc_share_fy26: float = 11.4   # % of interest capitalised (IDC) at FY26 CWIP level
    tax_rate: float = 18.5         # economic (current) tax % of PBT
    receivable_days: float = 80.0
    leverage_penalty_bp: float = 50.0   # extra bp on new debt if ND/EBITDA above threshold
    nd_ebitda_threshold: float = 4.5


@dataclass
class Policy:
    name: str = "Status quo (53% payout)"
    payout: float = 53.0           # % of PAT
    monetise_amt: float = 0.0      # ₹ Cr of operating assets sold to InvIT
    monetise_year: int = 2028
    monetise_ev_ebitda: float = 9.0
    equity_raise: float = 0.0      # ₹ Cr (constrained: GoI must stay ≥ 51%)
    trigger_nd: float = 0.0        # dynamic rule: if last year's ND/EBITDA > trigger, use payout_stressed (0 = off)
    payout_stressed: float = 40.0
    capex_scale: float = 1.0       # 0.9 = bid more selectively, 10% less capex than the base plan


@dataclass
class Base:
    fy: int = 2026
    revenue: float = 46732.87
    net_block: float = 176623.0
    cwip: float = 43654.29
    debt: float = 148009.01
    cash: float = 8920.41
    equity: float = 100494.03
    receivables: float = 11672.59
    capitalised_last: float = 17016.0
    depreciation: float = 13029.68
    shares: float = SHARES_NOW


def base_from(fin: pd.DataFrame, ratios: pd.DataFrame) -> Base:
    f = fin.sort_values("fy").iloc[-1]
    r = ratios.sort_values("fy").iloc[-1]
    return Base(fy=int(f.fy), revenue=f.revenue, net_block=f.ppe + (f.intangibles or 0), cwip=f.cwip,
                debt=r.debt, cash=(f.cash or 0) + (f.other_bank_balances or 0),
                equity=f.equity_capital + f.other_equity,
                receivables=f.trade_receivables if pd.notna(f.trade_receivables) else 0,
                capitalised_last=r.capitalised if pd.notna(r.capitalised) else 0,
                depreciation=f.depreciation)


def project(b: Base, s: Scenario, p: Policy, wacc_pct: float, ke_pct: float,
            n: int = 1, shocks: dict | None = None) -> dict:
    """Year-by-year projection. With `shocks` (arrays of length n) it runs n Monte-Carlo paths."""
    sh = shocks or {}
    one = np.ones(n)
    capex_mult = sh.get("capex_mult", one)
    cap_rate = s.cap_rate * sh.get("cap_mult", one)
    y_new = s.yield_new * sh.get("yield_mult", one)
    opex_pct = s.opex_pct + sh.get("opex_add", 0 * one)
    kd_new = s.kd_new + sh.get("kd_add", 0 * one)
    rdays = s.receivable_days + sh.get("rdays_add", 0 * one)

    net_block, cwip = b.net_block * one, b.cwip * one
    old_debt, new_debt = b.debt * one, 0 * one
    cash, equity, rec, rev = b.cash * one, b.equity * one, b.receivables * one, b.revenue * one
    cap_prev = b.capitalised_last * one
    dep_old, cum_new = b.depreciation * one, 0 * one
    kd_new_eff = kd_new.copy()
    nd_prev = (b.debt - b.cash) / (b.revenue * (1 - s.opex_pct / 100)) * one
    years = list(range(b.fy + 1, b.fy + 1 + len(s.capex)))
    out = {k: [] for k in ["fy", "capex", "capitalised", "cwip", "net_block", "revenue", "ebitda",
                           "depreciation", "interest", "pbt", "pat", "dividends", "div_to_govt",
                           "debt", "cash", "net_debt", "equity", "cfo", "new_borrowing",
                           "de", "nd_ebitda", "icr", "roe", "roce_post", "eva", "fcff", "fcfe", "eps", "dps",
                           "kd_marginal", "payout"]}
    for i, y in enumerate(years):
        capex = s.capex[i] * capex_mult * p.capex_scale
        payout = np.where((p.trigger_nd > 0) & (nd_prev > p.trigger_nd), p.payout_stressed, p.payout) * one
        cap = np.minimum(cap_rate * (cwip + capex), cwip + capex)
        cwip = cwip + capex - cap
        dep_old = dep_old * (1 - s.dep_existing_decay / 100)
        dep = dep_old + s.dep_rate_new / 100 * (cum_new + 0.5 * cap)
        cum_new = cum_new + cap
        # monetisation: sell operating assets to InvIT
        mon = p.monetise_amt if (p.monetise_amt and y == p.monetise_year) else 0.0
        lost_ebitda = mon / p.monetise_ev_ebitda if mon else 0.0
        lost_rev = lost_ebitda / 0.88 if mon else 0.0
        nb_removed = mon * 0.8 if mon else 0.0
        rev = rev + y_new * cap_prev - s.erosion * rev - lost_rev
        net_block = net_block + cap - dep - nb_removed
        ebitda = rev * (1 - opex_pct / 100)
        # debt dynamics
        repay = s.repay_rate / 100 * old_debt
        cwip_share = cwip / (net_block + cwip)
        idc = s.idc_share_fy26 / 100 * cwip_share / 0.198
        interest_paid_est = (s.kd_old / 100 * (old_debt - repay / 2)
                             + kd_new_eff / 100 * new_debt)  # new borrowing of the year added below
        rec_new = rev * rdays / 365
        d_rec = rec_new - rec
        rec = rec_new
        # first pass PAT to size borrowing (interest on this year's borrowing at half-year)
        ebit = ebitda - dep + s.other_income
        pre_int = interest_paid_est * (1 - idc)
        pbt0 = ebit - pre_int
        pat0 = pbt0 * (1 - s.tax_rate / 100)
        div = np.maximum(payout / 100 * pat0, 0)
        cfo = pat0 + dep - d_rec
        need = capex + div + repay - cfo - mon - p.equity_raise
        borrow = np.maximum(need, 0)
        surplus = np.maximum(-need, 0)
        interest = (interest_paid_est + kd_new_eff / 100 * borrow / 2) * (1 - idc)
        pbt = ebit - interest
        pat = pbt * (1 - s.tax_rate / 100)
        div = np.maximum(payout / 100 * pat, 0)
        old_debt = old_debt - repay
        new_debt = new_debt + borrow
        debt = old_debt + new_debt
        cash = cash + surplus
        gain_on_sale = mon - nb_removed
        equity = equity + pat - div + p.equity_raise + gain_on_sale * (1 - s.tax_rate / 100)
        nd = debt - cash
        nd_ebitda = nd / ebitda
        nd_prev = nd_ebitda
        # leverage penalty raises the marginal rate on future borrowing
        kd_new_eff = kd_new + np.where(nd_ebitda > s.nd_ebitda_threshold, s.leverage_penalty_bp / 100, 0)
        ce = debt + equity
        nopat = (ebit) * (1 - s.tax_rate / 100)
        fcff = nopat + dep - capex - d_rec
        fcfe = pat + dep - capex - d_rec + borrow - repay
        vals = dict(fy=y * one, capex=capex, capitalised=cap, cwip=cwip, net_block=net_block, revenue=rev,
                    ebitda=ebitda, depreciation=dep, interest=interest, pbt=pbt, pat=pat, dividends=div,
                    div_to_govt=div * 0.5134, debt=debt, cash=cash, net_debt=nd, equity=equity,
                    cfo=pat + dep - d_rec, new_borrowing=borrow, de=debt / equity, nd_ebitda=nd_ebitda,
                    icr=ebit / interest, roe=pat / (equity - (pat - div) / 2) * 100,
                    roce_post=nopat / ce * 100, eva=nopat - wacc_pct / 100 * ce,
                    fcff=fcff, fcfe=fcfe, eps=pat / b.shares, dps=div / b.shares,
                    kd_marginal=kd_new_eff, payout=payout)
        for k, v in vals.items():
            out[k].append(v)
        cap_prev = cap
    return {k: np.vstack(v) for k, v in out.items()}   # shape (years, n)


def to_frame(res: dict, path: int = 0) -> pd.DataFrame:
    return pd.DataFrame({k: v[:, path] for k, v in res.items()}).assign(fy=lambda d: d.fy.astype(int))


def valuation(res: dict, b: Base, ke_pct: float, g_ri: float = 3.0, g_div: float = 6.0,
              path: int | None = None) -> dict:
    """Equity value per share.
    * Residual income (RI): V0 = BV0 + Σ (PAT_t − Ke·BV_{t−1})/(1+Ke)^t + terminal RI growing at g_ri.
      Preferred for a regulated utility: tariff = return on book equity, so value hinges on ROE vs Ke.
      (FCFF-DCF is not used: during a capex build-out free cash flow is negative and the
       answer is dominated by terminal assumptions.)
    * Dividend discount (DDM): explicit dividends + Gordon terminal at g_div – the Government's lens."""
    T = res["fy"].shape[0]
    t = np.arange(1, T + 1)[:, None]
    ke, gr, gd = ke_pct / 100, g_ri / 100, g_div / 100
    disc = (1 + ke) ** t
    bv = res["equity"]
    bv_prev = np.vstack([np.full((1, bv.shape[1]), b.equity), bv[:-1]])
    ri = res["pat"] - ke * bv_prev
    tv_ri = ri[-1] * (1 + gr) / (ke - gr)
    v_ri = b.equity + (ri / disc).sum(axis=0) + tv_ri / disc[-1]
    tv_d = res["dividends"][-1] * (1 + gd) / (ke - gd)
    v_ddm = (res["dividends"] / disc).sum(axis=0) + tv_d / disc[-1]
    out = dict(ri_per_share=v_ri / b.shares, ddm_per_share=v_ddm / b.shares,
               pv_ri=(ri / disc).sum(axis=0), tv_ri_share=(tv_ri / disc[-1]) / v_ri,
               justified_pb=v_ri / b.equity)
    if path is not None:
        out = {k: float(np.asarray(v).ravel()[path]) for k, v in out.items()}
    return out


def implied_ke(res: dict, b: Base, price: float, g_ri: float = 3.0, lo=5.0, hi=25.0) -> float:
    """Cost of equity at which the residual-income value equals today's share price."""
    for _ in range(60):
        mid = (lo + hi) / 2
        v = valuation(res, b, mid, g_ri, path=0)["ri_per_share"]
        lo, hi = (mid, hi) if v > price else (lo, mid)
    return (lo + hi) / 2


SHOCK_SD = dict(capex_mult=0.10, cap_mult=0.15, yield_mult=0.10, opex_add=1.5, kd_add=0.50, rdays_add=12)


def draw_shocks(n: int, seed: int = 7, sd: dict | None = None) -> dict:
    sd = {**SHOCK_SD, **(sd or {})}
    rng = np.random.default_rng(seed)
    return dict(capex_mult=np.clip(rng.normal(1, sd["capex_mult"], n), 0.6, 1.5),
                cap_mult=np.clip(rng.normal(1, sd["cap_mult"], n), 0.5, 1.6),
                yield_mult=np.clip(rng.normal(1, sd["yield_mult"], n), 0.6, 1.4),
                opex_add=rng.normal(0, sd["opex_add"], n),
                kd_add=rng.normal(0, sd["kd_add"], n),
                rdays_add=rng.normal(0, sd["rdays_add"], n))


def monte_carlo(b: Base, s: Scenario, p: Policy, wacc_pct, ke_pct, n=2000, seed=7,
                de_limit=2.0, icr_floor=2.5) -> dict:
    sh = draw_shocks(n, seed)
    res = project(b, s, p, wacc_pct, ke_pct, n=n, shocks=sh)
    breach_nd = (res["nd_ebitda"] > s.nd_ebitda_threshold).any(axis=0)
    breach_de = (res["de"] > de_limit).any(axis=0)
    breach_icr = (res["icr"] < icr_floor).any(axis=0)
    roe_below_ke = res["roe"][-1] < ke_pct
    val = valuation(res, b, ke_pct)
    return dict(res=res, p_breach_nd=breach_nd.mean(), p_breach_de=breach_de.mean(),
                p_breach_icr=breach_icr.mean(), p_any_breach=(breach_nd | breach_de | breach_icr).mean(),
                p_roe_below_ke=roe_below_ke.mean(), ri=val["ri_per_share"], ddm=val["ddm_per_share"])


def max_sustainable_payout(b, s, p, wacc_pct, ke_pct, tolerance=0.20, n=1000,
                           grid=range(30, 81, 5), de_limit=2.0, icr_floor=2.5) -> pd.DataFrame:
    rows = []
    for po in grid:
        mc = monte_carlo(b, s, Policy(**{**asdict(p), "payout": po}), wacc_pct, ke_pct, n=n,
                         de_limit=de_limit, icr_floor=icr_floor)
        res = mc["res"]
        rows.append(dict(payout=po, p_any_breach=mc["p_any_breach"] * 100,
                         p_breach_nd=mc["p_breach_nd"] * 100,
                         median_nd_ebitda_peak=np.median(res["nd_ebitda"].max(axis=0)),
                         median_de_fy_end=np.median(res["de"][-1]),
                         cum_div_govt=np.median(res["div_to_govt"].sum(axis=0)),
                         median_ri_value=np.median(mc["ri"]), median_ddm=np.median(mc["ddm"]),
                         ok=mc["p_any_breach"] <= tolerance))
    return pd.DataFrame(rows)


SCENARIOS = {
    "Pessimistic": Scenario(name="Pessimistic", capex=(42000, 45000, 45000, 42000, 40000), cap_rate=0.30,
                            yield_new=0.145, erosion=0.025, opex_pct=17.0, kd_new=8.60, tax_rate=21.0,
                            receivable_days=95),
    "Expected": Scenario(),
    "Optimistic": Scenario(name="Optimistic", capex=(37000, 38000, 38000, 36000, 34000), cap_rate=0.42,
                           yield_new=0.18, erosion=0.010, opex_pct=14.0, kd_new=7.30, tax_rate=18.0,
                           receivable_days=65),
}

POLICIES = {
    "A · Status quo: 53% payout": Policy(name="A · Status quo: 53% payout", payout=53),
    "B · Retain more: 40% payout": Policy(name="B · Retain more: 40% payout", payout=40),
    "C · Higher payout: 65%": Policy(name="C · Higher payout: 65%", payout=65),
    "D · 53% + ₹15,000 Cr InvIT sale (FY28)": Policy(
        name="D · 53% + ₹15,000 Cr InvIT sale (FY28)", payout=53, monetise_amt=15000, monetise_year=2028),
    "E · Dynamic 55%→40% if ND/EBITDA>3.9x + ₹10,000 Cr InvIT": Policy(
        name="E · Dynamic 55%→40% if ND/EBITDA>3.9x + ₹10,000 Cr InvIT", payout=55, trigger_nd=3.9,
        payout_stressed=40, monetise_amt=10000, monetise_year=2028),
    "F · Selective bidding: 53% payout, capex −10%": Policy(
        name="F · Selective bidding: 53% payout, capex −10%", payout=53, capex_scale=0.9),
    "G · Recommended: dynamic 60%→40% (trigger 3.8x) + selective bidding": Policy(
        name="G · Recommended: dynamic 60%→40% (trigger 3.8x) + selective bidding", payout=60,
        trigger_nd=3.8, payout_stressed=40, capex_scale=0.9),
}


def policy_summary(b, s, p, wacc_pct, ke_pct, n_mc=1000, de_limit=2.0, icr_floor=2.5) -> dict:
    det = project(b, s, p, wacc_pct, ke_pct)
    df = to_frame(det)
    v = valuation(det, b, ke_pct, path=0)
    mc = monte_carlo(b, s, p, wacc_pct, ke_pct, n=n_mc, de_limit=de_limit, icr_floor=icr_floor)
    last = df.iloc[-1]
    return dict(policy=p.name, scenario=s.name,
                peak_nd_ebitda=df.nd_ebitda.max(), de_fy31=last.de, icr_min=df.icr.min(),
                roe_fy31=last.roe, eps_fy31=last.eps, dps_avg=df.dps.mean(),
                cum_dividends=df.dividends.sum(), cum_div_govt=df.div_to_govt.sum(),
                debt_fy31=last.debt, cum_eva=df.eva.sum(),
                ri_per_share=v["ri_per_share"], ddm_per_share=v["ddm_per_share"],
                p_any_breach=mc["p_any_breach"] * 100, p_roe_below_ke=mc["p_roe_below_ke"] * 100,
                path=df)


def calibrate_capitalisation(ratios: pd.DataFrame) -> pd.DataFrame:
    r = ratios.sort_values("fy").reset_index(drop=True)
    cw = r["_cwip"]
    d = pd.DataFrame({"fy": r.fy, "capex": r.capex_derived, "capitalised": r.capitalised,
                      "opening_cwip": cw.shift(1)})
    d["cap_rate"] = d.capitalised / (d.opening_cwip + d.capex)
    return d.dropna()


def fmt_cr(x):
    return "–" if pd.isna(x) else f"₹{x:,.0f} Cr"


def fmt_lcr(x):
    return "–" if pd.isna(x) else f"₹{x/1e5:,.2f} lakh Cr"
