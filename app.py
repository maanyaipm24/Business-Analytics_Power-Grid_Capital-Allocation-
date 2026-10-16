"""
POWERGRID Capital-Allocation Decision App
Finance Analytics project · Business Analytics course

Managerial decision supported
  "How should POWERGRID fund its ₹1.75 lakh Cr transmission build-out over FY27–FY31 –
   what dividend payout, borrowing and asset-monetisation policy keeps credit metrics safe,
   keeps ROE above the cost of equity, and still pays the Government a dependable dividend?"

Run:  streamlit run app.py
"""
from dataclasses import asdict, replace
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import finance_engine as F

st.set_page_config(page_title="POWERGRID Capital Allocation", page_icon="⚡", layout="wide")

PAGES = ["1 · Business overview", "2 · Data input & sources", "3 · Data quality",
         "4 · Financial performance", "5 · Diagnostics: DuPont & drivers", "6 · Cost of capital (WACC)",
         "7 · Forecast & risk", "8 · Scenario & policy lab", "9 · Recommendations & monitoring",
         "10 · Report & downloads"]
T = "plotly_white"
C = dict(blue="#1F5A96", teal="#2A9D8F", orange="#E07B39", red="#C8423B", grey="#8A8F98",
         purple="#6C4BA8", gold="#D9A520", green="#2E8B57")
SEG_COL = {"Transmission": C["blue"], "Consultancy": C["orange"], "Telecom": C["teal"]}


def cr(x, d=0):
    return "–" if x is None or pd.isna(x) else f"₹{x:,.{d}f} Cr"


def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}%"


def xf(x, d=2):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}x"


def fig_layout(fig, h=380, title=None, **kw):
    fig.update_layout(template=T, height=h, margin=dict(t=90 if title else 50, b=20, l=10, r=10),
                      title=dict(text=title, y=0.98, yanchor="top", font=dict(size=15)) if title else None,
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, font=dict(size=11)), **kw)
    return fig


def dcr(x):
    return f"{'-' if x < 0 else '+'}₹{abs(x):,.0f} Cr"


# =============================================================================
# DATA (cached)
# =============================================================================
@st.cache_data
def bundled():
    return F.load_bundled()


@st.cache_data(show_spinner="Fetching live prices from Yahoo Finance…")
def fetch_live_prices():
    import yfinance as yf  # optional dependency
    pg = yf.download("POWERGRID.NS", period="7y", interval="1mo", auto_adjust=False, progress=False)
    nf = yf.download("^NSEI", period="7y", interval="1mo", auto_adjust=False, progress=False)
    if pg.empty or nf.empty:
        raise RuntimeError("empty download")
    pg.columns = [c[0] if isinstance(c, tuple) else c for c in pg.columns]
    nf.columns = [c[0] if isinstance(c, tuple) else c for c in nf.columns]
    df = pd.DataFrame({"pg_close": pg["Close"], "pg_adj_close": pg["Adj Close"]}).join(
        nf["Close"].rename("nifty_close"), how="inner").dropna()
    df["month"] = df.index.strftime("%Y-%m")
    df["partial_month"] = 0
    df.iloc[-1, df.columns.get_loc("partial_month")] = 1
    return df.reset_index(drop=True)[["month", "pg_close", "pg_adj_close", "nifty_close", "partial_month"]]


B = bundled()
st.sidebar.title("⚡ POWERGRID\nCapital-Allocation App")
page = st.sidebar.radio("Workflow", PAGES, label_visibility="collapsed")
st.sidebar.divider()

# ---- data source --------------------------------------------------------------
src = st.sidebar.radio("Financial statements",
                       ["Bundled: POWERGRID audited filings FY17–FY26", "Upload statements (template CSV)"])
fin, upload_msgs = B["fin"].copy(), []
if src.startswith("Upload"):
    up = st.sidebar.file_uploader("Statements CSV (same columns as template)", type="csv")
    if up is not None:
        df, upload_msgs = F.validate_upload(pd.read_csv(up))
        if df is None:
            st.sidebar.error(upload_msgs[0] + " – using bundled data.")
        else:
            fin = df
    else:
        st.sidebar.info("No file yet – using bundled data. Download the template on page 2.")

px_src = st.sidebar.radio("Market prices", ["Bundled monthly prices (to Oct-2026)", "Fetch live (Yahoo Finance)",
                                            "Upload prices CSV"])
prices, price_note = B["px"].copy(), "Bundled: Yahoo Finance monthly history retrieved 15-Oct-2026"
if px_src.startswith("Fetch"):
    try:
        prices = fetch_live_prices()
        price_note = f"Live: Yahoo Finance, fetched {datetime.now():%d %b %Y %H:%M}"
    except Exception as e:  # no internet / package missing
        st.sidebar.warning(f"Live fetch unavailable ({type(e).__name__}); using bundled prices.")
elif px_src.startswith("Upload"):
    upp = st.sidebar.file_uploader("Prices CSV: month, pg_close, pg_adj_close, nifty_close", type="csv")
    if upp is not None:
        prices = pd.read_csv(upp)
        price_note = "Uploaded price file"

# ---- filters -----------------------------------------------------------------
st.sidebar.divider()
st.sidebar.caption("Filters (performance & diagnostic pages)")
fy_min, fy_max = int(fin.fy.min()), int(fin.fy.max())
fy_range = st.sidebar.slider("Fiscal years", fy_min, fy_max, (fy_min, fy_max))
segs = st.sidebar.multiselect("Business segments", ["Transmission", "Consultancy", "Telecom"],
                              default=["Transmission", "Consultancy", "Telecom"])
peers_sel = st.sidebar.multiselect("Peers for benchmarking", ["NTPC", "NHPC", "Tata Power"],
                                   default=["NTPC", "NHPC", "Tata Power"])
st.sidebar.caption(f"Statements refreshed: {datetime.now():%d %b %Y} · latest year FY{fy_max}\n\n{price_note}")

# ---- core computations ---------------------------------------------------------
R = F.build_ratios(fin, prices)
DP = F.dupont(fin)
A = B["assum"].set_index("key")["value"].to_dict()
Rv = R[(R.fy >= fy_range[0]) & (R.fy <= fy_range[1])]
last, prev = R.iloc[-1], R.iloc[-2]

if "cap_inputs" not in st.session_state:
    be0 = F.estimate_beta(prices, 60)
    st.session_state.cap_inputs = dict(rf=A["risk_free_pct"], default_spread=1.87, erp=A["india_erp_pct"],
                                       beta=round(float(be0["beta"]), 3), kd_spread=A["aaa_psu_spread_pct"],
                                       tax=A["tax_rate_pct"], price=A["share_price"])
ci_d = st.session_state.cap_inputs
CI = F.CapitalInputs(**ci_d, shares=A["shares_cr"], debt=float(last.debt),
                     cash=float(fin.iloc[-1].cash + fin.iloc[-1].other_bank_balances),
                     book_equity=float(last.equity))
W = F.wacc(CI)
BASE = F.base_from(fin, R)

if "limits" not in st.session_state:
    st.session_state.limits = dict(nd_ebitda=4.5, de=2.0, icr=2.5, tolerance=10)
LIM = st.session_state.limits

if "scen_tbl" not in st.session_state:
    rows = {}
    for n, s in F.SCENARIOS.items():
        d = asdict(s)
        rows[n] = {**{f"Capex FY{27 + i} (₹ Cr)": v for i, v in enumerate(d["capex"])},
                   "Capitalisation rate (share of CWIP+capex)": d["cap_rate"],
                   "Revenue yield on new assets (₹ per ₹)": d["yield_new"],
                   "Revenue erosion on old assets (%/yr)": d["erosion"] * 100,
                   "Opex % of revenue": d["opex_pct"], "Other income (₹ Cr)": d["other_income"],
                   "Marginal cost of new debt (%)": d["kd_new"], "Economic tax rate (%)": d["tax_rate"],
                   "Receivable days": d["receivable_days"]}
    st.session_state.scen_tbl = pd.DataFrame(rows)


def scenarios_from_table(tbl: pd.DataFrame) -> dict:
    out = {}
    for n in tbl.columns:
        c = tbl[n]
        out[n] = F.Scenario(name=n, capex=tuple(float(c[f"Capex FY{27 + i} (₹ Cr)"]) for i in range(5)),
                            cap_rate=float(c["Capitalisation rate (share of CWIP+capex)"]),
                            yield_new=float(c["Revenue yield on new assets (₹ per ₹)"]),
                            erosion=float(c["Revenue erosion on old assets (%/yr)"]) / 100,
                            opex_pct=float(c["Opex % of revenue"]), other_income=float(c["Other income (₹ Cr)"]),
                            kd_new=float(c["Marginal cost of new debt (%)"]),
                            tax_rate=float(c["Economic tax rate (%)"]),
                            receivable_days=float(c["Receivable days"]),
                            nd_ebitda_threshold=LIM["nd_ebitda"])
    return out


SCEN = scenarios_from_table(st.session_state.scen_tbl)

if "pol_tbl" not in st.session_state:
    st.session_state.pol_tbl = pd.DataFrame([dict(
        Policy=p.name, **{"Payout %": p.payout, "InvIT sale ₹ Cr": p.monetise_amt, "Sale year": p.monetise_year,
                          "Trigger ND/EBITDA (0=off)": p.trigger_nd, "Stressed payout %": p.payout_stressed,
                          "Capex scale": p.capex_scale}) for p in F.POLICIES.values()])


def policies_from_table(tbl):
    return {r.Policy: F.Policy(name=r.Policy, payout=float(r["Payout %"]), monetise_amt=float(r["InvIT sale ₹ Cr"]),
                               monetise_year=int(r["Sale year"]), trigger_nd=float(r["Trigger ND/EBITDA (0=off)"]),
                               payout_stressed=float(r["Stressed payout %"]), capex_scale=float(r["Capex scale"]))
            for _, r in tbl.iterrows()}


POL = policies_from_table(st.session_state.pol_tbl)


@st.cache_data(show_spinner="Running scenarios and 1,000 Monte-Carlo paths per policy…")
def run_grid(scen_items, pol_items, wacc_pct, ke_pct, base_tuple, de_lim, icr_floor):
    base = F.Base(*base_tuple)
    rows, paths = [], {}
    for sn, s in scen_items:
        for pn, p in pol_items:
            ps = F.policy_summary(base, s, p, wacc_pct, ke_pct, n_mc=1000, de_limit=de_lim, icr_floor=icr_floor)
            paths[(sn, pn)] = ps.pop("path")
            rows.append(ps)
    return pd.DataFrame(rows), paths


def grid():
    return run_grid(tuple(SCEN.items()), tuple(POL.items()), W["wacc"], W["ke"],
                    tuple(asdict(BASE).values()), LIM["de"], LIM["icr"])


# =============================================================================
# 1. BUSINESS OVERVIEW
# =============================================================================
if page == PAGES[0]:
    st.title("POWERGRID: funding the transmission build-out")
    st.caption("Power Grid Corporation of India Ltd · Maharatna CPSE · NSE: POWERGRID · "
               "India's largest transmission utility · 51.34% owned by the Government of India")
    k = st.columns(5)
    k[0].metric("FY26 revenue", cr(last.revenue), f"{last['revenue_growth_%']:+.1f}% YoY")
    k[1].metric("FY26 PAT (reported)", cr(last.pat), f"{last['pat_growth_%']:+.1f}% YoY")
    k[2].metric("Core PAT (ex one-offs)", cr(last.core_pat), f"{last['core_pat_growth_%']:+.1f}% YoY",
                delta_color="normal")
    k[3].metric("FY26 capex (cash flow)", cr(last.capex), f"{last.capex / prev.capex - 1:+.0%} vs FY25")
    k[4].metric("Core ROE vs cost of equity", pct(last["core_roe_%"]), f"spread {last['core_roe_%'] - W['ke']:+.1f} pp",
                delta_color="normal")

    st.subheader("Why this company, why now")
    st.markdown(f"""
* **Stable and well established:** a profit every year, about 99.8% system availability, and regulated
  returns set by CERC. Most coverage tracks quarterly earnings. Few studies look at its **capital-allocation
  choices**, which this app is about.
* **Highly topical:** India's 500 GW non-fossil target needs huge transmission investment. POWERGRID's
  capex has **tripled in two years** (₹11,410 Cr in FY24 to {cr(last.capex)} in FY26). Works in hand are
  ~₹1.75 lakh Cr, of which 83% is won in competitive bidding (TBCB), not cost-plus regulated (RTM) projects.
* **Rates are rising:** the 10-year G-sec is at **{A['risk_free_pct']}%**, a 3-year high, and the RBI has
  started hiking. Borrowings rose **{cr(last.debt - prev.debt)}** in FY26 alone. The share has fallen from
  ₹318 (Apr-26) to ₹250.
""")
    st.subheader("The managerial decision")
    st.info("**How should POWERGRID fund its FY27–FY31 capex: what dividend payout, borrowing and "
            "asset-monetisation policy keeps credit metrics inside comfort limits, keeps ROE above the cost of "
            "equity, and still gives the Government (51%) a dependable dividend?**")
    a, b = st.columns(2)
    with a:
        st.markdown("""
**Target users**
* **Director (Finance) / CFO**: recommends dividend and borrowing plans to the Board
* **Board of Directors / Audit Committee**: approves payout, borrowing limits and InvIT monetisation
* **Ministry of Power & DIPAM**: the majority shareholder and dividend recipient
* **Credit-rating agencies, lenders and bondholders**: watch leverage and interest coverage
* **Institutional investors**: price the stock as a dividend yield play

**How it is decided today**
The payout follows DIPAM's CPSE guideline (at least 30% of PAT or 4% of net worth) and the Government's
budget needs. Borrowing fills whatever capex is left. The two are decided **separately, year by year**,
without a forward view of leverage, rating headroom or value creation.
""")
    with b:
        st.markdown(f"""
**What a wrong decision costs**
* **Payout too high in a capex peak:** leverage climbs. A one-notch rating slip adds about 25–50 bp on
  new borrowing. On ₹25–30k Cr of new debt a year, that is about **₹75–150 Cr of extra interest each
  year, and it compounds**.
* **Payout too low:** the Government's dividend falls (each 10 pp of payout ≈ ₹800 Cr/yr to GoI). The stock
  is held for its yield, so investors re-price it.
* **Bidding for returns below the cost of equity:** TBCB projects without a regulated 15.5% RoE can
  **destroy value** even while profit grows.
* **Equity issue is not a free lever:** GoI must hold ≥51% and already holds 51.34%. Retained earnings
  and InvIT monetisation are the only real sources of equity.
""")
    st.subheader("Finance process: how a transmission rupee flows")
    st.markdown("""
`Project awarded (RTM by CERC / TBCB by bidding)` → `Capex funded ~70% debt : 30% equity` →
`Capital work-in-progress (earns nothing, interest capitalised)` → `Commissioning = capitalisation` →
`Tariff starts: RoE on equity + interest + depreciation + O&M` → `PAT` → `Dividend vs retention` →
`Retained equity funds the next 30% equity share`
""")
    st.subheader("Trade-offs between actions")
    st.table(pd.DataFrame([
        ["Higher dividend payout", "Higher GoI dividend and yield support for the stock", "More debt, weaker ratios, less equity for growth"],
        ["Lower payout / retain", "Stronger balance sheet, room for capex", "GoI dividend falls; retained equity is costlier than debt"],
        ["More borrowing", "Cheapest source after tax (~5.9%)", "Leverage and rating risk, sensitive to rising rates"],
        ["InvIT monetisation", "Upfront cash, cuts debt, recycles capital", "Gives up future EBITDA; valuation depends on market"],
        ["Selective TBCB bidding", "Protects returns above the cost of equity", "Slower growth, loses market share to private players"],
    ], columns=["Lever", "Benefit", "Cost / risk"]))
    st.subheader("KPI definitions and targets")
    st.dataframe(pd.DataFrame([
        ["Core ROE", "PBT before exceptional items − current tax, ÷ average equity", "> cost of equity (≈11.8%)"],
        ["ROE − Ke spread", "Core ROE minus CAPM cost of equity", "> +3 pp"],
        ["ROIC vs WACC (EVA)", "Post-tax operating return on capital minus WACC", "≥ 0"],
        ["Net debt / EBITDA", "(Borrowings − cash) ÷ EBITDA", f"≤ {LIM['nd_ebitda']}x (comfort limit, editable)"],
        ["Debt / equity", "Borrowings ÷ net worth", f"≤ {LIM['de']}x (normative ceiling 2.33x = 70:30)"],
        ["Interest coverage", "EBIT ÷ finance costs", f"≥ {LIM['icr']}x"],
        ["Payout ratio", "Dividend ÷ PAT", "≥ 30% (DIPAM floor)"],
        ["P(breach)", "Monte-Carlo probability that any limit is breached in FY27–31", f"≤ {LIM['tolerance']}%"],
        ["Value per share", "Residual-income value: book value + PV of (PAT − Ke × book value)", "maximise"],
    ], columns=["KPI", "Definition", "Target"]), hide_index=True, width="stretch")
    st.caption("Use the sidebar to move through the workflow. Every number traces back to the audited "
               "filings listed on page 2.")


# =============================================================================
# 2. DATA INPUT & SOURCES
# =============================================================================
elif page == PAGES[1]:
    st.title("Data input & sources")
    st.markdown("""
**Primary source:** POWERGRID's **audited consolidated financial results**, filed with BSE and NSE under
**SEBI (LODR) Regulation 33** and published on the company's investor page
([powergrid.in/en/annual-results](https://www.powergrid.in/en/annual-results)). Each year uses the latest
(restated) version of that year.
""")
    for m in upload_msgs:
        st.info(m)
    src_tbl = pd.DataFrame([
        ["Financial statements (P&L, balance sheet, cash flow)", "Audited results filed under SEBI LODR Reg. 33", "Company-year", "FY2017–FY2026", "Primary"],
        ["Segment information (Transmission, Consultancy, Telecom)", "Same filings, segment note", "Segment-year", "FY2022–FY2026", "Primary"],
        ["Current maturities of debt FY17–21; cash flows FY17–19", "Annual-report notes (compiled by Screener.in)", "Company-year", "FY2017–FY2021", "Secondary, flagged"],
        ["Share price & NIFTY 50 (monthly)", "Yahoo Finance (NSE data); live fetch available", "Month", "Oct-2019 → Oct-2026", "Market"],
        ["10-yr G-sec yield, RBI repo", "Trading Economics, 15-Oct-2026", "Point", "Current", "Market"],
        ["Equity risk premium, sovereign default spread", "Damodaran, Jan-2026 country risk update", "Point", "Current", "Academic"],
        ["Capex guidance, works in hand", "Company Q1 FY27 call (via ICICI Direct note)", "Point", "FY27", "Company guidance"],
        ["Peers: NTPC, NHPC, Tata Power", "Screener.in standardised consolidated data", "Company-year", "FY2025–FY2026", "Secondary"],
    ], columns=["Dataset", "Source", "Grain", "Period", "Type"])
    st.dataframe(src_tbl, hide_index=True, width="stretch")
    st.subheader("Filing used for each year")
    st.dataframe(fin[["fy", "source", "secondary_fields", "basis_note"]], hide_index=True, width="stretch")
    st.subheader("Statements loaded (₹ crore)")
    show = fin[(fin.fy >= fy_range[0]) & (fin.fy <= fy_range[1])].drop(columns=["source", "secondary_fields", "basis_note"])
    st.dataframe(show.set_index("fy").T, width="stretch", height=420)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download statements template (CSV)", B["fin"].to_csv(index=False).encode(),
                       "statements_template.csv", "text/csv")
    c2.download_button("⬇ Download price file template (CSV)", B["px"].to_csv(index=False).encode(),
                       "prices_template.csv", "text/csv")
    st.caption("To update the app when FY27 results are filed, add a row to the template from the new filing and "
               "upload it from the sidebar. The template also works for any other listed company.")
    st.subheader("Market & policy assumptions")
    st.dataframe(B["assum"], hide_index=True, width="stretch")


# =============================================================================
# 3. DATA QUALITY
# =============================================================================
elif page == PAGES[2]:
    st.title("Data-quality assessment")
    q = F.quality_report(fin, B["xchk"] if src.startswith("Bundled") else None)
    n_issue = int((~q["Treatment / result"].eq("Pass")).sum())
    c = st.columns(4)
    c[0].metric("Fiscal years", len(fin))
    c[1].metric("Checks run", len(q))
    c[2].metric("Checks needing treatment", n_issue)
    c[3].metric("Statement arithmetic", "Ties exactly" if q.iloc[0, 1] == 0 and q.iloc[1, 1] == 0 else "Gaps")
    st.dataframe(q, hide_index=True, width="stretch")
    st.subheader("Data issues found and how they were treated")
    st.markdown("""
| Issue | Years | Why it matters | Treatment |
|---|---|---|---|
| **Debt reclassification (Schedule III, 2021)**: current maturities of long-term debt sat inside "other current financial liabilities" | FY17–FY21 | Leverage understated by ₹6,000–12,000 Cr | Added back from annual-report notes so debt is comparable across years |
| **Restatement (Ind AS 8)**: deferred tax recoverable from beneficiaries reclassified as a regulatory deferral account | FY18 (restated in FY19); FY17 on old basis | Total assets and deferred tax liabilities not comparable | Latest restated figures used; FY17 balance-sheet ratios flagged |
| **Bonus issues 1:3** (Jul-2021, Sep-2023) | FY17–FY23 | Reported EPS/DPS not comparable | All per-share data restated to today's 930.06 Cr shares |
| **Exceptional items**: −₹1,079 Cr (FY21), +₹3,320 Cr (FY22) | FY21, FY22 | Distort ROE trend (FY22 ROE 23%) | Excluded from **core PAT** |
| **Deferred-tax remeasurement (Finance Act 2026 / new regime)**: −₹4,581 Cr deferred tax, offset by −₹2,774 Cr regulatory deferral | FY26 | Reported PAT +2.6% hides a **9.8% fall** in core profit | Core PAT = PBT before exceptional items − current tax |
| **Aggregator mismatch**: Screener.in folds regulatory deferral and tax items into "expenses" | FY19, FY26 | FY26 EBITDA margin would read 70% instead of 81% | Filing line items used; Screener kept only as a cross-check |
| **Scanned filings** (no machine-readable text) | FY15–FY16 | Cannot verify | Analysis window starts FY17 (10 years) |
| **Missing cash-flow lines** (capex, dividends) | FY17–FY19 | Cash-flow ratios incomplete | Capex derived as Δ(net block + CWIP) + depreciation; validated against reported capex where both exist |
""")
    cc = R[["fy", "capex_derived", "capex"]].dropna()
    cc = cc[cc.fy >= 2020]
    fig = go.Figure([go.Bar(x=cc.fy, y=cc.capex, name="Reported capex (cash flow)", marker_color=C["blue"]),
                     go.Bar(x=cc.fy, y=cc.capex_derived, name="Derived from balance sheet", marker_color=C["grey"])])
    st.plotly_chart(fig_layout(fig, 300, title="Validation: derived vs reported capex (₹ Cr)"), width="stretch")
    st.caption("Derived capex tracks reported capex, apart from capital advances and the 2021–22 transfer of "
               "subsidiaries to PGInvIT. So it is fit to fill FY17–FY19.")
    st.subheader("Suitability for analysis")
    st.success("Ten years of audited, internally consistent statements: P&L and tax arithmetic tie to ₹0 in every year, "
               "and there are no duplicates or missing key fields. Fit for ratio, DuPont and cost-of-capital analysis. "
               "FY17 balance-sheet ratios and FY22 / FY26 reported profits carry the caveats above.")
    st.subheader("Privacy, confidentiality & governance")
    st.markdown("""
* Only **public, company-level** disclosures are used. There is no personal data and no material
  non-public information, so there are no insider-trading concerns.
* Uploaded files stay in session memory and are not stored.
* Peer and market data come from third-party aggregators. They are flagged as secondary and never mixed into
  POWERGRID's primary figures.
""")


# =============================================================================
# 4. FINANCIAL PERFORMANCE (DESCRIPTIVE)
# =============================================================================
elif page == PAGES[3]:
    st.title("Financial performance: what happened?")
    k = st.columns(6)
    k[0].metric("EBITDA margin", pct(last["ebitda_margin_%"]), f"{last['ebitda_margin_%'] - prev['ebitda_margin_%']:+.1f} pp")
    k[1].metric("Core ROE", pct(last["core_roe_%"]), f"{last['core_roe_%'] - prev['core_roe_%']:+.1f} pp")
    k[2].metric("ROCE (pre-tax)", pct(last["roce_pretax_%"]), f"{last['roce_pretax_%'] - prev['roce_pretax_%']:+.1f} pp")
    k[3].metric("Net debt / EBITDA", xf(last.net_debt_ebitda_x), f"{last.net_debt_ebitda_x - prev.net_debt_ebitda_x:+.2f}x",
                delta_color="inverse")
    k[4].metric("Interest coverage", xf(last.interest_coverage_x), f"{last.interest_coverage_x - prev.interest_coverage_x:+.2f}x")
    k[5].metric("Dividend / share", f"₹{last.dps_adj:.2f}", f"payout {last['payout_%']:.0f}%", delta_color="off")

    t1, t2, t3, t4, t5 = st.tabs(["Trends", "Ratio explorer", "Segment drill-down", "Peer benchmark", "Valuation history"])
    with t1:
        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_bar(x=Rv.fy, y=Rv.revenue, name="Revenue", marker_color=C["blue"])
        fig.add_bar(x=Rv.fy, y=Rv.pat, name="Reported PAT", marker_color=C["teal"])
        fig.add_scatter(x=Rv.fy, y=Rv.core_pat, name="Core PAT", mode="lines+markers", line=dict(color=C["red"], width=3))
        a.plotly_chart(fig_layout(fig, title="Revenue and profit (₹ Cr)", barmode="group"), width="stretch")
        fig = go.Figure()
        fig.add_bar(x=Rv.fy, y=Rv.capex, name="Capex", marker_color=C["orange"])
        fig.add_bar(x=Rv.fy, y=Rv.capitalised, name="Capitalised (commissioned)", marker_color=C["grey"])
        fig.add_scatter(x=Rv.fy, y=Rv.cfo, name="Cash from operations", mode="lines+markers", line=dict(color=C["blue"]))
        b.plotly_chart(fig_layout(fig, title="Investment cycle (₹ Cr)", barmode="group"), width="stretch")
        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=Rv.fy, y=Rv["roe_%"], name="ROE (reported)", line=dict(color=C["grey"], dash="dot"))
        fig.add_scatter(x=Rv.fy, y=Rv["core_roe_%"], name="Core ROE", line=dict(color=C["blue"], width=3))
        fig.add_scatter(x=Rv.fy, y=Rv["roce_pretax_%"], name="ROCE pre-tax", line=dict(color=C["teal"]))
        fig.add_hline(y=W["ke"], line_dash="dash", line_color=C["red"], annotation_text=f"Cost of equity {W['ke']:.1f}%")
        a.plotly_chart(fig_layout(fig, title="Returns (%)"), width="stretch")
        fig = go.Figure()
        fig.add_bar(x=Rv.fy, y=Rv.debt, name="Borrowings", marker_color=C["purple"])
        fig.add_bar(x=Rv.fy, y=Rv.equity, name="Net worth", marker_color=C["teal"])
        fig.add_scatter(x=Rv.fy, y=Rv.net_debt_ebitda_x, name="Net debt/EBITDA (rhs)", yaxis="y2",
                        line=dict(color=C["red"], width=3))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="x", showgrid=False))
        b.plotly_chart(fig_layout(fig, title="Capital structure", barmode="group"), width="stretch")
        st.markdown(f"""
**What happened:** revenue grew {(R.revenue.iloc[-1] / R.revenue.iloc[0]) ** (1 / (len(R) - 1)) - 1:.1%} a year
from FY17 to FY26, but growth stalled after FY23 ({((R.revenue.iloc[-1] / R.set_index('fy').revenue.get(2023, R.revenue.iloc[-1])) ** (1 / 3) - 1) * 100:+.1f}% a year) as commissioning slowed. Core profit peaked in
FY24–25 and **fell {abs(last['core_pat_growth_%']):.1f}% in FY26**. Leverage fell steadily from 2.4x D/E to 1.4x
through FY24, as the company paid out up to 67% of profit while capex was low. **That cycle has now turned:**
capex is up 3x and borrowings have risen two years running (+₹{R.debt.iloc[-1] - R.set_index('fy').debt.get(2024, R.debt.iloc[-1]):,.0f} Cr since FY24).
""")
    with t2:
        grp = st.selectbox("Ratio group", list(F.RATIO_GROUPS))
        cols = [c for c in F.RATIO_GROUPS[grp] if c in Rv]
        tbl = Rv[["fy"] + cols].set_index("fy").T
        st.dataframe(tbl.style.format("{:,.2f}", na_rep="–").background_gradient(axis=1, cmap="RdYlGn"),
                     width="stretch")
        pick = st.multiselect("Plot", cols, default=cols[:2])
        if pick:
            fig = px.line(Rv, x="fy", y=pick, markers=True, template=T)
            st.plotly_chart(fig_layout(fig, 340), width="stretch")
        with st.expander("Ratio definitions"):
            st.table(pd.DataFrame([(c, F.RATIO_DEFS.get(c, "")) for c in cols], columns=["Ratio", "Definition"]))
    with t3:
        sg = B["seg"][B["seg"].segment.isin(segs) & B["seg"].fy.between(*fy_range)].copy()
        if sg.empty:
            st.info("Segment data is available for FY22–FY26. Widen the year filter or select a segment.")
        else:
            sg["ebit_margin_%"] = sg.ebit / sg.revenue * 100
            sg["return_on_segment_assets_%"] = sg.ebit / sg.assets * 100
            a, b = st.columns(2)
            fig = px.bar(sg, x="fy", y="revenue", color="segment", color_discrete_map=SEG_COL, template=T)
            a.plotly_chart(fig_layout(fig, title="Segment revenue (₹ Cr)"), width="stretch")
            fig = px.line(sg, x="fy", y="return_on_segment_assets_%", color="segment", markers=True,
                          color_discrete_map=SEG_COL, template=T)
            b.plotly_chart(fig_layout(fig, title="EBIT ÷ segment assets (%)"), width="stretch")
            st.dataframe(sg.round(1), hide_index=True, width="stretch")
            st.markdown("""
**Concentration:** Transmission is about 93% of revenue and 97% of segment EBIT, so group returns *are*
transmission returns. Consultancy revenue doubled in FY26 (₹1,137 Cr to ₹2,347 Cr) but earns a 17% margin.
It helps explain why **other expenses rose ₹1,962 Cr**. Telecom is small but earns the highest return on
assets (27%).
""")
    with t4:
        pe = B["peers"].copy()
        pe = pe[pe.company.isin(peers_sel + ["POWERGRID"])]
        pe["equity"] = pe.equity_capital + pe.reserves
        pe["ebitda_margin_%"] = pe.op_profit / pe.sales * 100
        pe["net_margin_%"] = pe.net_profit / pe.sales * 100
        pe["asset_turnover_x"] = pe.sales / pe.total_assets
        pe["equity_multiplier_x"] = pe.total_assets / pe.equity
        pe["roe_%"] = pe.net_profit / pe.equity * 100
        pe["debt_equity_x"] = pe.borrowings / pe.equity
        pe["interest_cover_x"] = (pe.pbt + pe.interest) / pe.interest
        p26 = pe[pe.fy == 2026]
        show = p26[["company", "sales", "ebitda_margin_%", "net_margin_%", "asset_turnover_x", "equity_multiplier_x",
                    "roe_%", "debt_equity_x", "interest_cover_x", "payout_pct", "pe", "div_yield_pct"]].round(2)
        st.dataframe(show, hide_index=True, width="stretch")
        fig = px.scatter(p26, x="debt_equity_x", y="roe_%", size="sales", color="company", text="company", template=T,
                         labels={"debt_equity_x": "Debt / equity (x)", "roe_%": "ROE (%)"})
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig_layout(fig, 380, title="Return vs leverage, FY26 (bubble = revenue)"), width="stretch")
        st.markdown("""
**Benchmark read-out:** POWERGRID has the **highest margins** in the group (EBITDA ~70–85% vs NTPC ~30%)
and the **highest dividend yield**. It earns a higher ROE than NTPC at similar leverage (D/E ≈1.5x vs 1.3x). Its weakness is
**asset turnover** (0.17x): transmission is the most capital-heavy part of the power chain. That makes
**speed of capitalisation and cost of capital** the decisive value drivers.
""")
        st.caption("Peer data are Screener.in standardised figures (secondary). POWERGRID is shown on the same basis "
                   "here for a like-for-like comparison, so its margin differs from the filing-based 81%.")
    with t5:
        vv = Rv.dropna(subset=["pe_x"])
        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=vv.fy, y=vv.pe_x, name="P/E", line=dict(color=C["blue"], width=3))
        fig.add_scatter(x=vv.fy, y=vv.pb_x, name="P/B", line=dict(color=C["teal"], width=3))
        fig.add_scatter(x=vv.fy, y=vv.ev_ebitda_x, name="EV/EBITDA", line=dict(color=C["purple"]))
        a.plotly_chart(fig_layout(fig, title="Valuation multiples at March year-end"), width="stretch")
        fig = px.bar(vv, x="fy", y="div_yield_%", template=T, color_discrete_sequence=[C["orange"]])
        b.plotly_chart(fig_layout(fig, title="Dividend yield at March year-end (%)"), width="stretch")
        st.markdown("The market **re-rated** POWERGRID from 7.5x to about 17x earnings between FY20 and FY25, as investors "
                    "began paying for the transmission growth story. The dividend yield fell from 6–7% (FY20–FY23) to about 3%.")


# =============================================================================
# 5. DIAGNOSTICS
# =============================================================================
elif page == PAGES[4]:
    st.title("Diagnostics: why are returns falling?")
    t1, t2, t3, t4 = st.tabs(["DuPont decomposition", "ROE change bridge", "Profit bridge (variance)",
                              "Bottleneck: capitalisation lag"])
    with t1:
        dpv = DP[DP.fy.between(*fy_range)]
        mode = st.radio("Model", ["5-step DuPont", "3-step DuPont"], horizontal=True)
        if mode.startswith("3"):
            cols = ["net_margin", "asset_turnover", "equity_multiplier", "roe"]
            st.latex(r"ROE=\frac{PAT}{Revenue}\times\frac{Revenue}{Avg\ assets}\times\frac{Avg\ assets}{Avg\ equity}")
        else:
            cols = ["tax_burden", "interest_burden", "ebit_margin", "asset_turnover", "equity_multiplier", "roe"]
            st.latex(r"ROE=\frac{PAT}{PBT}\times\frac{PBT}{EBIT}\times\frac{EBIT}{Revenue}\times\frac{Revenue}{Avg\ assets}\times\frac{Avg\ assets}{Avg\ equity}")
        st.dataframe(dpv[["fy"] + cols].set_index("fy").T.style.format("{:.3f}"), width="stretch")
        norm = dpv[["fy"] + cols].set_index("fy")
        norm = norm / norm.iloc[0] * 100
        fig = px.line(norm.reset_index(), x="fy", y=cols, markers=True, template=T)
        st.plotly_chart(fig_layout(fig, 360, title=f"DuPont factors indexed to FY{int(dpv.fy.iloc[0])} = 100"), width="stretch")
        st.markdown("""
**Reading:** ROE rose from 15% to 18% (FY17–FY21) by **sweating assets** (turnover 0.13 to 0.15) while leverage
was high. ROE then held up through FY25 while the **equity multiplier fell from 4.2x to 2.9x**, because
margins and turnover improved as old loans were repaid. In FY26 the **EBIT margin fell 5 points** (other
expenses and lower other income) and **turnover fell** (CWIP up ₹10,400 Cr). Only a one-off **tax-burden
jump** (0.82 to 0.92) from the deferred-tax remeasurement held reported ROE at 16.5%.
""")
    with t2:
        yrs = DP.fy.tolist()
        c1, c2, c3 = st.columns(3)
        y0 = c1.selectbox("From year", yrs, index=yrs.index(2025) if 2025 in yrs else 0)
        y1 = c2.selectbox("To year", yrs, index=len(yrs) - 1)
        five = c3.radio("Factors", ["5-step", "3-step"], horizontal=True) == "5-step"
        if y0 == y1:
            st.info("Pick two different years.")
        else:
            br = F.roe_bridge(DP, y0, y1, five)
            r0 = DP.set_index("fy").roe[y0] * 100
            r1 = DP.set_index("fy").roe[y1] * 100
            fig = go.Figure(go.Waterfall(
                x=[f"ROE FY{y0}"] + br.factor.str.replace("_", " ").tolist() + [f"ROE FY{y1}"],
                y=[r0] + br.contribution_pp.tolist() + [r1],
                measure=["absolute"] + ["relative"] * len(br) + ["total"],
                increasing=dict(marker_color=C["green"]), decreasing=dict(marker_color=C["red"]),
                totals=dict(marker_color=C["blue"]), text=[f"{v:+.2f}" for v in [r0] + br.contribution_pp.tolist() + [r1]]))
            st.plotly_chart(fig_layout(fig, 400, title=f"What moved ROE from FY{y0} to FY{y1} (percentage points)"),
                            width="stretch")
            st.dataframe(br.round(4), hide_index=True, width="stretch")
            st.caption("Attribution uses the log-share method, so contributions add exactly to the total change. "
                       "Try FY22 → FY26: falling leverage (equity multiplier) and EBIT margin explain the whole decline.")
    with t3:
        yrs = fin.fy.tolist()
        c1, c2 = st.columns(2)
        y0 = c1.selectbox("Base year", yrs, index=len(yrs) - 2, key="pb0")
        y1 = c2.selectbox("Compare year", yrs, index=len(yrs) - 1, key="pb1")
        if y0 != y1:
            items, tax_items, tot = F.pbt_bridge(fin, y0, y1)
            allit = pd.concat([items, tax_items])
            fig = go.Figure(go.Waterfall(
                x=[f"PBT FY{y0}"] + items.item.tolist() + [f"PBT FY{y1}"] + tax_items.item.tolist() + [f"PAT FY{y1}"],
                y=[tot["pbt0"]] + items.impact.tolist() + [tot["pbt1"]] + tax_items.impact.tolist() +
                  [tot["pat1"]],
                measure=["absolute"] + ["relative"] * len(items) + ["total"] + ["relative"] * 2 + ["total"],
                increasing=dict(marker_color=C["green"]), decreasing=dict(marker_color=C["red"]),
                totals=dict(marker_color=C["blue"])))
            st.plotly_chart(fig_layout(fig, 430, title=f"Profit variance FY{y0} → FY{y1} (₹ Cr)"), width="stretch")
            st.markdown(f"""
FY25 → FY26: **PBT fell ₹{tot['pbt0'] - tot['pbt1']:,.0f} Cr (−8.9%)** because **other expenses rose ₹1,962 Cr**
(partly the doubling of consultancy work) and **other income fell ₹716 Cr**. Revenue added only ₹941 Cr.
A **₹4,581 Cr deferred-tax credit** (new-regime remeasurement), partly reversed through the regulatory
deferral account, turned that into a *reported* PAT rise. The **alternative explanation** is that the FY26
opex jump is temporary consultancy pass-through cost. Watch whether FY27 other expenses fall back as a
share of revenue.
""")
    with t4:
        cap = F.calibrate_capitalisation(R)
        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_bar(x=R.fy, y=R["_cwip"], name="Capital work-in-progress", marker_color=C["orange"])
        fig.add_scatter(x=R.fy, y=R["cwip_share_%"], name="CWIP share of assets (rhs %)", yaxis="y2",
                        line=dict(color=C["red"], width=3))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
        a.plotly_chart(fig_layout(fig, title="Money spent but not yet earning (₹ Cr)"), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=R.fy, y=R["roic_posttax_%"], name="ROIC on all capital", line=dict(color=C["grey"], width=3))
        fig.add_scatter(x=R.fy, y=R["roic_ex_cwip_%"], name="ROIC on commissioned assets",
                        line=dict(color=C["blue"], width=3))
        fig.add_hline(y=W["wacc"], line_dash="dash", line_color=C["red"], annotation_text=f"WACC {W['wacc']:.1f}%")
        b.plotly_chart(fig_layout(fig, title="CWIP drag on returns (%)"), width="stretch")
        st.dataframe(cap.round(3), hide_index=True, width="stretch")
        st.markdown(f"""
**Bottleneck:** commissioned assets earn **{last['roic_ex_cwip_%']:.1f}%** after tax, above WACC
({W['wacc']:.1f}%). Once ₹{last['_cwip']:,.0f} Cr of unfinished work is included, the return drops to
**{last['roic_posttax_%']:.1f}%**, below WACC. The **capitalisation rate** (share of CWIP + capex
commissioned in a year) fell from ~42% (FY18–19) to **19–28% (FY25–26)**. Every ₹10,000 Cr commissioned
a year earlier adds about ₹1,660 Cr of revenue a year (fitted yield 0.166). **Execution speed (right-of-way,
approvals, supply chain) is a finance lever, not just an operations issue.**
""")
        rd = R.dropna(subset=["receivable_days"])
        fig = px.bar(rd, x="fy", y="receivable_days", template=T, color_discrete_sequence=[C["purple"]])
        st.plotly_chart(fig_layout(fig, 280, title="Receivable days: a second cash drag (DISCOM payments)"),
                        width="stretch")


# =============================================================================
# 6. COST OF CAPITAL
# =============================================================================
elif page == PAGES[5]:
    st.title("Cost of capital: is POWERGRID earning its keep?")
    a, b = st.columns([1, 2])
    with a:
        st.subheader("Inputs (editable)")
        win = st.selectbox("Beta estimation window", [36, 60, 84], index=1, format_func=lambda m: f"{m} months")
        be = F.estimate_beta(prices, win)
        use = st.radio("Beta to use", [f"Raw regression β = {be['beta']:.2f}", f"Blume-adjusted β = {be['adj_beta']:.2f}",
                                       "Manual"], index=0)
        beta = be["beta"] if use.startswith("Raw") else be["adj_beta"] if use.startswith("Blume") else \
            st.number_input("Manual beta", 0.2, 2.0, float(ci_d["beta"]), 0.01)
        rf = st.number_input("10-year G-sec yield (%)", 4.0, 10.0, float(ci_d["rf"]), 0.01)
        dsp = st.number_input("Sovereign default spread removed (%)", 0.0, 3.0, float(ci_d["default_spread"]), 0.01,
                              help="Damodaran: the INR G-sec contains India's default risk, which the country ERP also "
                                   "counts. Removing it avoids double-counting.")
        erp = st.number_input("India equity risk premium (%)", 3.0, 10.0, float(ci_d["erp"]), 0.01)
        kds = st.number_input("Spread of new AAA PSU debt over G-sec (%)", 0.0, 2.0, float(ci_d["kd_spread"]), 0.05)
        tax = st.number_input("Marginal tax rate (%)", 15.0, 35.0, float(ci_d["tax"]), 0.01)
        price = st.number_input("Share price (₹)", 50.0, 600.0, float(ci_d["price"]), 1.0)
        if st.button("Apply to whole app"):
            st.session_state.cap_inputs = dict(rf=rf, default_spread=dsp, erp=erp, beta=round(float(beta), 3),
                                               kd_spread=kds, tax=tax, price=price)
            st.rerun()
        w = F.wacc(F.CapitalInputs(rf=rf, default_spread=dsp, erp=erp, beta=beta, kd_spread=kds, tax=tax, price=price,
                                   shares=A["shares_cr"], debt=float(last.debt), book_equity=float(last.equity)))
    with b:
        m = st.columns(4)
        m[0].metric("Cost of equity (CAPM)", pct(w["ke"], 2), f"Rf {w['rf_riskless']:.2f}% + β×ERP", delta_color="off")
        m[1].metric("After-tax cost of debt", pct(w["kd_post"], 2), f"pre-tax {w['kd_pre']:.2f}%", delta_color="off")
        m[2].metric("WACC (market weights)", pct(w["wacc"], 2), f"E {w['we']:.0%} / D {w['wd']:.0%}", delta_color="off")
        m[3].metric("70:30 hurdle rate", pct(w["wacc_target"], 2), "new regulated projects",
                    delta_color="off")
        d = be["data"]
        fig = px.scatter(d, x="market", y="stock", trendline="ols", template=T,
                         labels={"market": "NIFTY 50 monthly log return", "stock": "POWERGRID monthly log return"})
        st.plotly_chart(fig_layout(fig, 300, title=f"Beta regression ({be['n']} months): β = {be['beta']:.2f}, "
                                                    f"95% CI {be['ci_low']:.2f}–{be['ci_high']:.2f}, R² = {be['r2']:.2f}"),
                        width="stretch")
        rb = F.rolling_beta(prices, 36)
        fig = px.line(rb, x="month", y="beta", template=T)
        fig.add_hline(y=1, line_dash="dot")
        st.plotly_chart(fig_layout(fig, 260, title="Rolling 36-month beta: from 'defensive utility' to market-like risk"),
                        width="stretch")
    st.subheader("ROE vs cost of equity, and ROIC vs WACC (EVA)")
    ev = F.eva_history(R, fin, w["wacc"], w["ke"])
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=ev.fy, y=ev.equity_spread_pp, marker_color=np.where(ev.equity_spread_pp > 0, C["green"], C["red"]),
                name="Core ROE − Ke (pp)")
    a.plotly_chart(fig_layout(fig, 320, title="Equity spread: core ROE minus cost of equity (pp)"), width="stretch")
    fig = go.Figure()
    fig.add_bar(x=ev.fy, y=ev.eva, marker_color=np.where(ev.eva > 0, C["green"], C["red"]), name="EVA")
    b.plotly_chart(fig_layout(fig, 320, title="Economic value added = (ROIC − WACC) × capital (₹ Cr)"), width="stretch")
    st.markdown(f"""
**Findings**
* The equity spread **narrowed from +6.8 pp (FY24) to +{ev.equity_spread_pp.iloc[-1]:.1f} pp (FY26)**. POWERGRID
  still creates value for shareholders, but its cushion has almost halved.
* EVA turned **negative in FY26** (₹{ev.eva.iloc[-1]:,.0f} Cr) because of non-earning CWIP. It should recover
  only if commissioning catches up.
* **Beta has risen** from about 0.6 (7-year) to about 1.0 (3-year). The stock now trades on interest rates
  and the growth story, not as a pure bond proxy, so the CAPM Ke has risen.
* For new projects the correct hurdle is the **70:30 target-structure WACC ({w['wacc_target']:.2f}%)**, or for
  the equity share, **Ke ({w['ke']:.2f}%)**. RTM projects at a 15.0–15.5% regulated RoE clear it.
  Competitively bid TBCB projects must be checked one by one.
""")
    st.subheader("Sensitivity of WACC (%) to beta and ERP")
    ci_tmp = F.CapitalInputs(rf=rf, default_spread=dsp, erp=erp, beta=beta, kd_spread=kds, tax=tax, price=price,
                             shares=A["shares_cr"], debt=float(last.debt), book_equity=float(last.equity))
    sens = F.wacc_sensitivity(ci_tmp, [0.6, 0.75, 0.88, 1.0, 1.15], [5.5, 6.5, 7.08, 8.0])
    st.dataframe(sens.style.format("{:.2f}").background_gradient(cmap="RdYlGn_r"), width="stretch")
    st.subheader("What cost of equity is the market using?")
    det = F.project(BASE, SCEN["Expected"], POL[list(POL)[0]], w["wacc"], w["ke"])
    ike = F.implied_ke(det, BASE, price)
    v = F.valuation(det, BASE, w["ke"], path=0)
    c = st.columns(3)
    c[0].metric("Market-implied cost of equity", pct(ike, 2), f"vs CAPM {w['ke']:.2f}%", delta_color="off")
    c[1].metric("Residual-income value at CAPM Ke", f"₹{v['ri_per_share']:.0f}", f"price ₹{price:.0f}", delta_color="off")
    c[2].metric("Justified P/B at CAPM Ke", xf(v["justified_pb"]), f"market P/B {price * A['shares_cr'] / last.equity:.2f}x",
                delta_color="off")
    st.caption(f"At ₹{price:.0f} the market is discounting POWERGRID's expected profits at about {ike:.1f}%, only about "
               f"{ike - rf:.1f} pp above the G-sec. That is why the share falls when bond yields rise. It is a "
               "correlation to test, not proven causation.")


# =============================================================================
# 7. FORECAST & RISK (PREDICTIVE)
# =============================================================================
elif page == PAGES[6]:
    st.title("Forecast & risk: what may happen?")
    t1, t2, t3 = st.tabs(["Revenue model vs baselines", "5-year financial projection", "Risk: Monte-Carlo"])
    with t1:
        bt = F.backtest_revenue(R)
        mdl = bt.attrs["model"]
        st.latex(r"Revenue_t = Revenue_{t-1} + y\cdot Capitalised_{t-1} - e\cdot Revenue_{t-1}")
        st.markdown(f"""
**Why this model:** under CERC tariffs an asset starts earning **after it is commissioned**, and tariff on old
assets **erodes** as their loans are repaid. So last year's commissioning drives this year's revenue.
Fitted on FY19–FY26: **y = {mdl['yield_new']:.3f}** (₹ of annual revenue per ₹ commissioned) and
**e = {mdl['erosion'] * 100:.1f}%** a year (R² = {mdl['r2']:.2f}).

**Validation:** one-year-ahead forecasts with an expanding window, so each year is forecast using only data
available before it. That is how a CFO's budget works.
""")
        m = bt.attrs["mape"]
        c = st.columns(4)
        for i, (k_, lbl) in enumerate([("driver_model", "Capitalisation-driven model"), ("naive_flat", "Baseline: no change"),
                                       ("naive_last_growth", "Baseline: repeat last growth"),
                                       ("naive_cagr3", "Baseline: 3-yr CAGR")]):
            c[i].metric(lbl, f"{m[k_]:.1f}% MAPE")
        fig = go.Figure()
        fig.add_scatter(x=bt.fy, y=bt.actual, name="Actual", line=dict(color="black", width=3))
        fig.add_scatter(x=bt.fy, y=bt.driver_model, name="Driver model", line=dict(color=C["blue"], width=3))
        fig.add_scatter(x=bt.fy, y=bt.naive_last_growth, name="Repeat last growth", line=dict(color=C["grey"], dash="dot"))
        fig.add_scatter(x=bt.fy, y=bt.naive_cagr3, name="3-yr CAGR", line=dict(color=C["orange"], dash="dot"))
        st.plotly_chart(fig_layout(fig, 340, title="Back-test FY22–FY26 (₹ Cr)"), width="stretch")
        st.dataframe(bt.round(1), hide_index=True, width="stretch")
        st.caption("Limitations: only 8 annual observations; the model needs a capitalisation forecast (from company "
                   "guidance); CERC tariff resets or true-up orders can shift revenue by 1–3% in a year.")
    with t2:
        sname = st.selectbox("Scenario", list(SCEN), index=1)
        pname = st.selectbox("Funding policy", list(POL), index=0)
        det = F.project(BASE, SCEN[sname], POL[pname], W["wacc"], W["ke"])
        pj = F.to_frame(det)
        hist = R[["fy", "revenue", "ebitda", "core_pat", "debt", "net_debt_ebitda_x", "interest_coverage_x",
                  "debt_equity_x"]].rename(columns={"core_pat": "pat", "net_debt_ebitda_x": "nd_ebitda",
                                                    "interest_coverage_x": "icr", "debt_equity_x": "de"})
        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_bar(x=hist.fy, y=hist.revenue, name="Revenue (actual)", marker_color=C["grey"])
        fig.add_bar(x=pj.fy, y=pj.revenue, name="Revenue (forecast)", marker_color=C["blue"])
        fig.add_scatter(x=hist.fy, y=hist.pat, name="Core PAT (actual)", line=dict(color=C["grey"], width=3))
        fig.add_scatter(x=pj.fy, y=pj.pat, name="PAT (forecast)", line=dict(color=C["red"], width=3))
        a.plotly_chart(fig_layout(fig, title="Revenue & profit (₹ Cr)"), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=hist.fy, y=hist.nd_ebitda, name="ND/EBITDA actual", line=dict(color=C["grey"], width=3))
        fig.add_scatter(x=pj.fy, y=pj.nd_ebitda, name="ND/EBITDA forecast", line=dict(color=C["purple"], width=3))
        fig.add_scatter(x=hist.fy, y=hist.icr, name="Interest cover actual", line=dict(color=C["grey"], dash="dot"))
        fig.add_scatter(x=pj.fy, y=pj.icr, name="Interest cover forecast", line=dict(color=C["teal"], width=3))
        fig.add_hline(y=LIM["nd_ebitda"], line_dash="dash", line_color=C["red"], annotation_text="ND/EBITDA limit")
        fig.add_hline(y=LIM["icr"], line_dash="dash", line_color=C["orange"], annotation_text="ICR floor")
        b.plotly_chart(fig_layout(fig, title="Credit metrics"), width="stretch")
        show = pj[["fy", "capex", "capitalised", "revenue", "ebitda", "depreciation", "interest", "pat", "eps", "dps",
                   "payout", "dividends", "div_to_govt", "new_borrowing", "debt", "equity", "de", "nd_ebitda", "icr", "roe",
                   "kd_marginal"]]
        st.dataframe(show.set_index("fy").T.style.format("{:,.2f}"), width="stretch")
        st.markdown("**Sanity check against broker consensus:** ICICI Direct (Q1 FY27 note) estimates PAT of "
                    "₹16,144 Cr (FY27E) and ₹17,567 Cr (FY28E). The Expected / status-quo path gives "
                    f"₹{F.to_frame(F.project(BASE, SCEN['Expected'], POL[list(POL)[0]], W['wacc'], W['ke'])).pat.iloc[0]:,.0f} Cr "
                    f"and ₹{F.to_frame(F.project(BASE, SCEN['Expected'], POL[list(POL)[0]], W['wacc'], W['ke'])).pat.iloc[1]:,.0f} Cr, "
                    "within about 1–6% of consensus.")
    with t3:
        sname = st.selectbox("Scenario ", list(SCEN), index=1)
        pname = st.selectbox("Funding policy ", list(POL), index=0)
        n = st.select_slider("Simulated paths", [500, 1000, 2000, 5000], value=2000)
        mc = F.monte_carlo(BASE, SCEN[sname], POL[pname], W["wacc"], W["ke"], n=n, de_limit=LIM["de"], icr_floor=LIM["icr"])
        res = mc["res"]
        c = st.columns(5)
        c[0].metric("P(any limit breached)", pct(mc["p_any_breach"] * 100, 0))
        c[1].metric(f"P(ND/EBITDA > {LIM['nd_ebitda']}x)", pct(mc["p_breach_nd"] * 100, 0))
        c[2].metric(f"P(D/E > {LIM['de']}x)", pct(mc["p_breach_de"] * 100, 0))
        c[3].metric(f"P(interest cover < {LIM['icr']}x)", pct(mc["p_breach_icr"] * 100, 0))
        c[4].metric("P(FY31 ROE < Ke)", pct(mc["p_roe_below_ke"] * 100, 0))
        yrs = res["fy"][:, 0].astype(int)
        fig = go.Figure()
        for lo, hi, op in [(5, 95, 0.15), (25, 75, 0.3)]:
            fig.add_scatter(x=np.r_[yrs, yrs[::-1]], y=np.r_[np.percentile(res["nd_ebitda"], hi, axis=1),
                                                             np.percentile(res["nd_ebitda"], lo, axis=1)[::-1]],
                            fill="toself", fillcolor=f"rgba(108,75,168,{op})", line=dict(width=0),
                            name=f"P{lo}–P{hi}")
        fig.add_scatter(x=yrs, y=np.median(res["nd_ebitda"], axis=1), name="Median", line=dict(color=C["purple"], width=3))
        fig.add_hline(y=LIM["nd_ebitda"], line_dash="dash", line_color=C["red"], annotation_text="limit")
        a, b = st.columns(2)
        a.plotly_chart(fig_layout(fig, 340, title="Net debt / EBITDA: forecast range"), width="stretch")
        fig = px.histogram(x=mc["ri"], nbins=60, template=T, color_discrete_sequence=[C["blue"]],
                           labels={"x": "Residual-income value per share (₹)"})
        fig.add_vline(x=ci_d["price"], line_dash="dash", line_color=C["red"], annotation_text="market price")
        b.plotly_chart(fig_layout(fig, 340, title="Distribution of value per share"), width="stretch")
        st.caption("Uncertain drivers simulated each path: capex ±10%, commissioning speed ±15%, revenue yield ±10%, "
                   "opex ±1.5 pp, new-debt rate ±50 bp, receivable days ±12. Limitations: the drivers are treated as "
                   "independent, and regulatory changes (an RoE cut) are not modelled; test them in the Scenario lab.")


# =============================================================================
# 8. SCENARIO & POLICY LAB (PRESCRIPTIVE)
# =============================================================================
elif page == PAGES[7]:
    st.title("Scenario & policy lab: which funding policy?")
    st.markdown("**Step 1. Environment scenarios** (edit any cell). These are things management does not control.")
    ed = st.data_editor(st.session_state.scen_tbl, width="stretch", key="scen_ed")
    st.markdown("**Step 2. Funding policies** (edit, or add a row). These are the levers the Board controls.")
    pe = st.data_editor(st.session_state.pol_tbl, width="stretch", num_rows="dynamic", key="pol_ed", hide_index=True)
    st.markdown("**Step 3. Limits & risk appetite**")
    c = st.columns(4)
    nd_l = c[0].number_input("Max net debt / EBITDA (x)", 3.0, 7.0, float(LIM["nd_ebitda"]), 0.1)
    de_l = c[1].number_input("Max debt / equity (x)", 1.0, 3.0, float(LIM["de"]), 0.05)
    icr_l = c[2].number_input("Min interest coverage (x)", 1.5, 4.0, float(LIM["icr"]), 0.1)
    tol = c[3].number_input("Max acceptable P(breach) %", 1, 50, int(LIM["tolerance"]), 1)
    if st.button("▶ Run scenarios", type="primary"):
        st.session_state.scen_tbl = ed
        st.session_state.pol_tbl = pe.dropna(subset=["Policy"])
        st.session_state.limits = dict(nd_ebitda=nd_l, de=de_l, icr=icr_l, tolerance=tol)
        st.rerun()
    G, paths = grid()
    st.subheader("Results: policy × scenario")
    metric = st.selectbox("Compare on", ["p_any_breach", "ri_per_share", "cum_div_govt", "peak_nd_ebitda", "icr_min",
                                         "roe_fy31", "de_fy31", "dps_avg", "ddm_per_share", "cum_eva"],
                          format_func=lambda m: {"p_any_breach": "Probability of breaching any limit (%)",
                                                 "ri_per_share": "Value per share (residual income, ₹)",
                                                 "cum_div_govt": "Cumulative dividend to Government FY27–31 (₹ Cr)",
                                                 "peak_nd_ebitda": "Peak net debt / EBITDA (x)",
                                                 "icr_min": "Lowest interest coverage (x)", "roe_fy31": "ROE FY31 (%)",
                                                 "de_fy31": "Debt / equity FY31 (x)", "dps_avg": "Average DPS (₹)",
                                                 "ddm_per_share": "Dividend-discount value (₹)",
                                                 "cum_eva": "Cumulative EVA (₹ Cr)"}[m])
    pv = G.pivot(index="policy", columns="scenario", values=metric)[["Pessimistic", "Expected", "Optimistic"]]
    cmap = "RdYlGn_r" if metric in ("p_any_breach", "peak_nd_ebitda", "de_fy31") else "RdYlGn"
    st.dataframe(pv.style.format("{:,.2f}").background_gradient(cmap=cmap, axis=None), width="stretch")
    ex = G[G.scenario == "Expected"].copy()
    ex["meets_risk_appetite"] = ex.p_any_breach <= LIM["tolerance"]
    fig = px.scatter(ex, x="p_any_breach", y="ri_per_share", size="cum_div_govt", color="meets_risk_appetite",
                     hover_name="policy", text=ex.policy.str[:1], template=T,
                     color_discrete_map={True: C["green"], False: C["red"]},
                     labels={"p_any_breach": "P(breach) %", "ri_per_share": "Value per share (₹)"})
    fig.add_vline(x=LIM["tolerance"], line_dash="dash", annotation_text="risk appetite")
    fig.update_traces(textposition="middle center", textfont_color="white")
    st.plotly_chart(fig_layout(fig, 400, title="Expected scenario: value vs risk (bubble = dividend to GoI)"), width="stretch")
    ok = ex[ex.meets_risk_appetite]
    if len(ok):
        best = ok.sort_values(["ri_per_share", "cum_div_govt"], ascending=False).iloc[0]
        st.success(f"**Best policy within risk appetite (Expected):** {best.policy}: value ₹{best.ri_per_share:.0f}/share, "
                   f"P(breach) {best.p_any_breach:.1f}%, ₹{best.cum_div_govt:,.0f} Cr to GoI over FY27–31. "
                   f"In the pessimistic case its P(breach) is "
                   f"{G[(G.scenario == 'Pessimistic') & (G.policy == best.policy)].p_any_breach.iloc[0]:.0f}%, which "
                   "is why a contingency trigger is needed.")
    else:
        st.error("No policy meets the risk appetite in the Expected scenario. Relax the limits or add levers.")
    st.subheader("How high can the payout go? (Expected scenario)")
    base_pol = st.selectbox("Other levers held as in", list(POL), index=0)
    ms = F.max_sustainable_payout(BASE, SCEN["Expected"], POL[base_pol], W["wacc"], W["ke"],
                                  tolerance=LIM["tolerance"] / 100, n=1000, de_limit=LIM["de"], icr_floor=LIM["icr"])
    fig = go.Figure()
    fig.add_scatter(x=ms.payout, y=ms.p_any_breach, name="P(breach) %", line=dict(color=C["red"], width=3))
    fig.add_bar(x=ms.payout, y=ms.cum_div_govt / 1000, name="Dividend to GoI FY27–31 (₹ '000 Cr, rhs)",
                marker_color=C["teal"], opacity=0.5, yaxis="y2")
    fig.add_hline(y=LIM["tolerance"], line_dash="dash")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False), xaxis_title="Payout %")
    st.plotly_chart(fig_layout(fig, 340), width="stretch")
    mx = ms[ms.ok].payout.max() if ms.ok.any() else None
    st.info(f"Maximum payout within a {LIM['tolerance']}% breach tolerance: **{mx:.0f}%**" if mx else
            "No payout level meets the tolerance under these settings.")
    st.subheader("Sensitivity: marginal cost of new debt")
    rows = []
    for kd in [7.0, 7.5, 7.84, 8.25, 8.75, 9.25]:
        s_ = replace(SCEN["Expected"], kd_new=kd)
        ps = F.policy_summary(BASE, s_, POL[base_pol], W["wacc"], W["ke"], n_mc=500, de_limit=LIM["de"], icr_floor=LIM["icr"])
        rows.append(dict(kd_new=kd, icr_min=ps["icr_min"], p_any_breach=ps["p_any_breach"], roe_fy31=ps["roe_fy31"],
                         ri_per_share=ps["ri_per_share"]))
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    st.caption("Each +50 bp on new borrowing cuts FY31 ROE by about 0.3 pp and raises breach risk. Locking in "
               "long-tenor fixed-rate funding early in a hiking cycle is a hedge worth pricing.")


# =============================================================================
# 9. RECOMMENDATIONS & MONITORING
# =============================================================================
elif page == PAGES[8]:
    st.title("Recommendations: what should management do?")
    G, paths = grid()
    ex = G[G.scenario == "Expected"].set_index("policy")
    pes = G[G.scenario == "Pessimistic"].set_index("policy")
    sq = ex.iloc[0]
    rec_name = [p for p in ex.index if p.startswith("G")] or [ex.p_any_breach.idxmin()]
    rec = ex.loc[rec_name[0]]
    c = st.columns(4)
    c[0].metric("Risk of breaching a limit", f"{rec.p_any_breach:.0f}%",
                f"{rec.p_any_breach - sq.p_any_breach:+.0f} pp vs status quo", delta_color="inverse")
    c[1].metric("Value per share", f"₹{rec.ri_per_share:.0f}", f"{rec.ri_per_share - sq.ri_per_share:+.1f} ₹ vs status quo")
    st.caption(f"Recommended policy: **{rec.name}** (Expected scenario, compared with A · status quo).")
    c[2].metric("Dividend to GoI FY27–31", cr(rec.cum_div_govt), f"{dcr(rec.cum_div_govt - sq.cum_div_govt)} vs status quo")
    c[3].metric("Debt in FY31", cr(rec.debt_fy31), f"{dcr(rec.debt_fy31 - sq.debt_fy31)} vs status quo",
                delta_color="inverse")
    t1, t2, t3 = st.tabs(["Action plan", "Hurdle-rate card for bids", "Monitoring dashboard"])
    with t1:
        int_saving = (paths[("Expected", sq.name)].interest.sum() - paths[("Expected", rec.name)].interest.sum())
        value_gain = (rec.ri_per_share - sq.ri_per_share) * A["shares_cr"]
        lock_amt = 0.55 * paths[("Expected", rec.name)].new_borrowing.iloc[:2].sum()
        value_gain = (rec.ri_per_share - sq.ri_per_share) * A["shares_cr"]
        plan = pd.DataFrame([
            [1, "Adopt a dynamic payout rule: 60% of PAT in normal years, cut automatically to 40% if net debt/EBITDA "
                "exceeds 3.8x (still above DIPAM's 30% floor). Announce the rule so investors and DIPAM can plan.",
             "Director (Finance) → Board", "FY27 interim dividend (Nov-26)",
             "Lower GoI dividend only in stress years",
             f"P(breach) {sq.p_any_breach:.0f}% → {rec.p_any_breach:.0f}%; GoI dividend {dcr(rec.cum_div_govt - sq.cum_div_govt)} over FY27–31"],
            [2, "Selective TBCB bidding: approve a bid only if project equity IRR ≥ Ke + 1.5 pp (≈13.3%). In practice "
                "this trims the capex plan by about 10% versus guidance.",
             "Board Investment Committee + Director (Projects)", "Every bid from Q3 FY27",
             "TBCB market share may slip from ~47%",
             f"Value {dcr(value_gain)} ({rec.ri_per_share - sq.ri_per_share:+.1f} ₹/share); FY31 debt {dcr(rec.debt_fy31 - sq.debt_fy31)}"],
            [3, "Pre-approve a contingency InvIT monetisation (₹10,000–15,000 Cr of commissioned TBCB SPVs to PGInvIT), "
                "executed only if ND/EBITDA > 4.2x or new-debt cost > 8.5%.",
             "Director (Finance) + PGInvIT manager", "Board approval FY27; execute on trigger",
             "Gives up about ₹1,100–1,700 Cr/yr of EBITDA when used",
             "Cuts pessimistic-case breach risk; debt −₹10,000–15,000 Cr"],
            [4, "Lock in long-tenor fixed-rate debt (10–15 yr bonds / ECBs) for 50–60% of FY27–28 borrowing while AAA "
                "spreads are tight.", "Treasury (GM Finance)", "H2 FY27",
             "Possible negative carry if rates fall",
             f"Locks ≈{cr(lock_amt)} of new debt; each +50 bp avoided saves ≈{cr(lock_amt * 0.005)} a year"],
            [5, "Commissioning drive: lift the capitalisation rate from 28% back to 35%+ (right-of-way task force, "
                "long-term supply contracts).", "Director (Projects) + COO", "FY27–FY28",
             "Project management cost", "Each ₹10,000 Cr commissioned a year earlier ≈ +₹1,660 Cr revenue a year"],
            [6, "Normalised-profit reporting: publish core PAT (excluding deferred-tax / RDA one-offs) next to reported PAT.",
             "CFO + Investor Relations", "Q3 FY27 results", "Minimal",
             "Avoids mis-reading FY26 (+2.6% reported vs −9.8% core)"],
            [7, "Receivables: enforce the late-payment surcharge rules; target ≤ 70 receivable days.",
             "Commercial department", "FY27", "Relationship with state DISCOMs",
             f"Each 10 days ≈ {cr(R.revenue.iloc[-1] * 10 / 365)} of cash released"],
        ], columns=["#", "Action", "Owner", "Timeline", "Cost / trade-off", "Expected benefit"])
        st.table(plan.set_index("#"))
        st.markdown(f"""
**Why not simply cut the payout to 40%?** It lowers risk, but it also lowers value (₹{ex.loc[[p for p in ex.index if p.startswith('B')][0]].ri_per_share:.0f} vs ₹{sq.ri_per_share:.0f}).
Retained equity (cost ≈ {W['ke']:.1f}%) replaces debt costing only ≈ {W['kd_post']:.1f}% after tax, and the Government
loses about {cr(sq.cum_div_govt - ex.loc[[p for p in ex.index if p.startswith('B')][0]].cum_div_govt)} over five years.
**Why not raise it to 65%?** Breach probability rises to {ex.loc[[p for p in ex.index if p.startswith('C')][0]].p_any_breach:.0f}%.
The recommended mix **pays more** in normal years, **cuts automatically** when leverage rises, and frees balance sheet
by not chasing bids below the hurdle rate. In the **pessimistic** case (debt at 8.6%, slow commissioning) no payout policy
alone keeps interest cover above {LIM['icr']}x: that is why the InvIT contingency (action 3) is pre-approved.
""")
    with t2:
        st.markdown("Use this card when deciding whether to bid for a new transmission project.")
        c = st.columns(3)
        eq_irr = c[0].slider("Expected project equity IRR (%)", 8.0, 20.0, 13.0, 0.25)
        proj = c[1].number_input("Project cost (₹ Cr)", 500, 20000, 5000, 500)
        eq_share = c[2].slider("Equity share of funding (%)", 20, 40, 30)
        spread = eq_irr - W["ke"]
        val = proj * eq_share / 100 * spread / 100 / (W["ke"] / 100)
        verdict = ("✅ Bid: clears cost of equity with margin" if spread >= 1.5 else
                   "⚠️ Marginal: bid only for strategic reasons" if spread >= 0 else "❌ Do not bid: destroys value")
        st.metric("Equity spread over Ke", f"{spread:+.2f} pp", verdict, delta_color="off")
        st.metric("Approximate value created (perpetuity of spread on equity invested)", cr(val))
        st.table(pd.DataFrame([["RTM (cost-plus, CERC)", "15.0–15.5% RoE on 30% equity", f"{15.25 - W['ke']:+.1f} pp"],
                               ["TBCB (competitive bid)", "Bid-dependent; assumed ~12–14% equity IRR (not disclosed)",
                                f"{13 - W['ke']:+.1f} pp at 13%"],
                               ["Hurdle (this app)", f"Ke {W['ke']:.2f}% + 1.5 pp risk margin", "≥ +1.5 pp"]],
                              columns=["Project type", "Return", "Spread vs Ke"]))
    with t3:
        pj = paths[("Expected", rec.name)]
        mon = pd.DataFrame([
            ["Net debt / EBITDA", last.net_debt_ebitda_x, pj.nd_ebitda.max(), f"≤ {LIM['nd_ebitda']}x; payout trigger 3.8x; InvIT trigger 4.2x", "Quarterly", "Director (Finance)"],
            ["Debt / equity", last.debt_equity_x, pj.de.iloc[-1], f"≤ {LIM['de']}x", "Quarterly", "Director (Finance)"],
            ["Interest coverage", last.interest_coverage_x, pj.icr.min(), f"≥ {LIM['icr']}x", "Quarterly", "Treasury"],
            ["Core ROE − Ke (pp)", last["core_roe_%"] - W["ke"], pj.roe.iloc[-1] - W["ke"], "≥ +3 pp", "Annual", "Board"],
            ["Capitalisation rate", R["capitalised"].iloc[-1] / (R["_cwip"].iloc[-2] + R.capex_derived.iloc[-1]), 0.37,
             "≥ 0.35", "Quarterly", "Director (Projects)"],
            ["Receivable days", last.receivable_days, 70, "≤ 70", "Monthly", "Commercial"],
            ["Payout ratio %", last["payout_%"], pj.payout.mean(), "60% normal / 40% stressed", "Each dividend", "Board"],
            ["Marginal borrowing cost %", W["kd_pre"], pj.kd_marginal.mean(), "≤ G-sec + 60 bp", "Each issue", "Treasury"],
        ], columns=["KPI", "FY26 actual", "Plan (FY27–31)", "Limit / target", "Frequency", "Owner"])
        st.dataframe(mon.round(2), hide_index=True, width="stretch")
        fig = go.Figure()
        hist = R[["fy", "net_debt_ebitda_x"]]
        fig.add_scatter(x=hist.fy, y=hist.net_debt_ebitda_x, name="Actual", line=dict(color="black", width=3))
        fig.add_scatter(x=pj.fy, y=pj.nd_ebitda, name="Plan (recommended)", line=dict(color=C["green"], width=3))
        fig.add_scatter(x=paths[("Expected", sq.name)].fy, y=paths[("Expected", sq.name)].nd_ebitda,
                        name="Status quo", line=dict(color=C["grey"], dash="dot"))
        fig.add_hline(y=3.8, line_dash="dot", line_color=C["orange"], annotation_text="payout trigger 3.8x")
        fig.add_hline(y=LIM["nd_ebitda"], line_dash="dash", line_color=C["red"], annotation_text="limit")
        st.plotly_chart(fig_layout(fig, 340, title="Leverage glide path vs plan"), width="stretch")
        st.markdown("**How we will know it worked:** compare each quarter's actual net debt/EBITDA, interest cover "
                    "and capitalisation with the plan. Variance above 0.3x for two quarters prompts a re-plan. Re-run "
                    "this app when each year's audited results are filed (upload the new row on page 2).")


# =============================================================================
# 10. REPORT & DOWNLOADS
# =============================================================================
elif page == PAGES[9]:
    st.title("Report & downloads")
    G, paths = grid()
    ex = G[G.scenario == "Expected"].set_index("policy")
    sq = ex.iloc[0]
    rec_name = [p for p in ex.index if p.startswith("G")] or [ex.p_any_breach.idxmin()]
    rec = ex.loc[rec_name[0]]
    ev = F.eva_history(R, fin, W["wacc"], W["ke"])
    bt = F.backtest_revenue(R)
    det = F.project(BASE, SCEN["Expected"], POL[list(POL)[0]], W["wacc"], W["ke"])
    ike = F.implied_ke(det, BASE, ci_d["price"])
    md = f"""# POWERGRID: capital-allocation decision brief
*Generated {datetime.now():%d %b %Y} · data: audited consolidated filings FY17–FY26 (SEBI LODR Reg. 33)*

## Decision
How should POWERGRID fund its FY27–FY31 transmission capex: what payout, borrowing and monetisation policy?

## Situation (FY26)
- Revenue {cr(last.revenue)}; reported PAT {cr(last.pat)} (+{last['pat_growth_%']:.1f}%), but **core PAT {cr(last.core_pat)} ({last['core_pat_growth_%']:+.1f}%)**.
  A one-off deferred-tax remeasurement masked the decline.
- Capex {cr(last.capex)} (3x FY24); borrowings up {cr(last.debt - prev.debt)}; CWIP is {last['cwip_share_%']:.0f}% of the asset base.
- Core ROE {last['core_roe_%']:.1f}% vs cost of equity {W['ke']:.1f}%: the spread fell from +6.8 pp (FY24) to {last['core_roe_%'] - W['ke']:+.1f} pp.
- WACC {W['wacc']:.2f}% (market weights); 70:30 hurdle {W['wacc_target']:.2f}%; beta {ci_d['beta']:.2f}; market-implied Ke ≈ {ike:.1f}%.

## Why (diagnostics)
- 5-step DuPont FY25→FY26: EBIT margin and asset turnover cut ROE by about 2.6 pp; the one-off tax burden added about 2.0 pp.
- Bottleneck: only {R['capitalised'].iloc[-1] / (R['_cwip'].iloc[-2] + R.capex_derived.iloc[-1]):.0%} of available CWIP + capex was commissioned in FY26
  (about 42% in FY18–19). Commissioned assets earn {last['roic_ex_cwip_%']:.1f}% (above WACC), but all capital together earns {last['roic_posttax_%']:.1f}%.

## Forecast
- Capitalisation-driven revenue model: one-year-ahead MAPE {bt.attrs['mape']['driver_model']:.1f}% vs {bt.attrs['mape']['naive_last_growth']:.1f}–{bt.attrs['mape']['naive_cagr3']:.1f}% for naive baselines.
- Expected / status quo: PAT FY27 ≈ {cr(F.to_frame(det).pat.iloc[0])} (ICICI Direct consensus ₹16,144 Cr); peak ND/EBITDA {sq.peak_nd_ebitda:.2f}x.

## Recommendation
**{rec.name}**
- Probability of breaching any credit limit (FY27–31, expected): {sq.p_any_breach:.0f}% → **{rec.p_any_breach:.0f}%**
- Value per share (residual income): ₹{sq.ri_per_share:.0f} → **₹{rec.ri_per_share:.0f}**
- Dividend to Government FY27–31: {cr(sq.cum_div_govt)} → **{cr(rec.cum_div_govt)}**
- FY31 debt: {cr(sq.debt_fy31)} → **{cr(rec.debt_fy31)}**

Actions: (1) dynamic payout 60%→40% at ND/EBITDA > 3.8x; (2) bid TBCB only at equity IRR ≥ Ke + 1.5 pp
(≈10% less capex); (3) pre-approved ₹10–15k Cr InvIT monetisation if ND/EBITDA > 4.2x or new debt > 8.5%;
(4) lock in long-tenor fixed-rate debt; (5) commissioning drive to a ≥35% capitalisation rate; (6) report core PAT;
(7) receivables ≤ 70 days.

## Limitations
10 annual observations; CERC 2024–29 RoE for new assets should be checked against the notified regulation;
TBCB project returns are not disclosed (assumed); Monte-Carlo drivers are independent; peer data are secondary.
"""
    st.markdown(md)
    st.download_button("⬇ Decision brief (Markdown)", md.encode(), "powergrid_decision_brief.md", "text/markdown")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        fin.to_excel(xw, sheet_name="Statements", index=False)
        R.drop(columns=[c for c in R if c.startswith("_")]).to_excel(xw, sheet_name="Ratios", index=False)
        DP.to_excel(xw, sheet_name="DuPont", index=False)
        ev.to_excel(xw, sheet_name="EVA & equity spread", index=False)
        pd.DataFrame([W]).T.rename(columns={0: "value"}).to_excel(xw, sheet_name="WACC")
        bt.to_excel(xw, sheet_name="Revenue backtest", index=False)
        G.to_excel(xw, sheet_name="Policy x scenario", index=False)
        for (sn, pn), dfp in list(paths.items()):
            if sn == "Expected":
                dfp.to_excel(xw, sheet_name=f"Path {pn[:1]} expected"[:31], index=False)
        F.quality_report(fin, B["xchk"]).to_excel(xw, sheet_name="Data quality", index=False)
        B["assum"].to_excel(xw, sheet_name="Assumptions & sources", index=False)
    st.download_button("⬇ Full analysis workbook (Excel)", buf.getvalue(), "powergrid_capital_allocation.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("Edit scenarios or policies on page 8 first: the brief and workbook use your current settings.")
