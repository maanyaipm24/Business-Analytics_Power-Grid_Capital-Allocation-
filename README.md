# POWERGRID Capital-Allocation Decision App
**Finance Analytics project · Business Analytics course (IPM, IIFT Kakinada)**

Company: **Power Grid Corporation of India Ltd** (NSE: POWERGRID · BSE: 532898). A Maharatna CPSE,
India's largest transmission utility and 51.34% Government-owned. It has been profitable every year and
earns regulated returns, so it is stable. Most coverage tracks its quarterly results; few analyses
look at its capital-allocation choices, and those choices matter right now.

## The managerial decision
> **How should POWERGRID fund its FY27–FY31 transmission build-out? What dividend payout, borrowing and
> asset-monetisation policy keeps credit metrics inside comfort limits, keeps ROE above the cost of
> equity, and still pays the Government a dependable dividend?**

Target users: Director (Finance)/CFO, Board, DIPAM/Ministry of Power (majority shareholder),
lenders and rating agencies, institutional investors.

Why now (Oct 2026):
- Capex has tripled: ₹11,410 Cr (FY24) to ₹37,279 Cr (FY26). FY27 guidance is ₹37,000 Cr and works in hand are ~₹1.75 lakh Cr, 83% won through competitive bidding (TBCB).
- Borrowings are up ₹24,600 Cr since FY24.
- The 10-year G-sec is at 7.39%, a 3-year high, and the RBI is hiking. The share has fallen from ₹318 to ₹250.

---

## 1. Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
For a public link for the demo: push the folder to GitHub, then go to **share.streamlit.io**, choose New app, and pick `app.py`.
Free hosting; the demo won't depend on one laptop. On a machine with internet, the sidebar's
**"Fetch live (Yahoo Finance)"** option downloads current prices with `yfinance`.

## 2. Data: legitimate, traceable sources
| Data | Source | Notes |
|---|---|---|
| P&L, balance sheet, cash flow FY17–FY26 (consolidated, audited) | POWERGRID results filed with BSE/NSE under **SEBI (LODR) Regulation 33**, published at https://www.powergrid.in/en/annual-results | Each year's filing URL is stored in `data/powergrid_financials.csv` |
| Segment data FY22–FY26 | Same filings (segment note) | Transmission / Consultancy / Telecom |
| Monthly prices POWERGRID & NIFTY 50, Oct-2019 to Oct-2026 | Yahoo Finance (NSE data) | Live refresh in the app |
| 10-yr G-sec 7.39%, repo 5.50% | Trading Economics, 15-Oct-2026 | Editable |
| India ERP 7.08%, default spread 1.87% | Damodaran, Jan-2026 | Editable |
| Capex / capitalisation guidance, works in hand | Company Q1 FY27 call (ICICI Direct note) | |
| Peers NTPC, NHPC, Tata Power | Screener.in (secondary, flagged) | |

`data/build_dataset.py` rebuilds every CSV and documents each number's origin. To add FY27, append one
row to the template (download it on page 2) and upload it in the sidebar. The same template works for any
listed company.

## 3. Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app, 10 pages following the required workflow |
| `finance_engine.py` | All analytics: ratios, DuPont, beta/CAPM/WACC/EVA, forecasting, projection, Monte Carlo, valuation |
| `data/*.csv` | Statements, segments, prices, peers, assumptions, Screener cross-check |

## 4. How the app meets the 9 mandatory components
| # | Requirement | Where |
|---|---|---|
| 1 | Business overview | Page 1: problem, users, decision, current practice, cost of a wrong decision, finance process, trade-offs, KPI definitions and targets |
| 2 | Data input | Sidebar: bundled vs **upload** statements; bundled / **live** / uploaded prices; year slider; segment and peer filters; refresh date. Page 2: source register and templates |
| 3 | Data quality | Page 3: 16 automated checks (P&L arithmetic ties to ₹0, duplicates, gaps, outliers, restatements, secondary fields, cross-source reconciliation) plus a table of 8 issues and their treatments |
| 4 | Descriptive | Page 4: KPI tiles, trends, 30+ ratios in 6 groups, **segment drill-down**, peer benchmark, valuation history |
| 5 | Diagnostic | Page 5: **3- and 5-step DuPont**, ROE change bridge (any two years), PBT/PAT variance waterfall, **capitalisation bottleneck** (CWIP drag, ROIC ex-CWIP), receivables |
| 5b | Core finance | Page 6: **beta regression, CAPM Ke, WACC** (market and 70:30 target), sensitivity grid, **EVA**, ROE−Ke spread, market-implied cost of equity |
| 6 | Predictive | Page 7: capitalisation-driven revenue model, **back-tested vs 3 naive baselines**; 5-year projected financials; **Monte-Carlo breach probabilities**; consensus sanity check |
| 7 | Scenario | Page 8: **editable** scenario table (pessimistic/expected/optimistic), editable policy table (add rows), editable limits, policy × scenario matrix, max sustainable payout curve, rate sensitivity |
| 8 | Recommendations | Page 9: 7 prioritised actions with owner, timeline, cost and benefit; hurdle-rate card for bids; monitoring dashboard with triggers |
| 9 | Reporting | Page 10: decision brief (Markdown) and 12-sheet Excel workbook |

## 5. Key findings (use in report and deck)
1. **FY26 profit is weaker than it looks.** Reported PAT rose 2.6% to ₹15,928 Cr, but **core PAT fell 9.8%**.
   A ₹4,581 Cr deferred-tax credit (remeasurement for the new tax regime) masked the fall. PBT fell
   ₹1,697 Cr, driven by other expenses (+₹1,962 Cr) and other income (−₹716 Cr).
2. **DuPont:** ROE held at 17–19% through FY21–FY25 even as the equity multiplier fell from 4.2x to 2.9x,
   because margins and asset turnover rose. In FY26 EBIT margin (−1.6 pp) and asset turnover (−1.0 pp) cut ROE,
   offset only by the one-off tax burden (+2.0 pp).
3. **Cost of capital:** beta is 0.88 (5-year) but **1.0 over 3 years**, up from 0.64 over 7 years. CAPM Ke is
   **11.75%** (local-currency method: G-sec minus default spread, plus β × India ERP). WACC is 9.46%; the 70:30
   hurdle is 7.63%.
   The **equity spread fell from +6.8 pp (FY24) to +2.9 pp (FY26)**. At ₹250 the market implies a Ke of
   only ~8.7%, so it prices POWERGRID like a bond. That explains why the share falls when yields rise
   (a correlation, not proven causation).
4. **Bottleneck:** commissioned assets earn **10.6% post-tax ROIC** (above WACC), but ₹43,654 Cr of CWIP pulls
   overall ROIC to 8.9%. The capitalisation rate fell from ~42% to 19–28%.
5. **Forecast:** the capitalisation-driven revenue model has a **2.2% MAPE** vs 3.2–4.0% for naive baselines.
   FY27 PAT ≈ ₹15,900 Cr (ICICI Direct consensus ₹16,144 Cr).
6. **Policy (expected case):** status quo (53% payout) has ~13–15% probability of breaching a limit. The
   **recommended policy G** (60% payout, cut to 40% automatically above 3.8x ND/EBITDA, plus selective
   TBCB bidding at IRR ≥ Ke + 1.5 pp) gives **~3–4% breach risk**, **+₹4–5/share value**, about
   **₹3,300 Cr more dividend to GoI** and **~₹11,000 Cr less debt** by FY31. In the pessimistic case no payout
   rule alone is enough, so a ₹10–15k Cr InvIT monetisation is pre-approved as a contingency.

## 6. Suggested split for 6 members
| Member | Report section | App pages | Slides |
|---|---|---|---|
| 1 | Exec summary; problem & decision; domain process & stakeholders | 1 | 1–3 |
| 2 | Data description, quality, governance & privacy | 2–3 | 4 |
| 3 | Descriptive: ratio analysis, segments, peers | 4 | 5 |
| 4 | Diagnostic: DuPont, bridges, bottleneck; cost of capital / WACC / EVA | 5–6 | 6–7 |
| 5 | Predictive & scenario: forecast, Monte Carlo, policy lab | 7–8 | 8–9 |
| 6 | Recommendations, business value, monitoring; app design; demo lead | 9–10 | 10–11 |

## 7. Presentation (12 min) and demo (8 min)
**Slides:** 1 Title and decision · 2 Why POWERGRID, why now · 3 Finance process and stakeholders · 4 Data sources and quality ·
5 What happened (ratios, peers) · 6 Why (DuPont, FY26 profit bridge, CWIP drag) · 7 Cost of capital and EVA ·
8 Forecast vs baselines and risk · 9 Policy comparison (value vs risk) · 10 Recommendations and value · 11 Governance and limitations

**Demo script:**
1. Page 3: show that the statements tie to ₹0.
2. Page 5: run the ROE bridge from FY22 to FY26.
3. Page 6: switch beta to Blume-adjusted and watch WACC change.
4. Page 8: raise "Marginal cost of new debt" to 9% in the Expected column and click Run, so breach risk jumps.
5. Page 8: add your own policy row.
6. Page 9: walk through the action plan.
7. Page 10: download the brief.

## 8. Limitations (state these openly)
- Only 10 annual observations. TBCB project returns are not disclosed, so the 12–14% equity IRR is assumed.
- Check the CERC 2024–29 RoE for new projects (15.0% draft) against the notified regulation.
- Monte-Carlo drivers are independent. Regulatory shocks (an RoE cut) are tested as scenarios, not simulated.
- Thresholds (4.5x ND/EBITDA, 2.0x D/E, 2.5x interest cover) are management comfort limits, not rating-agency covenants.
- Peer data are secondary (Screener.in). FY17 is on the pre-restatement basis.
