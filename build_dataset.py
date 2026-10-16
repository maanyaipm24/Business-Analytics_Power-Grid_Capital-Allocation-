"""
Builds the bundled dataset for the POWERGRID Capital-Allocation Decision App.

PRIMARY SOURCE (financial statements): audited consolidated financial results of
Power Grid Corporation of India Ltd (NSE: POWERGRID, BSE: 532898), filed with BSE/NSE
under SEBI (LODR) Regulation 33 and published at https://www.powergrid.in/en/annual-results
Each figure below was transcribed from the filing listed in SOURCES (₹ crore).

Rules followed:
  * Each year uses the LATEST published version of that year (e.g. FY18 is the
    restated figure shown in the FY19 filing; FY25 cash flow is the comparative in FY26 filing).
  * Nothing is estimated here. Gaps are left blank and handled (and disclosed) by the app.
  * Two fields are from secondary sources and flagged in column `secondary_fields`:
      - current_maturities_ltd FY17-FY21: before the 2021 Schedule III amendment the
        filings kept current maturities of long-term debt inside "other current financial
        liabilities". Value = total borrowings per annual-report notes (as compiled by
        Screener.in) minus the non-current + current borrowings shown in the filing.
      - cfo/cfi/cff FY17-FY19: results filings before FY20 did not include a cash-flow
        statement; taken from annual reports via Screener.in.

Run: python build_dataset.py   -> writes the CSVs next to this file.
"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent

SOURCES = {
    2017: "https://www.powergrid.in/sites/default/files/annual-results/NSE_AFR_dt290592018_0.pdf (FY18 filing, FY17 comparative)",
    2018: "https://www.powergrid.in/sites/default/files/annual-results/NSE_AFR_dt29052019_0.pdf (FY19 filing, FY18 restated)",
    2019: "https://www.powergrid.in/sites/default/files/annual-results/NSE_AFR_dt29052019_0.pdf",
    2020: "https://www.powergrid.in/sites/default/files/annual-results/NSEBSE_AFR_dt17062021_0.pdf (FY21 filing, FY20 comparative)",
    2021: "https://www.powergrid.in/sites/default/files/annual-results/NSEBSE_AFR_dt17062021_0.pdf",
    2022: "https://www.powergrid.in/sites/default/files/annual-results/32db4429-c645-449d-893b-d33f769b8236.pdf (FY23 filing, FY22 comparative)",
    2023: "https://www.powergrid.in/sites/default/files/annual-results/32db4429-c645-449d-893b-d33f769b8236.pdf",
    2024: "https://www.powergrid.in/sites/default/files/annual-results/BSENSEOutcomedt19052025_0.PDF (FY25 filing, FY24 comparative)",
    2025: "https://www.powergrid.in/sites/default/files/annual-results/BSENSEOutcomedt19052025_0.PDF ; CF & intangibles u/d from FY26 filing",
    2026: "https://www.powergrid.in/sites/default/files/annual-results/BSENSEReg30_AFSOutcome.pdf",
}

COLS = ["fy", "revenue", "other_income", "employee_exp", "finance_costs", "depreciation",
        "other_exp", "jv_share", "exceptional_items", "pbt", "current_tax", "deferred_tax",
        "total_tax", "rda_net", "pat", "eps_reported",
        "ppe", "cwip", "intangibles", "inv_jv", "investments", "trade_receivables", "cash",
        "other_bank_balances", "rda_debit", "total_assets",
        "equity_capital", "other_equity", "nc_borrowings", "c_borrowings",
        "current_maturities_ltd", "dtl",
        "cfo", "cfi", "cff", "capex_cf", "dividends_paid_cf", "finance_costs_paid"]

# exceptional_items: + = income, - = expense (as reported). Blank = not disclosed in filing.
ROWS = [
    # FY17 (original basis; DTL shown net of 'deferred tax asset recoverable')
    [2017, 25697.44, 585.52, 1377.13, 6203.55, 7722.25, 1734.71, 141.75, 0, 9387.07,
     1989.07, 17.20, 2006.27, 69.93, 7450.73, 14.24,
     136422.44, 38187.00, 1332.73, 1020.76, 141.70, 3131.93, 3241.45, 112.18, 107.60, 195645.00,
     5231.59, 44633.91, 111253.17, 1500.00, 6235.00, 2535.98,
     21575, -23836, 3959, None, None, None],
    # FY18 (restated in FY19 filing - Ind AS 8, deferred tax recoverable shown as RDA)
    [2018, 29953.62, 476.92, 1599.09, 7324.14, 9230.99, 2237.70, 147.94, 0, 10186.56,
     2197.98, 3068.77, 5266.75, 3284.19, 8204.00, 15.68,
     154831.38, 37603.90, 1431.53, 1095.19, 128.78, 3640.02, 1534.80, 654.22, 11304.22, 225316.05,
     5231.59, 49194.40, 122710.32, 1000.00, 7792.68, 13472.85,
     22710, -25701, 1285, None, None, None],
    [2019, 35059.12, 602.20, 1783.57, 8736.57, 10540.95, 3055.01, 128.82, 0, 11674.04,
     2568.02, -3454.37, -886.35, -2526.87, 10033.52, 19.18,
     171057.99, 37388.47, 1924.19, 1166.58, 129.84, 4728.10, 3647.73, 688.92, 8083.27, 247357.59,
     5231.59, 53856.80, 131329.51, 4300.00, 10746.49, 10018.48,
     23271, -18727, -2431, None, None, None],
    [2020, 37743.54, 927.42, 1959.75, 9509.00, 11607.04, 2843.55, 155.13, 0, 12906.75,
     2320.45, 1210.30, 3530.75, 1683.40, 11059.40, 21.14,
     179449.82, 34904.41, 1935.28, 1332.18, 98.90, 5040.71, 4849.59, 604.31, 10123.06, 256653.22,
     5231.59, 59463.76, 135421.11, 3000.00, 9848.89, 11228.78,
     30738.63, -10733.49, -18805.84, 11367.12, 4425.92, None],
    [2021, 39639.79, 1183.74, 2114.76, 8134.69, 12039.19, 2531.92, 214.70, -1078.64, 15139.03,
     2508.59, 955.66, 3464.25, 361.68, 12036.46, 23.01,
     182108.99, 24679.80, 1774.94, 1346.85, 138.70, 3675.53, 4487.23, 871.48, 10575.13, 255749.12,
     5231.59, 64704.48, 129174.79, 1800.00, 12257.21, 11836.62,
     29312.15, -8972.95, -20521.05, 9363.55, 6821.99, None],
    [2022, 41621.64, 1076.26, 2243.89, 8036.22, 12871.66, 2805.55, 78.47, 3320.33, 20139.38,
     3313.83, -528.26, 2785.57, -529.74, 16824.07, 24.12,
     190048.35, 12799.90, 1724.49, 1861.21, 1926.43, 9189.33, 2577.11, 2471.07, 9933.24, 250943.43,
     6975.45, 69271.68, 114199.40, 20465.87, 0, 11356.98,
     26123.52, 752.79, -28966.94, 7966.22, 10811.96, None],
    [2023, 45581.28, 1024.36, 2508.53, 9633.94, 13333.38, 3594.62, -82.12, 0, 17453.05,
     2944.52, -660.02, 2284.50, 248.57, 15417.12, 22.10,
     183777.39, 13741.24, 1659.60, 1714.65, 1774.46, 13694.53, 5192.17, 2192.41, 10234.44, 250295.55,
     6975.45, 76039.06, 113891.29, 12703.61, 0, 10678.87,
     38004.74, -6125.70, -29263.98, 6902.57, 8544.93, None],
    [2024, 45843.10, 1070.02, 2434.44, 8772.75, 13095.27, 3505.77, -19.55, 0, 19085.34,
     3338.21, -397.42, 2940.79, -571.39, 15573.16, 16.74,
     176222.51, 18081.73, 1538.04, 2038.98, 1515.93, 11834.11, 3463.95, 4031.17, 9542.08, 250890.52,
     9300.60, 77844.51, 105066.95, 18381.67, 0, 10298.57,
     37289.50, -13114.39, -25903.33, 11409.68, 11218.85, 9356.86],
    [2025, 45792.32, 1667.06, 2604.40, 8700.13, 12904.15, 4122.55, -110.26, 0, 19017.89,
     3362.27, 410.49, 3772.76, 276.31, 15521.44, 16.69,
     170885.68, 33268.69, 1750.87, 1883.06, 1233.59, 8008.80, 3796.79, 6280.45, 9876.87, 266107.19,
     9300.60, 83362.21, 109857.95, 21107.01, 0, 10705.42,
     36221.82, -23531.80, -12357.18, 24134.04, 9765.64, 9221.11],
    [2026, 46732.87, 951.56, 2668.90, 8447.58, 13029.68, 6084.90, -132.34, 0, 17321.03,
     3199.30, -4580.61, -1381.31, -2774.39, 15927.95, 17.13,
     174970.20, 43654.29, 1652.78, 1323.72, 1671.37, 11672.59, 5280.62, 3639.79, 6515.10, 294727.26,
     9300.60, 91193.43, 121071.80, 26937.21, 0, 6154.70,
     40930.58, -35305.46, -4006.26, 37279.11, 8370.55, 9535.27],
]

fin = pd.DataFrame(ROWS, columns=COLS)
fin["source"] = fin.fy.map(SOURCES)
fin["secondary_fields"] = ""
fin.loc[fin.fy <= 2021, "secondary_fields"] = "current_maturities_ltd"
fin.loc[fin.fy <= 2019, "secondary_fields"] = "current_maturities_ltd; cfo; cfi; cff"
fin["basis_note"] = ""
fin.loc[fin.fy == 2017, "basis_note"] = "Pre-restatement basis: DTL netted with deferred-tax recoverable; RDA debit small"
fin.loc[fin.fy == 2018, "basis_note"] = "Restated (Ind AS 8) in FY19 filing"
fin.loc[fin.fy == 2026, "basis_note"] = "Deferred tax remeasured (Finance Act 2026 / new regime); offset in RDA"
# dividend per share adjusted to today's 930.06 Cr shares (bonus 1:3 in Jul-2021 and Sep-2023)
# dividend payout % for the year (declared basis) - Screener.in, cross-checked FY23-FY26 with
# record-date dividends (FY26: 4.50 + 3.25 + 1.25 = Rs 9.00 per share; company press release)
fin["payout_pct_declared"] = [31, 33, 43, 47, 52, 61, 67, 67, 54, 53]
fin.to_csv(OUT / "powergrid_financials.csv", index=False)

# ----------------------------------------------------------------------------
# Segment information (consolidated, from the same filings; latest comparative used)
# revenue incl. allocable other income; result = profit before interest & tax
seg = pd.DataFrame([
    [2022, "Transmission", 40976.38, 23120.26, 227077.55, 14295.14],
    [2022, "Consultancy", 780.67, 351.47, 1899.37, 2743.96],
    [2022, "Telecom", 668.86, 223.53, 925.48, 648.01],
    [2023, "Transmission", 44965.81, 26618.13, 225499.48, 15733.66],
    [2023, "Consultancy", 536.62, 84.26, 1633.71, 2728.17],
    [2023, "Telecom", 813.74, 313.41, 1092.53, 637.64],
    [2024, "Transmission", 44902.27, 26168.39, 220672.49, 13971.09],
    [2024, "Consultancy", 718.00, 125.96, 1698.09, 2875.70],
    [2024, "Telecom", 910.28, 329.48, 1537.53, 778.31],
    [2025, "Transmission", 44776.80, 27124.86, 218619.23, 13009.35],
    [2025, "Consultancy", 1137.28, 150.12, 2445.81, 3831.64],
    [2025, "Telecom", 1128.10, 523.89, 1931.34, 1025.57],
    [2026, "Transmission", 44082.74, 26353.13, 230311.08, 10771.36],
    [2026, "Consultancy", 2347.36, 403.38, 4624.52, 4860.41],
    [2026, "Telecom", 1195.07, 565.99, 2122.61, 1309.86],
], columns=["fy", "segment", "revenue", "ebit", "assets", "liabilities"])
seg.to_csv(OUT / "segments.csv", index=False)

# ----------------------------------------------------------------------------
# Monthly prices: POWERGRID.NS (split-adjusted close, and dividend+split adjusted close)
# and NIFTY 50 (^NSEI) - Yahoo Finance history pages, retrieved 15-Oct-2026.
pg = """2019-10,111.57,77.44
2019-11,108.73,75.47
2019-12,107.02,74.28
2020-01,105.10,72.95
2020-02,102.09,70.86
2020-03,89.49,64.40
2020-04,91.15,65.60
2020-05,88.62,63.78
2020-06,98.35,70.78
2020-07,100.29,72.18
2020-08,100.66,72.44
2020-09,91.38,67.27
2020-10,96.19,70.81
2020-11,108.23,79.67
2020-12,106.79,80.67
2021-01,103.75,78.37
2021-02,120.54,91.06
2021-03,121.30,93.32
2021-04,123.78,95.23
2021-05,126.93,97.65
2021-06,130.73,100.57
2021-07,128.36,98.76
2021-08,131.51,101.18
2021-09,142.43,111.51
2021-10,138.79,108.66
2021-11,155.06,121.40
2021-12,153.30,124.23
2022-01,161.55,130.92
2022-02,156.86,130.65
2022-03,162.60,135.43
2022-04,170.81,142.27
2022-05,174.68,145.49
2022-06,158.93,132.37
2022-07,160.46,133.65
2022-08,172.20,144.83
2022-09,159.15,133.86
2022-10,171.11,143.92
2022-11,168.00,144.68
2022-12,160.27,138.03
2023-01,162.49,139.93
2023-02,166.69,146.94
2023-03,169.27,149.22
2023-04,177.94,156.85
2023-05,175.39,154.61
2023-06,191.36,168.69
2023-07,199.54,175.90
2023-08,183.41,164.84
2023-09,199.75,179.52
2023-10,202.15,181.68
2023-11,208.95,191.43
2023-12,237.20,217.32
2024-01,259.30,237.56
2024-02,282.85,263.47
2024-03,276.90,257.93
2024-04,301.85,281.17
2024-05,310.00,288.76
2024-06,330.95,308.28
2024-07,348.20,324.34
2024-08,337.40,316.90
2024-09,352.85,331.41
2024-10,320.80,301.31
2024-11,329.40,313.82
2024-12,308.70,294.10
2025-01,301.65,287.39
2025-02,250.85,241.78
2025-03,290.35,279.86
2025-04,307.45,296.34
2025-05,289.75,279.28
2025-06,299.90,289.06
2025-07,291.00,280.48
2025-08,275.25,266.45
2025-09,280.25,271.29
2025-10,288.15,278.94
2025-11,269.95,265.71
2025-12,264.60,260.45
2026-01,256.50,252.48
2026-02,298.65,297.26
2026-03,296.10,294.73
2026-04,318.35,316.87
2026-05,290.55,289.20
2026-06,286.30,284.97
2026-07,284.25,282.93
2026-08,263.55,263.55
2026-09,260.50,260.50
2026-10,248.95,248.95"""
nifty = """2019-10,11877.45
2019-11,12056.05
2019-12,12168.45
2020-01,11962.10
2020-02,11201.75
2020-03,8597.75
2020-04,9859.90
2020-05,9580.30
2020-06,10302.10
2020-07,11073.45
2020-08,11387.50
2020-09,11247.55
2020-10,11642.40
2020-11,12968.95
2020-12,13981.75
2021-01,13634.60
2021-02,14529.15
2021-03,14690.70
2021-04,14631.10
2021-05,15582.80
2021-06,15721.50
2021-07,15763.05
2021-08,17132.20
2021-09,17618.15
2021-10,17671.65
2021-11,16983.20
2021-12,17354.05
2022-01,17339.85
2022-02,16793.90
2022-03,17464.75
2022-04,17102.55
2022-05,16584.55
2022-06,15780.25
2022-07,17158.25
2022-08,17759.30
2022-09,17094.35
2022-10,18012.20
2022-11,18758.35
2022-12,18105.30
2023-01,17662.15
2023-02,17303.95
2023-03,17359.75
2023-04,18065.00
2023-05,18534.40
2023-06,19189.05
2023-07,19753.80
2023-08,19253.80
2023-09,19638.30
2023-10,19079.60
2023-11,20133.15
2023-12,21731.40
2024-01,21725.70
2024-02,21982.80
2024-03,22326.90
2024-04,22604.85
2024-05,22530.70
2024-06,24010.60
2024-07,24951.15
2024-08,25235.90
2024-09,25810.85
2024-10,24205.35
2024-11,24131.10
2024-12,23644.80
2025-01,23508.40
2025-02,22124.70
2025-03,23519.35
2025-04,24334.20
2025-05,24750.70
2025-06,25517.05
2025-07,24768.35
2025-08,24426.85
2025-09,24611.10
2025-10,25722.10
2025-11,26202.95
2025-12,26129.60
2026-01,25320.65
2026-02,25178.65
2026-03,22331.40
2026-04,23997.55
2026-05,23547.75
2026-06,23865.75
2026-07,24383.60
2026-08,24080.40
2026-09,22620.45
2026-10,22721.95"""
p = pd.DataFrame([r.split(",") for r in pg.split("\n")], columns=["month", "pg_close", "pg_adj_close"])
n = pd.DataFrame([r.split(",") for r in nifty.split("\n")], columns=["month", "nifty_close"])
px_ = p.merge(n, on="month")
for c in ["pg_close", "pg_adj_close", "nifty_close"]:
    px_[c] = px_[c].astype(float)
px_["partial_month"] = (px_.month == "2026-10").astype(int)
px_.to_csv(OUT / "market_monthly.csv", index=False)

# ----------------------------------------------------------------------------
# Peers - FY26 & FY25 consolidated, Screener.in standardised (secondary source), 15-Oct-2026
peers = pd.DataFrame([
    # company, fy, sales, op_profit, other_income, interest, dep, pbt, net_profit, payout, eq_cap, reserves, borrowings, total_assets, mcap, price, pe, bv, div_yield
    ["NTPC", 2026, 187385, 55438, 2662, 13801, 19629, 24669, 27546, 32, 9697, 193479, 271005, 558442, 304814, 315, 11.0, 210, 2.86],
    ["NTPC", 2025, 188138, 54355, 8526, 13282, 17401, 32198, 23953, 35, 9697, 174374, 250096, 523732, None, None, None, None, None],
    ["NHPC", 2026, 11615, 4097, 3196, 1423, 1976, 3894, 4220, 43, 10045, 31392, 52327, 120007, 73146, 72.7, 19.3, 41.2, 2.21],
    ["NHPC", 2025, 10380, 5523, 1626, 1189, 1193, 4767, 3412, 64, 10045, 29623, 39557, 102680, None, None, None, None, None],
    ["Tata Power", 2026, 62429, 13271, 3433, 5257, 4811, 6636, 5118, 21, 320, 39148, 76141, 174564, 111947, 350, 28.6, 124, 0.71],
    ["Tata Power", 2025, 65478, 12450, 2689, 4702, 4117, 6320, 4775, 18, 320, 35521, 62866, 156193, None, None, None, None, None],
    # POWERGRID on the same Screener basis, for like-for-like comparison
    ["POWERGRID", 2026, 46733, 32831, 3193, 8448, 13030, 14547, 15928, 53, 9301, 91193, 148071, 294674, 231631, 250, 14.6, 108, 3.60],
    ["POWERGRID", 2025, 45792, 38779, 2120, 8700, 12904, 19294, 15521, 54, 9301, 83362, 131030, 266019, None, None, None, None, None],
], columns=["company", "fy", "sales", "op_profit", "other_income", "interest", "depreciation", "pbt",
            "net_profit", "payout_pct", "equity_capital", "reserves", "borrowings", "total_assets",
            "mcap", "price", "pe", "book_value", "div_yield_pct"])
peers.to_csv(OUT / "peers.csv", index=False)

# Secondary cross-check (Screener.in consolidated, retrieved 15-Oct-2026) for reconciliation
xchk = pd.DataFrame({
    "fy": list(range(2017, 2027)),
    "sales": [25697, 29954, 35059, 37744, 39640, 41622, 45581, 45843, 45792, 46733],
    "net_profit": [7451, 8204, 10034, 11059, 12036, 16824, 15417, 15573, 15521, 15928],
    "borrowings": [118988, 131503, 146376, 148270, 143232, 134696, 126661, 123516, 131030, 148071],
    "reserves": [44634, 49194, 53857, 59464, 64704, 69272, 76039, 77845, 83362, 91193],
    "total_assets": [195662, 225354, 247380, 256671, 255708, 250950, 250303, 250829, 266019, 294674],
    "cfo": [21575, 22710, 23271, 30739, 29403, 26124, 38005, 37290, 36222, 40935],
    "expenses": [3247, 3993, 8140, 4858, 4734, 5796, 7001, 6599, 7013, 13901],
})
xchk.to_csv(OUT / "screener_crosscheck.csv", index=False)

# ----------------------------------------------------------------------------
# Market & policy assumptions (all editable in the app)
assum = pd.DataFrame([
    ["risk_free_pct", 7.39, "India 10-year G-sec yield, 15-Oct-2026", "https://tradingeconomics.com/india/government-bond-yield"],
    ["india_erp_pct", 7.08, "India total equity risk premium (mature 4.23% + country 2.85%), Damodaran Jan-2026", "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/ctryprem.html"],
    ["repo_rate_pct", 5.50, "RBI repo after 25 bp hike (Oct-2026); market pricing further hikes", "https://tradingeconomics.com/india/government-bond-yield"],
    ["aaa_psu_spread_pct", 0.45, "Assumed spread of AAA PSU bonds over G-sec (editable)", "Assumption"],
    ["tax_rate_pct", 25.17, "New tax regime rate (company moving to new regime per FY26 note 9)", "FY26 results filing, note 9"],
    ["share_price", 250.00, "NSE close 15-Oct-2026", "https://www.screener.in/company/POWERGRID/consolidated/"],
    ["shares_cr", 930.06, "Equity shares outstanding (₹9,300.60 Cr / ₹10 FV)", "FY26 results filing"],
    ["govt_holding_pct", 51.34, "Promoter (Government of India) holding, Jun-2026", "https://www.screener.in/company/POWERGRID/consolidated/"],
    ["cerc_roe_existing_pct", 15.5, "CERC base RoE on regulated (RTM) transmission assets", "CERC Tariff Regulations (verify final 2024-29 text)"],
    ["cerc_roe_new_pct", 15.0, "CERC base RoE proposed for new transmission projects from FY25 (draft 2024-29)", "https://powerline.net.in/2024/02/05/roe-adjustments-key-highlights-of-cercs-draft-tariff-regulations/"],
    ["normative_de", 2.333, "CERC normative debt:equity 70:30", "CERC Tariff Regulations"],
    ["capex_guidance_fy27", 37000, "FY27 capex guidance (₹ Cr)", "https://mailcontent.icicidirect.com/mailcontent/idirect_powergrid_q1fy27.pdf"],
    ["capitalisation_guidance_fy27", 30000, "FY27 capitalisation guidance (₹ Cr)", "https://mailcontent.icicidirect.com/mailcontent/idirect_powergrid_q1fy27.pdf"],
    ["work_in_hand", 175000, "Works in hand ~₹1.75 lakh Cr (83% TBCB, ~14% RTM)", "https://mailcontent.icicidirect.com/mailcontent/idirect_powergrid_q1fy27.pdf"],
    ["fy26_capex_reported", 40000, "FY26 capex ~₹40,000 Cr (above guidance)", "https://psuwatch.com/amp/story/newsupdates/power-grid-fy26-profit-rises-3-percent-as-capex-surges-past-guidance-2"],
], columns=["key", "value", "description", "source"])
assum.to_csv(OUT / "assumptions.csv", index=False)
print("written:", [f.name for f in OUT.glob("*.csv")])
