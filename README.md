# 🏦 FinSight Bank — Credit Risk Analytics & NPA Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Regression-4B8BBE?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Credit%20Risk%20%7C%20Banking-1A237E?style=flat-square)

> **End-Term Examination Project** | Post Graduate Certificate Programme in Big Data Analytics (PGCP-BDA)
> C-DAC Mumbai | Subject: Credit Risk Analytics & NPA Prediction | Maximum Marks: 100

---

## 📌 Problem Statement

FinSight Bank has experienced rising Non-Performing Assets (NPAs) across its retail loan portfolio.
The bank's dataset spans **2,000,000 loan records** across **nine integrated data sources** —
covering loan origination, borrower bureau data, credit card behaviour, collateral details,
EMI payment history, regional economic indicators, and macroeconomic variables.

**Analytical Objective:**
Build an end-to-end credit risk analytics pipeline to:
1. Assemble and clean the nine-source dataset
2. Perform comprehensive EDA to identify NPA drivers
3. Engineer 13 domain-specific risk features
4. Model **Loss Given Default (LGD%)** using OLS, Ridge, Lasso, and ElasticNet regression
5. Translate findings into five board-ready credit policy recommendations

---

## 📊 Dataset Overview

| File | Description | Key Columns |
|---|---|---|
| `loans_master.csv` | Loan origination details | loan_id, loan_amnt, int_rate, grade, issue_date |
| `loan_performance.csv` | NPA flags and risk tiers | npa_flag, dpd_bucket, risk_tier, lgd_pct |
| `customer_bureau.csv` | Borrower credit bureau data | cibil_score, dti_pct, annual_inc, revol_util |
| `credit_card_behavior.csv` | Card usage patterns | cc_utilization_pct, cc_late_payments_count |
| `loan_enquiry_bureau.csv` | Loan application history | num_enquiries_30d, rejection_rate_pct |
| `monthly_emi_track.csv` | EMI payment behaviour | emi_bounce_count, consecutive_missed_emis |
| `payment_history.csv` | Repayment records | loan_status, lgd_pct, total_pymnt_inr |
| `collateral_assets.csv` | Security details | collateral_value_inr, ltv_ratio_pct |
| `branch_region_economy.csv` | Geographic & macro data | branch_npa_rate, gdp_growth, covid_flag |

> **All 9 CSV files are linked by a common `loan_id` key.**
> **Target variable: `lgd_pct` (Loss Given Default %) — regression on defaulted loans only.**

---

## 🗂️ Project Structure

```
finsight-npa-prediction/
│
├── 📓 NOTEBOOKS
│   ├── FinSight_NPA_Complete_Solution.ipynb   ← Main solution notebook (5 questions)
│   └── Soln.ipynb                             ← Working/scratch notebook
│
├── 🐍 SCRIPTS
│   ├── build_notebook.py                      ← Auto-generates the solution notebook
│   ├── build_exec_summary.py                  ← Generates Executive Summary PDF (ReportLab)
│   └── build_doc.js                           ← Document builder utility
│
├── 📄 REPORTS
│   ├── FinSight_Executive_Summary.pdf         ← 5-page board-ready summary
│   ├── FinSight_Exam_Paper.docx               ← Original examination paper
│   ├── FinSight_Capstone_v2 (2).docx          ← Capstone project document
│   └── FinSight_NPA_Cell_Explanations.docx    ← Cell-by-cell code explanations
│
├── 📊 VISUALIZATIONS (13 Charts)
│   ├── q2a_loan_status_distribution.png       ← Bar + Pie: loan status
│   ├── q2b_cibil_kde.png                      ← KDE: CIBIL by default status
│   ├── q2c_histograms.png                     ← 12-panel histogram grid
│   ├── q2d_correlation_heatmap.png            ← Pearson correlation matrix
│   ├── q2e_boxplots.png                       ← 6 boxplots vs loan_status
│   ├── q2f_default_by_grade.png               ← Default rate by loan grade
│   ├── q2g_default_by_purpose.png             ← Default rate by loan purpose
│   ├── q2h_default_by_state.png               ← Top 10 states by default rate
│   ├── q2i_annual_default_trend.png           ← Annual trend + COVID shock
│   ├── q2j_repo_vs_default.png                ← Dual-axis: Repo rate vs default
│   ├── q2k_lgd_distribution.png               ← LGD histogram + KDE
│   ├── q2l_cibil_vs_lgd.png                   ← Scatter: CIBIL vs LGD
│   └── q4d_regression_diagnostics.png         ← 4-panel OLS diagnostics
│
├── 💾 DATA
│   ├── loans_master.csv
│   ├── loan_performance.csv
│   ├── customer_bureau.csv
│   ├── credit_card_behavior.csv
│   ├── loan_enquiry_bureau.csv
│   ├── monthly_emi_track.csv
│   ├── payment_history.csv
│   ├── collateral_assets.csv
│   ├── branch_region_economy.csv
│   ├── finsight_merged.parquet                ← Merged + cleaned master dataset
│   └── CSV INFO.txt                           ← Column schema reference
│
├── .gitignore
└── README.md
```

---

## 🔍 Analytical Sections (Notebook Structure)

### Q1 — Data Acquisition, Joining & Cleaning `[30 Marks]`

| Sub-question | Task |
|---|---|
| Q1(a) | Load 9 CSVs → downcast → Parquet export. Memory before/after report |
| Q1(b) | Sequential left-merges on `loan_id`. Row count assertions. Orphan analysis |
| Q1(c) | Detect 8 injected dirty-data issues. `dirty_flag` column. Imputation log |
| Q1(d) | MCAR / MAR / MNAR classification for 4 high-missing columns |
| Q1(e) | Winsorisation at P1/P99 on 6 most skewed columns |

### Q2 — Exploratory Data Analysis `[25 Marks]`

All 12 visualisations are compulsory. Each includes title, axis labels, and a 3-sentence business insight caption.

| Chart | Description |
|---|---|
| Q2(a) | Loan status distribution — Bar + Pie |
| Q2(b) | CIBIL KDE — Default vs Performing (Cohen's d, KS test) |
| Q2(c) | 12-panel histogram grid + log transformation skew reduction |
| Q2(d) | Pearson correlation heatmap — top 20 features |
| Q2(e) | 6 side-by-side boxplots vs `loan_status` |
| Q2(f) | Default rate by loan grade (A → G monotonicity check) |
| Q2(g) | Default rate by loan purpose — sorted bar chart |
| Q2(h) | Top 10 states by default rate — flagged vs portfolio average |
| Q2(i) | Annual default rate trend 2010–2024 + COVID-19 shock |
| Q2(j) | Dual-axis: RBI repo rate vs annual default rate |
| Q2(k) | LGD distribution (histogram + KDE) on defaulted loans |
| Q2(l) | Scatter plot: CIBIL score vs LGD with regression line |

### Q3 — Feature Engineering `[20 Marks]`

13 engineered features across 4 thematic groups:

```
Repayment Burden  →  emi_to_income_ratio, loan_to_income_ratio,
                      rate_spread_pct, real_interest_rate_pct

Bureau Behaviour  →  credit_util_composite, delinq_severity_score,
                      enq_velocity_score

Income & Collateral →  income_stability_ratio, credit_depth_score,
                        collateral_coverage_ratio

Transformations   →  log_annual_inc, log_loan_amnt, covid_issue_year_flag
```

Each feature: formula → domain rationale → `.describe()` → Pearson r with `lgd_pct`

### Q4 — Regression Modelling & Diagnostics `[15 Marks]`

| Step | Detail |
|---|---|
| Leakage removal | Drop post-default columns before modelling |
| VIF analysis | Remove all features with VIF > 10 |
| OLS baseline | Full `statsmodels` summary — R², Adj R², F-stat, DW, JB, CN |
| Ridge | GridSearchCV 5-fold CV — optimal alpha, CV RMSE, test R² |
| Lasso | Feature zeroing analysis — which features are excluded and why |
| ElasticNet | L1/L2 blend — optimal alpha + l1_ratio |
| Diagnostics | 4-panel figure: Residuals vs Fitted, QQ plot, Scale-Location, Cook's D |

### Q5 — Business Recommendations `[10 Marks]`

Five board-ready credit policy recommendations, each with:
- **Data evidence** (specific statistic from EDA/model)
- **Quantified impact** (₹ Crore or percentage points)
- **Implementation action** (concrete step with timeline)

| # | Recommendation |
|---|---|
| 1 | Implement CIBIL score floor of 650 for loan approvals |
| 2 | Cap EMI-to-Income Ratio (FOIR) at 50% |
| 3 | Restrict origination in Loan Grades F and G |
| 4 | Mandate collateral for loans above ₹5 lakh to high-risk segments |
| 5 | Build macroeconomic stress-testing framework using COVID & repo rate flags |

---

## 🧹 Data Cleaning Pipeline

A 5-phase cleaning pipeline (`build_notebook.py`) handles all quality issues:

| Phase | Action | Records Affected |
|---|---|---|
| Deduplication | Remove duplicate `transaction_id` | ~30,000 rows |
| Null Handling | MCAR/MAR/MNAR classification + targeted imputation | 4 columns |
| Type Casting | Parse mixed date formats; standardise booleans/categoricals | All rows |
| Logic Fixes | Delivery before order, price mismatches, return flag fixes | ~200,000 rows |
| Outlier Treatment | Winsorise 6 most skewed columns at P1/P99 | Varies |

**8 Injected Dirty-Data Issues Detected:**

| # | Column | Issue | Fix |
|---|---|---|---|
| 1 | `age` | Values < 18 or > 100 | Impute with median |
| 2 | `cibil_score` | Outside 300–900 range | Impute with median |
| 3 | `annual_inc_inr` | Negative income | Absolute value |
| 4 | `revol_util_pct` | > 100% utilisation | Cap at 100 |
| 5 | `dti_pct` | Negative DTI | Impute with median |
| 6 | `int_rate_pct` | Negative interest rate | Impute with median |
| 7 | `ltv_ratio_pct` | > 1000% LTV | Impute with median |
| 8 | `loan_term_months` | Non-standard term | Impute with mode |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/finsight-npa-prediction.git
cd finsight-npa-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy reportlab pyarrow jupyter
```

### 3. Place your datasets
```
Copy all 9 CSV files into the project root directory.
```

### 4. Launch the solution notebook
```bash
jupyter notebook FinSight_NPA_Complete_Solution.ipynb
```

### 5. Run all cells (Kernel → Restart & Run All)

The notebook will automatically:
- Merge all 9 CSV files into `finsight_merged.parquet`
- Clean, flag, and impute dirty records
- Generate all 13 visualisation charts
- Fit and compare all 4 regression models
- Print all business recommendations

### 6. (Optional) Regenerate the Executive Summary PDF
```bash
python build_exec_summary.py
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Pandas | 2.x | Data loading, cleaning, merging |
| NumPy | 1.24+ | Numerical operations |
| Matplotlib | 3.7+ | Base visualisations |
| Seaborn | 0.12+ | Statistical charts, heatmaps |
| Statsmodels | 0.14+ | OLS regression, VIF, diagnostics |
| Scikit-learn | 1.3+ | Ridge, Lasso, ElasticNet, GridSearchCV |
| SciPy | 1.11+ | KS test, t-test, Cohen's d |
| ReportLab | 4.x | Executive Summary PDF generation |
| PyArrow | 12+ | Parquet read/write |
| Jupyter | 7.x | Interactive notebook environment |

---

## 📋 Exam Marks Breakdown

| Question | Topic | Max Marks |
|---|---|---|
| Q1 | Data Acquisition, Joining & Cleaning | 30 |
| Q2 | Exploratory Data Analysis (12 charts) | 25 |
| Q3 | Feature Engineering (13 features) | 20 |
| Q4 | Regression Modelling & Diagnostics | 15 |
| Q5 | Business Recommendations | 10 |
| **TOTAL** | | **100** |

---

## 📁 Key Files Reference

| File | What it is |
|---|---|
| `FinSight_NPA_Complete_Solution.ipynb` | **Main notebook** — submit this for examination |
| `FinSight_Executive_Summary.pdf` | **5-page board summary** — submit alongside notebook |
| `finsight_merged.parquet` | Merged + cleaned master dataset (auto-generated) |
| `build_notebook.py` | Script that generates the notebook programmatically |
| `build_exec_summary.py` | Script that generates the PDF using ReportLab |
| `CSV INFO.txt` | Column schema and data dictionary for all 9 files |
| `FinSight_NPA_Cell_Explanations.docx` | Plain-language explanation of every notebook cell |

---

## ⚠️ Important Notes

- **Do NOT impute target variables** (`loan_status`, `lgd_pct`) — drop missing rows and log the count
- **Do NOT delete dirty records** — flag with `dirty_flag = 1` and impute
- **All modelling is on defaulted loans only** — `df[df["loan_status"] == 1]`
- **Remove post-default leakage columns** before fitting models (`total_pymnt`, `recoveries`, `net_loss_inr`)
- **random_state=42** is used throughout for reproducibility
- **All charts must have** title + axis labels + 3-sentence business insight caption

---

## 🔑 Key Design Decisions

**Why sentinel 999 for `mths_since_last_delinq`?**
Missing means the borrower has *never* been delinquent — this is informative (MNAR). Imputing with median would incorrectly assign a delinquency history to clean borrowers.

**Why winsorise instead of drop outliers?**
Exam instructions explicitly state: *"Do NOT delete dirty or outlier records."* Winsorisation caps extreme values at P1/P99 while preserving every row.

**Why model on defaulted loans only?**
LGD (Loss Given Default) is only meaningful for loans that have already defaulted. Including performing loans would introduce noise and misrepresent the regression target.

**Why remove high-VIF features before OLS?**
VIF > 10 signals severe multicollinearity — the affected coefficient's standard error is inflated more than 10× compared to a fully independent feature, making it statistically unreliable.

---

## 👤 Author

**Vedant**
PG-DBDA Student | C-DAC Kharghar, Navi Mumbai
[LinkedIn](https://www.linkedin.com/in/vedant-gajbhiye-264299205/) | [GitHub](https://github.com/Vedant560)

---

> ⭐ If this project helped you understand credit risk analytics, give it a star!

---

*Confidential — FinSight Bank NPA Study | C-DAC Mumbai 2024*
*For Academic Use Only*
