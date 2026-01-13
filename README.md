# 🎯 Promo Pulse Dashboard

**UAE Retail Analytics & Promotion Simulator**

A comprehensive business intelligence dashboard designed for UAE retail operations, featuring automated data cleaning, KPI tracking, promotional campaign simulation, and dual-perspective analytics (Manager & Executive views).

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Generation & Cleaning](#data-generation--cleaning)
- [KPI Categories](#kpi-categories)
- [Simulation Engine](#simulation-engine)
- [Critical Thinking & Assumptions](#critical-thinking--assumptions)

---

## ✨ Features

### Core Functionality
- **Synthetic Data Generation**: Generates realistic UAE retail data with intentional errors for demonstration
- **Automated Data Cleaning**: Background processing with comprehensive error logging
- **Dual POV Dashboard**: Manager (operational) and Executive (strategic) perspectives
- **Promotional Campaign Simulator**: Rule-based demand forecasting with constraint validation
- **AI-Powered Recommendations**: Intelligent insights based on simulation results

### Data Visualizations
- **BCG Matrix**: Channel positioning analysis (Stars, Cash Cows, Question Marks, Dogs)
- **Sunburst Chart**: Hierarchical revenue breakdown (City → Channel → Category)
- **Heatmap**: Category × Channel revenue intensity
- **Revenue Trends**: Time-series with 7-day moving average
- **Pareto Charts**: Issue distribution analysis

### Data Quality Features
- **13+ Error Types**: Intentional data quality issues for cleaning demonstration
- **<3% Outlier Rate**: Configurable outlier detection and capping
- **Department Attribution**: Issues traced to responsible departments
- **Comprehensive Error Logs**: Timestamp, table, record ID, issue type, original/corrected values, action taken

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/promo_pulse_dashboard.git
cd promo_pulse_dashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## 📖 Usage

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Navigation

1. **Dashboard Tab**: View KPIs and visualizations based on selected POV
2. **Simulation Tab**: Configure and run promotional campaign simulations
3. **Data Comparison Tab**: Side-by-side raw vs. cleaned data comparison
4. **Error Logs Tab**: Filter and download detailed error logs

### Controls

- **Data Source Toggle**: Switch between pre-built synthetic data and custom uploads
- **Data Type Toggle**: Use raw or cleaned data for analysis
- **POV Toggle**: Manager (operational) or Executive (strategic) view
- **Filters**: City, Channel, Category, Brand, Date Range

---

## 📁 Project Structure

```
promo_pulse_dashboard/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── app.py                    # Main Streamlit application
├── data_generator.py         # Synthetic data generation with errors
├── cleaner.py                # Data validation and cleaning engine
├── simulator.py              # KPI calculation and simulation engine
├── data/
│   ├── raw/                  # Generated raw data with errors
│   │   ├── products.csv
│   │   ├── stores.csv
│   │   ├── customers.csv
│   │   ├── sales.csv
│   │   ├── inventory.csv
│   │   ├── campaigns.csv
│   │   └── departments.csv
│   └── clean/                # Cleaned and validated data
│       ├── products.csv
│       ├── stores.csv
│       ├── customers.csv
│       ├── sales.csv
│       ├── inventory.csv
│       ├── campaigns.csv
│       ├── departments.csv
│       └── logs/
│           └── issues_log.csv
├── utils/                    # Utility functions (optional)
├── components/               # Reusable UI components (optional)
└── assets/                   # Images, logos, etc. (optional)
```

---

## 🔧 Data Generation & Cleaning

### Synthetic Data Tables

| Table | Records | Description |
|-------|---------|-------------|
| Products | 300 | SKU catalog with categories, brands, pricing |
| Stores | 18 | 6 stores × 3 channels across UAE cities |
| Customers | 5,000 | Customer demographics and segments |
| Sales | 35,000 | Transaction data for 2024 |
| Inventory | 54,000 | 30-day snapshot per product-store |
| Campaigns | 10 | Marketing campaign definitions |
| Departments | 6 | Organizational structure |

### Intentional Error Types

| Error Type | Rate | Description |
|------------|------|-------------|
| City Typos | 8% | Dubaai, Dubay, Abudhabi, SHARJAH, etc. |
| Wrong Year | 2% | 2025 dates (should be 2024) |
| Invalid Category | 2% | Typos like "Electronicss", "beautty" |
| Missing Unit Cost | 1.5% | Null values in cost field |
| Corrupted Timestamps | 1.5% | Invalid date formats |
| Missing Product ID | 1% | Null foreign keys |
| Duplicate Orders | 0.8% | Repeated transactions |
| Negative Stock | 0.5% | Invalid inventory levels |
| Outlier Quantity | 0.4% | Extreme order quantities |
| Outlier Price | 0.3% | Abnormal price values |
| Invalid Email | 3% | Malformed email addresses |

### Cleaning Rules

1. **City Standardization**: All variations mapped to `Dubai`, `Abu Dhabi`, `Sharjah`
2. **Year Correction**: 2025 → 2024 with logging
3. **Category Correction**: Typo patterns auto-corrected
4. **Email Validation**: Regex pattern matching
5. **Outlier Capping**: Quantity ≤20, Price within 0.2x-5x base
6. **Duplicate Removal**: Keep first by timestamp
7. **Missing Value Imputation**: Category median for costs, store median for discounts

---

## 📊 KPI Categories

### Business KPIs
- **Net Revenue**: Gross revenue - refunds (paid orders only)
- **Gross Margin %**: (Revenue - COGS) / Revenue × 100
- **Avg Order Value (AOV)**: Net revenue / total orders
- **Revenue Growth %**: YoY or period comparison

### Inventory KPIs
- **Stockout Rate %**: Zero-stock items / total items
- **Low Stock %**: Items below reorder point
- **Inventory Turnover**: COGS / avg inventory value

### Channel KPIs
- **Revenue by Channel**: App, Web, Marketplace breakdown
- **Channel Share %**: Contribution to total revenue
- **AOV by Channel**: Average order value per channel

### Customer KPIs
- **Unique Customers**: Distinct customer count
- **Repeat Rate %**: Customers with 2+ orders
- **Customer Segments**: Regular, Premium, VIP breakdown

### Promotion KPIs
- **Promo ROI**: Revenue per AED discount given
- **Budget Utilization %**: Spent / allocated × 100
- **Discounted Revenue %**: Sales with discount applied

### Data Quality KPIs
- **Total Issues**: Count of cleaned errors
- **Issues by Type**: Distribution of error categories
- **Issues by Department**: Accountability tracking

---

## 🎯 Simulation Engine

### Demand Uplift Model

The simulator uses a rule-based approach (no ML) with transparent assumptions:

```
Base Uplift = 1 + (discount% / 100) × 1.2 × (1 - discount% / 200)
```

**Sensitivity Factors:**

| Channel | Factor | Category | Factor | City | Factor |
|---------|--------|----------|--------|------|--------|
| Marketplace | 1.5× | Electronics | 1.4× | Dubai | 1.2× |
| App | 1.3× | Fashion | 1.3× | Abu Dhabi | 1.1× |
| Web | 1.0× | Grocery | 0.8× | Sharjah | 1.0× |

**Total Uplift Formula:**
```
Total Uplift = Base Uplift × √(Channel × Category × City)
Capped at 3× maximum
```

### Constraints

1. **Budget Constraint**: Total promo spend ≤ allocated budget
2. **Margin Floor**: Gross margin % ≥ configured floor
3. **Stock Constraint**: Cannot sell more than available inventory

### Violation Tracking

- `STOCKOUT_RISK`: Projected demand exceeds stock
- `MARGIN_FLOOR`: Projected margin below threshold
- `BUDGET_EXCEEDED`: Promo spend exceeds budget

---

## 🧠 Critical Thinking & Assumptions

### 1. Impact of Cleaning Rules on Analysis

**City Standardization**: Geographic analysis depends entirely on correct city mapping. While our rules handle common typos, novel variations would require rule updates. This could affect regional revenue attribution and store performance comparisons.

**Year Correction**: Auto-correcting 2025→2024 assumes all such dates are errors. In a real scenario, this might incorrectly modify legitimate future-dated records (e.g., scheduled campaigns). The logging preserves auditability.

**Outlier Capping**: Capping extreme values (rather than removing) preserves data points but may hide fraud or genuine high-value transactions. A real implementation should flag these for human review.

### 2. Simulation Assumptions

**Sensitivity Factors**: Channel and category sensitivities are based on typical retail patterns. UAE market specifics (Ramadan seasonality, summer slowdown, local preferences) may differ significantly from these assumptions.

**No Cannibalization**: The model assumes discounts create new demand rather than shifting existing purchases. In reality, some uplift would come from forward-buying or category switching.

**No Competition**: External factors (competitor promotions, market events) are not modeled.

### 3. Budget vs. Margin vs. Stockout Tradeoff

This is the central tension in promotional planning:

- **Deep Discounts**: Higher uplift but lower margins and higher stockout risk
- **Broad Reach**: Moderate discounts across more products, balanced risk
- **Targeted**: Focus on high-margin, well-stocked items for optimal ROI

**Recommendation**: Prioritize categories with:
- Above-average margin (>25%)
- Adequate stock levels (>30 days supply)
- High price elasticity (Electronics, Fashion)

### 4. Scope Exclusions

To meet project constraints, the following were excluded:

- **ML Forecasting**: Rule-based approach is more interpretable
- **Multi-Period Optimization**: Single simulation period only
- **Customer Segmentation Analysis**: Basic segment breakdown only
- **Supply Chain Simulation**: No replenishment modeling
- **A/B Testing Framework**: No experimental comparison

---

## 📝 Reference Values

| Metric | Reference | Source |
|--------|-----------|--------|
| Promo Budget | AED 1,000,000 | Configurable |
| Target Margin | 25% | Industry benchmark |
| Target Stockout Rate | <5% | Best practice |
| Target Return Rate | <5% | Industry average |
| Payment Failure Rate | <3% | Gateway benchmark |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is created for educational purposes as part of a Business Analytics course.

---

## 👤 Author

Created for Final Project - Business Analytics Dashboard Assignment

---

## 🙏 Acknowledgments

- Streamlit for the dashboard framework
- Plotly for interactive visualizations
- Pandas for data manipulation
