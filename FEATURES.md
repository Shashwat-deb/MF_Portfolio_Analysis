# MF Portfolio Pro - Features

A comprehensive Mutual Fund Portfolio Analysis and Optimization platform built with Streamlit.

## 🎯 Key Features

### 📊 Dashboard
- **KPI Cards** - At-a-glance metrics: Average CAGR, Sharpe Ratio, Total Funds, Top Performer
- **Metrics Table** - Complete fund analysis with sorting and filtering
- **Risk Distribution** - Visual breakdown of Conservative, Balanced, and Aggressive funds
- **Sharpe vs Volatility** - Interactive scatter plot for fund comparison

### 🎯 Recommendations
- **Risk-Matched Suggestions** - Top funds based on your risk profile
- **Fund Cards** - Detailed metrics for each recommended fund
- **Period-wise Returns** - Compare 1M, 3M, 6M, 1Y, 3Y, 5Y performance
- **NAV History Chart** - Interactive Plotly visualization

### 💼 Portfolio Builder
- **Custom Allocation** - Build portfolios with adjustable weight sliders
- **Real-time Metrics** - See portfolio return, volatility, and Sharpe as you adjust
- **Sharpe Optimization** - One-click optimal allocation using Modern Portfolio Theory
- **Correlation Heatmap** - Visualize fund correlations for diversification
- **Efficient Frontier** - Plot your portfolio against the optimal frontier

### 📈 Advanced Analytics
- **Drawdown Analysis** - Visualize historical drawdowns
- **Rolling Metrics** - 30-day rolling returns and volatility trends
- **Head-to-Head Comparison** - Compare any two funds side-by-side
- **Normalized NAV Overlay** - Compare performance from a common base

### 🎯 Goal Planner
- **SIP Calculator** - Project future value of systematic investments
- **Goal Tracker** - Set targets and calculate required monthly SIP
- **Monte Carlo Simulation** - Probability estimation for goal achievement
- **Gap Analysis** - Get actionable suggestions to reach your goals

### 📥 Export
- **CSV Downloads** - Export metrics and full NAV data
- **Summary Reports** - Generate markdown reports with key findings

---

## 📊 Metrics Calculated

| Metric | Description |
|--------|-------------|
| **CAGR** | Compound Annual Growth Rate |
| **Volatility** | Annualized standard deviation of returns |
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Calmar Ratio** | CAGR divided by Max Drawdown |
| **VaR (95%)** | Value at Risk at 95% confidence |
| **CVaR (95%)** | Conditional VaR / Expected Shortfall |
| **MF Score** | Composite score for fund ranking |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install streamlit plotly scipy pandas numpy matplotlib

# Run the app
streamlit run app.py
```

Open http://localhost:8501 and enable **Demo Mode** to explore with sample data!

---

## 🛠️ Technologies

- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **SciPy** - Portfolio optimization
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
