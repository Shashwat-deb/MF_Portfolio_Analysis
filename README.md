
# 📈 MF Portfolio Pro

> **Advanced Mutual Fund Portfolio Analysis & Optimization**

🚀 **[Live Demo](https://shashwat-mfportfolio.streamlit.app/)** | 📖 [Features](FEATURES.md) | 📝 [Walkthrough](WALKTHROUGH.md)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shashwat-mfportfolio.streamlit.app/)

A comprehensive Python-based platform for mutual fund portfolio analysis, optimization, and goal-based planning. Built with Streamlit and featuring quantitative institutional analytics powered by Modern Portfolio Theory.

## 🚀 Features

### Core Modules
* **01 Dashboard** - KPI cards, metrics table, risk distribution charts, and Sharpe vs Volatility scatter plot
* **02 Recommendations** - Risk-matched fund suggestions with period performance comparisons and NAV history
* **03 Portfolio Optimization** - Custom allocation weighting, SLSQP Sharpe optimization, correlation heatmap, and Efficient Frontier
* **04 Analytics** - Historical drawdown analysis, rolling metrics (returns & volatility), and head-to-head fund comparison
* **05 Goal Planner** - Compound SIP calculator and Monte Carlo wealth engine with goal probability estimation
* **06 Export & Reports** - Instant CSV downloads and professional markdown portfolio reports
* **07 Category Analysis** - Sector KPI matrix, return and risk dispersion boxplots, and cross-category correlation
* **08 Smart Advisor** - Strategic multi-category allocation based on risk appetite and investment horizon
* **09 Sector Exposure** - Look-through sector exposure, concentration (HHI) warnings and fund overlap, from an optional holdings CSV (`Scheme Name, Sector, Weight`). Module 03 can also cap sector and per-fund exposure when optimizing.

### Metrics Calculated
* CAGR, Volatility, Sharpe Ratio, Sortino Ratio
* Max Drawdown, Calmar Ratio, VaR (95%), CVaR (95%)
* Period-wise returns (1M, 3M, 6M, 1Y, 3Y, 5Y)
* Composite Multi-Factor Score

## 📁 Project Structure

* `app.py`: The main Streamlit web application.
* `mutual_funds_nav_dummy.csv`: Sample historical NAV data for demo mode and analysis.
* `mutual_funds_holdings_dummy.csv`: Sample sector holdings for the demo funds.
* `Mid_SEM.pdf`: Project documentation and methodology details.
* `requirements.txt`: Python package dependencies.
* `WALKTHROUGH.md`: Architecture and development walkthrough.
* `FEATURES.md`: Detailed breakdown of platform capabilities.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shashwat-deb/MF_Portfolio_Analysis.git
   cd MF_Portfolio_Analysis
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## 📊 Usage

1. **Load Data:** Pick a **Data Source** in the sidebar: **Demo** (preloaded sample data), **Live (AMFI)** (search real schemes and pull current NAV history from mfapi.in; AMFI categories are used automatically), or **Upload CSV** (your own NAV file).
2. **Configure Profile:** Adjust risk appetite (Conservative, Balanced, Aggressive) and recommendation parameters.
3. **Navigate:** Use the Institutional Command Center in the sidebar to explore all 9 quantitative modules.

## 🤝 Contributing

Contributions are welcome!

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewFeature`).
3. Commit your Changes (`git commit -m 'Add NewFeature'`).
4. Push to the Branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

*Developed by [Shashwat-deb](https://github.com/Shashwat-deb)*
