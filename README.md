
# 📈 MF Portfolio Pro

> **Advanced Mutual Fund Portfolio Analysis & Optimization**

🚀 **[Live Demo](https://shashwat-mfportfolio.streamlit.app/)** | 📖 [Features](FEATURES.md) | 📝 [Walkthrough](WALKTHROUGH.md)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shashwat-mfportfolio.streamlit.app/)

A comprehensive Python-based application for mutual fund portfolio analysis, optimization, and goal-based planning. Built with Streamlit and featuring advanced analytics powered by Modern Portfolio Theory.

## 🚀 Features

### Core Capabilities
* **📊 Dashboard** - KPI cards, metrics table, risk distribution charts
* **🎯 Recommendations** - Risk-matched fund suggestions with detailed analytics
* **💼 Portfolio Builder** - Custom allocation with Sharpe optimization & Efficient Frontier
* **📈 Advanced Analytics** - Drawdown analysis, rolling metrics, head-to-head comparison
* **🎯 Goal Planner** - SIP calculator with Monte Carlo simulation
* **📥 Export** - CSV downloads and summary reports

### Metrics Calculated
* CAGR, Volatility, Sharpe Ratio, Sortino Ratio
* Max Drawdown, Calmar Ratio, VaR (95%), CVaR (95%)
* Period-wise returns (1M, 3M, 6M, 1Y, 3Y, 5Y)


## 📁 Project Structure

* `app.py`: The main entry point for the application.
* `mutual_funds_nav_dummy.csv`: Sample historical NAV data for testing and analysis.
* `Mid_SEM.pdf`: Project documentation and methodology details.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Shashwat-deb/MF_Portfolio_Analysis.git](https://github.com/Shashwat-deb/MF_Portfolio_Analysis.git)
   cd MF_Portfolio_Analysis
```

2. **Install Dependencies:**
```bash
pip install pandas numpy matplotlib

```


3. **Run the App:**
```bash
python app.py

```



## 📊 Usage

1. **Upload/Load Data:** The app uses the included `.csv` files to pull historical NAV data.
2. **Select Mode:** Choose between creating a **New Portfolio** or **Assessing an Existing** one.
3. **Set Risk:** Define your risk appetite to see suggested mutual fund allocations.
4. **View Results:** Review the performance metrics and risk-return analysis.

## 🤝 Contributing

Contributions are welcome!

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewFeature`).
3. Commit your Changes (`git commit -m 'Add NewFeature'`).
4. Push to the Branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

*Developed by [Shashwat-deb*](https://github.com/Shashwat-deb)

```
