import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import io
from datetime import datetime, timedelta

# Set plotly_white as the global default template for ALL charts (light theme)
pio.templates.default = "plotly_white"

# ==================== CONSTANTS ====================
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04

# ==================== CYCLOPS CLUB INSPIRED STYLING ====================
def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* ========== ROOT VARIABLES ========== */
    :root {
        --bg-dark: #0a0a0a;
        --bg-card: #ffffff;
        --bg-muted: #f8f8f8;
        --bg-accent: #1a1a1a;
        --text-primary: #0a0a0a;
        --text-muted: #666666;
        --text-light: #999999;
        --border: #e5e5e5;
        --border-dark: #333333;
    }
    
    /* ========== GLOBAL STYLES ========== */
    .main {
        background: var(--bg-muted) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(180deg, #f8f8f8 0%, #ffffff 100%);
    }
    
    /* ========== STEP NAVIGATOR ========== */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 2rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    
    .step-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5rem;
        font-weight: 700;
        color: var(--bg-dark);
        line-height: 1;
        letter-spacing: -0.05em;
    }
    
    .step-content {
        flex: 1;
    }
    
    .step-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-light);
        margin-bottom: 0.25rem;
    }
    
    .step-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    
    /* ========== KPI CARDS - CYCLOPS STYLE ========== */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        padding: 1.5rem;
        border-radius: 0;
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .kpi-card:hover {
        border-color: var(--bg-dark);
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
    }
    
    .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-light);
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    
    .kpi-sublabel {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.1em;
        color: var(--text-light);
        margin-top: 0.5rem;
        text-transform: uppercase;
    }
    
    .kpi-indicator {
        position: absolute;
        top: 1rem;
        right: 1rem;
        width: 8px;
        height: 8px;
        background: #0a0a0a;
        border-radius: 50%;
    }
    
    .kpi-indicator.live {
        background: #00c853;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* ========== TECHNICAL LABELS ========== */
    .tech-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .tech-label::before {
        content: '●';
        font-size: 0.5rem;
        color: var(--text-light);
    }
    
    .tech-value {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* ========== HEADER - MINIMAL ========== */
    .cyclops-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    
    .cyclops-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }
    
    .cyclops-status {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    
    .rec-indicator {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: var(--text-muted);
    }
    
    .rec-dot {
        width: 6px;
        height: 6px;
        background: #ff3b30;
        border-radius: 50%;
        animation: blink 1.5s infinite;
    }
    
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.3; }
    }
    
    /* ========== TABS - STEP BASED ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-card);
        border: 1px solid var(--border);
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 1rem 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        border-right: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"]:last-child {
        border-right: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-dark) !important;
        color: white !important;
    }
    
    /* ========== CARDS & CONTAINERS ========== */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .data-grid {
        display: grid;
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
    }
    
    .data-cell {
        background: var(--bg-card);
        padding: 1rem;
    }
    
    /* ========== SIDEBAR - MINIMAL ========== */
    section[data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        background: var(--bg-dark) !important;
        color: white !important;
        border: none !important;
        border-radius: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        padding: 0.8rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: #333 !important;
        transform: translateY(-1px);
    }
    
    /* ========== PROGRESS BAR ========== */
    .progress-bar {
        height: 4px;
        background: var(--border);
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: var(--bg-dark);
        transition: width 0.5s ease;
    }
    
    /* ========== SECTION HEADER ========== */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-primary);
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border);
        margin: 2rem 0 1.5rem 0;
    }
    
    /* ========== FUND CARDS - MINIMAL ========== */
    .fund-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .fund-card:hover {
        border-color: var(--bg-dark);
    }
    
    .fund-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .fund-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border);
    }
    
    .fund-metric:last-child {
        border-bottom: none;
    }
    
    /* ========== ANIMATIONS ========== */
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ========== TEXT VISIBILITY FIXES ========== */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #0a0a0a !important;
    }
    
    .stMetric label, .stMetric [data-testid="stMetricLabel"] {
        color: #666666 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #0a0a0a !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #333333 !important;
    }
    
    .stCaption, .stCaption p {
        color: #666666 !important;
    }
    
    .stDataFrame, .stDataFrame th, .stDataFrame td {
        color: #0a0a0a !important;
    }
    
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #0a0a0a !important;
    }
    
    .stWarning, .stInfo, .stSuccess, .stError {
        color: #0a0a0a !important;
    }
    
    /* Fix text inputs */
    .stTextInput label, .stNumberInput label {
        color: #0a0a0a !important;
    }
    
    /* ========== HIDE STREAMLIT BRANDING ========== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)



# ==================== CORE METRIC FUNCTIONS ====================
def cagr(nav):
    """Calculate Compound Annual Growth Rate"""
    if len(nav) < 2:
        return np.nan
    start = nav.iloc[0]
    end = nav.iloc[-1]
    years = len(nav) / TRADING_DAYS
    if years == 0 or start == 0:
        return np.nan
    return (end / start) ** (1 / years) - 1


def annual_vol(ret):
    """Calculate Annualized Volatility"""
    if ret.dropna().empty:
        return np.nan
    return ret.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(c, v, rf=RISK_FREE_RATE):
    """Calculate Sharpe Ratio"""
    if v == 0 or np.isnan(v):
        return np.nan
    return (c - rf) / v


def sortino_ratio(ret, rf=RISK_FREE_RATE):
    """Calculate Sortino Ratio (uses downside deviation)"""
    if ret.dropna().empty:
        return np.nan
    negative_returns = ret[ret < 0]
    if negative_returns.empty:
        return np.nan
    downside_std = negative_returns.std() * np.sqrt(TRADING_DAYS)
    if downside_std == 0:
        return np.nan
    annual_ret = ret.mean() * TRADING_DAYS
    return (annual_ret - rf) / downside_std


def max_drawdown(nav):
    """Calculate Maximum Drawdown"""
    if nav.empty:
        return np.nan
    rm = nav.cummax()
    dd = nav / rm - 1
    return dd.min()


def calmar_ratio(nav, ret):
    """Calculate Calmar Ratio (CAGR / Max Drawdown)"""
    c = cagr(nav)
    mdd = abs(max_drawdown(nav))
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return c / mdd


def value_at_risk(ret, confidence=0.95):
    """Calculate Value at Risk at given confidence level"""
    if ret.dropna().empty:
        return np.nan
    return np.percentile(ret.dropna(), (1 - confidence) * 100)


def conditional_var(ret, confidence=0.95):
    """Calculate Conditional VaR (Expected Shortfall)"""
    if ret.dropna().empty:
        return np.nan
    var = value_at_risk(ret, confidence)
    return ret[ret <= var].mean()


def classify(vol):
    """Classify fund based on volatility"""
    if np.isnan(vol):
        return "Unknown"
    if vol < 0.05:
        return "Conservative"
    elif vol < 0.12:
        return "Balanced"
    else:
        return "Aggressive"


def calculate_rolling_metrics(nav, ret, window=30):
    """Calculate rolling returns and volatility"""
    rolling_ret = nav.pct_change(window).dropna() * (TRADING_DAYS / window)
    rolling_vol = ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return rolling_ret, rolling_vol


def calculate_period_returns(nav, periods_days):
    """Calculate returns for different periods"""
    period_returns = {}
    for name, days in periods_days.items():
        if len(nav) > days:
            period_nav = nav.iloc[-days:]
            ret = (period_nav.iloc[-1] / period_nav.iloc[0] - 1)
            period_returns[name] = ret
        else:
            period_returns[name] = np.nan
    return period_returns


# ==================== COMPUTE METRICS ====================
def compute_metrics(df):
    """Compute all metrics for each scheme"""
    out = []
    for scheme, grp in df.groupby("Scheme Name"):
        nav = grp["NAV"]
        ret = grp["Return"]
        c = cagr(nav)
        v = annual_vol(ret)
        s = sharpe_ratio(c, v)
        sort = sortino_ratio(ret)
        m = max_drawdown(nav)
        cal = calmar_ratio(nav, ret)
        var = value_at_risk(ret)
        cvar = conditional_var(ret)
        r = classify(v)
        
        # Period returns
        periods = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}
        period_rets = calculate_period_returns(nav, periods)
        
        out.append({
            "Scheme": scheme,
            "CAGR": c,
            "Volatility": v,
            "Sharpe": s,
            "Sortino": sort,
            "MaxDrawdown": m,
            "Calmar": cal,
            "VaR_95": var,
            "CVaR_95": cvar,
            "RiskClass": r,
            **{f"Return_{k}": v for k, v in period_rets.items()}
        })
    
    mdf = pd.DataFrame(out)
    mdf["MFScore"] = (
        0.35 * mdf["Sharpe"].fillna(0)
        + 0.25 * mdf["CAGR"].fillna(0)
        + 0.20 * mdf["Sortino"].fillna(0).clip(-2, 2) / 2
        - 0.20 * mdf["MaxDrawdown"].abs().fillna(0)
    )
    return mdf


# ==================== PORTFOLIO OPTIMIZATION ====================
def calculate_portfolio_metrics(weights, returns_df):
    """Calculate portfolio metrics given weights"""
    portfolio_returns = (returns_df * weights).sum(axis=1)
    port_annual_ret = portfolio_returns.mean() * TRADING_DAYS
    port_annual_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    port_sharpe = (port_annual_ret - RISK_FREE_RATE) / port_annual_vol if port_annual_vol != 0 else 0
    return port_annual_ret, port_annual_vol, port_sharpe


def optimize_portfolio(returns_df, target_return=None):
    """Optimize portfolio weights using Modern Portfolio Theory"""
    n = len(returns_df.columns)
    
    def neg_sharpe(weights):
        ret, vol, sharpe = calculate_portfolio_metrics(weights, returns_df)
        return -sharpe
    
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    if target_return:
        constraints.append({
            "type": "eq", 
            "fun": lambda x: calculate_portfolio_metrics(x, returns_df)[0] - target_return
        })
    
    bounds = tuple((0, 1) for _ in range(n))
    initial = np.array([1/n] * n)
    
    result = minimize(neg_sharpe, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else initial


def generate_efficient_frontier(returns_df, n_points=50):
    """Generate points along the efficient frontier"""
    n = len(returns_df.columns)
    
    def portfolio_vol(weights):
        return calculate_portfolio_metrics(weights, returns_df)[1]
    
    # Find min and max returns
    mean_returns = returns_df.mean() * TRADING_DAYS
    min_ret, max_ret = mean_returns.min(), mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    frontier_vols = []
    frontier_rets = []
    
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x, t=target: (returns_df * x).sum(axis=1).mean() * TRADING_DAYS - t}
        ]
        bounds = tuple((0, 1) for _ in range(n))
        initial = np.array([1/n] * n)
        
        result = minimize(portfolio_vol, initial, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            frontier_vols.append(result.fun)
            frontier_rets.append(target)
    
    return frontier_rets, frontier_vols


# ==================== SIP CALCULATOR ====================
def calculate_sip(monthly_amount, expected_return, years):
    """Calculate SIP returns"""
    months = years * 12
    monthly_rate = expected_return / 12
    
    # Future Value of SIP
    if monthly_rate == 0:
        future_value = monthly_amount * months
    else:
        future_value = monthly_amount * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    
    total_invested = monthly_amount * months
    wealth_gained = future_value - total_invested
    
    return future_value, total_invested, wealth_gained


def calculate_goal_probability(target, monthly_sip, years, historical_returns):
    """Monte Carlo simulation for goal achievement probability"""
    n_simulations = 1000
    months = years * 12
    successes = 0
    
    for _ in range(n_simulations):
        portfolio_value = 0
        for month in range(months):
            monthly_return = np.random.choice(historical_returns)
            portfolio_value = (portfolio_value + monthly_sip) * (1 + monthly_return/12)
        if portfolio_value >= target:
            successes += 1
    
    return successes / n_simulations


# ==================== LIGHT THEME CHART CONFIG ====================
# Using vibrant, high-contrast colors for readability on light backgrounds
VIBRANT_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ca8a04", "#0891b2", "#c026d3", "#ea580c"]

# Light theme layout configuration
LIGHT_LAYOUT = {
    "template": "plotly_white",
    "font": {"family": "Space Grotesk, sans-serif", "color": "#0a0a0a"},
    "title": {"font": {"size": 14, "weight": 600, "color": "#0a0a0a"}},
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff",
    "xaxis": {"gridcolor": "#e5e5e5", "linecolor": "#cccccc", "tickcolor": "#666666", "tickfont": {"color": "#333333"}},
    "yaxis": {"gridcolor": "#e5e5e5", "linecolor": "#cccccc", "tickcolor": "#666666", "tickfont": {"color": "#333333"}},
    "legend": {"font": {"color": "#0a0a0a"}}
}

# Keep MONO_COLORS for backward compatibility
MONO_COLORS = VIBRANT_COLORS

def apply_mono_layout(fig):
    """Apply light theme styling to Plotly figure"""
    fig.update_layout(**LIGHT_LAYOUT)
    return fig


# ==================== VISUALIZATION FUNCTIONS ====================
def create_nav_chart(navdf, selected_schemes):
    """Create interactive NAV chart - Light theme with vibrant colors"""
    fig = px.line(
        navdf, x="Date", y="NAV", color="Scheme Name",
        title="NAV HISTORY",
        color_discrete_sequence=VIBRANT_COLORS
    )
    # Make lines thicker for better visibility
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#0a0a0a")),
        height=500
    )
    return apply_mono_layout(fig)


def create_drawdown_chart(nav_series, scheme_name):
    """Create drawdown visualization - Light theme"""
    running_max = nav_series.cummax()
    drawdown = (nav_series / running_max - 1) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_series.index, y=drawdown,
        fill="tozeroy",
        fillcolor="rgba(220, 38, 38, 0.15)",
        line=dict(color="#dc2626", width=2),
        name="Drawdown"
    ))
    fig.update_layout(
        title=f"DRAWDOWN ANALYSIS — {scheme_name.upper()}",
        yaxis_title="Drawdown (%)",
        height=400
    )
    return apply_mono_layout(fig)


def create_correlation_heatmap(returns_df):
    """Create correlation heatmap - Light theme with readable colors"""
    corr = returns_df.corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[[0, "#eff6ff"], [0.5, "#3b82f6"], [1, "#1e3a8a"]],
        aspect="auto",
        title="CORRELATION MATRIX"
    )
    fig.update_layout(height=500)
    fig.update_traces(textfont=dict(color="#0a0a0a", size=12))
    return apply_mono_layout(fig)


def create_rolling_metrics_chart(dates, rolling_ret, rolling_vol, scheme_name):
    """Create rolling metrics chart - Light theme with vibrant colors"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("ROLLING RETURNS", "ROLLING VOLATILITY"))
    
    fig.add_trace(
        go.Scatter(x=dates, y=rolling_ret * 100, name="Return", 
                   line=dict(color="#2563eb", width=2.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=rolling_vol * 100, name="Volatility",
                   line=dict(color="#ea580c", width=2.5)),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"ROLLING METRICS (30-DAY) — {scheme_name.upper()}",
        height=500,
        showlegend=True
    )
    return apply_mono_layout(fig)


def create_period_comparison_chart(metrics_df, selected_schemes):
    """Create period return comparison chart - Light theme"""
    period_cols = ["Return_1M", "Return_3M", "Return_6M", "Return_1Y", "Return_3Y", "Return_5Y"]
    available_cols = [c for c in period_cols if c in metrics_df.columns]
    
    df_subset = metrics_df[metrics_df["Scheme"].isin(selected_schemes)][["Scheme"] + available_cols]
    df_melted = df_subset.melt(id_vars=["Scheme"], var_name="Period", value_name="Return")
    df_melted["Period"] = df_melted["Period"].str.replace("Return_", "")
    
    fig = px.bar(
        df_melted, x="Period", y="Return", color="Scheme",
        barmode="group",
        title="PERIOD-WISE RETURNS",
        color_discrete_sequence=VIBRANT_COLORS
    )
    fig.update_layout(
        yaxis_tickformat=".1%",
        height=400
    )
    return apply_mono_layout(fig)


def create_allocation_pie(weights, scheme_names):
    """Create portfolio allocation pie chart - Light theme"""
    fig = px.pie(
        values=weights * 100,
        names=scheme_names,
        title="PORTFOLIO ALLOCATION",
        hole=0.4,
        color_discrete_sequence=VIBRANT_COLORS
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", textfont=dict(color="white", size=12))
    return apply_mono_layout(fig)


def create_efficient_frontier_chart(frontier_rets, frontier_vols, current_ret, current_vol):
    """Create efficient frontier visualization - Light theme"""
    fig = go.Figure()
    
    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=np.array(frontier_vols) * 100,
        y=np.array(frontier_rets) * 100,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color="#2563eb", width=3)
    ))
    
    # Current portfolio
    fig.add_trace(go.Scatter(
        x=[current_vol * 100],
        y=[current_ret * 100],
        mode="markers",
        name="Current Portfolio",
        marker=dict(size=15, color="#dc2626", symbol="diamond")
    ))
    
    fig.update_layout(
        title="EFFICIENT FRONTIER",
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        height=500
    )
    return apply_mono_layout(fig)


# ==================== DISPLAY FUNCTIONS ====================
def display_kpi_cards(metrics_df):
    """Display KPI summary cards - Cyclops Club style"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_cagr = metrics_df["CAGR"].mean() * 100
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-indicator live"></div>
            <div class="kpi-label">METRIC: CAGR</div>
            <div class="kpi-value">{avg_cagr:.1f}%</div>
            <div class="kpi-sublabel">ANNUALIZED RETURN</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_sharpe = metrics_df["Sharpe"].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-indicator"></div>
            <div class="kpi-label">METRIC: SHARPE</div>
            <div class="kpi-value">{avg_sharpe:.2f}</div>
            <div class="kpi-sublabel">RISK-ADJUSTED</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_funds = len(metrics_df)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-indicator"></div>
            <div class="kpi-label">STATUS: ACTIVE</div>
            <div class="kpi-value">{total_funds}</div>
            <div class="kpi-sublabel">FUNDS ANALYZED</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        best_fund = metrics_df.loc[metrics_df["MFScore"].idxmax(), "Scheme"]
        best_score = metrics_df["MFScore"].max()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-indicator live"></div>
            <div class="kpi-label">TOP PERFORMER</div>
            <div class="kpi-value" style="font-size: 1.2rem;">{best_fund[:18]}</div>
            <div class="kpi-sublabel">SCORE: {best_score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)


# ==================== MAIN APP ====================
st.set_page_config(
    page_title="MF Portfolio Pro",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Cyclops-style Header
st.markdown("""
<div class="cyclops-header">
    <div class="cyclops-logo">MF PORTFOLIO PRO</div>
    <div class="cyclops-status">
        <span class="tech-label">SERVICE: <span class="tech-value">PORTFOLIO ANALYSIS</span></span>
        <span class="tech-label">MODE: <span class="tech-value">ACTIVE</span></span>
        <div class="rec-indicator">
            <div class="rec-dot"></div>
            <span>LIVE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Minimal
st.sidebar.markdown("""
<div style="padding: 1rem 0; border-bottom: 1px solid #e5e5e5; margin-bottom: 1rem;">
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; letter-spacing: -0.02em; color: #0a0a0a;">
        CONTROL PANEL
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; letter-spacing: 0.1em; color: #999; margin-top: 0.25rem;">
        SYSTEM CONFIGURATION
    </div>
</div>
""", unsafe_allow_html=True)

# Demo mode toggle
demo_mode = st.sidebar.checkbox("🎮 Demo Mode", value=True, help="Load sample data automatically")

if demo_mode:
    # Load sample data
    import os
    try:
        # Use relative path for cloud deployment
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "mutual_funds_nav_dummy.csv")
        df_raw = pd.read_csv(csv_path)
        st.sidebar.success("✅ Demo data loaded!")
    except:
        st.sidebar.error("Could not load demo data")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("📁 Upload NAV CSV", type=["csv"])
    if uploaded_file is None:
        st.info("👋 Upload a CSV with columns: **Scheme Name**, **Date**, **NAV** or enable Demo Mode!")
        st.stop()
    df_raw = pd.read_csv(uploaded_file)

# Process data
df_raw.columns = [c.strip() for c in df_raw.columns]

if not {"Scheme Name", "Date", "NAV"}.issubset(df_raw.columns):
    st.error("❌ CSV must contain: Scheme Name, Date, NAV")
    st.stop()

df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
df_raw = df_raw.dropna(subset=["Date"])
df_raw = df_raw.sort_values(["Scheme Name", "Date"])
df_raw["Return"] = df_raw.groupby("Scheme Name")["NAV"].pct_change()
df = df_raw.dropna(subset=["Return"])

if df.empty:
    st.error("❌ Not enough NAV history to compute metrics.")
    st.stop()

# Compute metrics
metrics_df = compute_metrics(df)

# Sidebar controls
st.sidebar.markdown("---")
risk_profile = st.sidebar.selectbox(
    "🎯 Risk Profile",
    ["Conservative", "Balanced", "Aggressive"],
    index=1
)
top_k = st.sidebar.slider("📊 Funds to Recommend", 1, 10, 5)

# Display KPI cards
display_kpi_cards(metrics_df)

st.markdown("---")

# Main tabs - Step based navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "01 DASHBOARD",
    "02 RECOMMENDATIONS", 
    "03 PORTFOLIO",
    "04 ANALYTICS",
    "05 GOALS",
    "06 EXPORT"
])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    st.markdown('<p class="section-header">Fund Metrics Overview</p>', unsafe_allow_html=True)
    
    # Filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Class",
            ["Conservative", "Balanced", "Aggressive", "Unknown"],
            default=["Conservative", "Balanced", "Aggressive"]
        )
    
    filtered_df = metrics_df[metrics_df["RiskClass"].isin(risk_filter)]
    
    # Display metrics table
    display_cols = ["Scheme", "CAGR", "Volatility", "Sharpe", "Sortino", "MaxDrawdown", "Calmar", "VaR_95", "RiskClass", "MFScore"]
    
    st.dataframe(
        filtered_df[display_cols].style.format({
            "CAGR": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}",
            "MaxDrawdown": "{:.2%}",
            "Calmar": "{:.2f}",
            "VaR_95": "{:.4f}",
            "MFScore": "{:.4f}",
        }),
        use_container_width=True,
        height=400
    )
    
    # Risk distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### RISK CLASS DISTRIBUTION")
        dist = metrics_df["RiskClass"].value_counts()
        fig = px.pie(values=dist.values, names=dist.index, hole=0.4, 
                     color_discrete_sequence=MONO_COLORS)
        fig = apply_mono_layout(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
    with col2:
        st.markdown("#### SHARPE VS VOLATILITY")
        fig = px.scatter(
            metrics_df, x="Volatility", y="Sharpe", color="RiskClass",
            size="CAGR", hover_name="Scheme",
            color_discrete_sequence=MONO_COLORS
        )
        fig.update_layout(
            xaxis_tickformat=".1%"
        )
        fig = apply_mono_layout(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)


# ==================== TAB 2: RECOMMENDATIONS ====================
with tab2:
    st.markdown(f'<p class="section-header">🎯 Top {top_k} Funds for {risk_profile} Investors</p>', unsafe_allow_html=True)
    
    subset = metrics_df[metrics_df["RiskClass"] == risk_profile].copy()
    subset = subset.sort_values("MFScore", ascending=False).head(top_k)
    
    if subset.empty:
        st.warning(f"No funds found matching {risk_profile} risk profile.")
    else:
        # Display recommended funds as cards
        for idx, row in subset.iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                with col1:
                    st.markdown(f"**{row['Scheme']}**")
                    st.caption(f"Score: {row['MFScore']:.4f}")
                with col2:
                    st.metric("CAGR", f"{row['CAGR']*100:.1f}%")
                with col3:
                    st.metric("Sharpe", f"{row['Sharpe']:.2f}")
                with col4:
                    st.metric("Sortino", f"{row['Sortino']:.2f}")
                with col5:
                    st.metric("Max DD", f"{row['MaxDrawdown']*100:.1f}%")
                st.markdown("---")
        
        # NAV Chart for recommended funds
        st.markdown("#### NAV History - Recommended Funds")
        rec_schemes = subset["Scheme"].tolist()
        navdf = df_raw[df_raw["Scheme Name"].isin(rec_schemes)][["Date", "Scheme Name", "NAV"]]
        fig = create_nav_chart(navdf, rec_schemes)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        
        # Period comparison
        st.markdown("#### Period-wise Returns")
        fig = create_period_comparison_chart(metrics_df, rec_schemes)
        st.plotly_chart(fig, use_container_width=True, theme=None)


# ==================== TAB 3: PORTFOLIO BUILDER ====================
with tab3:
    st.markdown('<p class="section-header">💼 Build Your Portfolio</p>', unsafe_allow_html=True)
    
    # Fund selection
    selected_funds = st.multiselect(
        "Select funds for your portfolio",
        metrics_df["Scheme"].tolist(),
        default=metrics_df.nlargest(3, "MFScore")["Scheme"].tolist()
    )
    
    if len(selected_funds) >= 2:
        # Equal weight allocation
        n_funds = len(selected_funds)
        
        # Create returns dataframe for selected funds
        returns_pivot = df_raw[df_raw["Scheme Name"].isin(selected_funds)].pivot(
            index="Date", columns="Scheme Name", values="Return"
        ).dropna()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Manual Allocation")
            weights = {}
            remaining = 100
            for i, fund in enumerate(selected_funds):
                if i == len(selected_funds) - 1:
                    weights[fund] = remaining
                    st.slider(f"{fund[:30]}...", 0, 100, remaining, disabled=True, key=f"w_{fund}")
                else:
                    w = st.slider(f"{fund[:30]}...", 0, remaining, min(remaining, 100//n_funds), key=f"w_{fund}")
                    weights[fund] = w
                    remaining -= w
            
            weight_array = np.array([weights[f]/100 for f in selected_funds])
            
            if abs(sum(weight_array) - 1.0) > 0.01:
                st.warning("Weights must sum to 100%")
            else:
                port_ret, port_vol, port_sharpe = calculate_portfolio_metrics(
                    weight_array, returns_pivot[selected_funds]
                )
                
                st.markdown("#### Portfolio Metrics")
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Expected Return", f"{port_ret*100:.1f}%")
                mcol2.metric("Volatility", f"{port_vol*100:.1f}%")
                mcol3.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
        
        with col2:
            st.markdown("#### Allocation Visualization")
            fig = create_allocation_pie(weight_array, selected_funds)
            st.plotly_chart(fig, use_container_width=True, theme=None)
        
        # Optimization
        st.markdown("---")
        st.markdown("#### 🧠 Portfolio Optimization")
        
        if st.button("🚀 Optimize for Maximum Sharpe"):
            optimal_weights = optimize_portfolio(returns_pivot[selected_funds])
            
            st.markdown("**Optimal Allocation:**")
            opt_df = pd.DataFrame({
                "Fund": selected_funds,
                "Current (%)": weight_array * 100,
                "Optimal (%)": optimal_weights * 100
            })
            st.dataframe(opt_df.style.format({"Current (%)": "{:.1f}", "Optimal (%)": "{:.1f}"}))
            
            opt_ret, opt_vol, opt_sharpe = calculate_portfolio_metrics(optimal_weights, returns_pivot[selected_funds])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Optimal Return", f"{opt_ret*100:.1f}%", f"{(opt_ret-port_ret)*100:.1f}%")
            col2.metric("Optimal Volatility", f"{opt_vol*100:.1f}%", f"{(opt_vol-port_vol)*100:.1f}%")
            col3.metric("Optimal Sharpe", f"{opt_sharpe:.2f}", f"{opt_sharpe-port_sharpe:.2f}")
        
        # Correlation analysis
        st.markdown("---")
        st.markdown("#### 🔗 Correlation Analysis")
        fig = create_correlation_heatmap(returns_pivot[selected_funds])
        st.plotly_chart(fig, use_container_width=True, theme=None)
        
        # Efficient frontier
        st.markdown("#### 📈 Efficient Frontier")
        if st.button("Generate Efficient Frontier"):
            with st.spinner("Calculating efficient frontier..."):
                frontier_rets, frontier_vols = generate_efficient_frontier(returns_pivot[selected_funds])
                if frontier_rets:
                    fig = create_efficient_frontier_chart(frontier_rets, frontier_vols, port_ret, port_vol)
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                else:
                    st.warning("Could not generate efficient frontier with current funds.")
    else:
        st.info("Select at least 2 funds to build a portfolio.")


# ==================== TAB 4: ANALYTICS ====================
with tab4:
    st.markdown('<p class="section-header">📈 Advanced Analytics</p>', unsafe_allow_html=True)
    
    # Fund selector
    selected_fund = st.selectbox("Select Fund for Analysis", metrics_df["Scheme"].tolist())
    
    fund_data = df_raw[df_raw["Scheme Name"] == selected_fund].copy()
    fund_data = fund_data.set_index("Date").sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Drawdown chart
        st.markdown("#### Drawdown Analysis")
        fig = create_drawdown_chart(fund_data["NAV"], selected_fund)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
    with col2:
        # Rolling metrics
        st.markdown("#### Rolling Metrics (30-day)")
        rolling_ret, rolling_vol = calculate_rolling_metrics(fund_data["NAV"], fund_data["Return"], 30)
        fig = create_rolling_metrics_chart(rolling_vol.index, rolling_ret, rolling_vol, selected_fund)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
    # Head-to-head comparison
    st.markdown("---")
    st.markdown("#### 🔄 Head-to-Head Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        fund_a = st.selectbox("Fund A", metrics_df["Scheme"].tolist(), key="funda")
    with col2:
        fund_b = st.selectbox("Fund B", metrics_df["Scheme"].tolist(), index=1, key="fundb")
    
    if fund_a != fund_b:
        metrics_a = metrics_df[metrics_df["Scheme"] == fund_a].iloc[0]
        metrics_b = metrics_df[metrics_df["Scheme"] == fund_b].iloc[0]
        
        comparison_metrics = ["CAGR", "Volatility", "Sharpe", "Sortino", "MaxDrawdown", "MFScore"]
        
        comp_df = pd.DataFrame({
            "Metric": comparison_metrics,
            fund_a: [metrics_a[m] for m in comparison_metrics],
            fund_b: [metrics_b[m] for m in comparison_metrics]
        })
        
        # Create comparison bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name=fund_a[:20], x=comparison_metrics, y=comp_df[fund_a], marker_color="#2563eb"))
        fig.add_trace(go.Bar(name=fund_b[:20], x=comparison_metrics, y=comp_df[fund_b], marker_color="#16a34a"))
        fig.update_layout(barmode="group", height=400)
        fig = apply_mono_layout(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        
        # NAV overlay
        st.markdown("#### NAV Comparison")
        nav_a = df_raw[df_raw["Scheme Name"] == fund_a][["Date", "NAV"]].copy()
        nav_a["Fund"] = fund_a
        nav_b = df_raw[df_raw["Scheme Name"] == fund_b][["Date", "NAV"]].copy()
        nav_b["Fund"] = fund_b
        
        # Normalize to 100
        nav_a["NAV_Normalized"] = nav_a["NAV"] / nav_a["NAV"].iloc[0] * 100
        nav_b["NAV_Normalized"] = nav_b["NAV"] / nav_b["NAV"].iloc[0] * 100
        
        combined = pd.concat([nav_a, nav_b])
        fig = px.line(combined, x="Date", y="NAV_Normalized", color="Fund", 
                      title="NORMALIZED NAV (BASE 100)", color_discrete_sequence=VIBRANT_COLORS)
        fig.update_traces(line=dict(width=2.5))
        fig = apply_mono_layout(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)


# ==================== TAB 5: GOAL PLANNER ====================
with tab5:
    st.markdown('<p class="section-header">🎯 Goal-Based Investment Planner</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 SIP Calculator")
        
        monthly_sip = st.number_input("Monthly SIP Amount (₹)", min_value=500, max_value=1000000, value=10000, step=500)
        investment_years = st.slider("Investment Period (Years)", 1, 30, 10)
        expected_return = st.slider("Expected Annual Return (%)", 1, 30, 12) / 100
        
        future_value, total_invested, wealth_gained = calculate_sip(monthly_sip, expected_return, investment_years)
        
        st.markdown("---")
        st.markdown("#### Results")
        
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Future Value", f"₹{future_value:,.0f}")
        mcol2.metric("Total Invested", f"₹{total_invested:,.0f}")
        mcol3.metric("Wealth Gained", f"₹{wealth_gained:,.0f}", f"{(wealth_gained/total_invested)*100:.0f}%")
        
        # Projection chart
        months = np.arange(1, investment_years * 12 + 1)
        monthly_rate = expected_return / 12
        projections = [monthly_sip * (((1 + monthly_rate) ** m - 1) / monthly_rate) * (1 + monthly_rate) for m in months]
        invested = [monthly_sip * m for m in months]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months/12, y=projections, name="Projected Value", 
                                 fill="tonexty", fillcolor="rgba(37, 99, 235, 0.2)",
                                 line=dict(color="#2563eb", width=2.5)))
        fig.add_trace(go.Scatter(x=months/12, y=invested, name="Amount Invested",
                                 line=dict(color="#dc2626", dash="dash", width=2.5)))
        fig.update_layout(
            title="SIP GROWTH PROJECTION",
            xaxis_title="Years",
            yaxis_title="Value (₹)",
            height=400
        )
        fig = apply_mono_layout(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
    with col2:
        st.markdown("#### 🎯 Goal Tracker")
        
        goal_name = st.text_input("Goal Name", value="Dream Home Down Payment")
        target_amount = st.number_input("Target Amount (₹)", min_value=10000, max_value=100000000, value=5000000, step=100000)
        target_years = st.slider("Target Years", 1, 30, 5, key="target_yrs")
        
        # Calculate required SIP
        monthly_rate = expected_return / 12
        months = target_years * 12
        if monthly_rate > 0:
            required_sip = target_amount / ((((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate))
        else:
            required_sip = target_amount / months
        
        st.markdown("---")
        st.markdown("#### Required Investment")
        
        st.metric("Required Monthly SIP", f"₹{required_sip:,.0f}")
        
        # Goal probability estimation
        st.markdown("---")
        st.markdown("#### 📊 Goal Achievement Analysis")
        
        # Historical returns from selected fund
        if len(metrics_df) > 0:
            hist_returns = df["Return"].dropna().values
            prob = calculate_goal_probability(target_amount, monthly_sip, target_years, hist_returns * TRADING_DAYS)
            
            st.metric("Achievement Probability", f"{prob*100:.0f}%")
            
            if prob >= 0.8:
                st.success("🎉 High probability of achieving your goal!")
            elif prob >= 0.5:
                st.warning("⚠️ Moderate probability. Consider increasing SIP or extending duration.")
            else:
                st.error("🚨 Low probability. Review your investment strategy.")
            
            # Gap analysis
            if monthly_sip < required_sip:
                gap = required_sip - monthly_sip
                st.info(f"💡 Increase SIP by ₹{gap:,.0f}/month to improve your chances.")


# ==================== TAB 6: EXPORT ====================
with tab6:
    st.markdown('<p class="section-header">📥 Export Data & Reports</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Export Metrics Data")
        
        # CSV export
        csv_buffer = io.StringIO()
        metrics_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Metrics (CSV)",
            data=csv_buffer.getvalue(),
            file_name="mf_metrics.csv",
            mime="text/csv"
        )
        
        # Full data export
        csv_full = io.StringIO()
        df_raw.to_csv(csv_full, index=False)
        st.download_button(
            label="📥 Download Full NAV Data (CSV)",
            data=csv_full.getvalue(),
            file_name="mf_full_data.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("#### 📋 Export Summary Report")
        
        # Generate summary report
        report = f"""
# Mutual Fund Portfolio Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Summary Statistics
- Total Funds Analyzed: {len(metrics_df)}
- Average CAGR: {metrics_df['CAGR'].mean()*100:.2f}%
- Average Sharpe Ratio: {metrics_df['Sharpe'].mean():.2f}
- Best Performer: {metrics_df.loc[metrics_df['MFScore'].idxmax(), 'Scheme']}

## Risk Distribution
{metrics_df['RiskClass'].value_counts().to_string()}

## Top 5 Funds by MF Score
{metrics_df.nlargest(5, 'MFScore')[['Scheme', 'CAGR', 'Sharpe', 'MFScore']].to_string()}
        """
        
        st.download_button(
            label="📥 Download Summary Report",
            data=report,
            file_name="mf_report.md",
            mime="text/markdown"
        )
    
    st.markdown("---")
    st.markdown("#### 📈 Data Preview")
    st.dataframe(metrics_df.head(10), use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p>Built with ❤️ using Streamlit | Mutual Fund Portfolio Pro v2.0</p>
    <p style="font-size: 0.8rem;">Disclaimer: This tool is for educational purposes only. Past performance is not indicative of future results.</p>
</div>
""", unsafe_allow_html=True)
