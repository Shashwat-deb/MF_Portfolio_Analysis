import numpy as np
import pandas as pd
import streamlit as st

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04


def cagr(nav):
    if len(nav) < 2:
        return np.nan
    start = nav.iloc[0]
    end = nav.iloc[-1]
    years = len(nav) / TRADING_DAYS
    return (end / start) ** (1 / years) - 1


def annual_vol(ret):
    if ret.dropna().empty:
        return np.nan
    return ret.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(c, v, rf=RISK_FREE_RATE):
    if v == 0 or np.isnan(v):
        return np.nan
    return (c - rf) / v


def max_drawdown(nav):
    if nav.empty:
        return np.nan
    rm = nav.cummax()
    dd = nav / rm - 1
    return dd.min()


def classify(vol):
    if np.isnan(vol):
        return "Unknown"
    if vol < 0.05:
        return "Conservative"
    elif vol < 0.12:
        return "Balanced"
    else:
        return "Aggressive"


def compute_metrics(df):
    out = []
    for scheme, grp in df.groupby("Scheme Name"):
        nav = grp["NAV"]
        ret = grp["Return"]
        c = cagr(nav)
        v = annual_vol(ret)
        s = sharpe_ratio(c, v)
        m = max_drawdown(nav)
        r = classify(v)
        out.append([scheme, c, v, s, m, r])
    mdf = pd.DataFrame(
        out,
        columns=["Scheme", "CAGR", "Volatility", "Sharpe", "MaxDrawdown", "RiskClass"],
    )
    mdf["MFScore"] = (
        0.5 * mdf["Sharpe"].fillna(0)
        + 0.3 * mdf["CAGR"].fillna(0)
        - 0.2 * mdf["MaxDrawdown"].abs().fillna(0)
    )
    return mdf


st.set_page_config(page_title="Mutual Fund Recommendation Engine", layout="wide")

st.title("📈 Mutual Fund Risk-Matched Recommendation Engine")

uploaded_file = st.sidebar.file_uploader("Upload Mutual Fund NAV CSV", type=["csv"])

risk_profile = st.sidebar.selectbox(
    "Select Risk Profile", ["Conservative", "Balanced", "Aggressive"]
)

top_k = st.sidebar.slider("Funds to Recommend", 1, 10, 3)

if uploaded_file is None:
    st.info("Upload a CSV with columns: Scheme Name, Date, NAV.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
df_raw.columns = [c.strip() for c in df_raw.columns]

if not {"Scheme Name", "Date", "NAV"}.issubset(df_raw.columns):
    st.error("CSV must contain: Scheme Name, Date, NAV")
    st.stop()

df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
df_raw = df_raw.dropna(subset=["Date"])
df_raw = df_raw.sort_values(["Scheme Name", "Date"])

df_raw["Return"] = df_raw.groupby("Scheme Name")["NAV"].pct_change()
df = df_raw.dropna(subset=["Return"])

if df.empty:
    st.error("Not enough NAV history to compute metrics.")
    st.stop()

metrics_df = compute_metrics(df)

st.subheader("📊 Mutual Fund Metrics")
st.dataframe(
    metrics_df.style.format(
        {
            "CAGR": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe": "{:.2f}",
            "MaxDrawdown": "{:.2%}",
            "MFScore": "{:.4f}",
        }
    ),
    use_container_width=True,
)

st.subheader(f"🎯 Recommended Funds for {risk_profile}")
subset = metrics_df[metrics_df["RiskClass"] == risk_profile].copy()
subset = subset.sort_values("MFScore", ascending=False).head(top_k)

st.table(
    subset[
        [
            "Scheme",
            "RiskClass",
            "MFScore",
            "CAGR",
            "Volatility",
            "Sharpe",
            "MaxDrawdown",
        ]
    ].style.format(
        {
            "CAGR": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe": "{:.2f}",
            "MaxDrawdown": "{:.2%}",
            "MFScore": "{:.4f}",
        }
    )
)

st.subheader("📉 NAV History")
schemes = metrics_df["Scheme"].tolist()
selected = st.multiselect("Select Schemes", schemes, default=subset["Scheme"].tolist())

if selected:
    navdf = df_raw[df_raw["Scheme Name"].isin(selected)][["Date", "Scheme Name", "NAV"]]
    navpivot = navdf.pivot(index="Date", columns="Scheme Name", values="NAV")
    st.line_chart(navpivot)

st.subheader("📌 Risk Class Distribution")
dist = metrics_df["RiskClass"].value_counts()
st.bar_chart(dist)
