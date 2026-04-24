# -*- coding: utf-8 -*-
# app.py

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import matplotlib.colors as mcolors
import traceback
import plotly.graph_objects as go


TRADING_DAYS = 252
st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("Portfolio Risk Dashboard")

from feature_engineering import (
    build_features_from_returns,
    build_stock_features_for_kmeans
)

@st.cache_resource
def load_models():
    feature_cols = joblib.load("feature_cols.joblib")
    ridge = joblib.load("ridge_model.joblib")
    rf = joblib.load("rf_model.joblib")
    return feature_cols, ridge, rf

feature_cols, ridge, rf = load_models()
WINDOW = 20  # must match training

def build_realtime_feature_vector(rets_assets: pd.DataFrame, rets_spy: pd.Series, weights: np.ndarray):
    X_rt = build_features_from_returns(rets_assets, rets_spy, weights, window=WINDOW)
    X_rt = X_rt.reindex(columns=feature_cols) 
    return X_rt

# UI Helpers & Fetching
def parse_tickers(tickers_text: str):
    raw = tickers_text.replace(",", " ").split()
    tickers = [t.strip().upper() for t in raw if t.strip()]
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            out.append(t); seen.add(t)
    return out
# Parses and normalizes user-inputted portfolio weights, defaulting to equal weights if input is empty.
def parse_weights(weights_text: str, n: int):
    if weights_text is None or weights_text.strip() == "":
        return np.ones(n) / n
    raw = weights_text.replace(",", " ").split()
    w = np.array([float(x) for x in raw], dtype=float)
    if len(w) != n:
        raise ValueError("Number of weights must match number of tickers.")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    if w.sum() == 0:
        raise ValueError("Sum of weights cannot be zero.")
    return w / w.sum()
# Downloads and caches historical adjusted closing prices for given tickers, dropping columns with too many missing values.
@st.cache_data(show_spinner=False)
def fetch_prices(tickers, period="3y"):
    df = yf.download(tickers=tickers, period=period, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Adj Close"].copy()
    else:
        prices = df[["Adj Close"]].copy()
        prices.columns = tickers
    prices = prices.dropna(how="all")
    prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices)))
    return prices

def compute_current_metrics(prices_assets: pd.DataFrame, weights: np.ndarray):
    rets = prices_assets.pct_change().dropna()
    tickers = list(rets.columns)

    if len(weights) != len(tickers):
        weights = np.ones(len(tickers)) / len(tickers)

    port_ret = rets @ weights
    ann_vol = port_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)

    corr = rets.corr()
    cov_ann = rets.cov() * TRADING_DAYS
    # --- force shapes to be correct ---
    weights = np.asarray(weights, dtype=float).reshape(-1)   
    sigma = np.asarray(cov_ann.values, dtype=float)          # cov matrix

    w_vec = weights.reshape(-1, 1)                           # column vector

    # sanity check
    if sigma.shape[0] != w_vec.shape[0]:
        raise ValueError(f"Dimension mismatch: sigma {sigma.shape}, weights {w_vec.shape}")

    port_var = (w_vec.T @ sigma @ w_vec).item()              
    port_vol = float(np.sqrt(port_var))

    mrc = sigma @ w_vec
    rc = (w_vec * mrc).flatten() / port_vol
    rc_pct = rc / port_vol

    rc_df = pd.DataFrame({
        "ticker": tickers,
        "risk_contribution_pct_of_vol": (rc_pct * 100).round(2)
    }).sort_values("risk_contribution_pct_of_vol", ascending=False)

    cum = (1 + port_ret).cumprod()

    return tickers, weights, ann_vol, corr, rc_df, cum, rets


# Plot Functions
def plot_cumulative(cum):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(cum.index, cum.values)
    plt.title("Portfolio Cumulative Return (Index)")
    plt.tight_layout()
    return fig
def compute_history_summary(cum: pd.Series, portfolio_return: pd.Series):
    #Returns summary stats for Page 2 header cards.
    # Cumulative return: final value of index (starts at 1) - 1
    cum_return = cum.iloc[-1] - 1

    # Annualized return
    n_days = len(portfolio_return)
    ann_return = (1 + cum_return) ** (TRADING_DAYS / n_days) - 1

    # Average 20D rolling vol
    rolling_vol = portfolio_return.rolling(20).std() * np.sqrt(TRADING_DAYS)
    avg_rolling_vol = rolling_vol.dropna().mean()

    return cum_return, ann_return, avg_rolling_vol, rolling_vol

def plot_corr_heatmap(corr):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bg_color = "#FFFFFF"  
    navy_color = "#08277B" 
    gold_accent = '#F3CA43' 
    primary_blue = '#08277B' 
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    colors = [gold_accent, bg_color, '#08277B']
    n_bins = 100
    cmap_name = 'auth_gold_navy'
    cm_auth = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # 5. Draw Heatmap
    cax = ax.imshow(corr.values, aspect="auto", cmap=cm_auth, vmin=-1.0, vmax=1.0)
    
    # Colorbar
    cbar = fig.colorbar(cax, orientation='vertical')
    cbar.ax.tick_params(colors='#08277B') 
    cbar.outline.set_edgecolor('#E0E0E0')   

    # Configure the coordinate axis labels and fonts
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="#0B1F3B", fontsize=10)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, color="#0B1F3B", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Correlation Heatmap", color='#0B1F3B', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def generate_correlation_explanation(corr_matrix):
    import numpy as np
    
    tickers = corr_matrix.columns
    explanations = []
    
    max_corr = 0
    max_pair = None
    
    min_corr = 1
    min_pair = None
    
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            val = corr_matrix.iloc[i, j]
            
            if val > max_corr:
                max_corr = val
                max_pair = (tickers[i], tickers[j])
            
            if val < min_corr:
                min_corr = val
                min_pair = (tickers[i], tickers[j])
    
    # explanation generator
    text = ""
    
    if max_pair:
        text += f"{max_pair[0]} and {max_pair[1]} tend to move very closely together, meaning they carry similar risk. "
    
    if min_pair:
        text += f"In contrast, {min_pair[0]} and {min_pair[1]} behave more independently, which helps improve diversification. "
    
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    
    if avg_corr > 0.6:
        text += "Overall, your portfolio contains many highly correlated assets, which may reduce diversification."
    elif avg_corr > 0.3:
        text += "Overall, your portfolio shows a moderate level of diversification."
    else:
        text += "Overall, your portfolio is well diversified, with assets that do not move strongly together."
    
    return text
# plot dsitribution of risk contribution percentages
def plot_risk_contrib(rc_df):
     fig, ax = plt.subplots(figsize=(7, 3.8))
     bg_color = '#F7F7FB'
     fig.patch.set_facecolor(bg_color)
     ax.set_facecolor(bg_color)
     colors = ["#F3CA43", '#0B1F3B', '#1D4ED8', '#60A5FA', '#93C5FD', '#DBEAFE']
     wedges, texts, autotexts = ax.pie(
        rc_df["risk_contribution_pct_of_vol"],
        labels=rc_df["ticker"],
        colors=colors[:len(rc_df)],
        autopct='%1.1f%%',
        startangle=90,          
        counterclock=False,    
        pctdistance=0.75,      
        textprops={'color': '#0B1F3B', 'fontweight': 'medium', 'fontsize': 11}, 
        wedgeprops={'edgecolor': bg_color, 'linewidth': 2.5, 'width': 0.45}   
        )
     for i, autotext in enumerate(autotexts):
        autotext.set_weight('bold')
        autotext.set_fontsize(10)
        if i == 0 or i >= 3:
            autotext.set_color('#0B1F3B') 
        else:
            autotext.set_color(bg_color)
            
     plt.tight_layout()
     return fig
# Calculate the 5 indicators required for the radar chart, calculate them separately by ticker
def compute_radar_metrics(prices_assets: pd.DataFrame, weights: np.ndarray, rf_annual: float = 0.04):
    
    rets = prices_assets.pct_change().dropna()
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    metrics = {}

    for ticker in rets.columns:
        r = rets[ticker].dropna()

        # Annualized Return
        ann_ret = r.mean() * TRADING_DAYS

        # Max Drawdown
        cum = (1 + r).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = drawdown.min()  

        # 3. Calmar Ratio = Annualized return/abs
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

        # 4. Sharpe-like = (Annualized return - rf)/Annualized volatility
        ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan

        # 5. CVaR (Expected Shortfall) at 95% — the average of the worst 5% of daily returns (negative)
        var_95 = np.percentile(r, 5)
        cvar = r[r <= var_95].mean()  

        metrics[ticker] = {
            "Ann. Return": ann_ret,
            "Max Drawdown": max_dd,       
            "Calmar Ratio": calmar,
            "Sharpe Ratio": sharpe,
            "CVaR (95%)": cvar,           
        }

    raw_df = pd.DataFrame(metrics).T  # shape: (n_tickers, 5)
    # normalization
    norm_df = raw_df.copy()
    def minmax(series, padding=0.1):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        return padding + (1 - padding) * (series - mn) / (mx - mn)
    norm_df["Ann. Return"]   = minmax(raw_df["Ann. Return"])
    norm_df["Calmar Ratio"]  = minmax(raw_df["Calmar Ratio"])
    norm_df["Sharpe Ratio"]  = minmax(raw_df["Sharpe Ratio"])
    norm_df["Max Drawdown"]  = minmax(-raw_df["Max Drawdown"])   
    norm_df["CVaR (95%)"]    = minmax(-raw_df["CVaR (95%)"])    

    return raw_df, norm_df

def plot_rolling_vol(rolling_vol):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(rolling_vol.index, rolling_vol.values)
    plt.title("20-Day Rolling Volatility")
    plt.tight_layout()
    return fig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def rolling_forecast(prices_assets: pd.DataFrame, horizon: int = 20, lookback: int = 60):
    rets = prices_assets.pct_change().dropna()
    avg_daily_ret = rets.tail(lookback).mean()
    daily_vol = rets.tail(lookback).std()
    latest_prices = prices_assets.iloc[-1]
    
    pred_ret = (1 + avg_daily_ret) ** horizon - 1
    pred_base = latest_prices * (1 + pred_ret)
    
    band = daily_vol * np.sqrt(horizon)
    pred_low = latest_prices * (1 + pred_ret - band)
    pred_high = latest_prices * (1 + pred_ret + band)
    
    # calculate logic of outlook
    outlooks = []
    for pr in pred_ret:
        if pr > 0.02:
            outlooks.append("Bullish")
        elif pr < -0.02:
            outlooks.append("Bearish")
        else:
            outlooks.append("Neutral")

    summary_df = pd.DataFrame({
        "Stock": prices_assets.columns,
        "Current Price": latest_prices.values.round(2),
        "Past 60D Avg Daily Return": avg_daily_ret.values,
        "Predicted 20D Return": pred_ret.values,
        "Predicted Low": pred_low.values.round(2),
        "Predicted Base": pred_base.values.round(2),
        "Predicted High": pred_high.values.round(2),
        "Outlook": outlooks
    })
    return latest_prices, avg_daily_ret, pred_ret, pred_low, pred_base, pred_high, summary_df
# Draw a chart of future forecast ranges
def plot_rolling_forecast(prices_assets: pd.DataFrame, pred_base: pd.Series, pred_low: pd.Series, pred_high: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ticker = prices_assets.columns[0]
    hist = prices_assets[ticker].tail(120)
    
    line_color = "#7c3aed" 
    fill_color = "#ddd6fe" 

    ax.plot(hist.index, hist.values, color=line_color, linewidth=2, label=f"Historical Price")
    
    future_idx = pd.date_range(start=hist.index[-1], periods=21, freq="B")[1:]
    
    ax.plot(
        [hist.index[-1], future_idx[-1]],
        [hist.iloc[-1], pred_base[ticker]],
        linestyle="--",
        color=line_color,
        linewidth=2,
        label="Forecast (Base)"
    )
    
    ax.fill_between(
        [hist.index[-1], future_idx[-1]],
        [hist.iloc[-1], pred_low[ticker]],
        [hist.iloc[-1], pred_high[ticker]],
        color=fill_color,
        alpha=0.4,
        label="Forecast Range"
    )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#64748b', which='both')
    ax.set_ylabel("Price (USD)", color='#64748b', fontsize=10, loc='top')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', ncol=3, frameon=False, fontsize=10)
    plt.tight_layout()
    return fig

# Computes CAPM metrics for each ticker relative to the market benchmark
def compute_capm_table(rets_assets: pd.DataFrame, rets_mkt: pd.Series, rf_annual: float):
    """
    rets_assets: DataFrame, columns=tickers, daily returns
    rets_mkt: Series, SPY daily returns
    rf_annual: annual risk-free rate (e.g., 0.04 for 4%)
    """
    # daily rf from annual rf
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    idx = rets_assets.index.intersection(rets_mkt.index)
    X = (rets_mkt.loc[idx] - rf_daily).astype(float)  
    mkt_prem_ann = X.mean() * TRADING_DAYS            

    out = []
    for t in rets_assets.columns:
        y = (rets_assets.loc[idx, t] - rf_daily).astype(float)  

        df_tmp = pd.concat([X.rename("mkt_excess"), y.rename("stk_excess")], axis=1).dropna()
        if len(df_tmp) < 30:
            continue

        x = df_tmp["mkt_excess"].values
        yy = df_tmp["stk_excess"].values

        # beta = cov/var (simple + stable)
        beta = np.cov(yy, x, ddof=1)[0, 1] / np.var(x, ddof=1)

        # alpha = mean(y) - beta*mean(x)
        alpha_daily = yy.mean() - beta * x.mean()
        alpha_ann = alpha_daily * TRADING_DAYS

        # R^2
        y_hat = alpha_daily + beta * x
        ss_res = ((yy - y_hat) ** 2).sum()
        ss_tot = ((yy - yy.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # CAPM expected return (annual)
        exp_ret_ann = rf_annual + beta * mkt_prem_ann

        out.append({
            "Ticker": t,
            "Beta": beta,
            "Alpha (ann.)": alpha_ann,
            "R^2": r2,
            "Market Risk Premium (ann.)": mkt_prem_ann,
            "CAPM Expected Return (ann.)": exp_ret_ann
        })

    capm_df = pd.DataFrame(out)
    if not capm_df.empty:
        capm_df = capm_df.sort_values("Beta", ascending=False).reset_index(drop=True)
    return capm_df

st.sidebar.header("Inputs")

tickers_text = st.sidebar.text_input("Tickers", value="AAPL MSFT NVDA")

period_options = {
    "1 Year — Focus on recent trends": "1y",
    "3 Years — Balanced view": "3y",
    "5 Years — Long-term stability": "5y",
}

selected_label = st.sidebar.selectbox(
    "Time Range for Analysis",
    list(period_options.keys()),
    index=1
)
period = period_options[selected_label]

run = st.sidebar.button("Run")

page = st.sidebar.radio(

    "Go to page",
    ["Page 1: Overview", "Page 2: Historical Performance", "Page 3: Rolling Forecast", "Page 4: CAPM Analysis", "Page 5: Diversification Risk", "Page 6: Individual Stock Performance"],
    index=0
)

st.sidebar.markdown("<p style='font-size:20px; font-weight:700; color:#475569;'>Enter tickers and click Run</p>", unsafe_allow_html=True)

if 'has_run' not in st.session_state:
    st.session_state['has_run'] = False

if run:
    st.session_state['has_run'] = True
    

if not st.session_state['has_run']:
    st.info("Enter tickers and click Run in the sidebar.")
else:
    try:
        # data preparation
        tickers = parse_tickers(tickers_text)
        if len(tickers) < 2:
            st.error("Please enter at least 2 tickers.")
            st.stop()
            
        n = len(tickers)

        # download user tickers ONLY
        prices_assets_all = fetch_prices(tickers, period=period)

        # drop tickers that failed (only for user assets)
        available = [t for t in tickers if t in prices_assets_all.columns]
        missing = [t for t in tickers if t not in prices_assets_all.columns]
        if missing:
            st.warning(f"Missing tickers dropped: {missing}")

        tickers_used_input = available
        if len(tickers_used_input) < 2:
            st.error("Not enough valid tickers after download. Please try different tickers.")
            st.stop()

        prices_assets = prices_assets_all[tickers_used_input].copy()
        tickers = tickers_used_input
        n = len(tickers)

        # download SPY separately
        prices_spy_df = fetch_prices(["SPY"], period=period)
        if "SPY" not in prices_spy_df.columns:
            st.error("SPY data missing. Needed for market_return feature.")
            st.stop()

        prices_spy = prices_spy_df["SPY"].copy()

        # Weight analysis
        if 'weight_input_str' not in st.session_state:
            st.session_state['weight_input_str'] = ""
            
        try:
            weights_used = parse_weights(st.session_state['weight_input_str'], n)
        except ValueError as e:
            st.warning(f"Weight input error: {e}. Defaulting to equal weights.")
            weights_used = np.ones(n) / n

        # calculation
        tickers_used, final_weights, ann_vol, corr, rc_df, cum, rets_assets = compute_current_metrics(
            prices_assets, weights_used
        )

        rets_spy = prices_spy.pct_change().dropna()
        common_idx = rets_assets.index.intersection(rets_spy.index)
        rets_assets = rets_assets.loc[common_idx]
        rets_spy = rets_spy.loc[common_idx]

        # ML Features & Prediction
        w_vec = np.array(final_weights[:len(tickers_used)], dtype=float)

        X_rt = build_realtime_feature_vector(
        rets_assets[tickers_used],
        rets_spy,
        w_vec
        )
        X_rt2 = X_rt.tail(1)

        pred_vol = float(np.asarray(ridge.predict(X_rt2)).ravel()[0])
        pred_prob = float(np.asarray(rf.predict_proba(X_rt2))[0, 1])

        portfolio_return = rets_assets[tickers_used].dot(final_weights[:len(tickers_used)])
        exp_return = portfolio_return.mean() * TRADING_DAYS
        avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
        div_score = (1 - avg_corr) * 100

        if pred_prob >= 0.67:
            risk_label = "High"
        elif pred_prob >= 0.33:
            risk_label = "Medium"
        else:
            risk_label = "Low"

        # UI design
        # content of page 1: overview
        if page == "Page 1: Overview":
            st.markdown("<div style='font-size:30px; font-weight:900; color:#1e293b; margin-bottom:6px;'>⚖️ Set Portfolio Weights</div>", unsafe_allow_html=True)
            st.info(
                "💡 **Instruction:** Enter relative weights separated by spaces (e.g., `2 3 5`). "
                "The app will automatically normalize them to 100%. Leave blank for equal weights."
            )
            new_weight_str = st.text_input(
                f"Enter {n} weights for [ {' | '.join(tickers)} ] :",
                value=st.session_state.get("weight_input_str", ""),
                key="weight_input_str_box"
            )
            if new_weight_str != st.session_state.get('weight_input_str', ''):
                st.session_state['weight_input_str'] = new_weight_str
                st.rerun()

            st.write("---")

            v_val = ann_vol * 100
            v_pct = min(v_val / 40 * 100, 100) 
            v_badge = "High" if v_val > 20 else "Low" if v_val < 10 else "Moderate"
            
            r_val = exp_return * 100
            r_pct = min(max(r_val, 0) / 35 * 100, 100)
            r_badge = "High" if r_val > 15 else "Moderate" if r_val > 5 else "Low"
            
            d_val = div_score
            d_pct = min(d_val, 100)
            d_badge = "Excellent" if d_val > 80 else "Good" if d_val > 60 else "Moderate" if d_val > 40 else "Low"
            
            m_val = pred_prob * 100
            m_pct = min(m_val, 100)
            m_badge = "High Risk" if risk_label == "High" else "Medium Risk" if risk_label == "Medium" else "Low Risk"

            # CSS for styling pages
            custom_css = """
            <style>
            .overview-title { font-size: 32px; font-weight: 800; color: #1e293b; margin-bottom: 6px; }
            .overview-sub { font-size: 15px; color: #64748b; margin-bottom: 28px; }
            .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
            .metric-card { background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02); display: flex; flex-direction: column;}
            .card-header { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
            .icon-box { width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 22px; }
            .card-title-en { font-size: 15px; font-weight: 700; color: #334155; }
            .card-title-cn { font-size: 13px; color: #64748b; margin-top: 2px;}
            .value-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
            .value-text { font-size: 34px; font-weight: 800; color: #0f172a; line-height: 1; }
            .badge { padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; }
            .desc-text { font-size: 13.5px; color: #475569; line-height: 1.6; margin-bottom: 24px; flex-grow: 1;}
            .scale-container { margin-top: auto; }
            .scale-track { height: 6px; background: #f1f5f9; border-radius: 3px; position: relative; margin-bottom: 10px; }

            .scale-fill {
                height: 100%;
                border-radius: 3px;
                position: absolute;
                left: 0; top: 0;
                width: 0%;
                animation: fillBar 1.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
            }
            .scale-indicator {
                width: 14px; height: 14px; border-radius: 50%;
                position: absolute; top: -4px;
                transform: translateX(-50%);
                border: 2.5px solid white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                left: 0%;
                animation: moveIndicator 1.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
            }
            @keyframes fillBar {
                from { width: 0%; }
                to   { width: var(--target-width); }
            }
            @keyframes moveIndicator {
                from { left: 0%; }
                to   { left: var(--target-width); }
            }

            .scale-labels { display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; text-align: center; }
            .scale-labels span { flex: 1; }
            .footer-note { font-size: 13px; color: #94a3b8; text-align: left; margin-top: 10px; }
            </style>
            """
            st.markdown(custom_css, unsafe_allow_html=True)

            html_content = f"""
<div class="overview-title">Portfolio Overview</div>
<div class="overview-sub">Key metrics at a glance, helping you quickly understand the risk and return profile of your portfolio.</div>

<div class="metric-grid">
<div class="metric-card">
<div class="card-header">
<div class="icon-box" style="background: #eff6ff; color: #3b82f6;">📉</div>
<div>
<div class="card-title-en">Current Annualized Volatility</div>
<div class="card-title-cn">Volatility Risk</div>
</div>
</div>
<div class="value-row">
<div class="value-text">{v_val:.2f}%</div>
<div class="badge" style="background: #eff6ff; color: #2563eb;">{v_badge}</div>
</div>
<div class="desc-text">Measures the price fluctuation of your portfolio. {v_val:.2f}% indicates your portfolio's volatility is {("relatively high, implying larger potential swings in returns" if v_val > 20 else "within a reasonable range")}, suitable for investors with a corresponding risk tolerance.</div>
<div class="scale-container">
<div class="scale-track">

<div class="scale-fill" style="--target-width: {v_pct}%; background: #3b82f6;"></div>
<div class="scale-indicator" style="--target-width: {v_pct}%; background: #3b82f6;"></div>

</div>
<div class="scale-labels"><span>Low<br>&lt;10%</span><span>Med<br>10-20%</span><span>High<br>20-30%</span><span>V.High<br>&gt;30%</span></div>
</div>
</div>

<div class="metric-card">
<div class="card-header">
<div class="icon-box" style="background: #ecfdf5; color: #10b981;">📈</div>
<div>
<div class="card-title-en">Expected Annual Return</div>
<div class="card-title-cn">Projected Return</div>
</div>
</div>
<div class="value-row">
<div class="value-text">{r_val:.2f}%</div>
<div class="badge" style="background: #ecfdf5; color: #059669;">{r_badge}</div>
</div>
<div class="desc-text">Projected annual return based on historical data and market models. {r_val:.2f}% indicates a {r_badge.lower()} expected return, implying {("higher potential upside, accompanied by greater risk" if r_val > 15 else "a relatively steady return expectation")}.</div>
<div class="scale-container">
<div class="scale-track">
<div class="scale-fill" style="--target-width: {r_pct}%; background: #10b981;"></div>
<div class="scale-indicator" style="--target-width: {r_pct}%; background: #10b981;"></div>

</div>
<div class="scale-labels"><span>Low<br>&lt;5%</span><span>Med<br>5-15%</span><span>High<br>15-25%</span><span>V.High<br>&gt;25%</span></div>
</div>
</div>

<div class="metric-card">
<div class="card-header">
<div class="icon-box" style="background: #f5f3ff; color: #8b5cf6;">🎯</div>
<div>
<div class="card-title-en">Diversification Score</div>
<div class="card-title-cn">Asset Allocation Spread</div>
</div>
</div>
<div class="value-row">
<div class="value-text">{d_val:.1f}</div>
<div class="badge" style="background: #f5f3ff; color: #7c3aed;">{d_badge}</div>
</div>
<div class="desc-text">Measures how well the assets are distributed. A score of {d_val:.1f} indicates {d_badge.lower()} diversification, meaning {("there is room for optimization to further reduce unsystematic risk" if d_val < 60 else "the risk is well dispersed across the portfolio")}.</div>
<div class="scale-container">
<div class="scale-track">
<div class="scale-fill" style="--target-width: {d_pct}%; background: #8b5cf6;"></div>
<div class="scale-indicator" style="--target-width: {d_pct}%; background: #8b5cf6;"></div>

</div>
<div class="scale-labels"><span>Low<br>&lt;40</span><span>Fair<br>40-60</span><span>Good<br>60-80</span><span>Great<br>&gt;80</span></div>
</div>
</div>

<div class="metric-card">
<div class="card-header">
<div class="icon-box" style="background: #fffbeb; color: #f59e0b;">🛡️</div>
<div>
<div class="card-title-en">ML Risk Prediction</div>
<div class="card-title-cn">Machine Learning Model</div>
</div>
</div>
<div class="value-row">
<div class="value-text">{risk_label} ({m_val:.0f}%)</div>
<div class="badge" style="background: #fffbeb; color: #d97706;">{m_badge}</div>
</div>
<div class="desc-text">Machine learning prediction of future risk. Evaluated as '{m_badge}', the probability of a major drawdown is {m_val:.0f}%. We recommend to {("stay highly vigilant and consider reducing exposure" if risk_label == "High" else "monitor closely and manage your risk exposure appropriately")}.</div>
<div class="scale-container">
<div class="scale-track">
<div class="scale-fill" style="--target-width: {m_pct}%; background: #f59e0b;"></div>
<div class="scale-indicator" style="--target-width: {m_pct}%; background: #f59e0b;"></div>
</div>
<div class="scale-labels"><span>Low<br>&lt;20%</span><span>Med<br>20-50%</span><span>High<br>50-75%</span><span>V.High<br>&gt;75%</span></div>
</div>
</div>
</div>
<div class="footer-note">Disclaimer: The above metrics are for reference only and do not constitute investment advice. Investing involves risk.</div>
"""
            st.markdown(html_content, unsafe_allow_html=True)

            # add animation effect
            st.markdown(f"""
            <script>
            setTimeout(function() {{
                var fills = document.querySelectorAll('.scale-fill');
                var indicators = document.querySelectorAll('.scale-indicator');
                var targets = [{v_pct}, {r_pct}, {d_pct}, {m_pct}];
                for (var i = 0; i < fills.length; i++) {{
                    fills[i].style.width = targets[i] + '%';
                    indicators[i].style.left = targets[i] + '%';
                }}
            }}, 100);
            </script>
            """, unsafe_allow_html=True)

        # content for page 2: historical performance
        elif page == "Page 2: Historical Performance":
            from datetime import date

            # Calculate summary indicators
            portfolio_return = rets_assets[tickers_used].dot(final_weights[:len(tickers_used)])
            cum_indexed = (1 + portfolio_return).cumprod() * 100 
            cum_return, ann_return, avg_rolling_vol, rolling_vol = compute_history_summary(
                (1 + portfolio_return).cumprod(), portfolio_return
            )

            start_date = portfolio_return.index[0].strftime("%b %d, %Y")
            end_date   = portfolio_return.index[-1].strftime("%b %d, %Y")
            final_idx  = cum_indexed.iloc[-1]
            cum_sign   = "+" if cum_return >= 0 else ""
            ann_sign   = "+" if ann_return >= 0 else ""
            cum_color  = "#16a34a" if cum_return >= 0 else "#dc2626"
            ann_color  = "#16a34a" if ann_return >= 0 else "#dc2626"

            st.markdown(f"""
            <style>
            .h2-header {{
                display: flex; justify-content: space-between; align-items: flex-start;
                margin-bottom: 24px;
            }}
            .h2-title {{ font-size: 30px; font-weight: 900; color: #1e293b; display:flex; align-items:center; gap:12px; }}
            .h2-sub {{ font-size: 14px; color: #64748b; margin-top: 6px; }}
            .h2-daterange {{
                background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
                padding: 10px 16px; font-size: 13px; color: #475569;
                display: flex; align-items: center; gap: 8px;
            }}
            /* Summary Cards */
            .summary-grid {{
                display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 28px;
            }}
            .summary-card {{
                background: white; border: 1px solid #e2e8f0; border-radius: 14px;
                padding: 22px 24px; display: flex; justify-content: space-between; align-items: flex-start;
                box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            }}
            .summary-card-label {{ font-size: 13px; font-weight: 600; color: #64748b; margin-bottom: 8px; }}
            .summary-card-value {{ font-size: 30px; font-weight: 800; line-height: 1; }}
            .summary-card-desc {{ font-size: 12px; color: #94a3b8; margin-top: 8px; line-height: 1.4; }}
            .summary-card-icon {{
                width: 44px; height: 44px; border-radius: 50%;
                display: flex; align-items: center; justify-content: center; font-size: 20px;
            }}
            /* Section Cards */
            .section-card {{
                background: white; border: 1px solid #e2e8f0; border-radius: 16px;
                padding: 28px; margin-bottom: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            }}
            .section-num {{
                width: 30px; height: 30px; border-radius: 50%; background: #6366f1;
                color: white; font-weight: 800; font-size: 15px;
                display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;
            }}
            .section-title {{ font-size: 18px; font-weight: 700; color: #1e293b; display: flex; align-items: center; }}
            .section-sub {{ font-size: 13px; color: #64748b; margin-top: 6px; margin-bottom: 20px; }}
            /* Insight Box */
            .insight-box {{
                display: flex; align-items: flex-start; gap: 14px;
                background: #f0fdf4; border-radius: 12px; padding: 16px 20px; margin-top: 20px;
            }}
            .insight-box-purple {{
                background: #f5f3ff;
            }}
            .insight-icon {{
                width: 38px; height: 38px; border-radius: 50%; background: #dcfce7;
                display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0;
            }}
            .insight-icon-purple {{
                background: #ede9fe;
            }}
            .insight-title {{ font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 4px; }}
            .insight-desc {{ font-size: 13px; color: #475569; line-height: 1.5; }}
            /* How to Interpret */
            .interpret-section {{
                background: #fafafa; border: 1px solid #f1f5f9; border-radius: 16px;
                padding: 24px 28px;
            }}
            .interpret-title {{ font-size: 17px; font-weight: 700; color: #1e293b; margin-bottom: 18px;
                display: flex; align-items: center; gap: 8px; }}
            .interpret-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
            .interpret-card {{
                display: flex; align-items: flex-start; gap: 12px;
            }}
            .interpret-icon {{
                width: 40px; height: 40px; border-radius: 10px;
                display: flex; align-items: center; justify-content: center;
                font-size: 18px; flex-shrink: 0;
            }}
            .interpret-card-title {{ font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 4px; }}
            .interpret-card-desc {{ font-size: 12px; color: #64748b; line-height: 1.5; }}
            </style>

            <div class="h2-header">
                <div>
                    <div class="h2-title">📈 Historical Portfolio Performance</div>
                    <div class="h2-sub">Explore how your portfolio has performed over time and how its risk has changed.</div>
                </div>
                <div class="h2-daterange">
                    📅 {start_date} &nbsp;→&nbsp; {end_date}
                </div>
            </div>

            <!-- Summary Cards -->
            <div class="summary-grid">
                <div class="summary-card">
                    <div>
                        <div class="summary-card-label">Cumulative Return (Index)</div>
                        <div class="summary-card-value" style="color:{cum_color};">{cum_sign}{cum_return:.2%}</div>
                        <div class="summary-card-desc">Total growth of your portfolio<br>during this period.</div>
                    </div>
                    <div class="summary-card-icon" style="background:#ecfdf5;">📈</div>
                </div>
                <div class="summary-card">
                    <div>
                        <div class="summary-card-label">Annualized Return</div>
                        <div class="summary-card-value" style="color:{ann_color};">{ann_sign}{ann_return:.2%}</div>
                        <div class="summary-card-desc">Average yearly return<br>(annualized).</div>
                    </div>
                    <div class="summary-card-icon" style="background:#ecfdf5;">📊</div>
                </div>
                <div class="summary-card">
                    <div>
                        <div class="summary-card-label">Volatility (20D Rolling Avg.)</div>
                        <div class="summary-card-value" style="color:#7c3aed;">{avg_rolling_vol:.2%}</div>
                        <div class="summary-card-desc">Average short-term volatility<br>of your portfolio.</div>
                    </div>
                    <div class="summary-card-icon" style="background:#f5f3ff;">〰️</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Cumulative Return
            st.markdown("""
            <div class="section-card">
                <div class="section-title">
                    <span class="section-num">1</span> Portfolio Cumulative Return (Index)
                </div>
                <div class="section-sub">Shows how your portfolio has grown over time. The value starts at 100.</div>
            </div>
            """, unsafe_allow_html=True)

            x_data = cum_indexed.index.tolist()
            y_data = cum_indexed.values.tolist()
            steps = 60
            frame_size = max(1, len(x_data) // steps)

            frames = []
            for i in range(1, steps + 1):
                end = min(i * frame_size, len(x_data))
                frames.append(go.Frame(
                    data=[go.Scatter(
                        x=x_data[:end],
                        y=y_data[:end],
                        mode='lines',
                        line=dict(color='#16a34a', width=2.5),
                        fill='tozeroy',
                        fillcolor='rgba(22,163,74,0.08)',
                    )]
                ))
            # Builds and renders an interactive Plotly line chart showing portfolio cumulative return indexed to 100, with a labeled endpoint annotation.
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=cum_indexed.index,
                y=cum_indexed.values,
                mode='lines',
                line=dict(color='#16a34a', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(22,163,74,0.08)',
                name='Portfolio Index',
                hovertemplate='%{x|%b %d, %Y}<br>Index: %{y:.2f}<extra></extra>'
            ))
            fig1.add_annotation(
                x=cum_indexed.index[-1], y=cum_indexed.iloc[-1],
                text=f"  {cum_indexed.iloc[-1]:.2f}",
                showarrow=False,
                font=dict(color="white", size=12, family="Arial Black"),
                bgcolor="#16a34a", borderpad=5,
                xanchor="left"
            )
            fig1.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=20, b=40, l=60, r=40),
                height=340,
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#94a3b8"), title=""),
                yaxis=dict(
                    showgrid=True, gridcolor='#f1f5f9',
                    tickfont=dict(size=11, color="#94a3b8"),
                    title=dict(text="Index", font=dict(size=12, color="#94a3b8"))
                ),
                hovermode='x unified',
                showlegend=False,
            )
            st.plotly_chart(fig1, use_container_width=True)
           
            # Injects a JavaScript animation to draw the cumulative return chart from left to right over 40 frames, then renders an insight box showing portfolio growth with a $10,000 investment example.
            st.markdown("""
            <script>
            setTimeout(function() {
                var plots = document.querySelectorAll('.js-plotly-plot');
                if (plots[0]) {
                    var trace = {
                        x: plots[0].data[0].x,
                        y: plots[0].data[0].y
                    };
                    var frames = [];
                    var steps = 40;
                    for (var i = 1; i <= steps; i++) {
                        var end = Math.floor(trace.x.length * i / steps);
                        frames.push({
                            data: [{
                                x: trace.x.slice(0, end),
                                y: trace.y.slice(0, end)
                            }]
                        });
                    }
                    Plotly.animate(plots[0], frames, {
                        transition: { duration: 30, easing: 'cubic-in-out' },
                        frame: { duration: 30, redraw: false }
                    });
                }
            }, 800);
            </script>
            """, unsafe_allow_html=True)

            invested = 10000
            final_value = invested * cum_indexed.iloc[-1] / 100
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-icon">⊕</div>
                <div>
                    <div class="insight-title">What does this mean?</div>
                    <div class="insight-desc">
                        Your portfolio grew by <strong>{cum_sign}{cum_return:.2%}</strong> during this period.
                        For example, if you invested <strong>${invested:,}</strong>,
                        it would now be worth about <strong>${final_value:,.0f}</strong>.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.write("")

            #  Chart of Rolling Volatility 
            st.markdown("""
            <div class="section-card">
                <div class="section-title">
                    <span class="section-num" style="background:#7c3aed;">2</span> 20-Day Rolling Volatility
                </div>
                <div class="section-sub">Measures how much your portfolio's daily returns fluctuate over the past 20 trading days.</div>
            </div>
            """, unsafe_allow_html=True)

            rv = rolling_vol.dropna() * 100 
            avg_rv = rv.mean()
            high_threshold = 40.0
            low_threshold = 10.0

            x_data2 = rv.index.tolist()
            y_data2 = rv.values.tolist()
            steps2 = 60
            frame_size2 = max(1, len(x_data2) // steps2)

            frames2 = []
            for i in range(1, steps2 + 1):
                end = min(i * frame_size2, len(x_data2))
                frames2.append(go.Frame(
                    data=[go.Scatter(
                        x=x_data2[:end],
                        y=y_data2[:end],
                        mode='lines',
                        line=dict(color='#7c3aed', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(124,58,237,0.08)',
                    )]
                ))
            # Builds and renders an interactive Plotly line chart showing 20-day rolling volatility with reference lines for high, average, and low volatility thresholds.
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=rv.index, y=rv.values,
                mode='lines',
                line=dict(color='#7c3aed', width=2),
                fill='tozeroy',
                fillcolor='rgba(124,58,237,0.08)',
                name='Rolling Vol',
                hovertemplate='%{x|%b %d, %Y}<br>Volatility: %{y:.2f}%<extra></extra>'
            ))
            fig2.add_hline(
                y=high_threshold, line_dash="dash", line_color="#ef4444", line_width=1.5,
                annotation_text="High Volatility", annotation_position="right",
                annotation_font_color="#ef4444", annotation_font_size=11
            )
            fig2.add_hline(
                y=avg_rv, line_dash="dot", line_color="#94a3b8", line_width=1.5,
                annotation_text="Average", annotation_position="right",
                annotation_font_color="#94a3b8", annotation_font_size=11
            )
            fig2.add_annotation(
                x=rv.index[-1], y=low_threshold,
                text="Low Volatility", showarrow=False,
                font=dict(color="#16a34a", size=11),
                xanchor="right"
            )
            fig2.add_annotation(
                x=rv.index[-1], y=rv.iloc[-1],
                text=f"  {rv.iloc[-1]:.2f}%",
                showarrow=False,
                font=dict(color="white", size=12),
                bgcolor="#7c3aed", borderpad=5,
                xanchor="left"
            )
            fig2.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=20, b=40, l=60, r=80),
                height=340,
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#94a3b8"), title=""),
                yaxis=dict(
                    showgrid=True, gridcolor='#f1f5f9',
                    tickfont=dict(size=11, color="#94a3b8"),
                    title=dict(text="Volatility (%)", font=dict(size=12, color="#94a3b8")),
                    range=[0, max(high_threshold + 15, rv.max() + 5)]
                ),
                hovermode='x unified',
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
            # Injects a JavaScript animation to draw the rolling volatility chart from left to right over 40 frames.
            st.markdown("""
            <script>
            setTimeout(function() {
                var plots = document.querySelectorAll('.js-plotly-plot');
                if (plots[1]) {
                    var trace = {
                        x: plots[1].data[0].x,
                        y: plots[1].data[0].y
                    };
                    var frames = [];
                    var steps = 40;
                    for (var i = 1; i <= steps; i++) {
                        var end = Math.floor(trace.x.length * i / steps);
                        frames.push({
                            data: [{
                                x: trace.x.slice(0, end),
                                y: trace.y.slice(0, end)
                            }]
                        });
                    }
                    Plotly.animate(plots[1], frames, {
                        transition: { duration: 30, easing: 'cubic-in-out' },
                        frame: { duration: 30, redraw: false }
                    });
                }
            }, 800);
            </script>
            """, unsafe_allow_html=True)
            # Renders a purple insight box explaining the current rolling volatility value and what it means for portfolio risk.
            st.markdown(f"""
            <div class="insight-box insight-box-purple">
                <div class="insight-icon insight-icon-purple">〰️</div>
                <div>
                    <div class="insight-title">What does this mean?</div>
                    <div class="insight-desc">
                        This shows the "ups and downs" of your portfolio.
                        Higher values mean bigger daily swings (more risk).
                        Lower values mean smoother performance (less risk).
                        Current rolling volatility is <strong>{rv.iloc[-1]:.2f}%</strong>.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.write("")

            # Renders a three-card educational section explaining how to interpret rising returns, volatility spikes, and long-term investing.
            st.markdown("""
            <div class="interpret-section">
                <div class="interpret-title">📈 How to Interpret These Charts</div>
                <div class="interpret-grid">
                    <div class="interpret-card">
                        <div class="interpret-icon" style="background:#ecfdf5;">📈</div>
                        <div>
                            <div class="interpret-card-title">Rising Portfolio Return</div>
                            <div class="interpret-card-desc">Good! Your portfolio is growing over time. Look at the long-term trend rather than short-term dips.</div>
                        </div>
                    </div>
                    <div class="interpret-card">
                        <div class="interpret-icon" style="background:#f5f3ff;">〰️</div>
                        <div>
                            <div class="interpret-card-title">Volatility Spikes</div>
                            <div class="interpret-card-desc">Market uncertainty or big news can cause sharp swings. It's normal and often temporary.</div>
                        </div>
                    </div>
                    <div class="interpret-card">
                        <div class="interpret-icon" style="background:#eff6ff;">🛡️</div>
                        <div>
                            <div class="interpret-card-title">Stay Focused on the Long Term</div>
                            <div class="interpret-card-desc">Short-term volatility is normal. Staying invested and diversified helps you ride out the ups and downs.</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # content of page 3: rolling forecast
        elif page == "Page 3: Rolling Forecast":

            st.markdown("<div style='font-size:30px; font-weight:900; color:#1e293b; margin-bottom:6px;'>📅 Rolling Forecast</div>", unsafe_allow_html=True)
            st.markdown("<p style='color:#64748b; font-size:14px; margin-top:-16px;'>We look at the past 60 trading days to estimate what might happen in the next 20 trading days.</p>", unsafe_allow_html=True)

            latest_prices, avg_daily_ret, pred_ret_20d, pred_low, pred_base, pred_high, summary_df = rolling_forecast(
                prices_assets[tickers_used], horizon=20, lookback=60
            )

            first_ticker = tickers_used[0]
            first_pred = pred_ret_20d[first_ticker]

            # Generate a trend signal (bullish, bearish, neutral) for the first asset based on the predicted 20-day return rate, and set the corresponding color, icon and descriptive text
            if first_pred > 0.02:
                signal = "Bullish"
                signal_color = "#16a34a"
                badge_bg = "#dcfce7"
                badge_color = "#166534"
                signal_icon = "📈"
                signal_desc = "Strong upward momentum."
            elif first_pred < -0.02:
                signal = "Bearish"
                signal_color = "#dc2626"
                badge_bg = "#fee2e2"
                badge_color = "#991b1b"
                signal_icon = "📉"
                signal_desc = "Downward trend expected."
            else:
                signal = "Neutral"
                signal_color = "#7c3aed"
                badge_bg = "#ede9fe"
                badge_color = "#5b21b6"
                signal_icon = "➡️"
                signal_desc = "No strong upward or downward trend right now."

            pred_sign = "+" if first_pred > 0 else ""

            # Renders three summary metric cards showing forecast horizon, predicted return, and trend signal for the first ticker.
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div style="background:white; border:1px solid #e2e8f0; border-radius:16px; padding:24px; box-shadow:0 2px 8px rgba(0,0,0,0.03); height:100%;">
                    <div style="font-size:13px; font-weight:700; color:#475569; margin-bottom:12px;">📅 Forecast Horizon</div>
                    <div style="font-size:28px; font-weight:800; color:#1e293b; margin-bottom:10px;">20 Trading Days</div>
                    <div style="font-size:13px; color:#94a3b8;">The next 20 trading days outlook.</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="background:white; border:1px solid #e2e8f0; border-radius:16px; padding:24px; box-shadow:0 2px 8px rgba(0,0,0,0.03); height:100%;">
                    <div style="font-size:13px; font-weight:700; color:#475569; margin-bottom:12px;">📈 Predicted Return ({first_ticker})</div>
                    <div style="font-size:28px; font-weight:800; margin-bottom:10px; color:{signal_color};">{pred_sign}{first_pred:.2%}</div>
                    <div style="font-size:13px; color:#94a3b8;">Estimated total return over next 20 days.</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style="background:white; border:1px solid #e2e8f0; border-radius:16px; padding:24px; box-shadow:0 2px 8px rgba(0,0,0,0.03); height:100%;">
                    <div style="font-size:13px; font-weight:700; color:#475569; margin-bottom:12px;">⚡ Trend Signal</div>
                    <div style="font-size:28px; font-weight:800; margin-bottom:10px; color:{signal_color};">{signal}</div>
                    <div style="font-size:13px; color:#94a3b8;">{signal_desc}</div>
                    <div style="display:inline-block; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600; margin-top:10px; background:{badge_bg}; color:{badge_color};">
                        🛡️ Moderate Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Forecast Summary Table
            st.markdown("<h3 style='font-size:20px; font-weight:700; color:#1e293b; margin-bottom:12px;'>📊 Forecast Summary</h3>", unsafe_allow_html=True)

            display_df = summary_df.copy()

            def style_forecast_table(df):
                def color_return(val):
                    if isinstance(val, float):
                        color = '#16a34a' if val > 0 else '#dc2626'
                        return f'color: {color}; font-weight: bold'
                    return ''
                def color_outlook(val):
                    if val == 'Bullish':
                        return 'background-color: #dcfce7; color: #166534; font-weight: bold'
                    elif val == 'Bearish':
                        return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                    else:
                        return 'background-color: #ede9fe; color: #5b21b6; font-weight: bold'
                return df.style.format({
                    "Past 60D Avg Daily Return": "{:+.2%}",
                    "Predicted 20D Return":      "{:+.2%}",
                    "Current Price":             "${:.2f}",
                    "Predicted Low":             "${:.2f}",
                    "Predicted Base":            "${:.2f}",
                    "Predicted High":            "${:.2f}",
                }).map(color_return, subset=["Past 60D Avg Daily Return", "Predicted 20D Return"])\
                .map(color_outlook, subset=["Outlook"])

            st.dataframe(style_forecast_table(display_df), use_container_width=True, hide_index=True)
            st.info("💡 **Base price** is our best estimate based on recent trends. Low and high prices show a possible range depending on market volatility.")

            # === Price Forecast Chart ===
            st.markdown(f"<h3 style='font-size:20px; font-weight:700; color:#1e293b; margin-bottom:4px;'>📈 Price Forecast Chart ({first_ticker})</h3>", unsafe_allow_html=True)
            st.caption("See how the stock might move in the next 20 trading days.")
            st.pyplot(
                plot_rolling_forecast(
                    prices_assets[tickers_used],
                    pred_base, pred_low, pred_high
                )
            )

            #Meaning of Results
            st.markdown("<h3 style='font-size:20px; font-weight:700; color:#1e293b; margin-bottom:12px;'>🤔 What Do These Results Mean?</h3>", unsafe_allow_html=True)
            # Renders three color-coded explanation cards describing positive, negative, and neutral forecast scenarios.
            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                st.markdown("""
                <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:18px;'>
                    <div style='font-size:14px; font-weight:700; color:#16a34a; margin-bottom:6px;'>↗ Positive Forecast (e.g. +2%)</div>
                    <div style='font-size:13px; color:#475569;'>The stock is expected to go up. For example, +2% means a $100 stock may rise to $102.</div>
                </div>
                """, unsafe_allow_html=True)
            with ic2:
                st.markdown("""
                <div style='background:#fef2f2; border:1px solid #fecaca; border-radius:12px; padding:18px;'>
                    <div style='font-size:14px; font-weight:700; color:#dc2626; margin-bottom:6px;'>↘ Negative Forecast (e.g. -4%)</div>
                    <div style='font-size:13px; color:#475569;'>The stock is expected to go down. For example, -4% means a $100 stock may drop to $96.</div>
                </div>
                """, unsafe_allow_html=True)
            with ic3:
                st.markdown("""
                <div style='background:#f5f3ff; border:1px solid #ddd6fe; border-radius:12px; padding:18px;'>
                    <div style='font-size:14px; font-weight:700; color:#7c3aed; margin-bottom:6px;'>↔ Neutral Forecast</div>
                    <div style='font-size:13px; color:#475569;'>No clear direction. The price may move sideways in the short term.</div>
                </div>
                """, unsafe_allow_html=True)
        
        # content of page 4: CAPM analysis
        elif page == "Page 4: CAPM Analysis":
            st.markdown("<div style='font-size:30px; font-weight:900; color:#1e293b; margin-bottom:6px;'>📐 CAPM Analysis</div>", unsafe_allow_html=True)
            st.write("---")

            # Creates three bordered containers for the CAPM page, controlling the visual order of chart, model assumptions, and results table.
            chart_container = st.container(border=True)
            assumptions_container = st.container(border=True)
            table_container = st.container(border=True)

            # Model Assumptions
            with assumptions_container:
                st.markdown("#### Model Assumptions")
                st.caption("Set your annual risk-free rate used in CAPM calculation.")
                col_input, _ = st.columns([1, 2])
                with col_input:
                    rf_pct = st.number_input(
                        "Annual Risk-Free Rate (Rf) %",
                        min_value=0.0, max_value=20.0, value=4.00, step=0.25, format="%.2f"
                    )
                rf_annual = rf_pct / 100.0

                # Computes the CAPM table using the selected assets, SPY returns as the market benchmark, and the user-defined risk-free rate.
                capm_assets = rets_assets[tickers_used].copy()
                capm_df = compute_capm_table(capm_assets, rets_spy, rf_annual)

            if capm_df.empty:
                st.warning("Not enough data to estimate CAPM (need at least ~30 overlapping daily observations).")
                st.stop()

                # CAPM Beta by Ticker
            with chart_container:
                col_title, col_badge = st.columns([3, 1])
                with col_title:
                    st.markdown("#### CAPM Beta by Ticker ⓘ")
                    st.caption("Beta measures a stock's sensitivity to the overall market (SPY).")
                with col_badge:
                    st.markdown(
                        "<div style='background-color:#F3F4F6; color:#4B5563; padding:5px 10px; border-radius:15px; font-size:12px; text-align:center; margin-top:10px;'>" +
                        "Higher Beta = Higher Market Sensitivity" +
                        "</div>",
                        unsafe_allow_html=True
                    )
                fig = go.Figure()
                # # Generates a list of hex colors interpolated from dark blue to light blue, one per ticker, to create a gradient effect across the CAPM bar chart.
                n = len(capm_df)
                start_color = np.array([29, 78, 216])    
                end_color   = np.array([147, 197, 253]) 

                gradient_colors = [
                    '#{:02x}{:02x}{:02x}'.format(
                        int(start_color[0] + (end_color[0] - start_color[0]) * i / max(n-1, 1)),
                        int(start_color[1] + (end_color[1] - start_color[1]) * i / max(n-1, 1)),
                        int(start_color[2] + (end_color[2] - start_color[2]) * i / max(n-1, 1)),
                    )
                    for i in range(n)
                ]
                # Renders a bar chart of Beta values per ticker with a light-to-dark blue colorscale mapped to Beta magnitude.
                fig.add_trace(go.Bar(
                    x=capm_df["Ticker"],
                    y=capm_df["Beta"],
                    text=capm_df["Beta"].apply(lambda x: f"{x:.2f}"),
                    textposition='outside',
                    marker=dict(
                        color=capm_df["Beta"],
                        colorscale=[
                            [0.0, "#93C5FD"],
                            [0.5, "#2563EB"],
                            [1.0, "#1D4ED8"],
                        ],
                        showscale=False,
                    ),
                    width=0.4
                ))
                # Adds a dashed reference line at Beta = 1 to indicate the market average benchmark.
                fig.add_hline(
                    y=1.0, line_dash="dash", line_color="#9CA3AF",
                    annotation_text="Market Average (Beta = 1)",
                    annotation_position="top right",
                    annotation_font_color="#6B7280"
                )
                # Sets the chart layout with a white background, dynamic y-axis range, and renders the Beta bar chart in Streamlit.
                fig.update_layout(
                    plot_bgcolor='white',
                    margin=dict(t=40, b=20, l=20, r=20),
                    yaxis=dict(
                        range=[0, max(2.5, capm_df["Beta"].max() + 0.5)], 
                        showgrid=True, gridcolor='#F3F4F6', title="Beta"
                    ),
                    xaxis=dict(showgrid=False),
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

            # CAPM Results Table
            with table_container:
                st.markdown("#### CAPM Results Table") 
                def style_beta(v): return 'color: #2563EB; font-weight: bold;' 
                def style_alpha(v): return 'color: #DC2626; font-weight: bold;' if v < 0 else 'color: #16A34A; font-weight: bold;' 
                def style_return(v): return 'color: #16A34A; font-weight: bold;' 
                styled_df = capm_df.style.format({
                    "Beta": "{:.3f}",
                    "Alpha (ann.)": "{:.2%}",
                    "R^2": "{:.3f}",
                    "Market Risk Premium (ann.)": "{:.2%}",
                    "CAPM Expected Return (ann.)": "{:.2%}"
                })\
                .map(style_beta, subset=['Beta'])\
                .map(style_alpha, subset=['Alpha (ann.)'])\
                .map(style_return, subset=['CAPM Expected Return (ann.)'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # How to Interpret Beta
            with st.container(border=True):
                st.markdown("#### 💡 Beta Investment Strategy Guide")
                st.write("") 
        
                c1, c2, c3 = st.columns(3)
                # Beta = 1 Module
                with c1:
                    st.markdown("### **Beta ≈ 1**")
                    st.markdown("<p style='color:#6B7280; font-size:14px;'>Market Benchmark: Moves in tandem with the market</p>", unsafe_allow_html=True)
                    st.markdown("⚖️ **Mixed Betas (Balanced):** More stability. Lower overall volatility in both up and down markets.")
                    st.write("")
                    st.info("""
                    **Allocation Suggestions:**
                    * **Risk-Averse:** 20%-40% of equity allocation
                    * **Balanced:** 50%-70% of equity allocation
                    * **Aggressive:** 10%-30% of equity allocation
            
                    ⚠️ **Note:** Absolutely avoid heavily overweighting Beta=1 assets during the early/mid-stages of a bull market, as you will miss out on excess returns (alpha).
                    """)
                # Beta > 1 Module
                with c2:
                    st.markdown("### **<span style='color:#2563EB'>Beta > 1</span>**", unsafe_allow_html=True)
                    st.markdown("<p style='color:#6B7280; font-size:14px;'>Offensive: More volatile than the market</p>", unsafe_allow_html=True)
                    st.markdown("📈 **All High Betas (>1):** Higher potential returns, but also higher risk. This is like adding leverage.")
                    st.write("") 
                    st.warning("""
                    **Risk Warning:**
                    * If your risk tolerance is low, high-beta stocks may not be suitable for you.
                    
                    **Characteristics:** Outperforms when the market rises, but experiences deeper drawdowns when the market falls.
                    """)
                # Beta < 1 Module
                with c3:
                    st.markdown("### **<span style='color:#16A34A'>Beta < 1</span>**", unsafe_allow_html=True)
                    st.markdown("<p style='color:#6B7280; font-size:14px;'>Defensive: Portfolio 'Stabilizer'</p>", unsafe_allow_html=True)
                    st.markdown("🛡️ **All Low Betas (<1):** More defensive. Lower risk, but may also mean lower growth potential.")
                    st.write("")
                    st.success("""
                    **Strategy Logic:**
                               
                    Low-beta stocks act as a stabilizer to hedge against the volatility of mid-to-high beta growth and cyclical stocks.
                    
                    * **High Valuation / High Volatility:** Increase allocation ⬆️
                    * **Low Valuation / Early Bull Market:** Decrease allocation, shift to mid/high-beta assets ⬇️
                    """)
                    # Dynamic Output: Portfolio Diagnostic 
                    low_beta_count = len(capm_df[capm_df["Beta"] < 1])
                    total_count = len(capm_df)
                    low_beta_ratio = low_beta_count / total_count if total_count > 0 else 0
                    st.markdown("---")
                    st.markdown("**🔍 Portfolio Diagnostic:**")
                    if low_beta_ratio < 0.3:
                        st.write("📍 Your current portfolio has weak defensive capabilities. If you anticipate high market volatility or overvaluation, we **recommend increasing** your exposure to these low-beta stocks.")
                    else:
                        st.write("📍 You currently hold a significant amount of defensive assets. If you anticipate an upcoming bull market, we **recommend decreasing** this allocation to capture more upside potential.")

            st.caption("Note: CAPM is a theoretical model and does not guarantee future performance.")

        # content of page 5: diversification risk
        elif page == "Page 5: Diversification Risk":
            from datetime import date

            # Pre-calculate the required data
            top_risk_asset = rc_df.iloc[0]["ticker"]
            top_risk_pct = rc_df.iloc[0]["risk_contribution_pct_of_vol"]
            explanation = generate_correlation_explanation(corr)

            div_label = "Low" if avg_corr > 0.6 else "Moderate" if avg_corr > 0.3 else "Strong"
            div_color = "#dc2626" if avg_corr > 0.6 else "#d97706" if avg_corr > 0.3 else "#16a34a"
            div_bg    = "#fee2e2" if avg_corr > 0.6 else "#fffbeb" if avg_corr > 0.3 else "#ecfdf5"
            div_desc  = "There is room to improve diversification." if avg_corr > 0.3 else "Your portfolio is well diversified."

            # UI design
            st.markdown(f"""
            <style>
            .d5-header {{
                display: flex; justify-content: space-between; align-items: flex-start;
                margin-bottom: 24px;
            }}
            .d5-title {{ font-size: 30px; font-weight: 900; color: #1e293b;
                        display: flex; align-items: center; gap: 12px; }}
            .d5-sub {{ font-size: 14px; color: #64748b; margin-top: 6px; }}
            .d5-daterange {{
                background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
                padding: 10px 16px; font-size: 13px; color: #475569;
            }}

            /* Top 3-col summary bar */
            .d5-summary {{
                display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 0;
                background: white; border: 1px solid #e2e8f0; border-radius: 16px;
                padding: 0; margin-bottom: 28px; overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            }}
            .d5-sum-cell {{
                padding: 24px 28px;
                border-right: 1px solid #f1f5f9;
            }}
            .d5-sum-cell:last-child {{ border-right: none; }}
            .d5-sum-label {{ font-size: 13px; font-weight: 600; color: #64748b; margin-bottom: 6px; }}
            .d5-sum-main {{ font-size: 15px; color: #1e293b; line-height: 1.5; }}
            .d5-sum-main b {{ color: #6366f1; }}
            .d5-sum-value {{ font-size: 28px; font-weight: 800; line-height: 1; }}
            .d5-sum-desc {{ font-size: 12px; color: #94a3b8; margin-top: 6px; }}
            .d5-icon-cell {{
                display: flex; align-items: center; gap: 16px;
            }}
            .d5-bulb {{
                width: 48px; height: 48px; background: #ede9fe; border-radius: 50%;
                display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0;
            }}

            /* Section titles */
            .d5-sec-title {{ font-size: 20px; font-weight: 800; color: #1e293b;
                            margin-bottom: 4px; margin-top: 8px; }}
            .d5-sec-sub {{ font-size: 13px; color: #64748b; margin-bottom: 16px; }}

            /* Heatmap legend card */
            .hm-legend {{
                background: white; border: 1px solid #e2e8f0; border-radius: 14px;
                padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            }}
            .hm-legend-title {{ font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 14px; }}
            .hm-legend-item {{ display: flex; align-items: flex-start; gap: 12px; margin-bottom: 14px; }}
            .hm-legend-icon {{ font-size: 20px; flex-shrink: 0; margin-top: 2px; }}
            .hm-legend-label {{ font-size: 13px; font-weight: 700; color: #1e293b; }}
            .hm-legend-desc {{ font-size: 12px; color: #64748b; margin-top: 2px; }}

            /* Insight box */
            .d5-insight {{
                background: white; border: 1px solid #e2e8f0; border-radius: 14px;
                padding: 22px 24px; display: flex; align-items: flex-start; gap: 20px;
                margin-top: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            }}
            .d5-insight-avatar {{ font-size: 48px; flex-shrink: 0; }}
            .d5-insight-title {{ font-size: 15px; font-weight: 700; color: #1e293b; margin-bottom: 8px; }}
            .d5-insight-text {{ font-size: 13px; color: #475569; line-height: 1.6; }}
            .d5-div-badge {{
                display: inline-block; padding: 8px 18px; border-radius: 10px;
                font-size: 13px; font-weight: 700; margin-top: 14px;
                border: 1px solid currentColor;
            }}

            /* Risk contribution list */
            .rc-card {{
                background: white; border: 1px solid #e2e8f0; border-radius: 14px;
                padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            }}
            .rc-row {{
                display: flex; align-items: center; gap: 14px;
                padding: 12px 0; border-bottom: 1px solid #f8fafc;
            }}
            .rc-row:last-child {{ border-bottom: none; }}
            .rc-dot {{ width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }}
            .rc-name {{ font-size: 14px; font-weight: 700; color: #1e293b; }}
            .rc-desc {{ font-size: 12px; color: #94a3b8; }}
            .rc-pct {{ font-size: 16px; font-weight: 800; color: #1e293b; margin-left: auto; }}

            /* Insight right panel */
            .rc-insight {{
                background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 14px;
                padding: 20px;
            }}
            .rc-insight-title {{ font-size: 14px; font-weight: 700; color: #6d28d9; margin-bottom: 10px; }}
            .rc-insight-text {{ font-size: 13px; color: #475569; line-height: 1.6; }}

            /* What Can You Do Next */
            .next-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 8px; }}
            .next-card {{
                background: white; border: 1px solid #e2e8f0; border-radius: 14px;
                padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            }}
            .next-icon {{ font-size: 24px; margin-bottom: 10px; }}
            .next-title {{ font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 6px; }}
            .next-desc {{ font-size: 12px; color: #64748b; line-height: 1.5; margin-bottom: 12px; }}

            /* Footer */
            .d5-footer {{
                background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px;
                padding: 12px 18px; font-size: 13px; color: #92400e; margin-top: 24px;
            }}
            </style>

            <div class="d5-header">
                <div>
                    <div class="d5-title">🛡️ Diversification Risk</div>
                    <div class="d5-sub">Understand how your holdings contribute to risk and how they move together.</div>
                </div>
                <div class="d5-daterange">📅 Data as of<br><strong>{date.today().strftime("%B %d, %Y")}</strong></div>
            </div>

            <!-- Summary Bar -->
            <div class="d5-summary">
                <div class="d5-sum-cell d5-icon-cell">
                    <div class="d5-bulb">💡</div>
                    <div>
                        <div class="d5-sum-label">Key Takeaway</div>
                        <div class="d5-sum-main">
                            Your portfolio's risk is mainly driven by <b>{top_risk_asset}</b>,
                            which contributes {top_risk_pct:.1f}% of total volatility.
                        </div>
                    </div>
                </div>
                <div class="d5-sum-cell">
                    <div class="d5-sum-label">Diversification Level</div>
                    <div class="d5-sum-value" style="color:{div_color};">{div_label}</div>
                    <div class="d5-sum-desc">{div_desc}</div>
                </div>
                <div class="d5-sum-cell">
                    <div class="d5-sum-label">Total Portfolio Volatility (Ann.)</div>
                    <div class="d5-sum-value" style="color:#6366f1;">{ann_vol*100:.2f}%</div>
                    <div class="d5-sum-desc">Based on historical data</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Correlation Heatmap Section
            st.markdown("""
            <div class="d5-sec-title">Correlation Heatmap</div>
            <div class="d5-sec-sub">This shows how closely the selected stocks move together.<br>
            Higher correlation means weaker diversification.</div>
            """, unsafe_allow_html=True)

            col_hm, col_legend = st.columns([3, 1])
            with col_hm:
                st.pyplot(plot_corr_heatmap(corr))

            with col_legend:
                st.markdown(f"""
                <div class="hm-legend">
                    <div class="hm-legend-title">How to Read This Heatmap</div>
                    <div class="hm-legend-item">
                        <div class="hm-legend-icon">😊</div>
                        <div>
                            <div class="hm-legend-label">Closer to 1 (Dark Blue)</div>
                            <div class="hm-legend-desc">Stocks move in the same direction.<br>Less diversification benefit.</div>
                        </div>
                    </div>
                    <div class="hm-legend-item">
                        <div class="hm-legend-icon">😐</div>
                        <div>
                            <div class="hm-legend-label">Around 0 (Light)</div>
                            <div class="hm-legend-desc">Stocks move independently.<br>Great for diversification.</div>
                        </div>
                    </div>
                    <div class="hm-legend-item">
                        <div class="hm-legend-icon">😟</div>
                        <div>
                            <div class="hm-legend-label">Closer to -1 (Yellow)</div>
                            <div class="hm-legend-desc">Stocks move in opposite directions.<br>Strong diversification benefit.</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Insight Box
            st.markdown(f"""
            <div class="d5-insight">
                <div class="d5-insight-avatar">🧑‍💼</div>
                <div style="flex:1;">
                    <div class="d5-insight-title">What does this mean for you?</div>
                    <div class="d5-insight-text">{explanation}</div>
                    <div class="d5-div-badge" style="background:{div_bg}; color:{div_color};">
                        ⚖️ &nbsp;{div_label} Diversification
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.write("")

            # Risk Contribution Section
            st.markdown("""
            <div class="d5-sec-title">Risk Contribution</div>
            <div class="d5-sec-sub">This shows how much each holding contributes to the portfolio's total volatility.</div>
            """, unsafe_allow_html=True)

            rc_colors = ["#1D4ED8", "#0B1F3B", "#F3CA43", "#60A5FA", "#10b981", "#ef4444"]

            col_pie, col_list, col_ri = st.columns([2, 2, 1.5])

            with col_pie:
                st.pyplot(plot_risk_contrib(rc_df))

            with col_list:
                rc_rows_html = '<div class="rc-card">'
                for idx, (_, row) in enumerate(rc_df.iterrows()):
                    desc = (
                        "Main driver of portfolio risk"
                        if row["risk_contribution_pct_of_vol"] == rc_df["risk_contribution_pct_of_vol"].max()
                        else "Lower risk contribution"
                        if row["risk_contribution_pct_of_vol"] == rc_df["risk_contribution_pct_of_vol"].min()
                        else "Moderate risk contribution"
                    )
                    color = rc_colors[idx % len(rc_colors)]
                    ticker = row['ticker']
                    pct = f"{row['risk_contribution_pct_of_vol']:.1f}%"

                    rc_rows_html += (
                        '<div class="rc-row">'
                        f'<div class="rc-dot" style="background:{color};"></div>'
                        '<div>'
                        f'<div class="rc-name">{ticker}</div>'
                        f'<div class="rc-desc">{desc}</div>'
                        '</div>'
                        f'<div class="rc-pct">{pct}</div>'
                        '</div>'
                    )
                rc_rows_html += '</div>'
                st.markdown(rc_rows_html, unsafe_allow_html=True)

            with col_ri:
                st.markdown(f"""
                <div class="rc-insight">
                    <div class="rc-insight-title">🎯 What Does This Mean?</div>
                    <div class="rc-insight-text">
                        <b>{top_risk_asset}</b> contributes more than half of the portfolio's total risk.
                        Consider balancing your positions if you'd like a lower volatility portfolio.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.write("")

            # What Can You Do Next
            st.markdown('<div class="d5-sec-title">What Can You Do Next?</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="next-grid">
                <div class="next-card">
                    <div class="next-icon">🥧</div>
                    <div class="next-title">Improve Diversification</div>
                    <div class="next-desc">Consider adding assets with low correlation to reduce overall risk.</div>
                </div>
                <div class="next-card">
                    <div class="next-icon">🛡️</div>
                    <div class="next-title">Manage Risk Exposure</div>
                    <div class="next-desc">Adjust position sizes to balance risk across your holdings.</div>
                </div>
                <div class="next-card">
                    <div class="next-icon">📈</div>
                    <div class="next-title">Monitor Regularly</div>
                    <div class="next-desc">Risk changes over time. Check in regularly to stay on track.</div>
                </div>
            </div>

            <div class="d5-footer">
                💡 <strong>Note:</strong> Past performance does not guarantee future results.
                Always consider your risk tolerance and investment objectives.
            </div>
            """, unsafe_allow_html=True)
           
        # content of page 6: radar chart for individual stock performance
        elif page == "Page 6: Individual Stock Performance":
            st.markdown("<div style='font-size:30px; font-weight:900; color:#1e293b; margin-bottom:6px;'>📡 Individual Stock Performance Across Key Metrics</div>", unsafe_allow_html=True)
            from datetime import date
            st.markdown(f"""
            <style>
            .radar-header {{
                display: flex; justify-content: space-between; align-items: flex-start;
                background: white; border: 1px solid #e2e8f0; border-radius: 16px;
                padding: 24px 28px; margin-bottom: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }}
            .radar-header-left {{ display: flex; align-items: center; gap: 16px; }}
            .radar-icon {{
                width: 48px; height: 48px; background: #eff6ff; border-radius: 12px;
                display: flex; align-items: center; justify-content: center; font-size: 24px;
            }}
            .radar-title {{ font-size: 28px; font-weight: 800; color: #1e293b; margin: 0; }}
            .radar-sub {{ font-size: 14px; color: #64748b; margin-top: 4px; }}
            .radar-date {{
                background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
                padding: 10px 16px; font-size: 13px; color: #475569; text-align: right;
            }}
            .radar-date strong {{ display: block; color: #1e293b; font-size: 14px; }}
            </style>
            <div class="radar-header">
                <div class="radar-header-left">
                    <div class="radar-icon">🛡️</div>
                    <div>
                        <div class="radar-title">Portfolio Risk Radar</div>
                        <div class="radar-sub">Compare key risk-return metrics across your portfolio holdings. Higher is better for all metrics.</div>
                    </div>
                </div>
                <div class="radar-date">
                    📅 Data as of<br>
                    <strong>{date.today().strftime("%B %d, %Y")}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Process of computing
            raw_df, norm_df = compute_radar_metrics(prices_assets[tickers_used], final_weights)

            categories = ["Ann. Return", "Max Drawdown", "Calmar Ratio", "Sharpe Ratio", "CVaR (95%)"]
            labels_display = [
                "Ann. Return\n(Normalized)",
                "Max Drawdown\n(Inverted & Normalized)",
                "Calmar Ratio\n(Normalized)",
                "Sharpe Ratio\n(Normalized)",
                "CVaR (95%)\n(Inverted & Normalized)",
            ]
            categories_closed = categories + [categories[0]]
            labels_closed = labels_display + [labels_display[0]]

            palette = ["#F3CA43", "#A5B4FC", "#60A5FA", "#10b981", "#f59e0b", "#ef4444"]
            fill_palette = ["rgba(243,202,67,0.2)", "rgba(165,180,252,0.25)", "rgba(96,165,250,0.2)",
                            "rgba(16,185,129,0.2)", "rgba(245,158,11,0.2)", "rgba(239,68,68,0.2)"]

            fig = go.Figure()

            for i, ticker in enumerate(norm_df.index):
                values = norm_df.loc[ticker, categories].tolist()
                values_closed = values + [values[0]]
                fig.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=labels_closed,
                    fill='toself',
                    name=ticker,
                    line=dict(color=palette[i % len(palette)], width=2.5),
                    fillcolor=fill_palette[i % len(fill_palette)],
                ))

            fig.update_layout(
                polar=dict(
                    bgcolor="#F8FAFC",
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        tickfont=dict(size=10, color="#94a3b8"),
                        gridcolor="#e2e8f0", linecolor="#e2e8f0",
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=12, color="#334155"),
                        gridcolor="#e2e8f0", linecolor="#cbd5e1",
                    )
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5,
                    font=dict(size=13, color="#1e293b"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="#FFFFFF",
                margin=dict(t=60, b=100, l=80, r=80),
                height=560,
                transition={
                    "duration": 800,
                    "easing": "cubic-in-out"
                },
                uirevision="radar_chart",
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Understanding the Metrics Table
            st.markdown("""
            <style>
            .metrics-section {{
                background: white; border: 1px solid #e2e8f0; border-radius: 16px;
                padding: 28px; margin-bottom: 24px;
            }}
            .metrics-section-title {{ font-size: 20px; font-weight: 700; color: #1e293b; margin-bottom: 4px; }}
            .metrics-section-sub {{ font-size: 13px; color: #64748b; margin-bottom: 20px; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; }}
            .metrics-table th {{
                text-align: left; font-size: 13px; font-weight: 600; color: #64748b;
                padding: 10px 14px; border-bottom: 1px solid #f1f5f9;
            }}
            .metrics-table td {{ padding: 14px; border-bottom: 1px solid #f8fafc; vertical-align: middle; }}
            .metric-name {{ font-weight: 700; font-size: 14px; color: #1e293b; }}
            .metric-desc {{ font-size: 13px; color: #475569; }}
            .formula-badge {{
                background: #f1f5f9; color: #475569; border-radius: 6px;
                padding: 4px 10px; font-size: 12px; font-family: monospace;
                display: inline-block; margin-left: 8px;
            }}
            .good-badge {{
                background: #ecfdf5; color: #059669; border-radius: 6px;
                padding: 4px 10px; font-size: 12px; font-weight: 600;
            }}
            .metric-icon {{
                width: 36px; height: 36px; border-radius: 8px;
                display: inline-flex; align-items: center; justify-content: center; font-size: 18px;
            }}
            .info-note {{
                background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
                padding: 12px 16px; font-size: 13px; color: #64748b; margin-top: 16px;
            }}
            </style>

            <div class="metrics-section">
                <div class="metrics-section-title" style="font-size:24px; font-weight:900; color:#1e293b;">📖 Understanding the Metrics</div>
                <div class="metrics-section-sub">
                    Each metric is normalized (0 to 1) across all holdings for fair comparison.<br>
                    The higher the score, the stronger the risk-return profile.
                </div>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>What It Measures</th>
                        <th>How It's Calculated</th>
                        <th>What's Good?</th>
                    </tr>
                    <tr>
                        <td>
                            <span class="metric-icon" style="background:#ecfdf5;">📈</span>
                            <span class="metric-name" style="margin-left:8px;">Annualized Return</span>
                        </td>
                        <td><span class="metric-desc">The average yearly return over the selected period.</span></td>
                        <td><span class="formula-badge">Formula</span> (Ending / Beginning Value)^(252/N) − 1</td>
                        <td><span class="good-badge">↑ Higher the better</span></td>
                    </tr>
                    <tr>
                        <td>
                            <span class="metric-icon" style="background:#fff1f2;">📉</span>
                            <span class="metric-name" style="margin-left:8px;">Max Drawdown</span>
                        </td>
                        <td><span class="metric-desc">The largest peak-to-trough decline. Lower drawdown is better.</span></td>
                        <td><span class="formula-badge">Formula</span> (Peak − Trough) / Peak &nbsp;(inverted for scoring)</td>
                        <td><span class="good-badge">↑ Higher the better</span></td>
                    </tr>
                    <tr>
                        <td>
                            <span class="metric-icon" style="background:#fffbeb;">⚖️</span>
                            <span class="metric-name" style="margin-left:8px;">Calmar Ratio</span>
                        </td>
                        <td><span class="metric-desc">Measures return relative to the maximum drawdown.</span></td>
                        <td><span class="formula-badge">Formula</span> Annualized Return / Max Drawdown</td>
                        <td><span class="good-badge">↑ Higher the better</span></td>
                    </tr>
                    <tr>
                        <td>
                            <span class="metric-icon" style="background:#f5f3ff;">📊</span>
                            <span class="metric-name" style="margin-left:8px;">Sharpe Ratio</span>
                        </td>
                        <td><span class="metric-desc">Measures return per unit of volatility (risk-adjusted return).</span></td>
                        <td><span class="formula-badge">Formula</span> (Ann. Return − Rf) / Ann. Volatility</td>
                        <td><span class="good-badge">↑ Higher the better</span></td>
                    </tr>
                    <tr>
                        <td>
                            <span class="metric-icon" style="background:#eff6ff;">🛡️</span>
                            <span class="metric-name" style="margin-left:8px;">CVaR (95%)</span>
                        </td>
                        <td><span class="metric-desc">Expected average loss in the worst 5% of cases. Lower loss is better.</span></td>
                        <td><span class="formula-badge">Formula</span> Average of worst 5% returns &nbsp;(inverted for scoring)</td>
                        <td><span class="good-badge">↑ Higher the better</span></td>
                    </tr>
                </table>
                <div class="info-note" style="margin-bottom:48px;">
                    ℹ️ All metrics are normalized for comparison. Higher normalized score = better performance.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Best / Worst Cards
            norm_df["Total Score"] = norm_df[categories].mean(axis=1)
            best_ticker = norm_df["Total Score"].idxmax()
            worst_ticker = norm_df["Total Score"].idxmin()
            best_score = norm_df.loc[best_ticker, "Total Score"]
            worst_score = norm_df.loc[worst_ticker, "Total Score"]

            st.markdown(f"""
            <style>
            .bw-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
            .best-card {{
                background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
                border: 1px solid #fde68a; border-radius: 16px; padding: 28px;
                display: flex; align-items: flex-start; gap: 20px; position: relative; overflow: hidden;
            }}
            .worst-card {{
                background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
                border: 1px solid #fed7aa; border-radius: 16px; padding: 28px;
                display: flex; align-items: flex-start; gap: 20px; position: relative; overflow: hidden;
            }}
            .bw-icon {{ font-size: 36px; }}
            .bw-label {{ font-size: 13px; font-weight: 600; color: #92400e; margin-bottom: 4px; }}
            .bw-ticker {{ font-size: 32px; font-weight: 900; color: #1e293b; line-height: 1; }}
            .bw-score {{ font-size: 14px; font-weight: 600; margin-top: 6px; }}
            .bw-desc {{ font-size: 13px; color: #64748b; margin-top: 8px; line-height: 1.5; }}
            .best-score {{ color: #d97706; }}
            .worst-score {{ color: #ea580c; }}
            </style>
            <div class="bw-grid">
                <div class="best-card">
                    <div class="bw-icon">🏆</div>
                    <div>
                        <div class="bw-label">Best Overall</div>
                        <div class="bw-ticker">{best_ticker}</div>
                        <div class="bw-score best-score">Average Normalized Score: {best_score:.2f}</div>
                        <div class="bw-desc">{best_ticker} has the strongest overall risk-return profile across all five metrics.</div>
                    </div>
                </div>
                <div class="worst-card">
                    <div class="bw-icon">⚠️</div>
                    <div>
                        <div class="bw-label">Weakest Overall</div>
                        <div class="bw-ticker">{worst_ticker}</div>
                        <div class="bw-score worst-score">Average Normalized Score: {worst_score:.2f}</div>
                        <div class="bw-desc">{worst_ticker} has the weakest overall risk-return profile across all five metrics.</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

           # What Do These Scores Mean
            st.markdown("""
            <style>
            .scores-wrapper {
                background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
                border: 1px solid #c7d2fe;
                border-radius: 20px;
                padding: 28px 32px;
                margin-bottom: 24px;
            }
            .scores-title {
                font-size: 20px; font-weight: 800; color: #1e1b4b; margin-bottom: 20px;
            }
            .scores-grid {
                display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0;
            }
            .score-card {
                display: flex; align-items: flex-start; gap: 16px;
                padding: 0 24px;
                border-right: 1px solid #c7d2fe;
            }
            .score-card:first-child { padding-left: 0; }
            .score-card:last-child { border-right: none; }
            .score-icon-box {
                width: 44px; height: 44px; border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                font-size: 20px; flex-shrink: 0; margin-top: 2px;
            }
            .score-card-title {
                font-size: 14px; font-weight: 700; color: #1e1b4b; margin-bottom: 6px;
            }
            .score-card-desc {
                font-size: 13px; color: #4338ca; line-height: 1.6;
            }
            .scores-footer {
                margin-top: 22px;
                padding-top: 16px;
                border-top: 1px solid #c7d2fe;
                font-size: 13px; color: #4338ca;
            }
            .scores-footer strong { color: #1e1b4b; }
            </style>

            <div class="scores-wrapper">
                <div class="scores-title">What Do These Scores Mean?</div>
                <div class="scores-grid">
                    <div class="score-card">
                        <div class="score-icon-box" style="background: #dcfce7;">📈</div>
                        <div>
                            <div class="score-card-title">Higher Score = Better</div>
                            <div class="score-card-desc">A higher overall score means the stock delivers better returns while taking less risk.</div>
                        </div>
                    </div>
                    <div class="score-card">
                        <div class="score-icon-box" style="background: #e0e7ff;">⚖️</div>
                        <div>
                            <div class="score-card-title">Holistic Comparison</div>
                            <div class="score-card-desc">No single metric tells the whole story. We combine all 5 normalized metrics with equal weight.</div>
                        </div>
                    </div>
                    <div class="score-card">
                        <div class="score-icon-box" style="background: #ede9fe;">🔍</div>
                        <div>
                            <div class="score-card-title">Use It Wisely</div>
                            <div class="score-card-desc">Use this view to guide portfolio adjustments based on your risk-return goals.</div>
                        </div>
                    </div>
                </div>
                <div class="scores-footer">
                    💡 <strong>Note:</strong> Past performance does not guarantee future results.
                    Always consider your risk tolerance and investment objectives.
                </div>
            </div>
            """, unsafe_allow_html=True)
    # Catches and displays any runtime errors with a full traceback for debugging.               
    except Exception as e:
        st.error("Error processing inputs")
        st.code(traceback.format_exc())