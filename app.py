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

# -----------------------
# UI Helpers & Fetching
# -----------------------
def parse_tickers(tickers_text: str):
    raw = tickers_text.replace(",", " ").split()
    tickers = [t.strip().upper() for t in raw if t.strip()]
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

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
    weights = np.asarray(weights, dtype=float).reshape(-1)   # 变成 1D
    sigma = np.asarray(cov_ann.values, dtype=float)          # cov matrix

    w_vec = weights.reshape(-1, 1)                           # column vector

    # sanity check (optional but recommended)
    if sigma.shape[0] != w_vec.shape[0]:
        raise ValueError(f"Dimension mismatch: sigma {sigma.shape}, weights {w_vec.shape}")

    port_var = (w_vec.T @ sigma @ w_vec).item()              # 强制取单个标量
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

# -----------------------
# Plot Functions
# -----------------------
def plot_cumulative(cum):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(cum.index, cum.values)
    plt.title("Portfolio Cumulative Return (Index)")
    plt.tight_layout()
    return fig

def plot_corr_heatmap(corr):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bg_color = "#FFFFFF"  # 浅灰白背景
    navy_color = "#08277B" # 海军蓝 (标题和字体)
    gold_accent = '#F3CA43' # 强调金 (替换后的金色)
    primary_blue = '#08277B' # 主色蓝\
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    colors = [gold_accent, bg_color, '#08277B']
    n_bins = 100
    cmap_name = 'auth_gold_navy'
    cm_auth = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # 5. 绘制 Heatmap
    cax = ax.imshow(corr.values, aspect="auto", cmap=cm_auth, vmin=-1.0, vmax=1.0)
    
    # 6. 配置颜色的颜色条 (Colorbar)
    cbar = fig.colorbar(cax, orientation='vertical')
    cbar.ax.tick_params(colors='#08277B')  # 颜色条刻度用海军蓝
    cbar.outline.set_edgecolor('#E0E0E0')   # 颜色条边框用很淡的灰色

    # 7. 配置坐标轴标签和字体
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="#0B1F3B", fontsize=10)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, color="#0B1F3B", fontsize=10)
    
    # 彭博风：移除不必要的四周黑边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 8. 标题用海军蓝加粗
    ax.set_title("Correlation Heatmap", color='#0B1F3B', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def generate_correlation_explanation(corr_matrix):
    import numpy as np
    
    tickers = corr_matrix.columns
    explanations = []
    
    # 找最高相关的pair
    max_corr = 0
    max_pair = None
    
    # 找最低相关的pair
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
    
    # 生成解释
    text = ""
    
    if max_pair:
        text += f"{max_pair[0]} and {max_pair[1]} tend to move very closely together, meaning they carry similar risk. "
    
    if min_pair:
        text += f"In contrast, {min_pair[0]} and {min_pair[1]} behave more independently, which helps improve diversification. "
    
    # 整体判断
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    
    if avg_corr > 0.6:
        text += "Overall, your portfolio contains many highly correlated assets, which may reduce diversification."
    elif avg_corr > 0.3:
        text += "Overall, your portfolio shows a moderate level of diversification."
    else:
        text += "Overall, your portfolio is well diversified, with assets that do not move strongly together."
    
    return text

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
        startangle=90,          # 从正上方12点钟方向开始
        counterclock=False,     # 顺时针排列
        pctdistance=0.75,       # 百分比数字的位置
        textprops={'color': '#0B1F3B', 'fontweight': 'medium', 'fontsize': 11}, # 外部标签用海军蓝
        wedgeprops={'edgecolor': bg_color, 'linewidth': 2.5, 'width': 0.45}   # 关键：width 控制环的厚度，实现环形图
        )
     for i, autotext in enumerate(autotexts):
        autotext.set_weight('bold')
        autotext.set_fontsize(10)
        # 金色(0)和浅蓝(3,4,5)背景用海军蓝字，深蓝(1,2)背景用背景色亮字
        if i == 0 or i >= 3:
            autotext.set_color('#0B1F3B') 
        else:
            autotext.set_color(bg_color)
            
     plt.tight_layout()
     return fig

def compute_radar_metrics(prices_assets: pd.DataFrame, weights: np.ndarray, rf_annual: float = 0.04):
    #计算雷达图所需的5个指标，按ticker分别计算（不是组合层面）返回 DataFrame，index=ticker，columns=5个指标（已归一化到0-1）
    rets = prices_assets.pct_change().dropna()
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    metrics = {}

    for ticker in rets.columns:
        r = rets[ticker].dropna()

        # 1. Annualized Return (过去全部数据年化)
        ann_ret = r.mean() * TRADING_DAYS

        # 2. Max Drawdown
        cum = (1 + r).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = drawdown.min()  # 负数

        # 3. Calmar Ratio = 年化收益 / abs(最大回撤)
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

        # 4. Sharpe-like = (年化收益 - rf) / 年化波动率
        ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan

        # 5. CVaR (Expected Shortfall) at 95% — 取最差5%日收益的均值（负数）
        var_95 = np.percentile(r, 5)
        cvar = r[r <= var_95].mean()  # 负数，越小越危险

        metrics[ticker] = {
            "Ann. Return": ann_ret,
            "Max Drawdown": max_dd,       # 负数
            "Calmar Ratio": calmar,
            "Sharpe Ratio": sharpe,
            "CVaR (95%)": cvar,           # 负数
        }

    raw_df = pd.DataFrame(metrics).T  # shape: (n_tickers, 5)
    # --- 归一化到 [0, 1]，让雷达图可比 ---
    # 注意：Max Drawdown 和 CVaR 是负数，越接近0越好 → 取反后再归一化
    norm_df = raw_df.copy()
    def minmax(series, padding=0.1):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        return padding + (1 - padding) * (series - mn) / (mx - mn)
    norm_df["Ann. Return"]   = minmax(raw_df["Ann. Return"])
    norm_df["Calmar Ratio"]  = minmax(raw_df["Calmar Ratio"])
    norm_df["Sharpe Ratio"]  = minmax(raw_df["Sharpe Ratio"])
    norm_df["Max Drawdown"]  = minmax(-raw_df["Max Drawdown"])   # 取反：回撤越小越好
    norm_df["CVaR (95%)"]    = minmax(-raw_df["CVaR (95%)"])     # 取反：尾损越小越好

    return raw_df, norm_df

def plot_rolling_vol(rolling_vol):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(rolling_vol.index, rolling_vol.values)
    plt.title("20-Day Rolling Volatility")
    plt.tight_layout()
    return fig

# === 👇 注意：下面是独立的辅助函数，必须顶格写 (0 个空格缩进) 👇 ===
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
    
    summary_df = pd.DataFrame({
        "Current Price": latest_prices.round(2),
        "Past 60d Avg Daily Return": avg_daily_ret.round(4),
        "Predicted 20d Return": pred_ret.round(4),
        "Predicted Low Price": pred_low.round(2),
        "Predicted Base Price": pred_base.round(2),
        "Predicted High Price": pred_high.round(2),
    })
    return latest_prices, avg_daily_ret, pred_ret, pred_low, pred_base, pred_high, summary_df

def plot_rolling_forecast(prices_assets: pd.DataFrame, pred_base: pd.Series, pred_low: pd.Series, pred_high: pd.Series):
    fig = plt.figure(figsize=(9, 5))
    ticker = prices_assets.columns[0]
    hist = prices_assets[ticker].tail(120)
    plt.plot(hist.index, hist.values, label=f"{ticker} Historical Price")
    
    future_idx = pd.date_range(start=hist.index[-1], periods=21, freq="B")[1:]
    plt.plot(
        [hist.index[-1], future_idx[-1]],
        [hist.iloc[-1], pred_base[ticker]],
        linestyle="--",
        label="Base Forecast"
    )
    plt.fill_between(
        [hist.index[-1], future_idx[-1]],
        [hist.iloc[-1], pred_low[ticker]],
        [hist.iloc[-1], pred_high[ticker]],
        alpha=0.2,
        label="Forecast Range"
    )
    plt.title(f"Rolling Forecast for {ticker}")
    plt.legend()
    plt.tight_layout()
    return fig

#-----------------------------------------
def compute_capm_table(rets_assets: pd.DataFrame, rets_mkt: pd.Series, rf_annual: float):
    """
    rets_assets: DataFrame, columns=tickers, daily returns
    rets_mkt: Series, SPY daily returns
    rf_annual: annual risk-free rate (e.g., 0.04 for 4%)
    """
    # daily rf from annual rf
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1

    # align
    idx = rets_assets.index.intersection(rets_mkt.index)
    X = (rets_mkt.loc[idx] - rf_daily).astype(float)  # market excess
    mkt_prem_ann = X.mean() * TRADING_DAYS            # annualized market risk premium

    out = []
    for t in rets_assets.columns:
        y = (rets_assets.loc[idx, t] - rf_daily).astype(float)  # stock excess

        # drop NA
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


# -----------------------
# Sidebar UI (干净利落，没有 weight 输入框了)
# -----------------------
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

# 👉 关键：后面仍然用 period = "1y"/"3y"/"5y"
period = period_options[selected_label]

run = st.sidebar.button("Run")

page = st.sidebar.radio(

    "Go to page",
    ["Page 1: Risk", "Page 2: History", "Page 3: Rolling Forecast", "Page 4: CAPM", "Page 5: Diversification", "Page 6: Individual Stock Performance Across Key Metrics"],
    index=0
)

st.sidebar.caption("Enter tickers and click Run.")


# -----------------------
# 主逻辑 (核心计算在顶部，渲染在底部)
# -----------------------
if 'has_run' not in st.session_state:
    st.session_state['has_run'] = False

if run:
    st.session_state['has_run'] = True
    # 如果用户重新点击了 run，我们把之前的权重记录清空，恢复均等权重
    

if not st.session_state['has_run']:
    st.info("Enter tickers and click Run in the sidebar.")
else:
    try:
        # === 1. 数据准备阶段 ===
        tickers = parse_tickers(tickers_text)
        if len(tickers) < 2:
            st.error("Please enter at least 2 tickers.")
            st.stop()
            
        n = len(tickers)

        # 1) download user tickers ONLY
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

        # 2) download SPY separately (do NOT mix with assets)
        prices_spy_df = fetch_prices(["SPY"], period=period)
        if "SPY" not in prices_spy_df.columns:
            st.error("SPY data missing. Needed for market_return feature.")
            st.stop()

        prices_spy = prices_spy_df["SPY"].copy()

        # === 2. 权重解析阶段 ===
        if 'weight_input_str' not in st.session_state:
            st.session_state['weight_input_str'] = ""
            
        try:
            weights_used = parse_weights(st.session_state['weight_input_str'], n)
        except ValueError as e:
            st.warning(f"Weight input error: {e}. Defaulting to equal weights.")
            weights_used = np.ones(n) / n

        # === 3. 全局核心计算阶段 (无论在哪个Page，这里都会被执行) ===
        tickers_used, final_weights, ann_vol, corr, rc_df, cum, rets_assets = compute_current_metrics(
            prices_assets, weights_used
        )

        rets_spy = prices_spy.pct_change().dropna()
        common_idx = rets_assets.index.intersection(rets_spy.index)
        rets_assets = rets_assets.loc[common_idx]
        rets_spy = rets_spy.loc[common_idx]

        # ML Features & Prediction
        # ML Features & Prediction (use the SAME feature pipeline as training)
       # ML Features & Prediction (use the SAME feature pipeline as training)
        w_vec = np.array(final_weights[:len(tickers_used)], dtype=float)

        X_rt = build_realtime_feature_vector(
        rets_assets[tickers_used],
        rets_spy,
        w_vec
        )

        # 只取最后一行，保证是 1-row
        X_rt2 = X_rt.tail(1)

        # 预测结果强制压成一维，再取第一个值
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

        # === 4. UI 页面分发阶段 ===
        if page == "Page 1: Risk":
            
            # --- 1. 权重输入区域 (保持功能性) ---
            st.markdown("### ⚖️ Set Portfolio Weights")
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

           # --- 2. 动态计算卡片所需的展示数据 (全部换为英文语境) ---
            v_val = ann_vol * 100
            v_pct = min(v_val / 40 * 100, 100) # 刻度条百分比位置
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

            # --- 3. 核心 CSS 样式 (保持不变) ---
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
            .scale-fill { height: 100%; border-radius: 3px; position: absolute; left: 0; top: 0; }
            .scale-indicator { width: 14px; height: 14px; border-radius: 50%; position: absolute; top: -4px; transform: translateX(-50%); border: 2.5px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
            .scale-labels { display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; text-align: center; }
            .scale-labels span { flex: 1; }
            .footer-note { font-size: 13px; color: #94a3b8; text-align: left; margin-top: 10px; }
            </style>
            """
            st.markdown(custom_css, unsafe_allow_html=True)

            # --- 4. 生成 HTML 结构 (已全部替换为英文文本，请顶格复制) ---
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
<div class="scale-fill" style="width: {v_pct}%; background: #3b82f6;"></div>
<div class="scale-indicator" style="left: {v_pct}%; background: #3b82f6;"></div>
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
<div class="scale-fill" style="width: {r_pct}%; background: #10b981;"></div>
<div class="scale-indicator" style="left: {r_pct}%; background: #10b981;"></div>
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
<div class="scale-fill" style="width: {d_pct}%; background: #8b5cf6;"></div>
<div class="scale-indicator" style="left: {d_pct}%; background: #8b5cf6;"></div>
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
<div class="scale-fill" style="width: {m_pct}%; background: #f59e0b;"></div>
<div class="scale-indicator" style="left: {m_pct}%; background: #f59e0b;"></div>
</div>
<div class="scale-labels"><span>Low<br>&lt;20%</span><span>Med<br>20-50%</span><span>High<br>50-75%</span><span>V.High<br>&gt;75%</span></div>
</div>
</div>
</div>
<div class="footer-note">Disclaimer: The above metrics are for reference only and do not constitute investment advice. Investing involves risk.</div>
"""
            st.markdown(html_content, unsafe_allow_html=True)

            
        elif page == "Page 2: History":
       
            st.subheader("Historical Portfolio Performance")
            st.pyplot(plot_cumulative(cum))

            rolling_vol = portfolio_return.rolling(20).std() * np.sqrt(TRADING_DAYS)
            st.markdown("### 20-Day Rolling Volatility")
            st.pyplot(plot_rolling_vol(rolling_vol))
        
        elif page == "Page 3: Rolling Forecast":
            st.subheader("Rolling Forecast")
            st.caption(
                "This page uses the past 60 trading days average return to estimate the next 20 trading days outlook."
            )

            latest_prices, avg_daily_ret, pred_ret_20d, pred_low, pred_base, pred_high, summary_df = rolling_forecast(
                prices_assets[tickers_used], horizon=20, lookback=60
            )

            first_ticker = tickers_used[0]
            first_pred = pred_ret_20d[first_ticker]

            if first_pred > 0.05:
                signal = "Bullish"
            elif first_pred > 0:
                signal = "Neutral"
            else:
                signal = "Weak"

            c1, c2, c3 = st.columns(3)
            c1.metric("Forecast Horizon", "20 Trading Days")
            c2.metric("Predicted Return (First Ticker)", f"{first_pred:.2%}")
            c3.metric("Trend Signal", signal)

            st.markdown("### Forecast Summary Table")
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("### Price Forecast Chart")
            st.pyplot(
                plot_rolling_forecast(
                    prices_assets[tickers_used],
                    pred_base,
                    pred_low,
                    pred_high
                )
            )

            st.markdown("### Interpretation")
            st.write("This forecast is based on recent return momentum over the past 60 trading days.")


        elif page == "Page 4: CAPM":
            # 1. 顶部标题区域
            st.markdown("✅ <span style='color:green; font-weight:bold'>CAPM model loaded</span>", unsafe_allow_html=True)
            st.write("---")

            # 2. 创建容器来控制视觉顺序 (UI顺序：图表 -> 参数假设 -> 表格)
            # 这样可以在代码逻辑上先获取参数进行计算，但在视觉上图表排在前面
            chart_container = st.container(border=True)
            assumptions_container = st.container(border=True)
            table_container = st.container(border=True)

            # 3. 在假设容器中获取参数 (Model Assumptions)
            with assumptions_container:
                st.markdown("#### Model Assumptions")
                st.caption("Set your annual risk-free rate used in CAPM calculation.")
                # 使用列布局让输入框不要占满整行，更精致
                col_input, _ = st.columns([1, 2])
                with col_input:
                    rf_pct = st.number_input(
                        "Annual Risk-Free Rate (Rf) %",
                        min_value=0.0, max_value=20.0, value=4.00, step=0.25, format="%.2f"
                    )
                rf_annual = rf_pct / 100.0

                # 4. 数据计算逻辑
                capm_assets = rets_assets[tickers_used].copy()
                capm_df = compute_capm_table(capm_assets, rets_spy, rf_annual)

            if capm_df.empty:
                st.warning("Not enough data to estimate CAPM (need at least ~30 overlapping daily observations).")
                st.stop()

                # 5. 在图表容器中渲染图表 (CAPM Beta by Ticker)
            with chart_container:
                col_title, col_badge = st.columns([3, 1])
                with col_title:
                    st.markdown("#### CAPM Beta by Ticker ⓘ")
                    st.caption("Beta measures a stock's sensitivity to the overall market (SPY).")
                with col_badge:
                        # 模拟右上角的灰色标签
                    st.markdown(
                        "<div style='background-color:#F3F4F6; color:#4B5563; padding:5px 10px; border-radius:15px; font-size:12px; text-align:center; margin-top:10px;'>" +
                        "Higher Beta = Higher Market Sensitivity" +
                        "</div>",
                        unsafe_allow_html=True
                    )
                # Plotly 柱状图
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=capm_df["Ticker"],
                    y=capm_df["Beta"],
                    text=capm_df["Beta"].apply(lambda x: f"{x:.2f}"),
                    textposition='outside',
                    marker_color='#60A5FA', # 现代清爽蓝
                    width=0.4
                ))

                #添加 Beta = 1 的虚线基准线
                fig.add_hline(
                    y=1.0, line_dash="dash", line_color="#9CA3AF",
                    annotation_text="Market Average (Beta = 1)",
                    annotation_position="top right",
                    annotation_font_color="#6B7280"
                )
                
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

                # 6. 在表格容器中渲染表格 (CAPM Results Table)
            with table_container:
                st.markdown("#### CAPM Results Table") 
                # 定义 Pandas Styler 的上色逻辑
                def style_beta(v): return 'color: #2563EB; font-weight: bold;' # 蓝色
                def style_alpha(v): return 'color: #DC2626; font-weight: bold;' if v < 0 else 'color: #16A34A; font-weight: bold;' # 红/绿
                def style_return(v): return 'color: #16A34A; font-weight: bold;' # 绿色
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

            # 7. 教学解释卡片 (How to Interpret Beta)
            with st.container(border=True):
                st.markdown("#### 💡 Beta Investment Strategy Guide")
                st.write("") 
        
                c1, c2, c3 = st.columns(3)
                # --- Beta = 1 Module ---
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
                # --- Beta > 1 Module ---
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
                # --- Beta < 1 Module ---
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
                    # --- Dynamic Output: Portfolio Diagnostic ---
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


        elif page == "Page 5: Diversification":
            
            st.subheader("Portfolio Risk Explanation")

            if not rc_df.empty:
                top_risk_asset = rc_df.iloc[0]["ticker"]
                top_risk_pct = rc_df.iloc[0]["risk_contribution_pct_of_vol"]

                st.markdown("### Interpretation")
                st.info(
                    f"The portfolio’s largest source of risk is **{top_risk_asset}**, "
                    f"which contributes about **{top_risk_pct:.1f}%** of total portfolio volatility. "
                )
           
            st.markdown("### Correlation Heatmap")
            st.write(
                    "This heatmap shows how closely the selected stocks move together. "
                    "Higher correlation means weaker diversification."
            )
            
            st.pyplot(plot_corr_heatmap(corr))
            st.write("") 
            st.write("")

            st.markdown("### What does this graph tell us?")

            explanation = generate_correlation_explanation(corr)
            st.write(explanation)
            if avg_corr > 0.6:
                st.error("⚠️ Low Diversification")
            elif avg_corr > 0.3:
                st.warning("⚖️ Moderate Diversification")
            else:
                st.success("✅ Strong Diversification")
            # === 2. Risk Contribution 环形图 ===
            st.markdown("### Risk Contribution")
            st.caption("Larger contribution means that stock is driving more risk.")
            st.pyplot(plot_risk_contrib(rc_df))
        
        elif page == "Page 6: Individual Stock Performance Across Key Metrics":
            from datetime import date
            # === Header ===
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

            # === Compute ===
            raw_df, norm_df = compute_radar_metrics(prices_assets[tickers_used], final_weights)

            # === Radar Chart ===
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
            )

            st.plotly_chart(fig, use_container_width=True)

            # === Understanding the Metrics Table ===
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
                <div class="metrics-section-title">📖 Understanding the Metrics</div>
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
                <div class="info-note">
                    ℹ️ All metrics are normalized for comparison. Higher normalized score = better performance.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # === Best / Worst Cards ===
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

           # === What Do These Scores Mean ===
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
                    
    except Exception as e:
        st.error("Error processing inputs")
        st.code(traceback.format_exc())