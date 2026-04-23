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

def plot_rolling_vol(rolling_vol):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(rolling_vol.index, rolling_vol.values)
    plt.title("20-Day Rolling Volatility")
    plt.tight_layout()
    return fig


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
    ["Page 1: Risk", "Page 2: History", "Page 3: Diversification"],
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

            # --- 2. 动态计算卡片所需的展示数据 ---
            v_val = ann_vol * 100
            v_pct = min(v_val / 40 * 100, 100) # 刻度条百分比位置
            v_badge = "较高" if v_val > 20 else "偏低" if v_val < 10 else "中等"
            
            r_val = exp_return * 100
            r_pct = min(max(r_val, 0) / 35 * 100, 100)
            r_badge = "较高" if r_val > 15 else "一般" if r_val > 5 else "偏低"
            
            d_val = div_score
            d_pct = min(d_val, 100)
            d_badge = "优秀" if d_val > 80 else "良好" if d_val > 60 else "中等" if d_val > 40 else "较低"
            
            m_val = pred_prob * 100
            m_pct = min(m_val, 100)
            m_badge = "高风险" if risk_label == "High" else "中风险" if risk_label == "Medium" else "低风险"

            # --- 3. 核心 CSS 样式 ---
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

            # --- 4. 生成 HTML 结构 ---
            # 注意：下面的 HTML 必须全部顶格写，绝对不能有开头的空格缩进，否则又会变成代码块！
            html_content = f"""
<div class="overview-title">Portfolio Overview</div>
<div class="overview-sub">关键指标一览，帮助您快速了解投资组合的风险与收益特征</div>

<div class="metric-grid">
<div class="metric-card">
<div class="card-header">
<div class="icon-box" style="background: #eff6ff; color: #3b82f6;">📉</div>
<div>
<div class="card-title-en">Current Annualized Volatility</div>
<div class="card-title-cn">当前年化波动率</div>
</div>
</div>
<div class="value-row">
<div class="value-text">{v_val:.2f}%</div>
<div class="badge" style="background: #eff6ff; color: #2563eb;">{v_badge}</div>
</div>
<div class="desc-text">衡量投资组合的价格波动程度。{v_val:.2f}% 表明您的投资组合波动{("较为明显，收益可能有较大起伏" if v_val > 20 else "在合理范围内")}，适合相应风险承受能力的投资者。</div>
<div class="scale-container">
<div class="scale-track">
<div class="scale-fill" style="width: {v_pct}%; background: #3b82f6;"></div>
<div class="scale-indicator" style="left: {v_pct}%; background: #3b82f6;"></div>
</div>
<div class="scale-labels"><span>低<br>&lt;10%</span><span>中<br>10-20%</span><span>高<br>20-30%</span><span>极高<br>&gt;30%</span></div>
</div>
</div>

<div class="metric-card">
<div class="card-header">
<div class="icon-box" style="background: #ecfdf5; color: #10b981;">📈</div>
<div>
<div class="card-title-en">Expected Annual Return</div>
<div class="card-title-cn">预期年化收益率</div>
</div>
</div>
<div class="value-row">
<div class="value-text">{r_val:.2f}%</div>
<div class="badge" style="background: #ecfdf5; color: #059669;">{r_badge}</div>
</div>
<div class="desc-text">基于历史数据与市场模型预测的潜在年化收益。{r_val:.2f}% 表明预期收益{r_badge}，意味着{("更高的潜在回报，但也伴随风险" if r_val > 15 else "较为稳健的回报预期")}。</div>
<div class="scale-container">
<div class="scale-track">
<div class="scale-fill" style="width: {r_pct}%; background: #10b981;"></div>
<div class="scale-indicator" style="left: {r_pct}%; background: #10b981;"></div>
</div>
<div class="scale-labels"><span>低<br>&lt;5%</span><span>中<br>5-15%</span><span>较高<br>15-25%</span><span>高<br>&gt;25%</span></div>
</div>
</div>

<div class="metric-card">
<div class="card-header">
<div class="icon-box" style="background: #f5f3ff; color: #8b5cf6;">🎯</div>
<div>
<div class="card-title-en">Diversification Score</div>
<div class="card-title-cn">多元化评分</div>
</div>
</div>
<div class="value-row">
<div class="value-text">{d_val:.1f}</div>
<div class="badge" style="background: #f5f3ff; color: #7c3aed;">{d_badge}</div>
</div>
<div class="desc-text">反映资产在不同类别的分散程度。{d_val:.1f} 表明多元化程度{d_badge}，{("仍有优化空间，进一步分散可降低风险" if d_val < 60 else "风险分散较好，非系统性风险较低")}。</div>
<div class="scale-container">
<div class="scale-track">
<div class="scale-fill" style="width: {d_pct}%; background: #8b5cf6;"></div>
<div class="scale-indicator" style="left: {d_pct}%; background: #8b5cf6;"></div>
</div>
<div class="scale-labels"><span>低<br>&lt;40</span><span>中<br>40-60</span><span>良好<br>60-80</span><span>优秀<br>&gt;80</span></div>
</div>
</div>

<div class="metric-card">
<div class="card-header">
<div class="icon-box" style="background: #fffbeb; color: #f59e0b;">🛡️</div>
<div>
<div class="card-title-en">ML Risk Prediction</div>
<div class="card-title-cn">机器学习风险预测</div>
</div>
</div>
<div class="value-row">
<div class="value-text">{risk_label} ({m_val:.0f}%)</div>
<div class="badge" style="background: #fffbeb; color: #d97706;">{m_badge}</div>
</div>
<div class="desc-text">基于机器学习模型对未来风险的预测。结果为“{m_badge}”，发生较大回撤的概率为 {m_val:.0f}%，建议{("重点警惕并立刻减少敞口" if risk_label == "High" else "保持关注并适当控制风险敞口")}。</div>
<div class="scale-container">
<div class="scale-track">
<div class="scale-fill" style="width: {m_pct}%; background: #f59e0b;"></div>
<div class="scale-indicator" style="left: {m_pct}%; background: #f59e0b;"></div>
</div>
<div class="scale-labels"><span>低<br>&lt;20%</span><span>中<br>20-50%</span><span>高<br>50-75%</span><span>极高<br>&gt;75%</span></div>
</div>
</div>
</div>
<div class="footer-note">提示：以上指标仅供参考，不构成投资建议。投资有风险，入市需谨慎。</div>
"""
            st.markdown(html_content, unsafe_allow_html=True)
            
        elif page == "Page 2: History":
       
            # === 修复重点 2：这五行必须保持统一的缩进深度 ===
            st.subheader("Historical Portfolio Performance")
            st.pyplot(plot_cumulative(cum))

            rolling_vol = portfolio_return.rolling(20).std() * np.sqrt(TRADING_DAYS)
            st.markdown("### 20-Day Rolling Volatility")
            st.pyplot(plot_rolling_vol(rolling_vol))


        elif page == "Page 3: Diversification":
            
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

    except Exception as e:
     st.error("Error processing inputs")
    st.code(traceback.format_exc())