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

st.markdown("""
<style>
/* Page title spacing */
.block-container {padding-top: 1.5rem;}

/* Card grid */
.metric-grid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-top: 8px;
}
@media(max-width: 980px){
  .metric-grid{grid-template-columns: 1fr;}
}

/* Card */
.card{
  border: 1px solid #EEF0F4;
  border-radius: 16px;
  padding: 18px 18px 14px 18px;
  background: #FFFFFF;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}

/* Card header */
.card-top{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.card-left{
  display:flex;
  align-items:center;
  gap: 12px;
}
.icon-wrap{
  width:44px;height:44px;border-radius: 999px;
  display:flex;align-items:center;justify-content:center;
  background: #F4F7FF;
  border: 1px solid #E7ECFF;
}
.icon{
  font-size:18px; line-height: 1;
}
.card-title{
  font-size: 14px;
  font-weight: 700;
  color:#0F172A;
  margin:0;
}
.card-sub{
  font-size: 12px;
  color:#64748B;
  margin:2px 0 0 0;
}

/* Big value */
.big{
  font-size: 36px;
  font-weight: 800;
  color:#0F172A;
  letter-spacing: -0.5px;
  margin: 8px 0 6px 0;
}

/* Badge */
.badge{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  border: 1px solid transparent;
  margin-left: 10px;
  transform: translateY(-6px);
}
.badge-blue{background:#EAF2FF;color:#1D4ED8;border-color:#D8E6FF;}
.badge-green{background:#E9F9F0;color:#16A34A;border-color:#D2F2DF;}
.badge-purple{background:#F3E8FF;color:#7C3AED;border-color:#E9D5FF;}
.badge-orange{background:#FFF1E6;color:#EA580C;border-color:#FFE2CC;}

/* Divider */
.hr{height:1px;background:#EEF0F4;margin: 10px 0 10px 0;}

/* Description */
.desc{
  font-size: 12.5px;
  color:#475569;
  line-height: 1.5;
  margin: 0 0 10px 0;
}

/* Scale bar */
.scale-wrap{margin-top: 4px;}
.scale{
  position:relative;
  height: 8px;
  background: #EEF2F7;
  border-radius: 999px;
  overflow:hidden;
}
.fill{
  position:absolute;left:0;top:0;bottom:0;
  border-radius: 999px;
}
.dot{
  position:absolute;top:50%;
  width:14px;height:14px;border-radius:999px;
  transform: translate(-50%,-50%);
  border: 3px solid white;
  box-shadow: 0 1px 2px rgba(15,23,42,0.15);
}
.scale-labels{
  display:flex;
  justify-content:space-between;
  font-size: 11px;
  color:#64748B;
  margin-top: 6px;
}
.scale-labels span{white-space:nowrap;}
.small-note{
  font-size: 11px;
  color:#94A3B8;
  margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


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
            st.markdown("## Portfolio Overview")
            st.markdown("Key metrics at a glance to help you quickly understand your portfolio’s risk and return profile.")

            # ---------- helper functions ----------
            def clamp01(x: float) -> float:
                return float(max(0.0, min(1.0, x)))

            def format_pct(x: float) -> str:
                return f"{x*100:.2f}%"

            def risk_badge_from_vol(v: float):
                if v < 0.10:
                    return ("Low", "badge-blue")
                elif v < 0.20:
                    return ("Medium", "badge-purple")
                elif v < 0.30:
                    return ("Elevated", "badge-orange")
                else:
                    return ("Very High", "badge-orange")

            def badge_from_return(r: float):
                if r < 0.05:
                    return ("Low", "badge-blue")
                elif r < 0.15:
                    return ("Moderate", "badge-purple")
                elif r < 0.25:
                    return ("High", "badge-green")
                else:
                    return ("Very High", "badge-green")

            def badge_from_divscore(s: float):
                if s < 40:
                    return ("Low", "badge-orange")
                elif s < 60:
                    return ("Medium", "badge-purple")
                elif s < 80:
                    return ("Good", "badge-green")
                else:
                    return ("Excellent", "badge-green")

            def badge_from_ml(prob: float):
                if prob < 0.20:
                    return ("Low Risk", "badge-green")
                elif prob < 0.50:
                    return ("Medium Risk", "badge-orange")
                elif prob < 0.75:
                    return ("High Risk", "badge-orange")
                else:
                    return ("Very High Risk", "badge-orange")

            def scale_html(value_0to1: float, color: str, labels: list[str]):
                p = clamp01(value_0to1)
                left = f"{p*100:.1f}%"
                return f"""
                <div class="scale-wrap">
                <div class="scale">
                    <div class="fill" style="width:{left}; background:{color}; opacity:0.35;"></div>
                    <div class="dot" style="left:{left}; background:{color};"></div>
                </div>
                <div class="scale-labels">
                    {''.join([f"<span>{lab}</span>" for lab in labels])}
                </div>
                </div>
                """

            def card_html(icon_bg: str, icon: str, title_en: str, subtitle: str,
                        big_value: str, badge_text: str, badge_class: str,
                        desc: str, scale_block: str):
                return f"""
                <div class="card">
                <div class="card-top">
                    <div class="card-left">
                    <div class="icon-wrap" style="background:{icon_bg}; border-color:#EEF0F4;">
                        <div class="icon">{icon}</div>
                    </div>
                    <div>
                        <div class="card-title">{title_en}</div>
                        <div class="card-sub">{subtitle}</div>
                    </div>
                    </div>
                </div>

                <div class="big">{big_value}
                    <span class="badge {badge_class}">{badge_text}</span>
                </div>

                <div class="hr"></div>
                <p class="desc">{desc}</p>
                {scale_block}
                </div>
                """

            # ---------- values ----------
            vol_val = float(ann_vol)
            ret_val = float(exp_return)
            div_val = float(div_score)
            ml_prob = float(pred_prob)

            vol_badge, vol_badge_class = risk_badge_from_vol(vol_val)
            ret_badge, ret_badge_class = badge_from_return(ret_val)
            div_badge, div_badge_class = badge_from_divscore(div_val)
            ml_badge,  ml_badge_class  = badge_from_ml(ml_prob)

            # normalize for scale bars
            vol_norm = clamp01(vol_val / 0.40)                 # treat 40% vol as max
            ret_norm = clamp01((ret_val - 0.00) / 0.40)        # treat 40% return as max
            div_norm = clamp01(div_val / 100.0)
            ml_norm  = clamp01(ml_prob)

            # descriptions
            vol_desc = f"Measures price variability of the portfolio. {format_pct(vol_val)} indicates noticeable fluctuations and potentially larger swings in outcomes."
            ret_desc = f"A simple annualized estimate based on historical returns. {format_pct(ret_val)} suggests higher potential return, typically accompanied by higher risk."
            div_desc = f"Captures how diversified the portfolio is (lower correlation implies better diversification). {div_val:.1f} indicates a moderate diversification level."
            ml_desc  = f"Machine-learning-based risk probability estimate. The model label is “{risk_label}”, with an estimated risk probability of about {ml_prob:.0%}."

            # scale labels
            vol_scale = scale_html(
                vol_norm, "#3B82F6",
                ["Low <10%", "Medium 10–20%", "High 20–30%", "Very High >30%"]
            )
            ret_scale = scale_html(
                ret_norm, "#22C55E",
                ["Low <5%", "Medium 5–15%", "High 15–25%", "Very High >25%"]
            )
            div_scale = scale_html(
                div_norm, "#A855F7",
                ["Low <40", "Medium 40–60", "Good 60–80", "Excellent >80"]
            )
            ml_scale = scale_html(
                ml_norm, "#F59E0B",
                ["Low <20%", "Medium 20–50%", "High 50–75%", "Very High >75%"]
            )

            # ---------- render grid ----------
            grid_html = f"""
            <div class="metric-grid">
            {card_html("#EAF2FF","📈","Current Annualized Volatility","Volatility (annualized)",
                        format_pct(vol_val), vol_badge, vol_badge_class, vol_desc, vol_scale)}

            {card_html("#E9F9F0","📊","Expected Annual Return","Return (annualized)",
                        format_pct(ret_val), ret_badge, ret_badge_class, ret_desc, ret_scale)}

            {card_html("#F3E8FF","🧩","Diversification Score","Correlation-based diversification",
                        f"{div_val:.1f}", div_badge, div_badge_class, div_desc, div_scale)}

            {card_html("#FFF1E6","🛡️","ML Risk Prediction","Model-based risk probability",
                        f"{risk_label} ({ml_prob:.0%})", ml_badge, ml_badge_class, ml_desc, ml_scale)}
            </div>
            <div class="small-note">Note: These metrics are for informational purposes only and do not constitute investment advice.</div>
            """
            st.markdown(grid_html, unsafe_allow_html=True)


        elif page == "Page 2: History":
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