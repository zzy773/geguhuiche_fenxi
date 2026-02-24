import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 禁用代理，防止云端请求干扰
os.environ['NO_PROXY'] = '*'

st.set_page_config(page_title="爆发增强策略 Pro", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.title("📈 爆发增强策略 Pro 回测系统 (云端稳定版)")
st.caption("逻辑严谨对齐：MA7 趋势、Q2 动能 及双重止损机制。")

# --- 侧边栏配置 ---
st.sidebar.header("回测配置")
symbol = st.sidebar.text_input("个股代码 (如 001255)", value="001255")
start_val = st.sidebar.date_input("开始日期", value=pd.to_datetime("2024-01-01"))
end_val = st.sidebar.date_input("结束日期", value=pd.to_datetime("2026-02-24"))

@st.cache_data(ttl=3600)
def fetch_stock_data(code, start, end):
    """带动态列名映射的数据抓取函数"""
    try:
        s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
        
        if df is None or df.empty: return None

        # 动态探测列名，免疫 API 变更导致的长度不匹配
        mapping = {'日期': 'date', '收盘': 'close', '最高': 'high', '最低': 'low', '涨跌幅': 'pct_chg'}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        
        # 补齐涨跌幅，防止 KeyError
        if 'pct_chg' not in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
            
        df['date'] = pd.to_datetime(df['date'])
        
        # 同步大盘指数
        idx = ak.stock_zh_index_daily(symbol="sh000001")
        idx['date'] = pd.to_datetime(idx['date'])
        df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
        return df
    except Exception as e:
        st.error(f"数据获取异常: {e}")
        return None

if st.sidebar.button("启动严谨回测"):
    df = fetch_stock_data(symbol, start_val, end_val)
    
    if df is not None:
        # --- 核心指标计算 (严格复刻通达信 11436 逻辑) ---
        df['ma7'] = df['close'].rolling(7).mean()
        df['idx_ma5'] = df['idx_c'].rolling(5).mean()
        
        # 复刻 Q2 动能：修正了 ABS 函数的嵌套逻辑
        q1 = df['close'].diff()
        q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        df['q2'] = 100 * q_ema / q_abs_ema
        
        # --- 爆发点判定 (XG) ---
        # 条件：大盘多头、30日内有过涨停、Q2抬头且>-20、MA7斜率向上、乖离率 <= 3%
        df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                   (df['pct_chg'].rolling(30).max() > 9.5) & \
                   (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                   (df['ma7'] > df['ma7'].shift(1)) & \
                   (df['close'] > df['high'].shift(1)) & \
                   ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)

        # --- 模拟交易 (双重止损策略) ---
        cash, shares, stop_low = 100000.0, 0, 0
        history, trades = [], []

        for i in range(len(df)):
            r = df.iloc[i]
            # 离场：收盘破 MA7 或 破信号日最低点
            if shares > 0:
                if r['close'] < stop_low or r['close'] < r['ma7']:
                    cash = shares * r['close']
                    shares = 0
                    trades.append({"日期": r['date'].date(), "动作": "卖出/止损", "价格": r['close']})
            # 进场：触发信号且空仓
            if r['xg'] and shares == 0:
                shares = cash / r['close']
                cash = 0
                stop_low = r['low'] # 锁定 11442.jpg 中的绿色止损底线
                trades.append({"日期": r['date'].date(), "动作": "买入/信号", "价格": r['close']})
            history.append(cash + shares * r['close'])

        df['balance'] = history
        
        # --- 可视化 ---
        st.subheader(f"📊 回测统计 - {symbol}")
        c1, c2, c3 = st.columns(3)
        final_v = df['balance'].iloc[-1]
        c1.metric("期末总资产", f"{final_v:.2f}")
        c2.metric("累计盈亏", f"{(final_v-100000)/1000:.2f}%")
        c3.metric("总信号数", len(df[df['xg']]))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(df['date'], df['close'], label='收盘价', alpha=0.5)
        ax1.plot(df['date'], df['ma7'], label='MA7 趋势防线', color='cyan')
        ax1.scatter(df[df['xg']]['date'], df[df['xg']]['close'], color='red', marker='^', s=100, label='★爆发点')
        ax1.set_title("信号点复盘分布")
        ax1.legend()

        ax2.plot(df['date'], df['balance'], label='账户净值', color='orange')
        ax2.axhline(100000, color='black', linestyle='--')
        st.pyplot(fig)
        
        if trades: st.subheader("交易日志"), st.table(pd.DataFrame(trades))
    else:
        st.error("数据加载失败，请检查代码或网络环境。")
