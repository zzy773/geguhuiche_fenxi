import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- 环境加固 ---
os.environ['NO_PROXY'] = '*' # 强制跳过所有代理干扰
st.set_page_config(page_title="爆发增强策略 Pro 交互回测", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.title("🚀 爆发增强策略 Pro - 自动化交互回测系统")
st.markdown("逻辑对齐：**MA7趋势**、**Q2动能**、**3%乖离限制**及**信号日低点止损**。")

# --- 侧边栏交互输入 ---
st.sidebar.header("回测配置")
stock_code = st.sidebar.text_input("个股代码 (如 001255)", value="001255").strip()
start_date = st.sidebar.date_input("起始日期", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime("2026-02-24"))
init_cash = st.sidebar.number_input("初始资金 (元)", value=100000)

@st.cache_data(ttl=3600)
def fetch_data_with_retry(code, start, end):
    """带伪装和自动重试的数据抓取，解决 RemoteDisconnected"""
    s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    for i in range(3): # 尝试 3 次抓取
        try:
            # 增加随机延迟模拟真人行为
            time.sleep(1)
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
            if df is not None and not df.empty:
                # 动态列名映射
                mapping = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '涨跌幅': 'pct_chg'}
                df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
                df['date'] = pd.to_datetime(df['date'])
                
                # 同步大盘指数
                idx = ak.stock_zh_index_daily(symbol="sh000001")
                idx['date'] = pd.to_datetime(idx['date'])
                df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
                return df
        except Exception as e:
            if i == 2: st.error(f"数据抓取最终失败: {e}")
    return None

# --- 执行回测流程 ---
if st.sidebar.button("启动逻辑回测"):
    df = fetch_data_with_retry(stock_code, start_date, end_date)
    
    if df is not None:
        # 1. 指标计算 (严格复刻通达信 11436 逻辑)
        df['ma7'] = df['close'].rolling(7).mean()
        df['idx_ma5'] = df['idx_c'].rolling(5).mean()
        
        # Q2 动能复刻：$Q_2 = 100 \times \frac{EMA(EMA(Q_1, 6), 6)}{EMA(EMA(|Q_1|, 6), 6)}$
        q1 = df['close'].diff()
        q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        df['q2'] = 100 * q_ema / q_abs_ema
        
        # 2. 信号判定 (XG)
        # 条件：大盘安全、30日内有涨停、Q2抬头、MA7趋势向上、乖离率 <= 3%
        df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                   (df['pct_chg'].rolling(30).max() > 9.5) & \
                   (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                   (df['ma7'] > df['ma7'].shift(1)) & \
                   (df['close'] > df['high'].shift(1)) & \
                   ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)

        # 3. 交易引擎：计算区间收益与止损
        cash, shares, stop_low = float(init_cash), 0, 0
        history, trades = [], []
        buy_date, buy_price = None, 0

        for i in range(len(df)):
            r = df.iloc[i]
            # 止损逻辑：收盘破 MA7 或 信号日最低价
            if shares > 0:
                if r['close'] < stop_low or r['close'] < r['ma7']:
                    sell_price = r['close']
                    ret = (sell_price - buy_price) / buy_price * 100
                    cash = shares * sell_price
                    trades.append({
                        "买入日期": buy_date.date(), "卖出日期": r['date'].date(),
                        "买入价格": f"{buy_price:.2f}", "卖出价格": f"{sell_price:.2f}",
                        "区间收益率": f"{ret:.2f}%"
                    })
                    shares = 0
            
            # 买入逻辑
            if r['xg'] and shares == 0:
                buy_date, buy_price = r['date'], r['close']
                shares = cash / buy_price
                cash = 0
                stop_low = r['low'] # 记录 11442.jpg 中的止损底线
            history.append(cash + shares * r['close'])

        df['balance'] = history
        
        # --- 结果展示 ---
        st.subheader("📊 策略收益汇总")
        final_v = df['balance'].iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("期末总资产", f"{final_v:.2f} 元")
        c2.metric("累积收益率", f"{(final_v - init_cash)/init_cash*100:.2f}%")
        c3.metric("爆发信号次数", len(df[df['xg']]))

        # 可视化图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(df['date'], df['close'], label='收盘价', alpha=0.5)
        ax1.plot(df['date'], df['ma7'], label='MA7趋势线', color='cyan')
        ax1.scatter(df[df['xg']]['date'], df[df['xg']]['close'], color='red', marker='^', s=100, label='爆发信号')
        ax1.set_title(f"{stock_code} 爆发点复盘图")
        ax1.legend()
        
        ax2.plot(df['date'], df['balance'], label='资产净值', color='orange')
        ax2.axhline(init_cash, color='black', linestyle='--')
        ax2.set_title("账户资产增长曲线")
        st.pyplot(fig)
        
        if trades:
            st.subheader("📋 详细区间交易收益表")
            st.dataframe(pd.DataFrame(trades), use_container_width=True)
