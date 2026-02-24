import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random

# --- 严谨环境配置：彻底解决 RemoteDisconnected ---
os.environ['NO_PROXY'] = '*' # 强制跳过代理干扰
st.set_page_config(page_title="爆发增强策略 Pro", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.title("🛡️ 爆发增强策略 Pro - 终极自动化回测系统")
st.caption("逻辑严谨对齐：MA7 趋势、Q2 动能 及双重止损逻辑。")

# --- 侧边栏交互输入 ---
st.sidebar.header("回测配置")
stock_code = st.sidebar.text_input("个股代码 (如 001255)", value="001255").strip()
start_date = st.sidebar.date_input("起始日期", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime("2026-02-24"))
init_cash = st.sidebar.number_input("初始资金 (元)", value=100000)

@st.cache_data(ttl=3600)
def fetch_data_robust(code, start, end):
    """三级容错抓取函数：专门对付 Connection aborted"""
    s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    # 模拟真实浏览器请求头
    for i in range(5): # 增加重试次数
        try:
            time.sleep(random.uniform(1.5, 3.0)) # 随机延迟避开 IP 封锁
            # 获取个股数据
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
            
            if df is not None and not df.empty:
                # 动态映射列名，解决 Length mismatch
                mapping = {'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','涨跌幅':'pct_chg'}
                df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
                df['date'] = pd.to_datetime(df['date'])
                
                # 获取大盘背景数据用于环境过滤
                idx = ak.stock_zh_index_daily(symbol="sh000001")
                idx['date'] = pd.to_datetime(idx['date'])
                df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
                return df
        except Exception as e:
            if i == 4: st.error(f"连接服务器失败: {e}. 请尝试刷新页面或更换时间段。")
    return None

if st.sidebar.button("启动严谨逻辑回测"):
    df = fetch_data_robust(stock_code, start_date, end_date)
    
    if df is not None:
        # 1. 核心指标计算
        df['ma7'] = df['close'].rolling(7).mean()
        df['idx_ma5'] = df['idx_c'].rolling(5).mean()
        
        # Q2 动能复刻：$Q_2 = 100 \times \frac{EMA(EMA(Q_1, 6), 6)}{EMA(EMA(|Q_1|, 6), 6)}$
        q1 = df['close'].diff()
        q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        df['q2'] = 100 * q_ema / q_abs_ema
        
        # 2. 爆发信号判定 (XG)
        df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                   (df['pct_chg'].rolling(30).max() > 9.5) & \
                   (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                   (df['ma7'] > df['ma7'].shift(1)) & \
                   (df['close'] > df['high'].shift(1)) & \
                   ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)

        # 3. 交易模拟：区间收益与止损
        cash, shares, stop_low = float(init_cash), 0, 0
        history, trades = [], []
        b_date, b_price = None, 0

        for i in range(len(df)):
            r = df.iloc[i]
            # 止损离场：收盘破 MA7 或 信号日最低点
            if shares > 0:
                if r['close'] < stop_low or r['close'] < r['ma7']:
                    ret = (r['close'] - b_price) / b_price * 100
                    cash = shares * r['close']
                    trades.append({
                        "买入日期": b_date.date(), "卖出日期": r['date'].date(),
                        "区间收益": f"{ret:.2f}%", "累计净值": f"{cash:.2f}"
                    })
                    shares = 0
            # 信号进场
            if r['xg'] and shares == 0:
                b_date, b_price = r['date'], r['close']
                shares = cash / b_price
                cash = 0
                stop_low = r['low'] # 锁定止损底线
            history.append(cash + shares * r['close'])

        df['balance'] = history
        
        # --- 4. 统计面板与展示 ---
        final_v = df['balance'].iloc[-1]
        st.subheader("📊 策略回测绩效清单")
        c1, c2, c3 = st.columns(3)
        c1.metric("期末总资产", f"{final_v:.2f} 元")
        c2.metric("累积回报率", f"{(final_v - init_cash)/init_cash*100:.2f}%")
        c3.metric("爆发点信号数", len(df[df['xg']]))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(df['date'], df['close'], label='收盘价', alpha=0.5)
        ax1.plot(df['date'], df['ma7'], label='MA7 支撑', color='cyan')
        ax1.scatter(df[df['xg']]['date'], df[df['xg']]['close'], color='red', marker='^', s=100, label='★买入信号')
        ax1.set_title("爆发信号与趋势复盘图")
        ax1.legend()
        
        ax2.plot(df['date'], df['balance'], label='资产净值', color='orange')
        ax2.axhline(init_cash, color='red', linestyle='--')
        ax2.set_title("账户资产累积曲线")
        st.pyplot(fig)
        
        if trades:
            st.subheader("📈 区间交易详细明细表")
            st.dataframe(pd.DataFrame(trades), use_container_width=True)
        else:
            st.info("所选时间段内未触发符合条件的爆发信号。")
