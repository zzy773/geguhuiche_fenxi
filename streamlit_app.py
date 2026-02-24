import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os

# --- 1. 底层环境强制补丁 ---
os.environ['NO_PROXY'] = '*' # 强制跳过代理
st.set_page_config(page_title="爆发增强策略 Pro", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.title("🛡️ 爆发增强策略 Pro - 自动化回测系统")
st.caption("策略逻辑：MA7 趋势、Q2 动能、信号日最低点止损。")

# --- 2. 侧边栏交互设置 ---
st.sidebar.header("回测参数")
code = st.sidebar.text_input("个股代码 (如 001255)", "001255").strip()
start_date = st.sidebar.date_input("回测起始", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("回测结束", pd.to_datetime("2026-02-24"))
init_fund = st.sidebar.number_input("初始资金", 100000)

@st.cache_data(ttl=600)
def fetch_data_safe(symbol, start, end):
    """三级抗封锁抓取逻辑，解决 RemoteDisconnected"""
    s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    for i in range(5): # 暴力重试 5 次
        try:
            # 随机休眠 2-4 秒，模仿真人点击避开封锁
            time.sleep(random.uniform(2, 4))
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
            if df is not None and not df.empty:
                # 动态映射列名，解决 Length mismatch
                name_map = {'日期':'date','收盘':'close','最高':'high','最低':'low','涨跌幅':'pct_chg'}
                df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
                df['date'] = pd.to_datetime(df['date'])
                
                # 同步大盘数据做环境过滤
                idx = ak.stock_zh_index_daily(symbol="sh000001")
                idx['date'] = pd.to_datetime(idx['date'])
                df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
                return df
        except Exception as e:
            if i == 4: st.error(f"连接服务器失败: {e}. 请尝试更换个股或稍后再试。")
    return None

# --- 3. 运行回测 ---
if st.sidebar.button("启动严谨回测"):
    with st.spinner("系统正在穿透数据拦截..."):
        df = fetch_data_safe(code, start_date, end_date)
        
        if df is not None:
            # 指标计算：MA7 与 Q2 动能
            df['ma7'] = df['close'].rolling(7).mean()
            df['idx_ma5'] = df['idx_c'].rolling(5).mean()
            
            # Q2 动能复刻
            q1 = df['close'].diff()
            q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            df['q2'] = 100 * q_ema / q_abs_ema
            
            # 信号判定 XG
            df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                       (df['pct_chg'].rolling(30).max() > 9.5) & \
                       (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                       (df['ma7'] > df['ma7'].shift(1)) & \
                       (df['close'] > df['high'].shift(1)) & \
                       ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)

            # 模拟交易引擎
            cash, shares, stop_low = float(init_fund), 0, 0
            history, trade_logs = [], []
            b_date, b_price = None, 0

            for i in range(len(df)):
                r = df.iloc[i]
                # 止损离场逻辑
                if shares > 0:
                    if r['close'] < stop_low or r['close'] < r['ma7']:
                        ret = (r['close'] - b_price) / b_price * 100
                        cash = shares * r['close']
                        trade_logs.append({
                            "买入时间": b_date.date(), "卖出时间": r['date'].date(),
                            "区间收益": f"{ret:.2f}%", "账户余额": f"{cash:.2f}"
                        })
                        shares = 0
                # 进场逻辑
                if r['xg'] and shares == 0:
                    b_date, b_price = r['date'], r['close']
                    shares = cash / b_price
                    cash = 0
                    stop_low = r['low'] # 锁定信号日低点止损
                history.append(cash + shares * r['close'])

            df['account'] = history
            
            # --- 4. 统计分析与绘图 ---
            st.subheader("📊 策略回测绩效总览")
            c1, c2, c3 = st.columns(3)
            final_v = df['account'].iloc[-1]
            c1.metric("最终净值", f"{final_v:.2f}")
            c2.metric("累计回报", f"{(final_v - init_fund)/init_fund*100:.2f}%")
            c3.metric("有效爆发信号", len(df[df['xg']]))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['date'], df['account'], color='orange', label='账户资产')
            ax.axhline(init_fund, color='red', linestyle='--')
            ax.set_title("账户资产累积增长曲线")
            st.pyplot(fig)

            if trade_logs:
                st.subheader("📋 区间交易详细明细表")
                st.dataframe(pd.DataFrame(trade_logs), use_container_width=True)
            else:
                st.info("该时段内未触发买入信号。")
