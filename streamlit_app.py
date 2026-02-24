import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- 1. 深度环境优化 (防封 IP) ---
os.environ['NO_PROXY'] = '*' # 强制跳过代理
st.set_page_config(page_title="爆发增强策略 Pro", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.title("🚀 爆发增强策略 Pro - 自动化交互回测系统")
st.caption("策略严谨对齐：MA7 趋势、Q2 动量抬头 及信号日最低价止损逻辑。")

# --- 2. 交互式侧边栏 ---
st.sidebar.header("回测配置")
stock_code = st.sidebar.text_input("个股代码 (如 001255)", value="001255").strip()
start_date = st.sidebar.date_input("起始日期", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime("2026-02-24"))
init_cash = st.sidebar.number_input("初始资金 (元)", value=100000)

@st.cache_data(ttl=3600)
def fetch_data_safe(code, start, end):
    """具备抗封锁和列名自动适配的数据抓取函数"""
    s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    # 模拟浏览器行为，减少 RemoteDisconnected 报错
    for i in range(3): 
        try:
            time.sleep(1.5) # 强制间隔，防止请求过快
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
            if df is not None and not df.empty:
                # 动态映射列名
                mapping = {'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','涨跌幅':'pct_chg'}
                df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
                df['date'] = pd.to_datetime(df['date'])
                
                # 同步大盘数据做过滤
                idx = ak.stock_zh_index_daily(symbol="sh000001")
                idx['date'] = pd.to_datetime(idx['date'])
                df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
                return df
        except Exception as e:
            if i == 2: st.error(f"数据抓取失败: {e}。建议检查代码或稍后重试。")
    return None

# --- 3. 核心计算与回测逻辑 ---
if st.sidebar.button("启动严谨回测"):
    with st.spinner("正在计算区间收益..."):
        df = fetch_data_safe(stock_code, start_date, end_date)
        
        if df is not None:
            # 指标计算
            df['ma7'] = df['close'].rolling(7).mean()
            df['idx_ma5'] = df['idx_c'].rolling(5).mean()
            
            # 修正后的 Q2 动能
            q1 = df['close'].diff()
            q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            df['q2'] = 100 * q_ema / q_abs_ema
            
            # 信号 XG
            df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                       (df['pct_chg'].rolling(30).max() > 9.5) & \
                       (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                       (df['ma7'] > df['ma7'].shift(1)) & \
                       (df['close'] > df['high'].shift(1)) & \
                       ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)

            # 交易模拟
            cash, shares, stop_low = float(init_cash), 0, 0
            history, trade_logs = [], []
            b_date, b_price = None, 0

            for i in range(len(df)):
                r = df.iloc[i]
                # 止损/卖出判断：破 MA7 或 破信号日最低价
                if shares > 0:
                    if r['close'] < stop_low or r['close'] < r['ma7']:
                        sell_price = r['close']
                        ret = (sell_price - b_price) / b_price * 100
                        cash = shares * sell_price
                        trade_logs.append({
                            "买入日期": b_date.date(), "卖出日期": r['date'].date(),
                            "买入价": f"{b_price:.2f}", "卖出价": f"{sell_price:.2f}",
                            "收益率": f"{ret:.2f}%"
                        })
                        shares = 0
                
                # 买入逻辑
                if r['xg'] and shares == 0:
                    b_date, b_price = r['date'], r['close']
                    shares = cash / b_price
                    cash = 0
                    stop_low = r['low'] # 锁定止损底线
                history.append(cash + shares * r['close'])

            df['balance'] = history
            
            # --- 4. 结果展示 ---
            final_val = df['balance'].iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("期末总资产", f"{final_val:.2f} 元")
            c2.metric("累积收益率", f"{(final_val - init_cash)/init_cash*100:.2f}%")
            c3.metric("爆发信号次数", len(df[df['xg']]))

            # 收益曲线图
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['date'], df['balance'], color='orange', label='资产增长')
            ax.axhline(init_cash, color='red', linestyle='--', label='初始资金')
            ax.set_title("策略资产累积曲线")
            ax.legend()
            st.pyplot(fig)

            # 区间收益表
            if trade_logs:
                st.subheader("📋 详细区间交易收益清单")
                st.dataframe(pd.DataFrame(trade_logs), use_container_width=True)
