import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random

# --- 严谨环境加固 ---
os.environ['NO_PROXY'] = '*' # 强制绕过代理干扰
st.set_page_config(page_title="爆发增强策略交互系统 Pro", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.title("🛡️ 爆发增强策略 Pro - 云端自动化回测系统")
st.markdown("该系统已针对 **RemoteDisconnected** 及 **Length mismatch** 错误进行了底层加固。")

# --- 侧边栏交互 ---
st.sidebar.header("回测核心配置")
stock_code = st.sidebar.text_input("输入 A 股代码 (如 001255)", value="001255").strip()
start_date = st.sidebar.date_input("回测起始日期", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("回测结束日期", value=pd.to_datetime("2026-02-24"))
init_cash = st.sidebar.number_input("初始模拟资金 (元)", value=100000)

@st.cache_data(ttl=3600)
def fetch_data_ultimate(code, start, end):
    """带呼吸机制的数据抓取，专门对付 RemoteDisconnected"""
    s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    for attempt in range(5): # 增加到 5 次尝试
        try:
            # 随机休眠 1-3 秒，模拟真人操作
            time.sleep(random.uniform(1, 3))
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
            
            if df is not None and not df.empty:
                # 动态映射列名，解决 Length mismatch
                mapping = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '涨跌幅': 'pct_chg'}
                df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
                
                # 补全可能缺失的涨跌幅
                if 'pct_chg' not in df.columns:
                    df['pct_chg'] = df['close'].pct_change() * 100
                    
                df['date'] = pd.to_datetime(df['date'])
                
                # 获取大盘背景数据
                idx = ak.stock_zh_index_daily(symbol="sh000001")
                idx['date'] = pd.to_datetime(idx['date'])
                df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
                return df
        except Exception as e:
            if attempt == 4: st.error(f"连接服务器失败: {e}. 这通常是由于接口封锁 IP，请稍后刷新重试。")
    return None

if st.sidebar.button("启动严谨逻辑回测"):
    with st.spinner("系统正在穿透数据迷雾..."):
        df = fetch_data_ultimate(stock_code, start_date, end_date)
        
        if df is not None:
            # 1. 核心算法复刻 (严格对齐 11436.jpg)
            df['ma7'] = df['close'].rolling(7).mean()
            df['idx_ma5'] = df['idx_c'].rolling(5).mean()
            
            # 修正后的 Q2 动能算法
            # $$Q_2 = 100 \times \frac{EMA(EMA(Q_1, 6), 6)}{EMA(EMA(|Q_1|, 6), 6)}$$
            q1 = df['close'].diff()
            q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            df['q2'] = 100 * q_ema / q_abs_ema
            
            # 2. 爆发信号 (XG) 判定
            df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                       (df['pct_chg'].rolling(30).max() > 9.5) & \
                       (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                       (df['ma7'] > df['ma7'].shift(1)) & \
                       (df['close'] > df['high'].shift(1)) & \
                       ((df['close'] - df['ma7'])/df['ma7']*100 <= 3) # 乖离限制

            # 3. 交易模拟引擎：区间收益与止损
            cash, shares, stop_low = float(init_cash), 0, 0
            history, trade_logs = [], []
            b_date, b_price = None, 0

            for i in range(len(df)):
                r = df.iloc[i]
                # 止损逻辑：收盘破 MA7 或 信号日最低点
                if shares > 0:
                    if r['close'] < stop_low or r['close'] < r['ma7']:
                        sell_price = r['close']
                        ret = (sell_price - b_price) / b_price * 100
                        cash = shares * sell_price
                        trade_logs.append({
                            "买入日期": b_date.date(), "卖出日期": r['date'].date(),
                            "买入价格": f"{b_price:.2f}", "卖出价格": f"{sell_price:.2f}",
                            "区间净收益": f"{ret:.2f}%"
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
            
            # 4. 交互展示
            final_v = df['balance'].iloc[-1]
            st.subheader("📋 回测绩效快报")
            c1, c2, c3 = st.columns(3)
            c1.metric("期末模拟总额", f"{final_v:.2f} 元")
            c2.metric("累积回报率", f"{(final_v - init_cash)/init_cash*100:.2f}%")
            c3.metric("有效爆发信号", len(df[df['xg']]))

            # 图表复盘
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['date'], df['balance'], color='orange', label='账户资产曲线')
            ax.axhline(init_cash, color='red', linestyle='--', label='初始本金')
            ax.set_title("策略资产累积增长曲线")
            ax.legend()
            st.pyplot(fig)
            
            if trade_logs:
                st.subheader("📈 交易区间收益详情")
                st.dataframe(pd.DataFrame(trade_records), use_container_width=True)
            else:
                st.info("在该时间段内未触发任何交易信号，系统已进入空仓观望模式。")
