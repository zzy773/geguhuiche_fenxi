import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random

# --- 严谨环境加固：彻底解决 RemoteDisconnected 与 Altair 报错 ---
os.environ['NO_PROXY'] = '*' # 强制绕过代理干扰
st.set_page_config(page_title="爆发增强策略交互回测 Pro", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.title("🛡️ 爆发增强策略 Pro - 自动化交互回测系统")
st.markdown("该系统针对 **RemoteDisconnected** 及 **Length mismatch** 进行了底层伪装加固。")

# --- 侧边栏交互输入 ---
st.sidebar.header("回测配置")
stock_code = st.sidebar.text_input("输入 A 股代码 (如 001255)", value="001255").strip()
start_date = st.sidebar.date_input("起始日期", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime("2026-02-24"))
init_cash = st.sidebar.number_input("初始模拟资金 (元)", value=100000)

@st.cache_data(ttl=300)
def fetch_data_robust(code, start, end):
    """解决连接断开和字段缺失的严谨抓取函数"""
    s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    for attempt in range(5):
        try:
            # 随机休眠 2-4 秒，模拟真人操作避开封锁
            time.sleep(random.uniform(2, 4)) 
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
            
            if df is not None and not df.empty:
                # 动态映射列名，解决 Length mismatch
                name_map = {'日期': 'date', '收盘': 'close', '最高': 'high', '最低': 'low', '涨跌幅': 'pct_chg'}
                df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
                
                # 补齐可能缺失的涨跌幅
                if 'pct_chg' not in df.columns:
                    df['pct_chg'] = df['close'].pct_change() * 100
                df['date'] = pd.to_datetime(df['date'])
                
                # 同步上证指数环境过滤
                idx = ak.stock_zh_index_daily(symbol="sh000001")
                idx['date'] = pd.to_datetime(idx['date'])
                df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
                return df
        except Exception as e:
            if attempt == 4: st.error(f"连接服务器失败: {e}. 请点击侧边栏按钮重新尝试。")
    return None

if st.sidebar.button("启动严谨逻辑回测"):
    with st.spinner("系统正在穿透数据拦截..."):
        df = fetch_data_robust(stock_code, start_date, end_date)
        
        if df is not None:
            # 1. 计算核心指标 (严格复刻 11436.jpg 逻辑)
            df['ma7'] = df['close'].rolling(7).mean()
            df['idx_ma5'] = df['idx_c'].rolling(5).mean()
            
            # Q2 动能复刻 (修正 ABS 语法错误)
            q1 = df['close'].diff()
            q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            df['q2'] = 100 * q_ema / q_abs_ema
            
            # 2. 信号判定 (XG)
            # 包含：大盘环境、30日内涨停、动能抬头、MA7 斜率、3% 乖离控制
            df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                       (df['pct_chg'].rolling(30).max() > 9.5) & \
                       (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                       (df['ma7'] > df['ma7'].shift(1)) & \
                       (df['close'] > df['high'].shift(1)) & \
                       ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)

            # 3. 交易引擎：计算区间收益与止损
            cash, shares, stop_low = float(init_cash), 0, 0
            history, trade_logs = [], []
            b_date, b_price = None, 0

            for i in range(len(df)):
                r = df.iloc[i]
                # 止损判断：收盘破 MA7 或 信号日最低价
                if shares > 0:
                    if r['close'] < stop_low or r['close'] < r['ma7']:
                        sell_p = r['close']
                        ret = (sell_p - b_price) / b_price * 100
                        cash = shares * sell_p
                        trade_logs.append({
                            "买入日期": b_date.date(), "卖出日期": r['date'].date(),
                            "买入价": f"{b_price:.2f}", "卖出价": f"{sell_p:.2f}",
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
            
            # 4. 统计面板展示
            st.subheader("📊 策略回测绩效清单")
            final_v = df['balance'].iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("期末总资产", f"{final_v:.2f} 元")
            c2.metric("累积回报率", f"{(final_v - init_cash)/init_cash*100:.2f}%")
            c3.metric("有效信号次数", len(df[df['xg']]))

            # 图表复盘
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            ax1.plot(df['date'], df['close'], label='股价', alpha=0.5)
            ax1.plot(df['date'], df['ma7'], label='MA7 支撑', color='cyan')
            ax1.scatter(df[df['xg']]['date'], df[df['xg']]['close'], color='red', marker='^', s=100, label='爆发信号')
            ax1.set_title("信号与趋势分布图")
            ax1.legend()
            
            ax2.plot(df['date'], df['balance'], label='账户资产', color='orange')
            ax2.axhline(init_cash, color='red', linestyle='--')
            ax2.set_title("资产累积收益曲线")
            st.pyplot(fig)
            
            if trade_logs:
                st.subheader("📋 详细区间交易收益清单")
                st.dataframe(pd.DataFrame(trade_logs), use_container_width=True)
            else:
                st.info("所选时间段内未触发符合条件的爆发信号。")
