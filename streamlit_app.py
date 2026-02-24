import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 基础配置 ---
st.set_page_config(page_title="爆发增强策略交互回测系统", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
os.environ['NO_PROXY'] = '*' # 强制跳过代理

st.title("🚀 爆发增强策略 Pro - 自动化交互回测系统")
st.markdown("该系统严谨对齐：**MA7趋势**、**Q2动能**及**信号日最低价止损**逻辑。")

# --- 侧边栏交互 ---
st.sidebar.header("回测配置")
stock_code = st.sidebar.text_input("输入个股代码 (如 001255)", value="001255").strip()
start_date = st.sidebar.date_input("起始日期", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime("2026-02-24"))
init_cash = st.sidebar.number_input("初始资金", value=100000)

@st.cache_data(ttl=3600)
def fetch_data(code, start, end):
    """带动态列名映射的数据抓取"""
    try:
        s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
        if df is None or df.empty: return None

        # 动态映射列名，解决 image_622378 报错
        mapping = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '涨跌幅': 'pct_chg'}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        
        # 兼容字段，解决 image_6c1fe0 报错
        if 'pct_chg' not in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
        df['date'] = pd.to_datetime(df['date'])
        
        # 同步上证大盘数据用于环境过滤
        idx = ak.stock_zh_index_daily(symbol="sh000001")
        idx['date'] = pd.to_datetime(idx['date'])
        df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
        return df
    except Exception as e:
        st.error(f"获取数据失败，请刷新重试: {e}")
        return None

# --- 执行回测 ---
if st.sidebar.button("启动逻辑回测"):
    with st.spinner("正在计算交易区间..."):
        df = fetch_data(stock_code, start_date, end_date)
        
        if df is not None:
            # 1. 计算核心指标 (复刻 11436.jpg 逻辑)
            df['ma7'] = df['close'].rolling(7).mean()
            df['idx_ma5'] = df['idx_c'].rolling(5).mean()
            
            # Q2 动能复刻 (修正了 ABS 语法错误)
            q1 = df['close'].diff()
            q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            df['q2'] = 100 * q_ema / q_abs_ema
            
            # 2. 信号判定 (XG)
            # 条件：大盘安全、30日内有过涨停、Q2抬头、MA7趋势、乖离率 <= 3%
            df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                       (df['pct_chg'].rolling(30).max() > 9.5) & \
                       (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                       (df['ma7'] > df['ma7'].shift(1)) & \
                       (df['close'] > df['high'].shift(1)) & \
                       ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)

            # 3. 交易模拟引擎 (实现区间收益计算)
            cash, shares, stop_low = float(init_cash), 0, 0
            history, trade_records = [], []
            buy_date, buy_price = None, 0

            for i in range(len(df)):
                r = df.iloc[i]
                # 止损/卖出判断：收盘破 MA7 或 破信号日最低点
                if shares > 0:
                    if r['close'] < stop_low or r['close'] < r['ma7']:
                        sell_price = r['close']
                        trade_return = (sell_price - buy_price) / buy_price * 100
                        cash = shares * sell_price
                        trade_records.append({
                            "买入日期": buy_date.date(),
                            "卖出日期": r['date'].date(),
                            "买入价格": f"{buy_price:.2f}",
                            "卖出价格": f"{sell_price:.2f}",
                            "区间收益率": f"{trade_return:.2f}%"
                        })
                        shares = 0
                
                # 买入判断：触发信号且当前空仓
                if r['xg'] and shares == 0:
                    buy_date, buy_price = r['date'], r['close']
                    shares = cash / buy_price
                    cash = 0
                    stop_low = r['low'] # 记录 11442.jpg 中的止损线
                
                history.append(cash + shares * r['close'])

            df['balance'] = history
            
            # --- 结果展示 ---
            final_val = df['balance'].iloc[-1]
            cum_return = (final_val - init_cash) / init_cash * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("初始资金", f"{init_cash} 元")
            c2.metric("期末总资产", f"{final_val:.2f} 元")
            c3.metric("累积净收益率", f"{cum_return:.2f}%")

            # 图表复盘
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            ax1.plot(df['date'], df['close'], label='收盘价', alpha=0.5)
            ax1.plot(df['date'], df['ma7'], label='MA7趋势线', color='cyan')
            ax1.scatter(df[df['xg']]['date'], df[df['xg']]['close'], color='red', marker='^', s=100, label='爆发信号')
            ax1.set_title(f"{stock_code} 爆发点复盘图")
            ax1.legend()

            ax2.plot(df['date'], df['balance'], label='账户净值', color='orange')
            ax2.axhline(init_cash, color='black', linestyle='--')
            ax2.set_title("策略资产累积曲线")
            ax2.legend()
            st.pyplot(fig)
            
            if trade_records:
                st.subheader("📋 详细交易区间收益表")
                st.dataframe(pd.DataFrame(trade_records), use_container_width=True)
            else:
                st.info("在该时间段内未触发任何爆发信号。")
