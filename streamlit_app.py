import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 基础页面设置
st.set_page_config(page_title="爆发增强策略 Pro", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.title("📈 爆发增强策略 Pro 自动化回测系统")
st.info("已集成：MA7 趋势、Q2 动能抬头及双重止损逻辑。")

# 2. 交互参数
st.sidebar.header("回测配置")
symbol = st.sidebar.text_input("个股代码 (如 001255)", "001255")
start_val = st.sidebar.date_input("开始日期", pd.to_datetime("2024-01-01"))
end_val = st.sidebar.date_input("结束日期", pd.to_datetime("2026-02-24"))

@st.cache_data(ttl=3600)
def fetch_data_robust(code, start, end):
    """严谨的数据抓取，解决 image_622378 列名偏移问题"""
    try:
        s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
        # 获取个股数据
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
        if df is None or df.empty: return None

        # 动态探测列名，免疫 API 变更
        mapping = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '涨跌幅': 'pct_chg'}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        
        # 补齐涨跌幅字段防止 KeyError
        if 'pct_chg' not in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
        df['date'] = pd.to_datetime(df['date'])
        
        # 获取大盘数据
        try:
            idx = ak.stock_zh_index_daily(symbol="sh000001")
            idx['date'] = pd.to_datetime(idx['date'])
            df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
        except:
            st.warning("大盘数据同步失败，将跳过指数过滤。")
            df['idx_c'] = 99999
        return df
    except Exception as e:
        st.error(f"数据获取异常: {e}")
        return None

# 3. 运行逻辑
if st.sidebar.button("启动严谨回测"):
    df = fetch_data_robust(symbol, start_val, end_val)
    
    if df is not None:
        # --- 计算指标 (复刻通达信 11436 逻辑) ---
        df['ma7'] = df['close'].rolling(7).mean()
        df['idx_ma5'] = df['idx_c'].rolling(5).mean()
        
        # $Q_2$ 动能复刻：修正 ABS 参数错误
        # $Q_2 = 100 \times \frac{EMA(EMA(Q_1, 6), 6)}{EMA(EMA(|Q_1|, 6), 6)}$
        q1 = df['close'].diff()
        q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        df['q2'] = 100 * q_ema / q_abs_ema
        
        # --- 信号判定 (XG) ---
        # 逻辑：30日内有涨停、动能抬头、MA7向上、乖离率 <= 3%
        df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                   (df['pct_chg'].rolling(30).max() > 9.5) & \
                   (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                   (df['ma7'] > df['ma7'].shift(1)) & \
                   (df['close'] > df['high'].shift(1)) & \
                   ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)

        # --- 交易模拟 (双重止损) ---
        cash, shares, stop_line = 100000.0, 0, 0
        history, logs = [], []

        for i in range(len(df)):
            r = df.iloc[i]
            # 卖出：跌破 MA7 或 信号日最低价
            if shares > 0:
                if r['close'] < stop_line or r['close'] < r['ma7']:
                    cash = shares * r['close']
                    shares = 0
                    logs.append({"日期": r['date'].date(), "动作": "止损卖出", "价格": r['close']})
            # 买入：触发信号
            if r['xg'] and shares == 0:
                shares = cash / r['close']
                cash = 0
                stop_line = r['low'] # 锁定 11442 中的止损底线
                logs.append({"日期": r['date'].date(), "动作": "信号买入", "价格": r['close']})
            history.append(cash + shares * r['close'])

        df['balance'] = history
        
        # --- 结果可视化 ---
        st.subheader(f"回测结果：{symbol}")
        c1, c2 = st.columns(2)
        final_v = df['balance'].iloc[-1]
        c1.metric("期末净值", f"{final_v:.2f}")
        c2.metric("累计收益率", f"{(final_v-100000)/1000:.2f}%")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(df['date'], df['close'], label='股价', alpha=0.5)
        ax1.plot(df['date'], df['ma7'], label='MA7 支撑线', color='cyan')
        ax1.scatter(df[df['xg']]['date'], df[df['xg']]['close'], color='red', marker='^', s=100, label='爆发信号')
        ax1.set_title("信号位置复盘")
        ax1.legend()

        ax2.plot(df['date'], df['balance'], label='账户净值', color='orange')
        ax2.axhline(100000, color='black', linestyle='--')
        ax2.set_title("账户资产曲线")
        st.pyplot(fig)
        
        if logs: st.table(pd.DataFrame(logs))
