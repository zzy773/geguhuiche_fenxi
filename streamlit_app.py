import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置页面配置
st.set_page_config(page_title="爆发增强策略 Pro 回测系统", layout="wide")

st.title("🚀 爆发增强策略 Pro 自动化回测系统")
st.markdown("""
本系统通过云端服务器直接抓取数据，解决了本地运行时的 `RemoteDisconnected` 错误。
其核心逻辑严格复刻了通达信 **MA7 支撑** 与 **Q2 动能抬头** 算法。
""")

# --- 侧边栏配置 ---
st.sidebar.header("回测参数设置")
symbol = st.sidebar.text_input("个股代码 (如 001255)", value="001255")
start_date = st.sidebar.date_input("开始日期", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime("2026-02-24"))
initial_cash = st.sidebar.number_input("初始资金", value=100000)

@st.cache_data(ttl=3600)
def get_data_robust(code, start, end):
    """严谨的数据获取函数，解决列名变化和连接问题"""
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
    try:
        # 获取个股历史数据
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        if df.empty: return None

        # 动态映射列名，防止 Length mismatch
        mapping = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'vol', '涨跌幅': 'pct_chg'}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        
        # 兼容字段缺失逻辑
        if 'pct_chg' not in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
        
        df['date'] = pd.to_datetime(df['date'])
        
        # 获取大盘数据用于环境过滤
        idx = ak.stock_zh_index_daily(symbol="sh000001")
        idx['date'] = pd.to_datetime(idx['date'])
        idx = idx[['date', 'close']].rename(columns={'close': 'idx_c'})
        
        return pd.merge(df, idx, on='date', how='left')
    except Exception as e:
        st.error(f"数据抓取失败: {e}")
        return None

if st.sidebar.button("开始运行回测"):
    df = get_data_robust(symbol, start_date, end_date)
    
    if df is not None:
        # --- 1. 计算核心指标 ---
        df['ma7'] = df['close'].rolling(7).mean()
        df['idx_ma5'] = df['idx_c'].rolling(5).mean()
        
        # 复刻 Q2 动能 (修正 ABS 嵌套错误)
        q1 = df['close'].diff()
        q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
        df['q2'] = 100 * q_ema / q_abs_ema
        
        # --- 2. 信号生成 (XG) ---
        # 包含：大盘多头、30日内涨停、动能抬头、MA7趋势、乖离率控制
        df['xg'] = (df['idx_c'] > df['idx_ma5']) & \
                   (df['pct_chg'].rolling(30).max() > 9.5) & \
                   (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) & \
                   (df['ma7'] > df['ma7'].shift(1)) & \
                   (df['close'] > df['high'].shift(1)) & \
                   ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)

        # --- 3. 回测执行 ---
        cash, shares, stop_low = float(initial_cash), 0, 0
        history = []
        trades_log = []

        for i in range(len(df)):
            row = df.iloc[i]
            # 止损逻辑：收盘跌破 MA7 或 信号日最低价
            if shares > 0:
                if row['close'] < stop_low or row['close'] < row['ma7']:
                    cash = shares * row['close']
                    shares = 0
                    trades_log.append({"日期": row['date'], "动作": "卖出/止损", "价格": row['close']})
            # 买入逻辑
            if row['xg'] and shares == 0:
                shares = cash / row['close']
                cash = 0
                stop_low = row['low'] # 记录信号日止损底线
                trades_log.append({"日期": row['date'], "动作": "买入", "价格": row['close']})
            
            history.append(cash + shares * row['close'])

        df['balance'] = history

        # --- 4. 结果展示 ---
        col1, col2, col3 = st.columns(3)
        final_val = df['balance'].iloc[-1]
        col1.metric("最终净值", f"{final_val:.2f}")
        col2.metric("累计收益率", f"{(final_val - initial_cash)/initial_cash*100:.2f}%")
        col3.metric("信号次数", len(df[df['xg']]))

        # 图表绘制
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # 价格图
        ax1.plot(df['date'], df['close'], label='收盘价', alpha=0.5)
        ax1.plot(df['date'], df['ma7'], label='MA7 支撑', color='cyan')
        buys = df[df['xg']]
        ax1.scatter(buys['date'], buys['close'], color='red', marker='^', s=100, label='爆发买点')
        ax1.set_title(f"个股 {symbol} 信号分布图")
        ax1.legend()
        
        # 收益图
        ax2.plot(df['date'], df['balance'], label='账户净值', color='orange')
        ax2.axhline(initial_cash, color='black', linestyle='--')
        ax2.set_title("策略收益曲线")
        ax2.legend()
        
        st.pyplot(fig)
        
        if trades_log:
            st.subheader("详细交易日志")
            st.table(pd.DataFrame(trades_log))
    else:
        st.error("无法加载数据，请检查代码输入是否正确。")
