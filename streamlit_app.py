import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
from datetime import datetime

# --- 环境加固 ---
os.environ['NO_PROXY'] = '*'
st.set_page_config(page_title="爆发增强策略交互回测 Pro", layout="wide")

# --- 中文支持（兼容 Linux 容器）---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

st.title("🛡️ 爆发增强策略 Pro - 自动化交互回测系统")
st.markdown("该系统针对 **RemoteDisconnected** 及 **字段变更** 进行了底层加固，适配 GitHub + Streamlit Cloud 部署。")

# --- 侧边栏配置 ---
st.sidebar.header("回测配置")
stock_code = st.sidebar.text_input("输入 A 股代码 (如 001255)", value="001255").strip()
start_date = st.sidebar.date_input("起始日期", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime("2026-02-24"))
init_cash = st.sidebar.number_input("初始模拟资金 (元)", value=100000, min_value=1000)

# --- 数据抓取函数（带字段兼容与重试）---
@st.cache_data(ttl=600)
def fetch_data_robust(code: str, start, end):
    s_str = start.strftime('%Y%m%d')
    e_str = end.strftime('%Y%m%d')
    
    for attempt in range(3):
        try:
            time.sleep(random.uniform(1.5, 3.0))  # 模拟人工延迟
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=s_str,
                end_date=e_str,
                adjust="qfq"
            )
            
            if df is None or df.empty:
                continue

            # 字段映射：兼容 AkShare 不同版本（2024-2026）
            col_map = {
                '日期': 'date',
                '收盘': 'close',
                '收盘价': 'close',
                '最高': 'high',
                '最高价': 'high',
                '最低': 'low',
                '最低价': 'low',
                '涨跌幅': 'pct_chg'
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            required_cols = ['date', 'close', 'high', 'low']
            if not all(col in df.columns for col in required_cols):
                st.warning(f"数据缺失关键字段: {df.columns.tolist()}")
                return None

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # 补全 pct_chg
            if 'pct_chg' not in df.columns:
                df['pct_chg'] = df['close'].pct_change() * 100

            # 合并上证指数（用于大盘环境判断）
            try:
                idx_df = ak.stock_zh_index_daily(symbol="sh000001")
                idx_df['date'] = pd.to_datetime(idx_df['date'])
                idx_df = idx_df[['date', 'close']].rename(columns={'close': 'idx_c'})
                df = pd.merge(df, idx_df, on='date', how='left')
                df['idx_c'] = df['idx_c'].fillna(method='ffill')  # 前向填充避免 NaN
            except Exception as e:
                st.warning("⚠️ 无法获取上证指数，使用股价自身替代大盘信号（策略效果可能下降）")
                df['idx_c'] = df['close']

            return df

        except Exception as e:
            if attempt == 2:
                st.error(f"❌ 数据获取失败（{code}）: {str(e)[:200]}")
                return None
    return None

# --- 主逻辑：回测按钮触发 ---
if st.sidebar.button("🚀 启动严谨逻辑回测"):
    if not stock_code.isdigit() or len(stock_code) != 6:
        st.error("请输入有效的 6 位 A 股代码（如 001255）")
    else:
        with st.spinner("📡 正在穿透数据拦截...（首次加载较慢，请耐心等待）"):
            df = fetch_data_robust(stock_code, start_date, end_date)

        if df is not None and not df.empty:
            # === 1. 计算技术指标 ===
            df['ma7'] = df['close'].rolling(window=7, min_periods=1).mean()
            df['idx_ma5'] = df['idx_c'].rolling(window=5, min_periods=1).mean()

            # Q2 动能指标（双EMA平滑）
            q1 = df['close'].diff()
            q_ema1 = q1.ewm(span=6, adjust=False).mean()
            q_ema2 = q_ema1.ewm(span=6, adjust=False).mean()
            q_abs_ema1 = q1.abs().ewm(span=6, adjust=False).mean()
            q_abs_ema2 = q_abs_ema1.ewm(span=6, adjust=False).mean()
            df['q2'] = 100 * q_ema2 / (q_abs_ema2 + 1e-8)  # 防除零

            # === 2. 信号生成 (XG) ===
            df['xg'] = (
                (df['idx_c'] > df['idx_ma5']) &
                (df['pct_chg'].rolling(window=30, min_periods=1).max() > 9.5) &
                (df['q2'] > df['q2'].shift(1)) &
                (df['q2'] > -20) &
                (df['ma7'] > df['ma7'].shift(1)) &
                (df['close'] > df['high'].shift(1)) &
                (((df['close'] - df['ma7']) / df['ma7'] * 100) <= 3)
            )

            # === 3. 交易模拟引擎 ===
            cash = float(init_cash)
            shares = 0.0
            stop_low = 0.0
            history = []
            trade_logs = []
            buy_date, buy_price = None, 0.0

            for i in range(len(df)):
                row = df.iloc[i]
                current_balance = cash + shares * row['close']
                history.append(current_balance)

                # 卖出条件：持仓中 & 触发止损
                if shares > 0:
                    if row['close'] < stop_low or row['close'] < row['ma7']:
                        sell_price = row['close']
                        ret_pct = (sell_price - buy_price) / buy_price * 100
                        cash = shares * sell_price
                        trade_logs.append({
                            "买入日期": buy_date.date(),
                            "卖出日期": row['date'].date(),
                            "买入价": f"{buy_price:.2f}",
                            "卖出价": f"{sell_price:.2f}",
                            "区间净收益": f"{ret_pct:.2f}%"
                        })
                        shares = 0.0

                # 买入条件：信号触发 & 无持仓
                if row['xg'] and shares == 0:
                    buy_date = row['date']
                    buy_price = row['close']
                    shares = cash / buy_price
                    cash = 0.0
                    stop_low = row['low']  # 止损设为当日最低价

            df['balance'] = history

            # === 4. 结果展示 ===
            final_value = df['balance'].iloc[-1]
            total_return = (final_value - init_cash) / init_cash * 100
            signal_count = df['xg'].sum()

            st.subheader("📊 策略回测绩效清单")
            col1, col2, col3 = st.columns(3)
            col1.metric("期末总资产", f"¥{final_value:,.2f}")
            col2.metric("累积回报率", f"{total_return:.2f}%")
            col3.metric("有效信号次数", int(signal_count))

            # 图表绘制
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            ax1.plot(df['date'], df['close'], label='股价', alpha=0.7, linewidth=1)
            ax1.plot(df['date'], df['ma7'], label='MA7 支撑', color='cyan', linewidth=1)
            signals = df[df['xg']]
            if not signals.empty:
                ax1.scatter(signals['date'], signals['close'], color='red', marker='^', s=80, label='爆发信号')
            ax1.set_title(f"{stock_code} 信号与趋势分布图", fontsize=14)
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.plot(df['date'], df['balance'], label='账户资产', color='orange', linewidth=1.5)
            ax2.axhline(init_cash, color='red', linestyle='--', label='初始资金')
            ax2.set_title("资产累积收益曲线", fontsize=14)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)

            st.pyplot(fig)

            # 交易记录
            if trade_logs:
                st.subheader("📋 详细区间交易收益清单")
                st.dataframe(pd.DataFrame(trade_logs), use_container_width=True)
            else:
                st.info("所选时间段内未触发符合条件的爆发信号。")

        else:
            st.error("❌ 未能获取有效股票数据，请检查代码或日期范围。")
