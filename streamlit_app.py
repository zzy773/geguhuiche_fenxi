import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random

# --- 环境配置 ---
os.environ['NO_PROXY'] = '*'
st.set_page_config(page_title="爆发增强策略交互回测系统", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.title("🛡️ 爆发增强策略 - 自动化交互回测系统")
st.markdown("该系统针对 A 股市场进行策略回测分析。")

# --- 侧边栏交互输入 ---
st.sidebar.header("回测配置")
stock_code = st.sidebar.text_input("输入 A 股代码 (如 001255)", value="001255").strip()
start_date = st.sidebar.date_input("起始日期", value=pd.to_datetime("2024-08-01"))
end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime("2024-11-24"))
init_cash = st.sidebar.number_input("初始模拟资金 (元)", value=100000)

@st.cache_data(ttl=300)
def fetch_data_robust(code, start, end):
    """获取股票数据"""
    s_str, e_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
    for attempt in range(3):
        try:
            time.sleep(random.uniform(1, 2))
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_str, end_date=e_str, adjust="qfq")
            
            if df is not None and not df.empty:
                # 映射列名
                name_map = {'日期': 'date', '收盘': 'close', '最高': 'high', '最低': 'low', '开盘': 'open', '成交量': 'volume', '涨跌幅': 'pct_chg'}
                df = df.rename(columns=name_map)
                
                # 如果没有涨跌幅，则计算
                if 'pct_chg' not in df.columns:
                    df['pct_chg'] = df['close'].pct_change() * 100
                df['date'] = pd.to_datetime(df['date'])
                
                # 获取上证指数数据
                idx = ak.stock_zh_index_daily(symbol="sh000001")
                idx['date'] = pd.to_datetime(idx['date'])
                df = pd.merge(df, idx[['date', 'close']].rename(columns={'close': 'idx_c'}), on='date', how='left')
                return df
        except Exception as e:
            if attempt == 2: 
                st.error(f"获取数据失败: {e}")
                return None
    return None

if st.sidebar.button("启动回测"):
    with st.spinner("正在获取数据..."):
        df = fetch_data_robust(stock_code, start_date, end_date)
        
        if df is not None and len(df) > 0:
            # 计算技术指标
            df['ma7'] = df['close'].rolling(7).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            df['idx_ma5'] = df['idx_c'].rolling(5).mean()
            
            # 计算 Q2 动能指标
            q1 = df['close'].diff()
            q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
            df['q2'] = 100 * q_ema / q_abs_ema
            
            # 生成交易信号
            df['xg'] = (
                (df['idx_c'] > df['idx_ma5']) &  # 大盘环境
                (df['pct_chg'].rolling(30).max() > 9.5) &  # 30日内涨停
                (df['q2'] > df['q2'].shift(1)) &  # 动能抬头
                (df['q2'] > -20) &  # 动能阈值
                (df['ma7'] > df['ma7'].shift(1)) &  # MA7 上升
                (df['close'] > df['high'].shift(1)) &  # 突破前高
                ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)  # 乖离率控制
            )

            # 交易执行逻辑
            cash = float(init_cash)
            shares = 0
            stop_loss_price = 0
            balance_history = []
            trades = []

            for i in range(len(df)):
                current_row = df.iloc[i]
                
                # 止损逻辑
                if shares > 0:
                    if current_row['close'] < stop_loss_price or current_row['close'] < current_row['ma7']:
                        sell_price = current_row['close']
                        profit_pct = (sell_price - buy_price) / buy_price * 100
                        cash = shares * sell_price
                        trades.append({
                            "买入日期": buy_date.date(),
                            "卖出日期": current_row['date'].date(),
                            "买入价": round(buy_price, 2),
                            "卖出价": round(sell_price, 2),
                            "收益率": f"{profit_pct:.2f}%"
                        })
                        shares = 0
                
                # 买入逻辑
                if current_row['xg'] and shares == 0:
                    buy_date = current_row['date']
                    buy_price = current_row['close']
                    shares = cash / buy_price
                    cash = 0
                    stop_loss_price = current_row['low']  # 设置止损价为当日最低价
                
                # 记录账户余额
                current_balance = cash + shares * current_row['close']
                balance_history.append(current_balance)

            df['balance'] = balance_history
            
            # 统计结果
            final_balance = df['balance'].iloc[-1]
            total_return = (final_balance - init_cash) / init_cash * 100
            signal_count = df['xg'].sum()
            
            # 展示结果
            st.subheader("📊 回测结果")
            col1, col2, col3 = st.columns(3)
            col1.metric("期末总资产", f"¥{final_balance:,.2f}")
            col2.metric("总收益率", f"{total_return:.2f}%")
            col3.metric("触发信号数", int(signal_count))
            
            # 绘制图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # 价格和信号图
            ax1.plot(df['date'], df['close'], label='股价', linewidth=1.5)
            ax1.plot(df['date'], df['ma7'], label='MA7', linestyle='--', alpha=0.7)
            ax1.scatter(df[df['xg']]['date'], df[df['xg']]['close'], 
                       color='red', marker='^', s=100, label='买入信号', zorder=5)
            ax1.set_title(f'{stock_code} 股票价格走势及策略信号', fontsize=16)
            ax1.set_ylabel('价格 (元)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 资产曲线
            ax2.plot(df['date'], df['balance'], label='账户资产', color='orange', linewidth=2)
            ax2.axhline(y=init_cash, color='red', linestyle='--', label='初始资金')
            ax2.set_title('账户资产变化', fontsize=16)
            ax2.set_ylabel('资产 (元)', fontsize=12)
            ax2.set_xlabel('日期', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 交易记录
            if len(trades) > 0:
                st.subheader("📈 交易记录")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("在选定的时间段内没有触发买入信号。")
        else:
            st.error("无法获取股票数据，请检查股票代码和日期范围。")
else:
    st.info("请在侧边栏配置参数并点击“启动回测”开始分析。")
