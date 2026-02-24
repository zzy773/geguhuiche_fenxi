import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
from datetime import datetime, timedelta

# ======================== 基础配置 (适配GitHub/Streamlit云环境) ========================
# 强制绕过代理干扰，解决RemoteDisconnected问题
os.environ['NO_PROXY'] = '*'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# 页面配置 - 适配Streamlit云显示
st.set_page_config(
    page_title="爆发增强策略交互回测 Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 解决Matplotlib中文显示问题 (兼容Linux/macOS/Windows)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'  # 适配Streamlit白色背景

# ======================== 页面标题与说明 ========================
st.title("🛡️ 爆发增强策略 Pro - 自动化交互回测系统")
st.markdown("""
该系统针对 A 股爆发性行情进行策略回测，核心优化点：
- 解决 RemoteDisconnected 连接断开问题
- 修复 Altair/Length mismatch 报错
- 适配 GitHub + Streamlit Cloud 部署环境
""")
st.divider()

# ======================== 侧边栏配置 ========================
with st.sidebar:
    st.header("📌 回测配置")
    
    # 股票代码输入（增加提示和验证）
    stock_code = st.text_input(
        "输入 A 股代码 (如 001255)",
        value="001255",
        help="请输入6位A股代码，支持沪深京市场"
    ).strip()
    
    # 日期选择（设置合理范围，避免无效日期）
    default_start = datetime.now() - timedelta(days=365)
    default_end = datetime.now()
    start_date = st.date_input(
        "起始日期",
        value=default_start,
        min_value=datetime(2010, 1, 1),
        max_value=default_end
    )
    end_date = st.date_input(
        "结束日期",
        value=default_end,
        min_value=start_date,
        max_value=default_end
    )
    
    # 初始资金（增加范围限制）
    init_cash = st.number_input(
        "初始模拟资金 (元)",
        value=100000,
        min_value=10000,
        max_value=10000000,
        step=10000
    )
    
    # 回测按钮
    run_backtest = st.button("🚀 启动严谨逻辑回测", type="primary")

# ======================== 数据获取函数 (增强鲁棒性) ========================
@st.cache_data(ttl=300, show_spinner="正在获取数据...")
def fetch_data_robust(code, start, end):
    """
    稳健的股票数据获取函数
    解决：连接断开、字段缺失、格式不统一问题
    """
    # 格式转换
    s_str = start.strftime('%Y%m%d')
    e_str = end.strftime('%Y%m%d')
    
    # 最多重试5次
    for attempt in range(5):
        try:
            # 随机休眠，模拟真人操作
            time.sleep(random.uniform(1, 3))
            
            # 获取股票数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=s_str,
                end_date=e_str,
                adjust="qfq"
            )
            
            # 数据有效性检查
            if df is None or df.empty:
                st.warning("未获取到有效股票数据，请检查代码或日期范围")
                return None
            
            # 动态列名映射（解决不同版本字段名不一致）
            name_map = {
                '日期': 'date', '收盘': 'close', '最高': 'high',
                '最低': 'low', '涨跌幅': 'pct_chg', '开盘': 'open', '成交量': 'volume'
            }
            df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
            
            # 处理缺失字段
            if 'pct_chg' not in df.columns:
                df['pct_chg'] = df['close'].pct_change() * 100
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna(subset=['close', 'high', 'low'])  # 删除关键字段缺失行
            
            # 获取上证指数数据
            idx_df = ak.stock_zh_index_daily(symbol="sh000001")
            idx_df['date'] = pd.to_datetime(idx_df['date'])
            idx_df = idx_df.rename(columns={'收盘': 'idx_c'})
            
            # 合并数据
            df = pd.merge(df, idx_df[['date', 'idx_c']], on='date', how='left')
            df['idx_c'] = df['idx_c'].fillna(method='ffill')  # 填充指数缺失值
            
            return df
        
        except Exception as e:
            if attempt == 4:  # 最后一次重试失败
                st.error(f"数据获取失败: {str(e)}")
                st.info("建议检查：1.股票代码是否正确 2.网络连接 3.日期范围是否合理")
                return None
            continue

# ======================== 回测核心逻辑 ========================
if run_backtest:
    # 输入验证
    if not stock_code or len(stock_code) != 6:
        st.error("请输入有效的6位A股代码！")
    elif start_date >= end_date:
        st.error("结束日期必须晚于起始日期！")
    else:
        with st.spinner("系统正在穿透数据拦截并执行回测..."):
            # 1. 获取数据
            df = fetch_data_robust(stock_code, start_date, end_date)
            
            if df is not None and not df.empty:
                # 2. 计算核心指标
                # MA7均线
                df['ma7'] = df['close'].rolling(7).mean()
                # 上证指数MA5
                df['idx_ma5'] = df['idx_c'].rolling(5).mean()
                
                # Q2动能指标
                q1 = df['close'].diff()
                q_ema = q1.ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
                q_abs_ema = q1.abs().ewm(span=6, adjust=False).mean().ewm(span=6, adjust=False).mean()
                df['q2'] = 100 * q_ema / q_abs_ema.replace(0, np.nan)  # 避免除零错误
                
                # 3. 信号判定 (XG)
                df['xg'] = (
                    (df['idx_c'] > df['idx_ma5']) &  # 大盘环境向好
                    (df['pct_chg'].rolling(30).max() > 9.5) &  # 30日内有涨停
                    (df['q2'] > df['q2'].shift(1)) & (df['q2'] > -20) &  # 动能抬头
                    (df['ma7'] > df['ma7'].shift(1)) &  # MA7斜率向上
                    (df['close'] > df['high'].shift(1)) &  # 突破前日高点
                    ((df['close'] - df['ma7'])/df['ma7']*100 <= 3)  # 乖离率控制
                ).fillna(False)
                
                # 4. 交易引擎
                cash = float(init_cash)
                shares = 0
                stop_low = 0
                trade_logs = []
                history = []
                b_date, b_price = None, 0
                
                for i in range(len(df)):
                    row = df.iloc[i]
                    
                    # 止损逻辑
                    if shares > 0:
                        if row['close'] < stop_low or row['close'] < row['ma7']:
                            sell_price = row['close']
                            return_rate = (sell_price - b_price) / b_price * 100
                            cash = shares * sell_price
                            
                            # 记录交易
                            trade_logs.append({
                                "买入日期": b_date.date(),
                                "卖出日期": row['date'].date(),
                                "买入价": f"{b_price:.2f}",
                                "卖出价": f"{sell_price:.2f}",
                                "区间净收益": f"{return_rate:.2f}%"
                            })
                            shares = 0
                    
                    # 买入逻辑
                    if row['xg'] and shares == 0:
                        b_date = row['date']
                        b_price = row['close']
                        shares = cash / b_price
                        cash = 0
                        stop_low = row['low']  # 设定止损底线
                    
                    # 记录账户资产
                    current_value = cash + shares * row['close']
                    history.append(current_value)
                
                df['balance'] = history
                
                # 5. 结果展示
                st.success("✅ 回测完成！")
                st.divider()
                
                # 绩效指标
                final_value = df['balance'].iloc[-1]
                total_return = (final_value - init_cash) / init_cash * 100
                signal_count = len(df[df['xg']])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("期末总资产", f"{final_value:.2f} 元", f"{final_value - init_cash:.2f} 元")
                with col2:
                    st.metric("累积回报率", f"{total_return:.2f}%", delta_color="normal")
                with col3:
                    st.metric("有效信号次数", signal_count)
                
                # 可视化图表
                st.subheader("📈 策略表现可视化")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
                
                # 股价与信号图
                ax1.plot(df['date'], df['close'], label='股价', alpha=0.7, color='#1f77b4')
                ax1.plot(df['date'], df['ma7'], label='MA7 支撑', color='cyan', linewidth=2)
                ax1.scatter(
                    df[df['xg']]['date'], 
                    df[df['xg']]['close'], 
                    color='red', marker='^', s=100, 
                    label='爆发信号', zorder=5
                )
                ax1.set_title('股价走势与交易信号', fontsize=12, fontweight='bold')
                ax1.set_ylabel('价格 (元)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 资产收益曲线
                ax2.plot(df['date'], df['balance'], label='账户资产', color='orange', linewidth=2)
                ax2.axhline(init_cash, color='red', linestyle='--', label='初始资金', alpha=0.7)
                ax2.set_title('账户资产变化', fontsize=12, fontweight='bold')
                ax2.set_ylabel('资产 (元)')
                ax2.set_xlabel('日期')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 交易记录
                if trade_logs:
                    st.subheader("📋 详细交易记录")
                    trade_df = pd.DataFrame(trade_logs)
                    st.dataframe(trade_df, use_container_width=True)
                else:
                    st.info("ℹ️ 所选时间段内未触发符合条件的交易信号")
