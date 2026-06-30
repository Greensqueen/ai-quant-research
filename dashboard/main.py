import os
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data.yahoo_finance import YahooFinance
from data.sample_data import generate_sample_stock_data, get_sample_stock_info, generate_multiple_stocks
from analysis.technical_indicators import TechnicalIndicators
from analysis.statistical_analysis import StatisticalAnalysis
from ml.xgboost_model import XGBoostModel
from ml.random_forest import RandomForestModel
from portfolio.efficient_frontier import EfficientFrontier
from portfolio.risk_metrics import RiskMetrics

st.set_page_config(page_title="AI Quant Research Platform", layout="wide")

st.title("📈 AI Quant Research Platform")

st.sidebar.title("导航")
page = st.sidebar.radio("选择功能模块", [
    "股票数据",
    "技术分析",
    "统计分析",
    "机器学习预测",
    "投资组合分析",
    "LLM解释"
])

@st.cache_data
def get_stock_data(symbol, start_date, end_date):
    try:
        yf = YahooFinance()
        df = yf.get_stock_data(symbol, start_date, end_date)
        if df.empty:
            st.warning(f"无法获取真实数据，使用示例数据")
            return generate_sample_stock_data(symbol, days=365)
        return df
    except Exception as e:
        st.warning(f"获取数据失败: {str(e)}，使用示例数据")
        return generate_sample_stock_data(symbol, days=365)

@st.cache_data
def get_stock_info(symbol):
    try:
        yf = YahooFinance()
        info = yf.get_stock_info(symbol)
        if not info or info.get('name') == '':
            return get_sample_stock_info(symbol)
        return info
    except Exception as e:
        return get_sample_stock_info(symbol)

if page == "股票数据":
    st.header("📊 股票数据获取")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("股票代码", "AAPL")
    with col2:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    with col3:
        end_date = st.date_input("结束日期", datetime.now())
    
    if st.button("获取数据"):
        with st.spinner("获取数据中..."):
            try:
                df = get_stock_data(symbol, str(start_date), str(end_date))
                st.success(f"成功获取 {symbol} 的数据，共 {len(df)} 条记录")
                
                info = get_stock_info(symbol)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("公司名称", info.get("name", "N/A"))
                with col2:
                    st.metric("行业", info.get("sector", "N/A"))
                with col3:
                    st.metric("市盈率", f"{info.get('pe_ratio', 0):.2f}")
                with col4:
                    st.metric("Beta系数", f"{info.get('beta', 0):.2f}")
                
                st.subheader("股价走势图")
                fig = px.line(df, x='Date', y='Close', title=f'{symbol} 收盘价走势')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("数据表格")
                st.dataframe(df.tail(20), use_container_width=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "下载数据",
                    csv,
                    f"{symbol}_data.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"获取数据失败: {str(e)}")

elif page == "技术分析":
    st.header("📈 技术指标分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("股票代码", "AAPL")
    with col2:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    with col3:
        end_date = st.date_input("结束日期", datetime.now())
    
    indicators = st.multiselect(
        "选择技术指标",
        ["SMA", "EMA", "RSI", "MACD", "布林带", "ATR", "OBV", "Momentum", "Stochastic", "ADX"],
        ["SMA", "RSI", "MACD"]
    )
    
    if st.button("分析"):
        with st.spinner("计算技术指标中..."):
            try:
                df = get_stock_data(symbol, str(start_date), str(end_date))
                ti = TechnicalIndicators()
                
                if "SMA" in indicators:
                    df = ti.add_sma(df, window=20)
                    df = ti.add_sma(df, window=50)
                if "EMA" in indicators:
                    df = ti.add_ema(df, window=12)
                    df = ti.add_ema(df, window=26)
                if "RSI" in indicators:
                    df = ti.add_rsi(df)
                if "MACD" in indicators:
                    df = ti.add_macd(df)
                if "布林带" in indicators:
                    df = ti.add_bollinger_bands(df)
                if "ATR" in indicators:
                    df = ti.add_atr(df)
                if "OBV" in indicators:
                    df = ti.add_obv(df)
                if "Momentum" in indicators:
                    df = ti.add_momentum(df)
                if "Stochastic" in indicators:
                    df = ti.add_stochastic(df)
                if "ADX" in indicators:
                    df = ti.add_adx(df)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='blue')))
                
                if "SMA" in indicators:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='green', dash='dash')))
                if "布林带" in indicators:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(color='orange', dash='dot')))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(color='orange', dash='dot')))
                
                fig.update_layout(title=f'{symbol} 技术分析', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                if "RSI" in indicators:
                    fig_rsi = px.line(df, x='Date', y='RSI', title='RSI指标')
                    fig_rsi.add_hline(y=70, line_color='red', line_dash='dash')
                    fig_rsi.add_hline(y=30, line_color='green', line_dash='dash')
                    fig_rsi.update_layout(height=300)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                if "MACD" in indicators:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')))
                    fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
                    fig_macd.update_layout(title='MACD指标', height=300)
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                st.subheader("数据表格")
                st.dataframe(df.tail(20), use_container_width=True)
            except Exception as e:
                st.error(f"分析失败: {str(e)}")

elif page == "统计分析":
    st.header("📊 统计分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("股票代码", "AAPL")
    with col2:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    with col3:
        end_date = st.date_input("结束日期", datetime.now())
    
    if st.button("分析"):
        with st.spinner("计算统计指标中..."):
            try:
                df = get_stock_data(symbol, str(start_date), str(end_date))
                sa = StatisticalAnalysis()
                df = sa.calculate_returns(df)
                
                stats = sa.get_descriptive_stats(df)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("平均收益(日)", f"{stats['mean']*100:.2f}%")
                with col2:
                    st.metric("标准差", f"{stats['std']*100:.2f}%")
                with col3:
                    st.metric("夏普比率", f"{stats['sharpe_ratio']:.2f}")
                with col4:
                    st.metric("偏度", f"{stats['skew']:.2f}")
                with col5:
                    st.metric("峰度", f"{stats['kurtosis']:.2f}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最大回撤", f"{sa.calculate_max_drawdown(df) * 100:.2f}%")
                with col2:
                    st.metric("胜率", f"{sa.calculate_win_rate(df) * 100:.2f}%")
                with col3:
                    st.metric("利润因子", f"{sa.calculate_profit_factor(df):.2f}")
                with col4:
                    st.metric("VaR(95%)", f"{sa.calculate_var(df) * 100:.2f}%")
                
                fig = px.histogram(df, x='Returns', title='收益率分布', nbins=50)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.line(df, x='Date', y='Returns', title='每日收益率')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("统计摘要")
                st.write(stats)
            except Exception as e:
                st.error(f"分析失败: {str(e)}")

elif page == "机器学习预测":
    st.header("🤖 机器学习预测")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("股票代码", "AAPL")
    with col2:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=730))
    with col3:
        end_date = st.date_input("结束日期", datetime.now())
    
    model_type = st.selectbox("选择模型", ["XGBoost", "Random Forest"])
    
    if st.button("训练模型"):
        with st.spinner("训练模型中..."):
            try:
                df = get_stock_data(symbol, str(start_date), str(end_date))
                ti = TechnicalIndicators()
                df = ti.add_all_indicators(df)
                
                df = df.dropna()
                
                if model_type == "XGBoost":
                    model = XGBoostModel()
                else:
                    model = RandomForestModel()
                
                X, y = model.prepare_features(df)
                X_train, X_test, y_train, y_test = model.split_data(X)
                
                model.train(X_train, y_train)
                train_metrics = model.evaluate(X_train, y_train)
                test_metrics = model.evaluate(X_test, y_test)
                
                st.subheader("模型评估指标")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**训练集**")
                    st.write(f"MSE: {train_metrics['mse']:.4f}")
                    st.write(f"RMSE: {train_metrics['rmse']:.4f}")
                    st.write(f"MAE: {train_metrics['mae']:.4f}")
                    st.write(f"R2: {train_metrics['r2']:.4f}")
                with col2:
                    st.write("**测试集**")
                    st.write(f"MSE: {test_metrics['mse']:.4f}")
                    st.write(f"RMSE: {test_metrics['rmse']:.4f}")
                    st.write(f"MAE: {test_metrics['mae']:.4f}")
                    st.write(f"R2: {test_metrics['r2']:.4f}")
                
                predictions = model.predict(X_test)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, name='真实值', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=y_test.index, y=predictions, name='预测值', line=dict(color='red')))
                fig.update_layout(title='预测结果对比', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                importance = model.get_feature_importance(X.columns.tolist())
                fig_importance = px.bar(importance.head(10), x='feature', y='importance', title='特征重要性')
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"模型训练失败: {str(e)}")

elif page == "投资组合分析":
    st.header("💰 投资组合分析")
    
    symbols_input = st.text_input("输入股票代码（逗号分隔）", "AAPL, MSFT, GOOG, META")
    symbols = [s.strip() for s in symbols_input.split(",")]
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("结束日期", datetime.now())
    
    if st.button("分析投资组合"):
        with st.spinner("分析投资组合中..."):
            try:
                returns_df = pd.DataFrame()
                
                for symbol in symbols:
                    df = get_stock_data(symbol, str(start_date), str(end_date))
                    sa = StatisticalAnalysis()
                    df = sa.calculate_returns(df)
                    returns_df[symbol] = df['Returns']
                
                returns_df = returns_df.dropna()
                
                ef = EfficientFrontier(returns_df)
                frontier = ef.generate_efficient_frontier()
                optimal = ef.get_optimal_portfolios()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=frontier['std'], 
                    y=frontier['return'], 
                    mode='markers',
                    marker=dict(color=frontier['sharpe_ratio'], colorscale='Viridis'),
                    name='Efficient Frontier'
                ))
                fig.add_trace(go.Scatter(
                    x=[optimal['minimum_variance']['std']], 
                    y=[optimal['minimum_variance']['return']],
                    mode='markers',
                    marker=dict(color='red', size=15, symbol='star'),
                    name='Minimum Variance'
                ))
                fig.add_trace(go.Scatter(
                    x=[optimal['maximum_sharpe']['std']], 
                    y=[optimal['maximum_sharpe']['return']],
                    mode='markers',
                    marker=dict(color='green', size=15, symbol='star'),
                    name='Maximum Sharpe'
                ))
                
                fig.update_layout(
                    title='有效前沿',
                    xaxis_title='波动率',
                    yaxis_title='预期收益',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("最优投资组合")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**最小方差组合**")
                    weights = dict(zip(symbols, optimal['minimum_variance']['weights']))
                    st.write(pd.DataFrame.from_dict(weights, orient='index', columns=['权重']).style.format("{:.2%}"))
                    st.write(f"预期收益: {optimal['minimum_variance']['return']*100:.2f}%")
                    st.write(f"波动率: {optimal['minimum_variance']['std']*100:.2f}%")
                
                with col2:
                    st.write("**最大夏普组合**")
                    weights = dict(zip(symbols, optimal['maximum_sharpe']['weights'] ))
                    st.write(pd.DataFrame.from_dict(weights, orient='index', columns=['权重']).style.format("{:.2%}"))
                    st.write(f"预期收益: {optimal['maximum_sharpe']['return']*100:.2f}%")
                    st.write(f"波动率: {optimal['maximum_sharpe']['std']*100:.2f}%")
                    st.write(f"夏普比率: {optimal['maximum_sharpe']['sharpe_ratio']:.2f}")
                
            except Exception as e:
                st.error(f"分析失败: {str(e)}")

elif page == "LLM解释":
    st.header("💬 LLM智能分析")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("股票代码", "AAPL")
    with col2:
        question = st.text_input("输入您的问题", "为什么今天股价下跌？")
    
    if st.button("获取分析"):
        with st.spinner("分析中..."):
            try:
                from llm.news_analyzer import NewsAnalyzer
                from llm.question_answering import StockQA
                
                analyzer = NewsAnalyzer()
                news = analyzer.get_stock_news(symbol, days=7)
                news_analysis = analyzer.analyze_news(news, symbol)
                
                qa = StockQA()
                answer = qa.answer_question(question, news_analysis=news_analysis)
                
                st.subheader("AI分析结果")
                st.write(answer)
                
                st.subheader("市场情绪分析")
                sentiment_colors = {
                    'positive': 'green',
                    'negative': 'red',
                    'neutral': 'gray'
                }
                st.markdown(f"**情绪状态**: <span style='color:{sentiment_colors[news_analysis['sentiment']]}'>{news_analysis['sentiment']}</span>", unsafe_allow_html=True)
                
                st.subheader("关键影响因素")
                for factor in news_analysis['key_factors']:
                    st.write(f"- {factor}")
                
                st.subheader("相关新闻")
                for n in news[:3]:
                    st.write(f"**{n['title']}**")
                    st.write(f"来源: {n['source']}")
                    st.write(f"{n['description']}")
                    st.write("---")
                    
            except Exception as e:
                st.error(f"分析失败: {str(e)}")
                st.info("注意：LLM功能需要配置OpenAI API密钥才能正常工作。")