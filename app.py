import os
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from streamlit_searchbox import st_searchbox
import warnings
warnings.filterwarnings('ignore')


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="📈 Stock Trading Dashboard", layout="wide")
st.title("📊 Smart Stock Trading Dashboard (Interactive)")

# ---------------------------
# Settings
# ---------------------------
LOCAL_ALL_STOCKS = r"D:\Data Science\DSA Codes\Python-Basics\Project-Stock-App\all_stocks_yahoo.csv"
GITHUB_RAW_ALL = "https://raw.githubusercontent.com/sangram2002/Stock-Price-App/main/all_stocks_yahoo.csv"
MAX_SUGGESTIONS = 60  

# ---------------------------
# Load stock universe
# ---------------------------
@st.cache_data(show_spinner=False)
def load_stock_universe(local_path: str = LOCAL_ALL_STOCKS, remote_url: str = GITHUB_RAW_ALL) -> pd.DataFrame:
    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, dtype=str)
            st.info(f"✅ Loaded stock list from local file.")
        else:
            r = requests.get(remote_url, timeout=10)
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text), dtype=str)
            st.info("✅ Loaded stock list from GitHub.")
    except Exception as e:

        st.error(f"❌ Could not load stock list: {e}")
        return pd.DataFrame()

    # Normalize columns
    colmap = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc == "yahoosymbol": colmap[c] = "YahooSymbol"
        elif lc in ("symbol", "ticker"): colmap[c] = "Symbol"
        elif lc in ("company", "name"): colmap[c] = "Company"
        elif lc in ("market", "index"): colmap[c] = "Market"
    df = df.rename(columns=colmap)

    for col in ["YahooSymbol", "Symbol", "Company", "Market"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["search"] = (df["YahooSymbol"] + " " + df["Symbol"] + " " + df["Company"]).str.lower()
    df["search_nospace"] = df["search"].str.replace(r"\s+", "", regex=True)
    df["symbol_lower"] = df["Symbol"].str.lower()
    return df.drop_duplicates(subset=["YahooSymbol", "Symbol", "Company"]).reset_index(drop=True)

stocks_df = load_stock_universe()

# ---------------------------
# Search function
# ---------------------------
def search_stocks_for_box(query: str) -> list:
    if not query or stocks_df.empty:
        return []
    q = query.strip().lower()
    q_nospace = re.sub(r"\s+", "", q)

    df = stocks_df
    mask = (
        df["symbol_lower"].str.contains(q, na=False, regex=False) |
        df["search"].str.contains(q, na=False, regex=False) |
        df["search_nospace"].str.contains(q_nospace, na=False, regex=False)
    )
    matches = df[mask].copy()

    if matches.empty and re.fullmatch(r"[A-Za-z0-9\.\-]{1,8}", query.strip()):
        return [query.strip().upper()]

    return matches.apply(lambda r: f"{r['Company']} ({r['YahooSymbol'] or r['Symbol']})", axis=1).drop_duplicates().tolist()[:MAX_SUGGESTIONS]

selection = st_searchbox(search_function=search_stocks_for_box, placeholder="Search company/ticker...", key="stock_searchbox")

# ---------------------------
# Resolve symbol
# ---------------------------
if not selection:
    st.info("🔍 Start typing to get suggestions (e.g., 'AAPL', 'Tesla', 'RELIANCE')")
    st.stop()

m = re.search(r"\(([^)]+)\)", selection)
if m:
    selected_symbol = m.group(1).strip()
else:
    selected_symbol = selection.strip().upper()

def normalize_yahoo_symbol(sym: str) -> str:
    if not stocks_df.empty:
        row = stocks_df[
            (stocks_df["Symbol"].str.lower() == sym.lower()) | 
            (stocks_df["YahooSymbol"].str.lower() == sym.lower())
        ]
        if not row.empty:
            return row.iloc[0]["YahooSymbol"] or row.iloc[0]["Symbol"]
    
    # For Indian stocks, add .NS if not present
    if re.fullmatch(r"[A-Za-z]{1,6}", sym) and not sym.endswith('.NS'):
        return sym + ".NS"
    return sym

selected_symbol = normalize_yahoo_symbol(selected_symbol)
st.success(f"✅ Selected ticker: **{selected_symbol}**")

# ---------------------------
# Date inputs
# ---------------------------
today = date.today()
default_start = today - timedelta(days=365)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("📅 Start date", value=default_start, max_value=today)
with col2:
    end_date = st.date_input("📅 End date", value=today, min_value=start_date, max_value=today)

# ---------------------------
# Fetch data with proper error handling
# ---------------------------
@st.cache_data(ttl=3600, max_entries=128)
def fetch_stock_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch stock data and handle MultiIndex columns properly"""
    try:
        # Download data
        data = yf.download(ticker, start=start, end=end, progress=False)
        
        if data.empty:
            return pd.DataFrame()
        
        # Handle MultiIndex columns (common with yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex by taking the first level (price type)
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        # Ensure we have the basic OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                st.warning(f"⚠️ Missing {col} data")
                return pd.DataFrame()
        
        # Convert to numeric and handle any remaining issues
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Drop rows with NaN close prices
        data = data.dropna(subset=['Close'])
        
        return data
    
    except Exception as e:
        st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Fetch the data
with st.spinner(f"📥 Fetching data for {selected_symbol}..."):
    df = fetch_stock_data(selected_symbol, start_date, end_date)

if df.empty:
    st.error(f"❌ No data found for {selected_symbol}. Please check the symbol and try again.")
    st.stop()

st.success(f"✅ Successfully loaded {len(df)} trading days of data")

# ---------------------------
# Technical Indicators Calculation
# ---------------------------
def calculate_technical_indicators(df):
    """Calculate technical indicators with proper error handling"""
    data = df.copy()
    
    try:
        # Moving Averages
        data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        
        # Bollinger Bands
        rolling_std = data['Close'].rolling(window=20, min_periods=1).std()
        data['BB_Upper'] = data['MA20'] + (2 * rolling_std)
        data['BB_Lower'] = data['MA20'] - (2 * rolling_std)
        
        # RSI Calculation
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI'] = data['RSI'].fillna(50)
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        
        # Enhanced Trading Signals
        data['Trade_Signal'] = 'Hold'
        
        # Buy conditions (more sophisticated)
        buy_condition = (
            (data['Close'] > data['MA20']) & 
            (data['MA20'] > data['MA50']) &
            (data['RSI'] > 30) & (data['RSI'] < 70) &
            (data['MACD'] > data['Signal_Line']) &
            (data['Close'] > data['BB_Lower'])
        )
        
        # Sell conditions
        sell_condition = (
            (data['Close'] < data['MA20']) & 
            (data['MA20'] < data['MA50']) &
            (data['RSI'] > 70) |
            (data['MACD'] < data['Signal_Line']) &
            (data['Close'] < data['BB_Upper'])
        )
        
        data.loc[buy_condition, 'Trade_Signal'] = 'Buy'
        data.loc[sell_condition, 'Trade_Signal'] = 'Sell'
        
        # Price change and returns
        data['Price_Change'] = data['Close'].diff()
        data['Price_Change_Pct'] = data['Close'].pct_change() * 100
        
        return data
    
    except Exception as e:
        st.error(f"❌ Error calculating indicators: {str(e)}")
        return data

# Calculate indicators
df = calculate_technical_indicators(df)

# ---------------------------
# Key Metrics Display
# ---------------------------
st.subheader("📊 Key Metrics")
latest_data = df.iloc[-1]
previous_data = df.iloc[-2] if len(df) > 1 else latest_data

col1, col2, col3, col4 = st.columns(4)

with col1:
    current_price = latest_data['Close']
    price_change = latest_data['Price_Change']
    price_change_pct = latest_data['Price_Change_Pct']
    
    color = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "⚪"
    st.metric(
        label="💰 Current Price", 
        value=f"₹{current_price:.2f}" if selected_symbol.endswith('.NS') else f"${current_price:.2f}",
        delta=f"{price_change_pct:.2f}%"
    )

with col2:
    st.metric(
        label="📈 RSI (14)", 
        value=f"{latest_data['RSI']:.1f}",
        delta="Overbought" if latest_data['RSI'] > 70 else "Oversold" if latest_data['RSI'] < 30 else "Neutral"
    )

with col3:
    volume_change = ((latest_data['Volume'] - previous_data['Volume']) / previous_data['Volume']) * 100 if previous_data['Volume'] != 0 else 0
    st.metric(
        label="🔊 Volume", 
        value=f"{latest_data['Volume']:,.0f}",
        delta=f"{volume_change:.1f}%"
    )

with col4:
    trend = "📈 Bullish" if latest_data['MA20'] > latest_data['MA50'] else "📉 Bearish"
    st.metric(
        label="📊 Trend (MA20/MA50)", 
        value=trend,
        delta=f"MA20: {latest_data['MA20']:.2f}"
    )

# ---------------------------
# Main Price Chart
# ---------------------------
st.subheader(f"📈 Price Chart — {selected_symbol}")

price_fig = go.Figure()

# Candlestick chart
price_fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="Price",
    increasing_line_color='#00ff88',
    decreasing_line_color='#ff4444'
))

# Moving averages
price_fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['MA20'], 
    mode='lines', 
    name='MA20',
    line=dict(color='orange', width=2, dash='dash')
))

price_fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['MA50'], 
    mode='lines', 
    name='MA50',
    line=dict(color='purple', width=2, dash='dot')
))

# Bollinger Bands
price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df['BB_Upper'],
    mode='lines',
    name='BB Upper',
    line=dict(width=1, color='rgba(128,128,128,0.5)'),
    showlegend=False
))

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df['BB_Lower'],
    mode='lines',
    name='Bollinger Bands',
    line=dict(width=1, color='rgba(128,128,128,0.5)'),
    fill='tonexty',
    fillcolor='rgba(128,128,128,0.1)'
))

# Buy/Sell signals
buy_signals = df[df['Trade_Signal'] == 'Buy']
sell_signals = df[df['Trade_Signal'] == 'Sell']

if not buy_signals.empty:
    price_fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', size=12, color='green')
    ))

if not sell_signals.empty:
    price_fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', size=12, color='red')
    ))

price_fig.update_layout(
    template='plotly_dark',
    hovermode='x unified',
    xaxis_title='Date',
    yaxis_title='Price',
    height=500
)

st.plotly_chart(price_fig, use_container_width=True)

# ---------------------------
# Technical Indicator Charts
# ---------------------------
col1, col2 = st.columns(2)

# RSI Chart
with col1:
    st.subheader("📉 RSI (Relative Strength Index)")
    rsi_fig = go.Figure()
    
    # Color code RSI line based on levels
    rsi_colors = ['red' if rsi > 70 else 'green' if rsi < 30 else 'yellow' for rsi in df['RSI']]
    
    rsi_fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['RSI'], 
        mode='lines',
        name='RSI',
        line=dict(color='cyan', width=2)
    ))
    
    # Add reference lines
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    rsi_fig.add_hline(y=50, line_dash="dot", line_color="white", annotation_text="Neutral (50)")
    
    rsi_fig.update_layout(
        template='plotly_dark',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100]),
        height=300
    )
    st.plotly_chart(rsi_fig, use_container_width=True)

# MACD Chart
with col2:
    st.subheader("📊 MACD")
    macd_fig = go.Figure()
    
    # MACD line
    macd_fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['MACD'], 
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    # Signal line
    macd_fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Signal_Line'], 
        mode='lines',
        name='Signal Line',
        line=dict(color='red', width=2)
    ))
    
    # Histogram
    colors = ['green' if h >= 0 else 'red' for h in df['MACD_Histogram']]
    macd_fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_Histogram'],
        name='Histogram',
        marker_color=colors,
        opacity=0.6
    ))
    
    macd_fig.update_layout(
        template='plotly_dark',
        yaxis_title='MACD',
        height=300
    )
    st.plotly_chart(macd_fig, use_container_width=True)

# ---------------------------
# Volume Chart
# ---------------------------
st.subheader("🔊 Trading Volume")
vol_fig = go.Figure()

# Color volume bars based on price movement
vol_colors = ['green' if close >= open_price else 'red' 
              for close, open_price in zip(df['Close'], df['Open'])]

vol_fig.add_trace(go.Bar(
    x=df.index,
    y=df['Volume'],
    name='Volume',
    marker_color=vol_colors,
    opacity=0.7
))

# Add volume moving average
vol_ma = df['Volume'].rolling(window=20).mean()
vol_fig.add_trace(go.Scatter(
    x=df.index,
    y=vol_ma,
    mode='lines',
    name='Volume MA20',
    line=dict(color='white', width=2)
))

vol_fig.update_layout(
    template='plotly_dark',
    yaxis_title='Volume',
    height=300
)
st.plotly_chart(vol_fig, use_container_width=True)

# ---------------------------
# Advanced Trading Insights
# ---------------------------
st.subheader("💡 Advanced Trading Analysis")

# Calculate additional metrics
def get_trading_insights(data):
    latest = data.iloc[-1]
    insights = []
    
    # Price vs Moving Averages
    if latest['Close'] > latest['MA20'] > latest['MA50']:
        insights.append(("🟢", "Strong Bullish Trend", "Price above both MA20 and MA50"))
    elif latest['Close'] < latest['MA20'] < latest['MA50']:
        insights.append(("🔴", "Strong Bearish Trend", "Price below both MA20 and MA50"))
    elif latest['Close'] > latest['MA20']:
        insights.append(("🟡", "Short-term Bullish", "Price above MA20 but below MA50"))
    else:
        insights.append(("🟡", "Mixed Signals", "Price action unclear"))
    
    # RSI Analysis
    if latest['RSI'] > 80:
        insights.append(("🔴", "Extremely Overbought", f"RSI at {latest['RSI']:.1f} - Strong sell signal"))
    elif latest['RSI'] > 70:
        insights.append(("🟠", "Overbought", f"RSI at {latest['RSI']:.1f} - Consider taking profits"))
    elif latest['RSI'] < 20:
        insights.append(("🟢", "Extremely Oversold", f"RSI at {latest['RSI']:.1f} - Strong buy signal"))
    elif latest['RSI'] < 30:
        insights.append(("🟢", "Oversold", f"RSI at {latest['RSI']:.1f} - Consider buying"))
    else:
        insights.append(("🟡", "RSI Neutral", f"RSI at {latest['RSI']:.1f} - Wait for clear signal"))
    
    # MACD Analysis
    if latest['MACD'] > latest['Signal_Line'] and latest['MACD_Histogram'] > 0:
        insights.append(("🟢", "MACD Bullish", "MACD above signal line with positive momentum"))
    elif latest['MACD'] < latest['Signal_Line'] and latest['MACD_Histogram'] < 0:
        insights.append(("🔴", "MACD Bearish", "MACD below signal line with negative momentum"))
    
    # Bollinger Band Analysis
    bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
    if bb_position > 0.8:
        insights.append(("🔴", "Near Upper BB", "Price near upper band - potential reversal"))
    elif bb_position < 0.2:
        insights.append(("🟢", "Near Lower BB", "Price near lower band - potential bounce"))
    
    # Volume Analysis
    recent_vol_avg = data['Volume'].tail(10).mean()
    long_vol_avg = data['Volume'].tail(50).mean()
    if recent_vol_avg > long_vol_avg * 1.5:
        insights.append(("🟠", "High Volume", "Above-average trading activity"))
    elif recent_vol_avg < long_vol_avg * 0.5:
        insights.append(("🟡", "Low Volume", "Below-average trading activity"))
    
    return insights

insights = get_trading_insights(df)

# Display insights in a nice format
for emoji, title, description in insights:
    st.markdown(f"{emoji} **{title}**: {description}")

# ---------------------------
# Risk Metrics
# ---------------------------
st.subheader("⚠️ Risk Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    # Volatility (annualized)
    daily_returns = df['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    st.metric("📊 Volatility (Annual)", f"{volatility:.1f}%")

with col2:
    # Maximum drawdown
    rolling_max = df['Close'].expanding().max()
    drawdown = (df['Close'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    st.metric("📉 Max Drawdown", f"{max_drawdown:.1f}%")

with col3:
    # Support/Resistance levels
    recent_high = df['High'].tail(30).max()
    recent_low = df['Low'].tail(30).min()
    current_position = (latest_data['Close'] - recent_low) / (recent_high - recent_low) * 100
    st.metric("📍 Position in Range", f"{current_position:.0f}%")

# ---------------------------
# Trading Recommendation
# ---------------------------
st.subheader("🎯 Trading Recommendation")

def get_overall_recommendation(data):
    latest = data.iloc[-1]
    score = 0
    factors = []
    
    # Trend score
    if latest['Close'] > latest['MA20'] > latest['MA50']:
        score += 2
        factors.append("Strong uptrend (+2)")
    elif latest['Close'] > latest['MA20']:
        score += 1
        factors.append("Short-term uptrend (+1)")
    elif latest['Close'] < latest['MA20'] < latest['MA50']:
        score -= 2
        factors.append("Strong downtrend (-2)")
    elif latest['Close'] < latest['MA20']:
        score -= 1
        factors.append("Short-term downtrend (-1)")
    
    # RSI score
    if 30 <= latest['RSI'] <= 70:
        score += 1
        factors.append("RSI in healthy range (+1)")
    elif latest['RSI'] < 30:
        score += 1
        factors.append("RSI oversold - potential bounce (+1)")
    else:
        score -= 1
        factors.append("RSI overbought (-1)")
    
    # MACD score
    if latest['MACD'] > latest['Signal_Line']:
        score += 1
        factors.append("MACD bullish (+1)")
    else:
        score -= 1
        factors.append("MACD bearish (-1)")
    
    # Volume confirmation
    recent_vol = data['Volume'].tail(5).mean()
    avg_vol = data['Volume'].mean()
    if recent_vol > avg_vol:
        score += 0.5
        factors.append("Volume confirmation (+0.5)")
    
    return score, factors

score, factors = get_overall_recommendation(df)


if score >= 2:
    st.success(f"🟢 **STRONG BUY** (Score: {score})")
    recommendation = "Strong upward momentum with multiple positive indicators"
elif score >= 1:
    st.success(f"🟢 **BUY** (Score: {score})")
    recommendation = "Positive momentum, good entry opportunity"
elif score >= -1:
    st.warning(f"🟡 **HOLD** (Score: {score})")
    recommendation = "Mixed signals, wait for clearer direction"
elif score >= -2:
    st.error(f"🔴 **SELL** (Score: {score})")
    recommendation = "Negative momentum, consider exit"
else:
    st.error(f"🔴 **STRONG SELL** (Score: {score})")
    recommendation = "Strong downward pressure, exit recommended"

st.write(f"**Analysis**: {recommendation}")

with st.expander("📋 Scoring Breakdown"):
    for factor in factors:
        st.write(f"• {factor}")

# ---------------------------
# Performance Summary
# ---------------------------
st.subheader("📈 Performance Summary")

col1, col2 = st.columns(2)

with col1:
    st.write("**Period Performance:**")
    total_return = ((latest_data['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
    st.write(f"• Total Return: {total_return:.2f}%")
    
    # Calculate Sharpe ratio (simplified)
    if len(daily_returns) > 30:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        st.write(f"• Sharpe Ratio: {sharpe:.2f}")

with col2:
    st.write("**Recent Signals:**")
    recent_signals = df[df['Trade_Signal'] != 'Hold'].tail(5)
    if not recent_signals.empty:
        for idx, row in recent_signals.iterrows():
            signal_emoji = "🟢" if row['Trade_Signal'] == 'Buy' else "🔴"
            st.write(f"• {idx.strftime('%Y-%m-%d')}: {signal_emoji} {row['Trade_Signal']} @ {row['Close']:.2f}")
    else:
        st.write("• No recent signals")

# ---------------------------
# Data table
# ---------------------------
with st.expander("📋 Historical Data (Last 30 days)"):
    display_df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD', 'Trade_Signal']].tail(30)
    display_df = display_df.round(2)
    st.dataframe(display_df, use_container_width=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
**⚠️ Disclaimer**: This dashboard is for educational purposes only. 
Always do your own research and consider consulting with a financial advisor before making investment decisions.
Technical indicators are not guaranteed predictors of future price movements.
""")