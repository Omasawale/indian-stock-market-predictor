import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ------------------- Page config -------------------
st.set_page_config(page_title="Indian Stock Predictor", layout="wide")
st.title("📈 Indian Stock Market Predictor (AI Model)")
st.markdown("Select an NSE stock – AI predicts next 3 months using multivariate linear regression.")

# ------------------- Indian Stocks -------------------
INDIAN_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Bharti Airtel": "BHARTIARTL.NS"
}

# ------------------- Data fetching -------------------
@st.cache_data(ttl=3600)
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None
    df.reset_index(inplace=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.set_index('Date', inplace=True)
    return df[['Close', 'Volume']]

# ------------------- Feature engineering -------------------
def create_features(df):
    """Create features for linear regression: lags, moving averages, volatility, time features"""
    df = df.copy()
    # Lag features (previous day close)
    df['Close_lag1'] = df['Close'].shift(1)
    # Moving averages
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    # Volatility (20-day std)
    df['Volatility'] = df['Close'].rolling(20).std()
    # Day of week (0=Monday ... 4=Friday)
    df['DayOfWeek'] = df.index.dayofweek
    # Month of year
    df['Month'] = df.index.month
    # Drop NaN rows
    df = df.dropna()
    return df

def train_linear_regression(X, y):
    """Multivariate linear regression using normal equation (no sklearn)"""
    # Add bias term (column of ones)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Normal equation: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
    return theta

def predict_with_model(theta, X_new):
    """Predict using trained parameters"""
    X_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
    return X_b @ theta

def forecast_future(model_theta, df, feature_names, future_days=90):
    """
    Recursively predict next 'future_days' days.
    For each future day, we need to create features.
    """
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days, freq='B')
    
    # Prepare initial known features from last row of df
    last_row = df.iloc[-1]
    predictions = []
    
    # Keep a rolling window of past closes to compute moving averages and volatility
    recent_closes = df['Close'].values[-50:].tolist()  # store last 50 closes
    
    for i, future_date in enumerate(future_dates):
        # Create features for this future day
        # Lag1 = previous close (last prediction or last actual)
        if i == 0:
            lag1 = last_row['Close']
        else:
            lag1 = predictions[-1]
        
        # SMA10: average of last 10 closes (including predicted ones)
        sma10 = np.mean(recent_closes[-10:]) if len(recent_closes) >= 10 else np.mean(recent_closes)
        # SMA50: average of last 50 closes
        sma50 = np.mean(recent_closes[-50:]) if len(recent_closes) >= 50 else np.mean(recent_closes)
        # Volatility: std of last 20 closes
        volatility = np.std(recent_closes[-20:]) if len(recent_closes) >= 20 else 0.0
        # Day of week (0=Monday)
        dayofweek = future_date.dayofweek
        month = future_date.month
        
        # Feature vector in the same order as training
        features = np.array([[lag1, sma10, sma50, volatility, dayofweek, month]])
        pred = predict_with_model(model_theta, features)[0]
        predictions.append(pred)
        
        # Update rolling closes with this prediction
        recent_closes.append(pred)
    
    future_df = pd.DataFrame({'Predicted': predictions}, index=future_dates)
    # Add simple confidence bands (±5% of predicted)
    future_df['Lower'] = future_df['Predicted'] * 0.95
    future_df['Upper'] = future_df['Predicted'] * 1.05
    return future_df

# ------------------- Plotting -------------------
def compute_sma(df, window):
    return df['Close'].rolling(window=window).mean()

def plot_4_charts(df_hist, future_df, stock_name, exchange_rate=83.0):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "1. Historical vs AI Predicted Prices",
            "2. Moving Averages (10 & 50 day)",
            "3. Recent 30 Days (Actual vs Forecast)",
            "4. Volume & Price Relationship"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        specs=[[{}, {}], [{"secondary_y": True}, {"secondary_y": True}]]
    )
    # Convert to INR
    price_hist = df_hist['Close'] * exchange_rate
    price_pred = future_df['Predicted'] * exchange_rate
    lower_pred = future_df['Lower'] * exchange_rate
    upper_pred = future_df['Upper'] * exchange_rate

    # Graph 1: Historical + Forecast with confidence band
    fig.add_trace(go.Scatter(x=df_hist.index, y=price_hist, name='Historical Price', line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(future_df.index) + list(future_df.index)[::-1],
        y=list(upper_pred) + list(lower_pred)[::-1],
        fill='toself', fillcolor='rgba(255,127,14,0.2)',
        line=dict(color='rgba(0,0,0,0)'), name='Forecast Confidence'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=future_df.index, y=price_pred, name='AI 3-Month Forecast', line=dict(color='#ff7f0e', dash='dot')), row=1, col=1)

    # Graph 2: Moving averages
    sma10 = compute_sma(df_hist, 10)
    sma50 = compute_sma(df_hist, 50)
    fig.add_trace(go.Scatter(x=df_hist.index, y=price_hist, name='Close Price', line=dict(color='#1f77b4', width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_hist.index, y=sma10 * exchange_rate, name='10-Day SMA', line=dict(color='#2ca02c')), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_hist.index, y=sma50 * exchange_rate, name='50-Day SMA', line=dict(color='#d62728')), row=1, col=2)

    # Graph 3: Last 30 days actual vs next 30 days forecast
    last_30 = df_hist.index[-30:]
    pred_30 = future_df.index[:30]
    fig.add_trace(go.Scatter(x=last_30, y=price_hist[-30:], name='Actual (last 30d)', line=dict(color='#1f77b4')), row=2, col=1)
    fig.add_trace(go.Scatter(x=pred_30, y=price_pred[:30], name='AI Forecast (next 30d)', line=dict(color='#ff7f0e')), row=2, col=1)

    # Graph 4: Volume + Price (dual axis)
    fig.add_trace(go.Bar(x=df_hist.index, y=df_hist['Volume'], name='Volume', marker_color='#7f7f7f'), row=2, col=2)
    fig.add_trace(go.Scatter(x=df_hist.index, y=price_hist, name='Price', line=dict(color='#1f77b4')), row=2, col=2, secondary_y=True)

    fig.update_layout(
        height=900, width=1200, title_text=f"{stock_name} – AI 3‑Month Forecast (₹)",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    for i in [1,2]:
        for j in [1,2]:
            fig.update_xaxes(title_text="Date", row=i, col=j, showgrid=True)
    for j in [1,2]:
        fig.update_yaxes(title_text="Price (₹)", row=1, col=j, tickprefix="₹")
    fig.update_yaxes(title_text="Price (₹)", row=2, col=1, tickprefix="₹")
    fig.update_yaxes(title_text="Volume", row=2, col=2, tickformat=",.0f", secondary_y=False)
    fig.update_yaxes(title_text="Price (₹)", row=2, col=2, tickprefix="₹", secondary_y=True)
    return fig

# ------------------- Main App -------------------
with st.sidebar:
    st.header("⚙️ Settings")
    selected_stock = st.selectbox("Select Indian Stock", list(INDIAN_STOCKS.keys()))
    ticker = INDIAN_STOCKS[selected_stock]
    years_back = st.slider("Historical data (years)", 1, 5, 2)
    forecast_days = st.slider("Forecast days (3 months ≈ 90)", 30, 180, 90, 30)
    run_btn = st.button("🚀 Analyze with AI", use_container_width=True)

if run_btn:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years_back*365)
    
    with st.spinner(f"Fetching {selected_stock} data..."):
        df_raw = fetch_data(ticker, start_date, end_date)
    
    if df_raw is None or df_raw.empty:
        st.error("No data found. Check ticker symbol.")
    else:
        st.success(f"✅ Fetched {len(df_raw)} days.")
        
        # Prepare features and target
        df_feat = create_features(df_raw)
        # Target: next day's close (shift -1)
        df_feat['Target'] = df_feat['Close'].shift(-1)
        df_feat = df_feat.dropna()
        
        if len(df_feat) < 100:
            st.warning("Not enough data for reliable AI. Choose more years.")
        else:
            # Features (excluding target and raw close)
            feature_cols = ['Close_lag1', 'SMA10', 'SMA50', 'Volatility', 'DayOfWeek', 'Month']
            X = df_feat[feature_cols].values
            y = df_feat['Target'].values
            
            # Train/test split (80/20 chronological)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Train model
            theta = train_linear_regression(X_train, y_train)
            
            # Evaluate on test set
            y_pred_test = predict_with_model(theta, X_test)
            mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            st.info(f"📊 AI Model Accuracy (MAPE on test data): {mape:.1f}%")
            
            # Forecast future
            future_df = forecast_future(theta, df_feat, feature_cols, future_days=forecast_days)
            
            # Plot
            fig = plot_4_charts(df_raw, future_df, selected_stock, exchange_rate=83.0)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show forecast table
            with st.expander("📅 View AI Forecast Table (next 30 days)"):
                forecast_table = future_df[['Predicted']].head(30).copy()
                forecast_table['Predicted (₹)'] = forecast_table['Predicted'] * 83
                st.dataframe(forecast_table[['Predicted (₹)']], use_container_width=True)
else:
    st.info("👈 Select a stock and click **Analyze with AI** to see dynamic 3‑month predictions.")