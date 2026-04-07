# 🇮🇳 Indian Stock Market Predictor (AI)

**Live 3‑month stock forecasts for NSE‑listed companies**  
Built with multivariate linear regression, Streamlit, and Plotly.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Features

- Predicts next **3 months** (adjustable up to 180 days) for 15 popular Indian stocks
- Uses **multivariate linear regression** with engineered features:
  - Lag price (`Close_lag1`)
  - Moving averages (10‑day & 50‑day SMA)
  - 20‑day rolling volatility
  - Day of week & month
- Interactive **4‑panel dashboard**:
  1. Historical price + AI forecast with confidence band (±5%)
  2. 10‑day & 50‑day simple moving averages
  3. Last 30 days actual vs next 30 days forecast
  4. Volume & price relationship (dual axis)
- Model accuracy displayed as **MAPE** on test data

##  Supported Stocks

| Company | Ticker |
|---------|--------|
| Reliance Industries | `RELIANCE.NS` |
| Tata Consultancy Services (TCS) | `TCS.NS` |
| Infosys | `INFY.NS` |
| HDFC Bank | `HDFCBANK.NS` |
| ICICI Bank | `ICICIBANK.NS` |
| State Bank of India | `SBIN.NS` |
| Wipro | `WIPRO.NS` |
| HCL Technologies | `HCLTECH.NS` |
| Bajaj Finance | `BAJFINANCE.NS` |
| Tata Motors | `TATAMOTORS.NS` |
| Maruti Suzuki | `MARUTI.NS` |
| Asian Paints | `ASIANPAINT.NS` |
| Hindustan Unilever | `HINDUNILVR.NS` |
| Sun Pharma | `SUNPHARMA.NS` |
| Bharti Airtel | `BHARTIARTL.NS` |

##  How It Works

1. **Data Fetching** – Downloads historical data (1‑5 years) from Yahoo Finance.
2. **Feature Engineering** – Creates lag, rolling averages, volatility, and time features.
3. **Model Training** – Multivariate linear regression using the normal equation (least squares).
4. **Forecasting** – Recursively predicts future days, updating rolling windows dynamically.
5. **Visualization** – Displays forecast with confidence intervals and interactive Plotly charts.

##  Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/indian-stock-market-predictor.git
cd indian-stock-market-predictor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
