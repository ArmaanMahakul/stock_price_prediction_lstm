import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.keras')

# Function to fetch stock information
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.info
    return stock_info

# Function to plot stock price history
def plot_stock_price(ticker, period):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    plt.figure(figsize=(10, 4))
    plt.plot(hist.index, hist['Close'], label='Close Price')
    plt.title(f"{ticker} - Stock Price History ({period})")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    return plt

# Function to fetch daily market data
def get_daily_market_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    return hist

# Function to make stock price predictions
def make_stock_prediction(ticker, days_forward):
    df = yf.download(ticker, "2019-01-01", datetime.today().strftime('%Y-%m-%d')).reset_index()
    df = df.drop(['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

    data_training = df['Close'][:int(len(df) * 0.70)]
    data_testing = df['Close'][int(len(df) * 0.70):]

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_training.values.reshape(-1, 1))

    # Prepare testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    final_df_scaled = scaler.transform(final_df.values.reshape(-1, 1))

    x_test = []
    for i in range(100, len(final_df_scaled)):
        x_test.append(final_df_scaled[i - 100:i])

    x_test = np.array(x_test)

    # Use the model to make predictions for the actual testing period
    y_pred = model.predict(x_test)
    y_pred_inverse = scaler.inverse_transform(y_pred)

    return data_testing.values, y_pred_inverse

# Load stock symbols from the CSV file
stock_symbols_df = pd.read_csv('Stock_Symbols.csv')

# Create a dictionary from the DataFrame
companies = dict(zip(stock_symbols_df['Company_Name'], stock_symbols_df['Scrip']))

# Sidebar for navigation and input
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Stock Information", "Stock Prediction"])

# Sidebar inputs for stock information
if page == "Stock Information":
    st.sidebar.title("Stock Selection")

    # Dropdown for selecting company name
    company_name = st.sidebar.selectbox("Select Company", options=list(companies.keys()), index=0)

    # Dropdown for selecting exchange
    exchange = st.sidebar.selectbox("Select Exchange", ["BSE", "NSE"])

    # Get the stock symbol for the selected company
    stock_symbol = companies[company_name]

    # Generate the final stock ticker based on the selected exchange
    if exchange == "BSE":
        ticker = f"{stock_symbol}.BO"
    else:
        ticker = f"{stock_symbol}.NS"

    # Display the stock ticker
    st.sidebar.write(f"Selected Ticker: {ticker}")

    # Timeframe selection for stock price history
    st.sidebar.title("Select Timeframe for Stock Price History")
    period = st.sidebar.selectbox("Timeframe", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

    # Main page content
    st.title("Stock Information Page")

    # Fetch and display stock information
    try:
        with st.spinner('Fetching stock information...'):
            stock_info = get_stock_info(ticker)
        st.subheader(f"Stock Information for {ticker}")

        # Displaying basic information
        st.write("### Basic Information")
        st.write(f"**Name**: {stock_info.get('shortName', 'N/A')}")
        st.write(f"**Sector**: {stock_info.get('sector', 'N/A')}")
        st.write(f"**Industry**: {stock_info.get('industry', 'N/A')}")
        st.write(f"**Website**: [{stock_info.get('website', 'N/A')}]({stock_info.get('website', 'N/A')})")

        # Displaying financial information
        st.write("### Financials")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Market Cap**: {stock_info.get('marketCap', 'N/A')}")
            st.write(f"**PE Ratio**: {stock_info.get('trailingPE', 'N/A')}")
            st.write(f"**Dividend Yield**: {stock_info.get('dividendYield', 'N/A')}")
            st.write(f"**EPS**: {stock_info.get('trailingEps', 'N/A')}")
        with col2:
            st.write(f"**Revenue**: {stock_info.get('totalRevenue', 'N/A')}")
            st.write(f"**Gross Profit**: {stock_info.get('grossProfits', 'N/A')}")
            st.write(f"**Operating Margin**: {stock_info.get('operatingMargins', 'N/A')}")
            st.write(f"**Profit Margin**: {stock_info.get('profitMargins', 'N/A')}")

        # Displaying daily market data
        st.write("### Daily Market Data")
        daily_data = get_daily_market_data(ticker)
        if not daily_data.empty:
            st.write(f"**Open**: {daily_data['Open'][0]}")
            st.write(f"**High**: {daily_data['High'][0]}")
            st.write(f"**Low**: {daily_data['Low'][0]}")
            st.write(f"**Close**: {daily_data['Close'][0]}")
            st.write(f"**Volume**: {daily_data['Volume'][0]}")
        else:
            st.write("No daily market data available.")

        # Displaying stock price history
        st.write("### Stock Price History")
        stock_price_plot = plot_stock_price(ticker, period)
        st.pyplot(stock_price_plot)

    except Exception as e:
        st.error(f"Failed to fetch stock information for {ticker}. Please try again.")
        st.error(str(e))

# Page: Stock Prediction
elif page == "Stock Prediction":
    st.title("Stock Prediction Page")

    # Sidebar inputs for stock prediction
    st.sidebar.title("Stock Selection")

    # Dropdown for selecting company name
    company_name = st.sidebar.selectbox("Select Company", options=list(companies.keys()), index=0)

    # Dropdown for selecting exchange
    exchange = st.sidebar.selectbox("Select Exchange", ["BSE", "NSE"])

    # Get the stock symbol for the selected company
    stock_symbol = companies[company_name]

    # Generate the final stock ticker based on the selected exchange
    if exchange == "BSE":
        ticker = f"{stock_symbol}.BO"
    else:
        ticker = f"{stock_symbol}.NS"

    st.sidebar.write(f"Selected Ticker: {ticker}")

    # Input for the number of days forward
    days_forward = st.sidebar.slider("Select number of days forward for prediction", 1, 50, 10)

    # Fetch and display the stock prediction
    try:
        with st.spinner('Fetching stock prediction...'):
            actual_prices, predicted_prices = make_stock_prediction(ticker, days_forward)

        st.write("### Predicted Stock Prices")

        # Plot the actual and predicted prices
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(actual_prices)), actual_prices, label='Actual Prices')
        plt.plot(range(100, len(predicted_prices) + 100), predicted_prices, label='Predicted Prices')
        plt.title(f'Stock Price Prediction for {ticker}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

        # Display future predictions on the sidebar
        st.sidebar.write("### Future Predictions")
        for i in range(len(predicted_prices)):
            st.sidebar.write(f"Day {i + 1}: {predicted_prices[i][0]:.2f}")

    except Exception as e:
        st.error(f"Failed to fetch stock prediction for {ticker}. Please try again.")
        st.error(str(e))
