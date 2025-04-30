import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

st.set_page_config(page_title="Cocoa Market Dashboard", layout="wide")
st.title("Cocoa Market Dashboard")

st.markdown("""
This dashboard analyzes historical cocoa futures prices and provides 
Buy/Sell signals based on moving averages, as well as a 30-day price forecast.
""")

uploaded_file = st.file_uploader("Upload your cocoa futures CSV", type="csv")

if uploaded_file is not None:
    # Load and clean the data
    data = pd.read_csv(uploaded_file)
    data.rename(columns={"Price": "Close", "Vol.": "Volume", "Change %": "ChangePercent"}, inplace=True)
    for col in ["Close", "Open", "High", "Low"]:
        data[col] = data[col].astype(str).str.replace(",", "").astype(float)
    data["Date"] = pd.to_datetime(data["Date"])
    data.sort_values("Date", inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Calculate MAs
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    # Generate trading signals
    data["Signal"] = np.nan
    data.loc[data["MA20"] > data["MA50"], "Signal"] = "BUY"
    data.loc[data["MA20"] < data["MA50"], "Signal"] = "SELL"
    data.loc[data["MA20"] == data["MA50"], "Signal"] = "HOLD"

    # Layout: 2 columns
    col1, col2 = st.columns(2)
    with col1:
        latest_signal = data.dropna(subset=["Signal"])["Signal"].iloc[-1]
        st.metric(label="Latest Trading Signal", value=latest_signal)
    with col2:
        latest_price = data["Close"].iloc[-1]
        st.metric(label="Latest Closing Price", value=f"${latest_price:,.2f}")

    st.divider()

    # Interactive Plot
    st.subheader("Cocoa Price with Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MA20"], name="MA20", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MA50"], name="MA50", line=dict(dash='dash')))
    fig.update_layout(title="Cocoa Futures Price Chart", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Forecast with Prophet
    st.subheader("30-Day Forecast")
    forecast_df = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(forecast_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.divider()

    # Display Data Table
    st.subheader("Recent Data with Signals")
    st.dataframe(data[["Date", "Close", "MA20", "MA50", "Signal"]].tail(30), use_container_width=True)
else:
    st.info("Please upload a CSV file to begin.")
