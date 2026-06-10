import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Tesla Stock Price Prediction", page_icon="🚗", layout="wide"
)

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("../data/TSLA.csv")

# -------------------------
# Feature Engineering
# -------------------------

df["MA50"] = df["Close"].rolling(window=50).mean()
df["MA200"] = df["Close"].rolling(window=200).mean()

# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("🚀 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Dataset",
        "Visualizations",
        "Model Performance",
        "Forecasting",
        "Conclusion",
    ],
)

# -------------------------
# Home Page
# -------------------------

if page == "Home":

    st.title("🚗 Tesla Stock Price Prediction Using Deep Learning")

    st.markdown("""
    ### Project Overview

    This project predicts Tesla stock prices using Deep Learning models.

    ### Features

    - Data Analysis
    - Data Visualization
    - Feature Engineering
    - SimpleRNN Model
    - LSTM Model
    - GridSearchCV Hyperparameter Tuning
    - Multi-Step Forecasting
    - Model Evaluation
    - Performance Comparison
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Dataset Rows", len(df))

    with c2:
        st.metric("Dataset Columns", len(df.columns))

    with c3:
        st.metric("Models Used", 2)

# -------------------------
# Dataset Page
# -------------------------

elif page == "Dataset":

    st.title("📊 Dataset Information")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe())

# -------------------------
# Visualization Page
# -------------------------

elif page == "Visualizations":

    st.title("📈 Tesla Stock Visualizations")

    # Closing Price

    st.subheader("Tesla Closing Price Trend")

    fig1, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df["Close"])

    ax1.set_title("Tesla Closing Price")

    ax1.set_xlabel("Days")

    ax1.set_ylabel("Price")

    st.pyplot(fig1)

    # Moving Average

    st.subheader("Moving Average Analysis")

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(df["Close"], label="Close Price")

    ax2.plot(df["MA50"], label="50-Day MA")

    ax2.plot(df["MA200"], label="200-Day MA")

    ax2.set_title("Tesla Price with Moving Averages")

    ax2.legend()

    st.pyplot(fig2)

# -------------------------
# Model Performance Page
# -------------------------

elif page == "Model Performance":

    st.title("🤖 Model Performance Comparison")

    performance_df = pd.DataFrame(
        {
            "Metric": ["MSE", "RMSE", "MAE", "R² Score"],
            "SimpleRNN": [273.76, 16.55, 10.32, 0.9522],
            "LSTM": [554.95, 23.56, 16.03, 0.9032],
        }
    )

    st.dataframe(performance_df)

    st.subheader("Best Model")

    st.success("SimpleRNN achieved better performance than LSTM.")

    st.subheader("GridSearchCV Best Parameters")

    grid_df = pd.DataFrame(
        {"Parameter": ["Units", "Dropout Rate"], "Best Value": [25, 0.1]}
    )

    st.dataframe(grid_df)

# -------------------------
# Forecasting Page
# -------------------------

elif page == "Forecasting":

    st.title("🔮 Multi-Step Forecasting")

    st.subheader("1-Day Forecast")

    st.metric("Predicted Tesla Price", "670.63")

    st.subheader("5-Day Forecast")

    forecast_5 = pd.DataFrame(
        {
            "Day": [1, 2, 3, 4, 5],
            "Predicted Price": [670.63, 696.78, 651.11, 641.31, 610.62],
        }
    )

    st.dataframe(forecast_5)

    st.success("The model predicts Tesla stock prices for the next 5 trading days.")

    st.subheader("10-Day Forecast")

    forecast_10 = pd.DataFrame(
        {
            "Day": list(range(1, 11)),
            "Predicted Price": [
                670.63,
                696.78,
                651.11,
                641.31,
                610.62,
                604.81,
                568.89,
                568.29,
                517.13,
                546.57,
            ],
        }
    )

    st.dataframe(forecast_10)

    st.success("The model predicts Tesla stock prices for the next 10 trading days.")

    st.subheader("Forecast Trend")

    st.line_chart(forecast_10.set_index("Day"))

    st.info(
        "Forecasts were generated using the trained "
        "SimpleRNN model for 1-day, 5-day, and "
        "10-day future stock price prediction."
    )

# -------------------------
# Conclusion Page
# -------------------------

elif page == "Conclusion":

    st.title("✨ Conclusion")

    st.markdown("""
    ### Project Summary

    - Tesla stock data was analyzed using Exploratory Data Analysis (EDA).
    - Feature Engineering techniques such as Moving Averages were applied.
    - Deep Learning models (SimpleRNN and LSTM) were developed.
    - GridSearchCV was used for hyperparameter tuning.
    - Multi-step forecasting was performed for 1-day, 5-day, and 10-day horizons.
    - SimpleRNN achieved the best overall performance.

    ### Best Hyperparameters

    - Units: 25
    - Dropout Rate: 0.1

    ### Final Result

    **Best Model: SimpleRNN**
    """)
