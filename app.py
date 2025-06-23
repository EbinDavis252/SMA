import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

# ========== Page Config ==========
st.set_page_config(page_title="Adani Ports Stock Dashboard", layout="wide", page_icon="📈")

# ========== Custom Styling ==========
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        h1, h2, h3 { color: #2e4053; }
        .stSidebar { background-color: #D6EAF8; }
    </style>
""", unsafe_allow_html=True)

# ========== Sidebar ==========
st.sidebar.title("📊 Dashboard Settings")
uploaded_file = st.sidebar.file_uploader("Upload ADANIPORTS CSV", type=["csv"])

show_eda = st.sidebar.checkbox("Show EDA Summary", True)
show_visuals = st.sidebar.checkbox("Show Visualizations", True)
show_metrics = st.sidebar.checkbox("Show Key Metrics", True)

# ========== Load Data ==========
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(subset=['Trades'], inplace=True)
    df['Volume'] = winsorize(df['Volume'], limits=[0.01, 0.01])
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['High_Low_Difference'] = df['High'] - df['Low']
    df['Rolling_Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
    df['Rolling_Mean_20'] = df['Daily_Return'].rolling(window=20).mean()
    df['Month'] = df['Date'].dt.month
    return df

if uploaded_file:
    df = load_data(uploaded_file)

    st.title("📈 Adani Ports Stock Market Dashboard")

    if show_eda:
        st.header("🔍 Data Overview")
        st.subheader("First 5 Rows of the Dataset")
        st.dataframe(df.head())

        st.subheader("Dataset Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Shape:", df.shape)
            st.write("Missing Values:")
            st.write(df.isnull().sum())
        with col2:
            st.write("Data Types:")
            st.write(df.dtypes)

        st.subheader("Descriptive Statistics")
        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

    if show_visuals:
        st.header("📉 Visual Explorations")

        st.subheader("Price Distribution Histograms")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        df[['Open', 'High', 'Low', 'Close', 'Volume']].hist(bins=30, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Stock Prices Over Time")
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(df['Date'], df['Open'], label='Open', color='blue')
        ax2.plot(df['Date'], df['High'], label='High', color='green')
        ax2.plot(df['Date'], df['Low'], label='Low', color='red')
        ax2.plot(df['Date'], df['Close'], label='Close', color='purple')
        ax2.set_title("Stock Prices Over Time")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("Trading Volume Over Time")
        fig3, ax3 = plt.subplots(figsize=(14, 4))
        ax3.plot(df['Date'], df['Volume'], color='orange')
        ax3.set_title("Trading Volume Over Time")
        ax3.grid(True)
        st.pyplot(fig3)

        st.subheader("Moving Averages")
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        ax4.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        ax4.plot(df['Date'], df['MA5'], label='5-day MA', linestyle='--', color='red')
        ax4.plot(df['Date'], df['MA20'], label='20-day MA', linestyle='-.', color='green')
        ax4.plot(df['Date'], df['MA50'], label='50-day MA', linestyle=':', color='purple')
        ax4.set_title("Close Price with Moving Averages")
        ax4.legend()
        ax4.grid(True)
        st.pyplot(fig4)

        st.subheader("Distribution of Returns & Volume Change")
        fig5, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].hist(df['Daily_Return'].dropna(), bins=50, color='skyblue', edgecolor='black')
        ax[0].set_title('Daily Return')
        ax[1].hist(df['Volume_Change'].dropna(), bins=50, color='salmon', edgecolor='black')
        ax[1].set_title('Volume Change')
        st.pyplot(fig5)

        st.subheader("Correlation Matrix Heatmap")
        corr_cols = ['Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Daily_Return', 'MA5', 'MA20', 'MA50', 'Volume_Change', 'High_Low_Difference']
        corr = df[corr_cols].corr()
        fig6, ax6 = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax6)
        st.pyplot(fig6)

        st.subheader("Volatility Trend Over Time")
        fig7, ax7 = plt.subplots(figsize=(14, 6))
        ax7.plot(df['Date'], df['Rolling_Volatility_20'], color='red')
        ax7.set_title("Rolling Volatility (20 Days)")
        st.pyplot(fig7)

        st.subheader("Volume vs Return Scatter")
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        ax8.scatter(df['Volume_Change'], df['Daily_Return'], alpha=0.5)
        ax8.set_title("Volume Change vs Daily Return")
        st.pyplot(fig8)

        st.subheader("Average Monthly Return")
        monthly_returns = df.groupby('Month')['Daily_Return'].mean()
        fig9, ax9 = plt.subplots()
        ax9.bar(monthly_returns.index, monthly_returns.values, color='lightblue')
        ax9.set_xticks(range(1, 13))
        ax9.set_title("Average Daily Return by Month")
        st.pyplot(fig9)

    if show_metrics:
        st.header("📌 Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Daily Return", f"{df['Daily_Return'].mean():.4f}")
        col2.metric("Volatility (Std Dev)", f"{df['Daily_Return'].std():.4f}")
        col3.metric("Avg Trading Volume", f"{df['Volume'].mean():,.0f}")

else:
    st.warning("📂 Please upload a CSV file to begin.")
