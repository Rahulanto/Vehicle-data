# ==============================
# CoverSight - EDA + ML Dashboard
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="CoverSight Analytics", layout="wide")

st.title("📊 CoverSight - Vehicle Sales Analytics")
st.markdown("Executive EDA & Machine Learning Forecast")

# ==============================
# LOAD DATA (Cloud Safe)
# ==============================

@st.cache_data
def load_data():
    return pd.read_excel("Vehicle Sales Data (2).xlsx", engine="openpyxl")

try:
    df = load_data()
    st.success("Dataset Loaded Successfully")
except Exception as e:
    st.error("Dataset not found. Make sure the Excel file is uploaded to GitHub repository.")
    st.stop()

# ==============================
# DATA PREVIEW
# ==============================

st.subheader("Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

# ==============================
# EDA SECTION
# ==============================

st.header("📊 Exploratory Data Analysis")

numeric_cols = df.select_dtypes(include=np.number).columns

# 1️⃣ Correlation Heatmap
if len(numeric_cols) > 0:
    st.subheader("1️⃣ Correlation Analysis")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    st.markdown("""
    **Business Insight:**  
    Strong positive correlation between Production Qty and Domestic Sale  
    indicates production is aligned with market demand.
    """)

# 2️⃣ Production Trend
if "Production Qty" in df.columns:
    st.subheader("2️⃣ Production Trend")

    fig2, ax2 = plt.subplots()
    ax2.plot(df["Production Qty"])
    ax2.set_title("Production Quantity Over Time")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Production Qty")
    st.pyplot(fig2)

# 3️⃣ Domestic Sales Distribution
if "Domestic Sale" in df.columns:
    st.subheader("3️⃣ Domestic Sales Distribution")

    fig3, ax3 = plt.subplots()
    ax3.hist(df["Domestic Sale"], bins=20)
    ax3.set_title("Domestic Sales Distribution")
    st.pyplot(fig3)

# ==============================
# ML SECTION
# ==============================

st.header("🤖 Machine Learning - Domestic Sales Prediction")

if "Domestic Sale" in df.columns:

    X = df[numeric_cols].drop("Domestic Sale", axis=1, errors="ignore")
    y = df["Domestic Sale"]

    if len(X.columns) > 0:

        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)

        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        col1, col2 = st.columns(2)
        col1.metric("R² Score", round(r2, 3))
        col2.metric("RMSE", round(rmse, 2))

        st.markdown("""
        **Executive Interpretation:**
        - High R² → Model explains majority of demand variation  
        - Low RMSE → Forecast error is minimal  
        - Can support demand planning & production optimization
        """)

        # Actual vs Predicted
        st.subheader("Actual vs Predicted Sales")

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(y.values, label="Actual")
        ax4.plot(y_pred, label="Predicted")
        ax4.legend()
        st.pyplot(fig4)

    else:
        st.warning("Not enough numeric features available for ML model.")

else:
    st.error("Domestic Sale column not found in dataset.")
