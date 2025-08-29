import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# -----------------------------
# Load the trained model and scaler
# -----------------------------
kmeans = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(page_title="Customer Segmentation App", layout="centered")

st.title("🛒 Customer Segmentation with ML")
st.write("""
This app segments customers based on their **Annual Income** and **Spending Score** 
using **K-Means Clustering**.
""")

# -----------------------------
# Sidebar Inputs (Single Customer Prediction)
# -----------------------------
st.sidebar.header("📌 Input Customer Data")

age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)
annual_income = st.sidebar.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending_score = st.sidebar.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.sidebar.button("🔍 Predict Segment"):
    # Prepare the input
    new_data = np.array([[annual_income, spending_score]])
    new_scaled = scaler.transform(new_data)
    cluster = kmeans.predict(new_scaled)[0]

    # Define cluster names
    cluster_names = {
        0: "🟢 Budget Shoppers",
        1: "🔵 Premium Customers",
        2: "🟡 Average Shoppers",
        3: "🟣 Young Spenders",
        4: "🔴 Elderly Careful Spenders"
    }

    st.success(f"✅ This customer belongs to **{cluster_names[cluster]}** (Cluster {cluster})")

    # -----------------------------
    # Visualization
    # -----------------------------
    st.subheader("📊 Cluster Visualization (2D Projection)")
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot centroids
    sns.scatterplot(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        s=200, color="red", label="Centroids", ax=ax
    )
    # Plot new customer
    ax.scatter(new_scaled[:, 0], new_scaled[:, 1], c="blue", s=150, label="New Customer", marker="X")
    plt.xlabel("Annual Income (scaled)")
    plt.ylabel("Spending Score (scaled)")
    plt.legend()
    st.pyplot(fig)

# -----------------------------
# Bulk Prediction with CSV
# -----------------------------
st.subheader("📂 Upload CSV to Cluster Multiple Customers")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### 🔎 Preview of Uploaded Data")
    st.dataframe(data.head())

    # Required columns check
    required_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    if all(col in data.columns for col in required_cols):
        # Scale and Predict
        scaled_data = scaler.transform(data[required_cols])
        data["Cluster"] = kmeans.predict(scaled_data)

        cluster_names = {
            0: "🟢 Budget Shoppers",
            1: "🔵 Premium Customers",
            2: "🟡 Average Shoppers",
            3: "🟣 Young Spenders",
            4: "🔴 Elderly Careful Spenders"
        }
        data["Cluster Name"] = data["Cluster"].map(cluster_names)

        st.write("### ✅ Clustered Data")
        st.dataframe(data.head(10))

        # Save and Download
        csv_out = data.to_csv(index=False)
        st.download_button("📥 Download Clustered Data", csv_out, file_name="clustered_customers.csv")

         # Visualization
        st.subheader("Customer Segmentation Visualization")
        fig = px.scatter(
            data,
            x="Age",
            y="Annual Income (k$)",
            color="Cluster Name",
            size="Spending Score (1-100)",
            hover_data=["CustomerID", "Spending Score (1-100)"],
            title="Customer Segments (Age vs Income)"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f"⚠️ CSV must contain columns: {required_cols}")
