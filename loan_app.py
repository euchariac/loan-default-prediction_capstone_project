import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime

#  Load Model & Helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "final_model_pipeline.pkl")
KMEANS_PATH = os.path.join(BASE_DIR, "kmeans_location.pkl")
CLUSTER_PATH = os.path.join(BASE_DIR, "cluster_states.pkl")

# Validate model files
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Please ensure 'final_model_pipeline.pkl' is in the same directory.")
else:
    model = joblib.load(MODEL_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    cluster_states = joblib.load(CLUSTER_PATH)

st.title("🏦 Loan Default Prediction App")
st.write("Predict the likelihood of loan default using customer demographics, loan history, and location data.")

# HELPER FUNCTIONS #
def calculate_age(birthdate, creationdate):
    """Calculate age in years based on birthdate and loan creation date."""
    return creationdate.year - birthdate.year - (
        (creationdate.month, creationdate.day) < (birthdate.month, birthdate.day)
    )



def get_bank_state(longitude, latitude):
    """Predict cluster from GPS and map to bank state."""
    cluster = kmeans.predict([[longitude, latitude]])[0]
    return cluster_states.get(cluster, "Unknown")

# APP MODE #
mode = st.radio("Select Prediction Mode:", ["Single Prediction", "Batch Prediction"])

# SINGLE PREDICTION #
if mode == "Single Prediction":
    st.sidebar.header("Customer & Loan Inputs")

    customerid = st.sidebar.text_input("Customer ID")

    loannumber = st.sidebar.number_input("Loan Number", min_value=1, step=1)
    loanamount = st.sidebar.number_input("Loan Amount", min_value=0, step=1000)
    totaldue = st.sidebar.number_input("Total Due", min_value=0, step=1000)
    termdays = st.sidebar.number_input("Loan Term (days)", min_value=1, step=1)

    bank_account_type = st.sidebar.selectbox(
        "Bank Account Type", ["Savings", "Current", "Fixed Deposit", "Other"]
    )
    bank_name_clients = st.sidebar.selectbox(
        "Bank Name",
        [
            "GT Bank", "Sterling Bank", "Fidelity Bank", "Access Bank", "EcoBank",
            "FCMB", "Skye Bank", "UBA", "Zenith Bank", "Diamond Bank", "First Bank",
            "Union Bank", "Stanbic IBTC", "Standard Chartered", "Heritage Bank",
            "Keystone Bank", "Unity Bank", "Wema Bank"
        ]
    )
    employment_status_clients = st.sidebar.selectbox(
        "Employment Status",
        ["Permanent", "Student", "Self-Employed", "Unemployed", "Retired", "Contract"]
    )

    # GPS inputs
    longitude = st.sidebar.number_input("Longitude (GPS)", value=0.0, format="%.6f")
    latitude = st.sidebar.number_input("Latitude (GPS)", value=0.0, format="%.6f")

    # Dates
    birthdate = st.sidebar.date_input("Birthdate")
    creationdate = st.sidebar.date_input("Loan Creation Date")

    # Derived features
    age = calculate_age(birthdate, creationdate)

    bank_state = get_bank_state(longitude, latitude)

    input_data = {
        "customerid": customerid,
        "loannumber": loannumber,
        "loanamount": loanamount,
        "totaldue": totaldue,
        "termdays": termdays,
        "bank_account_type": bank_account_type,
        "bank_name_clients": bank_name_clients,
        "employment_status_clients": employment_status_clients,
        "age": age,
        "bank_state": bank_state
    }

    input_df = pd.DataFrame([input_data])
    st.write("### Processed Input Data", input_df)

    if st.button("🔍 Predict"):
        X_input = input_df.drop(columns=["customerid"])
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        st.subheader("Prediction Result")
        st.write("**Customer ID:**", customerid)
        st.write("**Prediction:**", "✅ Good (No Default)" if pred == 1 else "⚠️ Bad (Default)")
        st.write(f"**Probability of Default:** {proba[0]:.2f}")
        st.write(f"**Probability of No Default:** {proba[1]:.2f}")

# ----------------- BATCH PREDICTION ----------------- #
else:
    st.subheader("📂 Batch Prediction")
    st.write("Upload the **Demographics file** and **Performance file** separately. They will be merged on `customerid`.")

    demo_file = st.file_uploader("Upload Demographics File (CSV/XLSX)", type=["csv", "xlsx"], key="demo")
    perf_file = st.file_uploader("Upload Performance File (CSV/XLSX)", type=["csv", "xlsx"], key="perf")

    if demo_file and perf_file:
        def load_file(f):
            return pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)

        demo_df = load_file(demo_file)
        perf_df = load_file(perf_file)

        df = pd.merge(demo_df, perf_df, on="customerid", how="inner")

        df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
        df["creationdate"] = pd.to_datetime(df["creationdate"], errors="coerce")

        df["age"] = df.apply(
            lambda row: calculate_age(row["birthdate"], row["creationdate"])
            if pd.notnull(row["birthdate"]) and pd.notnull(row["creationdate"])
            else None,
            axis=1,
        )
        df["age_catagory"] = df["age"].apply(lambda x: assign_age_category(x) if pd.notnull(x) else None)
        df["bank_state"] = df.apply(
            lambda row: get_bank_state(row["longitude_gps"], row["latitude_gps"])
            if pd.notnull(row["longitude_gps"]) and pd.notnull(row["latitude_gps"])
            else "Unknown",
            axis=1,
        )

        expected_features = [
            "loannumber", "loanamount", "totaldue", "termdays",
            "bank_account_type", "bank_name_clients", "employment_status_clients",
            "age", "age_catagory", "bank_state"
        ]
        X = df.reindex(columns=expected_features, fill_value=0)

        preds = model.predict(X)
        probas = model.predict_proba(X)

        results = df.copy()
        results["prediction"] = [" Good (No Default)" if p == 1 else "Bad (Default)" for p in preds]
        results["prob_default"] = probas[:, 0]
        results["prob_no_default"] = probas[:, 1]

        st.write("### Sample Predictions", results[["customerid", "prediction", "prob_default", "prob_no_default"]].head())
        st.download_button(
            "⬇️ Download Full Results",
            results.to_csv(index=False).encode("utf-8"),
            "batch_predictions.csv",
            "text/csv"
        )
