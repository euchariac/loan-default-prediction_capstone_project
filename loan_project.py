import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load the model 
model = joblib.load("final_gb_pipeline.pkl")
kmeans = joblib.load("kmeans_location.pkl")
cluster_states = joblib.load("cluster_states.pkl")

st.title("Loan Default Prediction App")
st.write("Predict the likelihood of loan default using customer demographics and loan history.")


# ----------------- Helper Functions ----------------- #
def clean_columns(df):
    """Standardize column names to lowercase, no spaces."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def merge_datasets(demo, perf, previousloan):
    demo = clean_columns(demo)
    perf = clean_columns(perf)
    previousloan = clean_columns(previousloan)

    merged = demo.merge(perf, on="customerid", how="left")
    merged = merged.merge(previousloan, on="customerid", how="left")
    return clean_columns(merged)

def engineer_features(df, max_duration=365):
    # Ensure required columns exist
    required = [
        "loanamount", "totaldue", "termdays", "num_loans",
        "num_fully_repaid", "num_ongoing", "total_loan_amount", "total_due_amount"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0

    # Age 
    if "creationdate" in df.columns and "birthdate" in df.columns:
        df["creationdate"] = pd.to_datetime(df["creationdate"], errors="coerce")
        df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
        df["age"] = (df["creationdate"] - df["birthdate"]).dt.days // 365

    # Ratios / Terms 
    df["avg_repayment_ratio"] = np.where(df["loanamount"] > 0,
                                         df["totaldue"] / df["loanamount"], 0)
    df["avg_loan_term"] = df["termdays"]
    df["avg_loan_duration"] = df["termdays"]

    # Credit Score components 
    df["repayment_rate"] = np.where(df["num_loans"] > 0,
                                    df["num_fully_repaid"] / df["num_loans"], 0)
    df["repayment_ratio_score"] = 1 - df["avg_repayment_ratio"]
    df["ongoing_score"] = 1 - np.where(df["num_loans"] > 0,
                                       df["num_ongoing"] / df["num_loans"], 0)
    df["duration_score"] = 1 - (df["avg_loan_duration"] / max_duration)

    # Weighted Credit Score 
    df["credit_score"] = (
        0.5 * df["repayment_rate"] +
        0.3 * df["repayment_ratio_score"] +
        0.1 * df["ongoing_score"] +
        0.1 * df["duration_score"]
    )

    # Clustering bank_state from GPS 
    if "latitude_gps" in df.columns and "longitude_gps" in df.columns:
        coords = df[["latitude_gps", "longitude_gps"]].dropna()
        if not coords.empty:
            clusters = kmeans.predict(coords)
            df.loc[coords.index, "location_cluster"] = clusters
            df.loc[coords.index, "bank_state"] = df.loc[coords.index, "location_cluster"].map(cluster_states)
        else:
            df["bank_state"] = "Unknown"
    else:
        df["bank_state"] = "Unknown"

    return df


# ----------------- Mode Selection ----------------- #
mode = st.radio("Select Prediction Mode:", ["Single Prediction", "Batch Prediction"])


# ----------------- SINGLE PREDICTION ----------------- #
if mode == "Single Prediction":
    st.sidebar.header("Customer & Loan Inputs")

    # Inputs
    loanamount = st.sidebar.number_input("Loan Amount", min_value=0, step=1000)
    totaldue = st.sidebar.number_input("Total Due", min_value=0, step=1000)
    termdays = st.sidebar.number_input("Loan Term (days)", min_value=0, step=1)

    birthdate = st.sidebar.date_input("Birthdate", min_value=datetime(1900, 1, 1))
    creationdate = st.sidebar.date_input("Loan Creation Date", min_value=datetime(2000, 1, 1))

    bank_account_type = st.sidebar.selectbox("Bank Account Type", ["Savings", "Current", "Fixed Deposit", "Other"])
    bank_name_clients = st.sidebar.selectbox("Bank Name", 
                                             ["GT Bank", "Sterling Bank", "Fidelity Bank", "Access Bank", "EcoBank",
                                              "FCMB", "Skye Bank", "UBA", "Zenith Bank", "Diamond Bank", "First Bank",
                                              "Union Bank", "Stanbic IBTC", "Standard Chartered", "Heritage Bank",
                                              "Keystone Bank", "Unity Bank", "Wema Bank"])
    employment_status_clients = st.sidebar.selectbox("Employment Status", 
                                                     ["Permanent", "Student", "Self-Employed", "Unemployed", "Retired", "Contract"])

    longitude_gps = st.sidebar.number_input("Longitude", 3.00, 15.00, 7.00)
    latitude_gps = st.sidebar.number_input("Latitude", 4.00, 14.00, 9.00)

    num_loans = st.sidebar.number_input("Previous Loans", min_value=0, step=1)
    num_fully_repaid = st.sidebar.number_input("Fully Repaid Loans", min_value=0, step=1)
    num_ongoing = st.sidebar.number_input("Ongoing Loans", min_value=0, step=1)
    total_loan_amount = st.sidebar.number_input("Total Loan Amount (Hist.)", min_value=0, step=1000)
    total_due_amount = st.sidebar.number_input("Total Due Amount (Hist.)", min_value=0, step=1000)

    # Put into dataframe
    input_data = {
        "loanamount": loanamount,
        "totaldue": totaldue,
        "termdays": termdays,
        "bank_account_type": bank_account_type,
        "bank_name_clients": bank_name_clients,
        "employment_status_clients": employment_status_clients,
        "num_loans": num_loans,
        "num_fully_repaid": num_fully_repaid,
        "num_ongoing": num_ongoing,
        "total_loan_amount": total_loan_amount,
        "total_due_amount": total_due_amount,
        "birthdate": pd.to_datetime(birthdate),
        "creationdate": pd.to_datetime(creationdate),
        "longitude_gps": longitude_gps,
        "latitude_gps": latitude_gps,
    }

    input_df = pd.DataFrame([input_data])
    input_df = clean_columns(input_df)
    input_df = engineer_features(input_df)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    st.write("Processed Input Data", input_df)

    if st.button("Predict"):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]  # [P(Default=0), P(No Default=1)]

        st.subheader("Prediction Result")
        st.write("Prediction:", "Default" if pred == 0 else "No Default")
        st.write(f"Probability of Default: {proba[0]:.2f}")
        st.write(f"Probability of No Default: {proba[1]:.2f}")


# BATCH PREDICTION #
else:
    st.subheader("Batch Prediction")
    demo_file = st.file_uploader("Upload Demo", type=["csv", "xlsx"])
    perf_file = st.file_uploader("Upload Perf", type=["csv", "xlsx"])
    prev_file = st.file_uploader("Upload Previous Loans", type=["csv", "xlsx"])

    if demo_file and perf_file and prev_file:
        def load_file(f):
            return pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)

        demo = clean_columns(load_file(demo_file))
        perf = clean_columns(load_file(perf_file))
        prev = clean_columns(load_file(prev_file))

        merged = merge_datasets(demo, perf, prev)
        processed = engineer_features(merged)
        processed = processed.reindex(columns=model.feature_names_in_, fill_value=0)

        preds = model.predict(processed)
        probas = model.predict_proba(processed)

        results = processed.copy()
        results["prediction"] = np.where(preds == 0, "Default", "No Default")
        results["prob_default"] = probas[:, 0]      # since 0 = Default
        results["prob_no_default"] = probas[:, 1]   # since 1 = No Default

        st.write("Sample Predictions", results.head())
        st.download_button("Download Results", results.to_csv(index=False).encode("utf-8"),
                           "batch_predictions.csv", "text/csv")
