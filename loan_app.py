import streamlit as st
import cloudpickle
import pandas as pd
import os
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize app
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# Load models with detailed error reporting
@st.cache_resource
def load_models():
    model_files = {
        'model': 'final_model_pipeline.pkl',
        'kmeans': 'kmeans_location.pkl', 
        'cluster_states': 'cluster_states.pkl'
    }
    
    loaded_models = {}
    
    for name, filename in model_files.items():
        try:
            if not os.path.exists(filename):
                if name == 'model':  # Main model is required
                    st.error(f"❌ Required file not found: {filename}")
                    return None, None, None
                else:  # KMeans files are optional
                    st.warning(f"⚠️ Optional file not found: {filename}")
                    loaded_models[name] = None
                    continue
                    
            st.write(f"🔄 Loading {name} from {filename}...")
            with open(filename, 'rb') as f:
                loaded_models[name] = cloudpickle.load(f)
            st.success(f"✅ {name} loaded successfully")
            
        except Exception as e:
            if name == 'model':  # Main model is required
                st.error(f"❌ Failed to load {name} from {filename}: {str(e)}")
                return None, None, None
            else:  # KMeans files are optional
                st.warning(f"⚠️ Failed to load optional {name}: {str(e)}")
                loaded_models[name] = None
    
    return loaded_models['model'], loaded_models.get('kmeans'), loaded_models.get('cluster_states')

# Load models with progress indication
st.write("🔄 Loading machine learning models...")
model, kmeans, cluster_states = load_models()

# Check if main model loaded successfully
if model is None:
    st.error("🚫 Main model failed to load. The app cannot continue.")
    st.stop()

# Check if KMeans models loaded
kmeans_available = kmeans is not None and cluster_states is not None
if not kmeans_available:
    st.warning("""
    ⚠️ Location clustering models not available. 
    The app will use default values for bank state.
    You can still use the app for predictions!
    """)

st.success("🎉 Main model loaded successfully! App is ready.")

# Helper functions
def calculate_age(birthdate, creationdate):
    return creationdate.year - birthdate.year - (
        (creationdate.month, creationdate.day) < (birthdate.month, birthdate.day)
    )

def get_bank_state(longitude, latitude):
    if kmeans_available:
        try:
            cluster = kmeans.predict([[longitude, latitude]])[0]
            return cluster_states.get(cluster, "Unknown")
        except Exception as e:
            st.error(f"Error predicting bank state: {e}")
            return "Default_State"
    else:
        # Return a default state if KMeans is not available
        return "Default_State"

# App title
st.title("🏦 Loan Default Prediction App built by Eucharia")
st.write("Predict the likelihood of loan default using customer demographics, loan history, and location data.")

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
        ["GT Bank","Sterling Bank","Fidelity Bank","Access Bank","EcoBank",
         "FCMB","Skye Bank","UBA","Zenith Bank","Diamond Bank","First Bank",
         "Union Bank","Stanbic IBTC","Standard Chartered","Heritage Bank",
         "Keystone Bank","Unity Bank","Wema Bank"]
    )
    employment_status_clients = st.sidebar.selectbox(
        "Employment Status",
        ["Permanent","Student","Self-Employed","Unemployed","Retired","Contract"]
    )

    longitude = st.sidebar.number_input("Longitude (GPS)", value=0.0, format="%.6f")
    latitude = st.sidebar.number_input("Latitude (GPS)", value=0.0, format="%.6f")

    birthdate = st.sidebar.date_input("Birthdate")
    creationdate = st.sidebar.date_input("Loan Creation Date")

    # Calculate derived features
    age = calculate_age(birthdate, creationdate)
    
    # Only show bank state input if KMeans is available, otherwise use default
    if kmeans_available:
        bank_state = get_bank_state(longitude, latitude)
        st.sidebar.write(f"**Predicted Bank State:** {bank_state}")
    else:
        bank_state = "Default_State"
        st.sidebar.write("**Bank State:** Using default value (location clustering not available)")

    st.sidebar.write(f"**Calculated Age:** {age}")

    input_df = pd.DataFrame([{
        "loannumber": loannumber,
        "loanamount": loanamount,
        "totaldue": totaldue,
        "termdays": termdays,
        "bank_account_type": bank_account_type,
        "bank_name_clients": bank_name_clients,
        "employment_status_clients": employment_status_clients,
        "age": age,
        "bank_state": bank_state
    }])

    if st.button("🔍 Predict"):
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            st.subheader("Prediction Results")
            st.write("✅ **Good (No Default)**" if pred == 1 else "⚠️ **Bad (Default)**")
            st.write(f"**Probability of Default:** {proba[0]:.2f}")
            st.write(f"**Probability of No Default:** {proba[1]:.2f}")
            
            # Visualize probabilities
            st.progress(proba[1])
            st.write(f"Confidence: {max(proba):.1%}")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# BATCH PREDICTION #
else:
    st.subheader("Batch Prediction")
    st.write("Upload demographics and performance files (merged on `customerid`).")
    
    if not kmeans_available:
        st.warning("⚠️ Location clustering not available. Using default bank states for batch processing.")

    demo_file = st.file_uploader("Upload Demographics File", type=["csv", "xlsx"])
    perf_file = st.file_uploader("Upload Performance File", type=["csv", "xlsx"])

    if demo_file and perf_file:
        def load_file(f): 
            if f.name.endswith(".csv"):
                return pd.read_csv(f)
            else:
                return pd.read_excel(f)

        try:
            demo_df = load_file(demo_file)
            perf_df = load_file(perf_file)
            
            # Check if customerid exists in both files
            if 'customerid' not in demo_df.columns or 'customerid' not in perf_df.columns:
                st.error("Error: 'customerid' column not found in one or both files.")
            else:
                df = pd.merge(demo_df, perf_df, on="customerid", how="inner")
                
                if df.empty:
                    st.error("No matching records found. Check if customerid values match in both files.")
                else:
                    # Process dates
                    df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
                    df["creationdate"] = pd.to_datetime(df["creationdate"], errors="coerce")

                    # Calculate age
                    df["age"] = df.apply(
                        lambda r: calculate_age(r["birthdate"], r["creationdate"])
                        if pd.notnull(r["birthdate"]) and pd.notnull(r["creationdate"]) else None,
                        axis=1
                    )

                    # Get bank state - use KMeans if available, otherwise default
                    if kmeans_available:
                        df["bank_state"] = df.apply(
                            lambda r: get_bank_state(r["longitude_gps"], r["latitude_gps"])
                            if pd.notnull(r["longitude_gps"]) and pd.notnull(r["latitude_gps"]) else "Unknown",
                            axis=1
                        )
                    else:
                        df["bank_state"] = "Default_State"

                    expected_features = [
                        "loannumber", "loanamount", "totaldue", "termdays",
                        "bank_account_type", "bank_name_clients", "employment_status_clients",
                        "age", "bank_state"
                    ]

                    # Check if all expected features are present
                    missing_features = [f for f in expected_features if f not in df.columns]
                    if missing_features:
                        st.error(f"Missing required columns: {missing_features}")
                    else:
                        X = df[expected_features]
                        
                        try:
                            preds = model.predict(X)
                            probas = model.predict_proba(X)

                            df["prediction"] = ["Good" if p == 1 else "Default" for p in preds]
                            df["prob_default"] = probas[:, 0]
                            df["prob_no_default"] = probas[:, 1]

                            st.subheader("Prediction Results")
                            st.write(f"Processed {len(df)} records")
                            
                            # Show summary statistics
                            good_loans = sum(preds == 1)
                            default_loans = sum(preds == 0)
                            st.write(f"✅ Good Loans: {good_loans}")
                            st.write(f"⚠️ Default Loans: {default_loans}")
                            st.write(f"📊 Default Rate: {default_loans/len(df):.1%}")
                            
                            st.write("Preview of results:")
                            st.write(df.head())
                            
                            st.download_button(
                                "⬇️ Download Predictions",
                                df.to_csv(index=False).encode("utf-8"),
                                "loan_predictions.csv",
                                "text/csv"
                            )
                        except Exception as e:
                            st.error(f"Batch prediction failed: {e}")
        except Exception as e:
            st.error(f"Error processing uploaded files: {e}")