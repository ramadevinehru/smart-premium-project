import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np

# Load trained XGBoost model
model = joblib.load("./xgboost_model.pkl")

train_sample = pd.read_csv("./train.csv", nrows=5000)

if 'Policy Start Date' in train_sample.columns:
    train_sample['Policy Start Date'] = pd.to_datetime(train_sample['Policy Start Date'], errors='coerce')
    train_sample['Policy Start Year'] = train_sample['Policy Start Date'].dt.year
    train_sample.drop(['Policy Start Date'], axis=1, inplace=True)

columns_to_drop = ['Premium Amount', 'Customer Feedback']
train_sample.drop([col for col in columns_to_drop if col in train_sample.columns], axis=1, inplace=True)

# Apply one-hot encoding to get expected feature columns
train_sample_encoded = pd.get_dummies(train_sample)
expected_features = train_sample_encoded.columns.tolist()

# Function to dynamically match one-hot encoded categorical inputs
def create_categorical_dict(columns, user_selections):
    cat_dict = {}
    for col in columns:
        if any(col.startswith(key + "_") for key in user_selections):
            for key, value in user_selections.items():
                if col == f"{key}_{value}":
                    cat_dict[col] = 1
    return cat_dict

def get_user_input():
    st.sidebar.header("Enter Customer Details")
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    annual_income = st.sidebar.number_input("Annual Income", min_value=10000, max_value=1000000, value=50000)
    credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=650)
    health_score = st.sidebar.slider("Health Score", min_value=1, max_value=10, value=5)
    previous_claims = st.sidebar.number_input("Previous Claims", min_value=0, max_value=10, value=0)
    vehicle_age = st.sidebar.number_input("Vehicle Age", min_value=0, max_value=30, value=5)

    # Categorical Inputs
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    smoking_status = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
    policy_type = st.sidebar.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
    education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.sidebar.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
    exercise_frequency = st.sidebar.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    property_type = st.sidebar.selectbox("Property Type", ["House", "Apartment", "Condo"])

    policy_start_year = st.sidebar.number_input("Policy Start Year", min_value=2000, max_value=2025, value=2023)

    # Numerical data dictionary
    user_data = {
        "Age": age,
        "Annual Income": annual_income,
        "Credit Score": credit_score,
        "Health Score": health_score,
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Policy Start Year": policy_start_year,  # ✅ corrected key name
    }

    # One-hot encoded categorical dictionary
    user_selections = {
        "Gender": gender,
        "Marital Status": marital_status,
        "Smoking Status": smoking_status,
        "Policy Type": policy_type,
        "Location": location,
        "Education Level": education_level,
        "Occupation": occupation,
        "Exercise Frequency": exercise_frequency,
        "Property Type": property_type,
    }

    categorical_data = create_categorical_dict(expected_features, user_selections)

    user_input_df = pd.DataFrame([{**user_data, **categorical_data}])
    user_input_df = user_input_df.reindex(columns=expected_features, fill_value=0)

    return user_input_df

st.title("Insurance Premium Prediction App")
st.write("This app predicts the insurance premium based on customer details.")

user_input_df = get_user_input()

if st.sidebar.button("Predict Premium"):
    prediction = model.predict(user_input_df)

    # Use actual mean and std from original train.csv before scaling
    premium_mean = 1234.567   # Replace with your real value
    premium_std = 456.789     # Replace with your real value

    actual_premium = (prediction[0] * premium_std) + premium_mean

    st.success(f"### Predicted Premium Amount: **₹{actual_premium:,.2f}**")

st.sidebar.write("**Adjust input values and click 'Predict Premium' to see the result.**")
