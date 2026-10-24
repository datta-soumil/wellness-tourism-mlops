"""Streamlit application for Wellness Tourism Package predictions."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Visit with Us | Package Predictor", page_icon="✈️")
MODEL_PATH = Path(__file__).with_name("model.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("✈️ Wellness Tourism Package Predictor")
st.caption("Estimate the likelihood that a customer will buy the new package.")
if not MODEL_PATH.exists():
    st.error("The trained model is not available yet. Run the GitHub Actions pipeline first.")
    st.stop()
model = load_model()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 35)
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City tier", [1, 2, 3], index=1)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        people = st.number_input("Number of persons visiting", 1, 10, 2)
        followups = st.number_input("Number of follow-ups", 0, 20, 4)
        product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        property_star = st.selectbox("Preferred property star", [3.0, 4.0, 5.0], index=1)
    with col2:
        pitch_duration = st.number_input("Pitch duration (minutes)", 1.0, 120.0, 15.0)
        marital = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
        trips = st.number_input("Number of trips per year", 0.0, 50.0, 3.0)
        passport = st.selectbox("Has passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
        pitch_score = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=2)
        own_car = st.selectbox("Owns a car", [0, 1], format_func=lambda x: "Yes" if x else "No")
        children = st.number_input("Children visiting", 0.0, 10.0, 0.0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = st.number_input("Monthly income", 1.0, 500000.0, 25000.0)
    submitted = st.form_submit_button("Predict purchase likelihood")

if submitted:
    customer = pd.DataFrame([{
        "Age": age, "TypeofContact": contact, "CityTier": city_tier,
        "DurationOfPitch": pitch_duration, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": people, "NumberOfFollowups": followups,
        "ProductPitched": product, "PreferredPropertyStar": property_star,
        "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": passport,
        "PitchSatisfactionScore": pitch_score, "OwnCar": own_car,
        "NumberOfChildrenVisiting": children, "Designation": designation,
        "MonthlyIncome": income,
    }])
    probability = model.predict_proba(customer)[0, 1]
    st.metric("Purchase probability", f"{probability:.1%}")
    if probability >= 0.5:
        st.success("Recommended: prioritize this customer for the package campaign.")
    else:
        st.info("Recommended: lower priority; consider a different engagement strategy.")
    with st.expander("Submitted customer data"):
        st.dataframe(customer, use_container_width=True)
