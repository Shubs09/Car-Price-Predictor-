import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------
# Load model and columns used during training
# ----------------------------------------------------------
model = joblib.load("car_price.pkl")
columns = joblib.load("columns.pkl")   # contains feature names used while training

st.title("🚗 Car Price Prediction App")
st.markdown("### Predict the selling price of a used car based on its details")

# ----------------------------------------------------------
# User Input Section
# ----------------------------------------------------------
car_name = st.selectbox("Car Brand", ["Maruti", "Hyundai", "Honda", "Tata", "Mahindra", "Ford", "Toyota", "BMW", "Audi"], key="brand")
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2017, key="year")
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=200000, value=30000, key="km")
owner = st.selectbox("Number of Previous Owners", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"], key="owner")
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=18.0, key="mileage")
engine = st.number_input("Engine (CC)", min_value=600, max_value=5000, value=1200, key="engine")
max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=400.0, value=90.0, key="power")
seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8], key="seats")
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"], key="fuel")
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"], key="seller")
transmission = st.selectbox("Transmission", ["Manual", "Automatic"], key="transmission")

# ----------------------------------------------------------
# Create Input DataFrame
# ----------------------------------------------------------
input_data = pd.DataFrame({
    'name': [car_name],
    'year': [year],
    'km_driven': [km_driven],
    'owner': [owner],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission]
})

# ----------------------------------------------------------
# One-Hot Encode input like training
# ----------------------------------------------------------
input_encoded = pd.get_dummies(input_data)

# Add missing columns and reorder to match training data
for col in columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[columns]

# ----------------------------------------------------------
# Predict Price
# ----------------------------------------------------------
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_encoded)[0]
        st.success(f"💰 Estimated Selling Price: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
