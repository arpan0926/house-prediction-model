import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model

@st.cache_resource
def load_model():
    return joblib.load('advanced_house_model.pkl')

model = load_model()

# 2. Build the Web App UI
st.title("🏡 Advanced House Price Predictor")
st.write("Enter the details of the house below to get an estimated market value.")

# Create layout columns for a cleaner, side-by-side interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Details")

    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=2000)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=100000, value=5000)

with col2:
    st.subheader("Location & Condition")
    
    yr_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
    zipcode = st.selectbox("Zipcode", options=['98001', '98002', '98003', '98004'])
    condition = st.selectbox("Condition", options=['Poor', 'Fair', 'Good', 'Excellent'])
    view = st.selectbox("View", options=['None', 'City', 'Water'])

st.markdown("---")

# 3. Prediction Logic

if st.button("Predict Price 💰", type="primary"):
    
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'yr_built': [yr_built],
        'zipcode': [zipcode],
        'condition': [condition],
        'view': [view]
    })

    try:
        # Pass the DataFrame to the model's predict function
        prediction = model.predict(input_data)[0]
        
        # Display the result beautifully using a Streamlit metric component
        st.success("Prediction Complete!")
        st.metric(label="Estimated House Price", value=f"${prediction:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Double-check that your 'advanced_house_model.pkl' is in the exact same folder as this app.py file.")
