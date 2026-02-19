import streamlit as st
import requests

st.set_page_config(page_title="Virtual Health Assistant", page_icon="💊")

st.title("💊 Virtual Health Assistant with XAI Feedback")
st.write("Enter your health details to get a prediction.")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
glucose_level = st.number_input("Glucose Level", min_value=50, max_value=300, value=90)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=180)

if st.button("Predict"):
    data = {
        "features": [age, blood_pressure, glucose_level, cholesterol]
    }
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=data)
        result = response.json()
        if "prediction" in result:
            if result["prediction"] == 1:
                st.error("⚠️ Risk of Disease Detected!")
            else:
                st.success("✅ You are Healthy!")
        else:
            st.warning(f"Error: {result.get('error')}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
