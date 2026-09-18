import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("polynomial_regression_ac_fan_model.pkl")

poly = pipeline["poly"]
model = pipeline["model"]

st.title("Electric Bill Prediction")

st.write(
    "Predict your electric bill using AC and Fan electricity consumption."
)

ac_units = st.number_input(
    "AC Units",
    min_value=0.0,
    value=100.0,
    step=1.0
)

fan_units = st.number_input(
    "Fan Units",
    min_value=0.0,
    value=50.0,
    step=1.0
)

if st.button("Predict Electric Bill"):

    # Validate inputs
    if ac_units <= 0 or fan_units <= 0:

        st.error(
            "AC Units and Fan Units must both be greater than 0."
        )

    else:

        input_data = pd.DataFrame({
            "AC_Units": [ac_units],
            "Fan_Units": [fan_units]
        })

        input_poly = poly.transform(input_data)

        prediction = model.predict(input_poly)

        st.success(
            f"Predicted Electric Bill: ₹{prediction[0]:.2f}"
        )
