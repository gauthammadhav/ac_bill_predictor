import streamlit as st
import pandas as pd
import joblib

# Load model
pipeline = joblib.load("polynomial_regression_ac_model.pkl")

poly = pipeline["poly"]
model = pipeline["model"]

st.title("Electric Bill Prediction")

st.write(
    "Predict your electric bill based on AC electricity consumption."
)

ac_units = st.number_input(
    "Enter AC Units",
    min_value=0.0,
    value=100.0,
    step=1.0
)

if st.button("Predict Electric Bill"):

    if ac_units <= 9 or ac_units >= 150:

        st.error(
            "AC Units out of range."
        )

    else:

        input_data = pd.DataFrame({
            "AC_Units": [ac_units]
        })

        input_poly = poly.transform(input_data)

        prediction = model.predict(input_poly)

        st.success(
            f"Predicted Electric Bill: ₹{prediction[0]:.2f}"
        )
