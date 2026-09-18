import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Electric Bill Predictor",
    page_icon="⚡",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0d12;
        color: #f5f5f5;
    }

    /* Main container */
    .block-container {
        max-width: 850px;
        padding-top: 4rem;
        padding-bottom: 4rem;
    }

    /* Remove Streamlit header space */
    header {
        visibility: hidden;
    }

    /* Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -2px;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Main card */
    .prediction-card {
        background: #151820;
        border: 1px solid #292d38;
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
    }

    /* Section label */
    .section-label {
        color: #a1a1aa;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    /* Input */
    div[data-baseweb="input"] {
        background-color: #20232d;
        border: 1px solid #343946;
        border-radius: 12px;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #ffffff;
    }

    input {
        color: white !important;
        font-size: 1.1rem !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 12px;
        border: none;
        background: white;
        color: #0b0d12;
        font-size: 1rem;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #e5e7eb;
        transform: translateY(-2px);
    }

    /* Result card */
    .result-card {
        background: #151820;
        border: 1px solid #343946;
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-label {
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    .result-value {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0.5rem 0;
    }

    .result-description {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #525866;
        font-size: 0.8rem;
        margin-top: 3rem;
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model
# -----------------------------

pipeline = joblib.load(
    "polynomial_regression_ac_model.pkl"
)

poly = pipeline["poly"]
model = pipeline["model"]


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title">⚡ Electric Bill Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate your monthly electricity bill using AC consumption.'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Input Card
# -----------------------------

st.markdown(
    '<div class="prediction-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-label">AC Consumption</div>',
    unsafe_allow_html=True
)

ac_units = st.number_input(
    "AC Units",
    min_value=0.0,
    value=72.0,
    step=1.0,
    label_visibility="collapsed"
)

st.write("")

predict = st.button("Predict Electric Bill")

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Prediction
# -----------------------------

if predict:

    input_data = pd.DataFrame({
        "AC_Units": [ac_units]
    })

    input_poly = poly.transform(input_data)

    prediction = model.predict(input_poly)

    bill = prediction[0]

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-label">
                Estimated Monthly Bill
            </div>

            <div class="result-value">
                ₹{bill:,.2f}
            </div>

            <div class="result-description">
                Estimated for {ac_units:.0f} AC units of consumption
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Footer
# -----------------------------

st.markdown(
    """
    <div class="footer">
        Polynomial Regression · Degree 2
    </div>
    """,
    unsafe_allow_html=True
)
