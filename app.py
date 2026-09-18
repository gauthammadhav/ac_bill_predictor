import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Billwise — Electric Bill Predictor",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LOAD MODEL
# ============================================================

pipeline = joblib.load("polynomial_regression_ac_model.pkl")

poly = pipeline["poly"]
model = pipeline["model"]

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================
   IMPORT FONT
========================= */

@import url(
    'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap'
);

/* =========================
   GLOBAL
========================= */

.stApp {
    background: #F6F6F3;
    color: #111111;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 900px;
    padding-top: 0rem;
    padding-bottom: 4rem;
}

/* Hide Streamlit elements */

header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* =========================
   NAVBAR
========================= */

.navbar {
    height: 78px;
    display: flex;
    align-items: center;
    justify-content: space-between;

    border-bottom: 1px solid #E2E2DE;

    margin-bottom: 85px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;

    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 700;

    letter-spacing: -0.5px;
}

.brand-icon {
    width: 34px;
    height: 34px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #111111;
    color: #FFFFFF;

    border-radius: 10px;

    font-size: 16px;
}

.nav-label {
    font-size: 12px;
    color: #888888;
}

/* =========================
   HERO
========================= */

.hero {
    text-align: center;
    margin-bottom: 55px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 7px 12px;

    border: 1px solid #DDDDD8;
    border-radius: 100px;

    background: #FFFFFF;

    color: #777777;

    font-size: 11px;
    font-weight: 600;

    letter-spacing: 1.2px;

    margin-bottom: 22px;
}

.status-dot {
    width: 6px;
    height: 6px;

    background: #35A853;

    border-radius: 50%;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;

    font-size: 58px;
    line-height: 1.02;

    font-weight: 600;

    letter-spacing: -3px;

    margin: 0;

    color: #111111;
}

.hero-title span {
    color: #999993;
}

.hero-description {
    max-width: 540px;

    margin: 20px auto 0;

    color: #777777;

    font-size: 15px;

    line-height: 1.6;
}

/* =========================
   MAIN CARD
========================= */

.main-card {
    background: #FFFFFF;

    border: 1px solid #E1E1DC;

    border-radius: 24px;

    padding: 32px;

    box-shadow:
        0 1px 2px rgba(0,0,0,0.02),
        0 10px 35px rgba(0,0,0,0.035);
}

/* =========================
   CARD HEADER
========================= */

.card-top {
    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 35px;
}

.card-label {
    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.5px;

    color: #999999;
}

.card-step {
    width: 28px;
    height: 28px;

    border-radius: 50%;

    border: 1px solid #E0E0DB;

    display: flex;

    align-items: center;
    justify-content: center;

    font-size: 10px;

    color: #888888;
}

/* =========================
   INPUT SECTION
========================= */

.input-heading {
    font-family: 'Space Grotesk', sans-serif;

    font-size: 26px;

    font-weight: 600;

    letter-spacing: -1px;

    margin-bottom: 7px;
}

.input-description {
    color: #888888;

    font-size: 13px;

    margin-bottom: 22px;
}

/* Remove Streamlit label */

div[data-testid="stNumberInput"] label {
    display: none;
}

/* Number input */

div[data-baseweb="input"] {
    height: 68px !important;

    background: #F7F7F5 !important;

    border: 1px solid #DADAD5 !important;

    border-radius: 14px !important;

    box-shadow: none !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #111111 !important;

    box-shadow: none !important;
}

div[data-testid="stNumberInput"] input {
    font-family: 'Space Grotesk', sans-serif !important;

    font-size: 25px !important;

    font-weight: 600 !important;

    color: #111111 !important;
}

/* =========================
   BUTTON
========================= */

.stButton {
    margin-top: 18px;
}

.stButton > button {
    width: 100%;

    height: 56px;

    border-radius: 13px;

    border: none;

    background: #111111;

    color: #FFFFFF;

    font-family: 'DM Sans', sans-serif;

    font-size: 14px;

    font-weight: 600;

    transition:
        transform 0.2s ease,
        background 0.2s ease;
}

.stButton > button:hover {
    background: #292929;

    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0);
}

/* =========================
   RESULT
========================= */

.result {
    margin-top: 30px;

    padding-top: 30px;

    border-top: 1px solid #E8E8E3;

    text-align: center;
}

.result-label {
    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.6px;

    color: #999999;

    margin-bottom: 12px;
}

.result-value {
    font-family: 'Space Grotesk', sans-serif;

    font-size: 54px;

    line-height: 1;

    font-weight: 600;

    letter-spacing: -3px;

    color: #111111;
}

.result-value .currency {
    font-size: 32px;

    vertical-align: 8px;

    margin-right: 3px;

    color: #777777;
}

.result-subtitle {
    color: #888888;

    font-size: 12px;

    margin-top: 12px;
}

/* =========================
   RESULT DETAILS
========================= */

.details {
    display: flex;

    justify-content: center;

    gap: 35px;

    margin-top: 25px;

    padding-top: 20px;

    border-top: 1px solid #EEEEEA;
}

.detail {
    text-align: center;
}

.detail-label {
    font-size: 9px;

    text-transform: uppercase;

    letter-spacing: 1.2px;

    color: #AAAAAA;
}

.detail-value {
    font-size: 13px;

    font-weight: 600;

    color: #333333;

    margin-top: 4px;
}

/* =========================
   EMPTY RESULT
========================= */

.empty-result {
    margin-top: 30px;

    padding: 25px;

    border-radius: 16px;

    background: #F7F7F5;

    border: 1px dashed #DCDCD7;

    text-align: center;
}

.empty-icon {
    font-size: 22px;

    color: #999999;

    margin-bottom: 8px;
}

.empty-title {
    font-size: 13px;

    font-weight: 600;

    color: #444444;
}

.empty-description {
    font-size: 11px;

    color: #999999;

    margin-top: 5px;
}

/* =========================
   MODEL INFORMATION
========================= */

.model-info {
    display: grid;

    grid-template-columns: 1fr 1fr 1fr;

    margin-top: 18px;

    border-top: 1px solid #E2E2DE;

    border-bottom: 1px solid #E2E2DE;
}

.model-item {
    padding: 20px 0;
}

.model-item + .model-item {
    border-left: 1px solid #E2E2DE;

    padding-left: 22px;
}

.model-label {
    font-size: 9px;

    text-transform: uppercase;

    letter-spacing: 1.2px;

    color: #AAAAAA;
}

.model-value {
    font-size: 12px;

    font-weight: 600;

    margin-top: 5px;

    color: #333333;
}

/* =========================
   FOOTER
========================= */

.footer {
    display: flex;

    justify-content: space-between;

    margin-top: 25px;

    color: #AAAAAA;

    font-size: 10px;
}

/* =========================
   MOBILE
========================= */

@media (max-width: 650px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .navbar {
        margin-bottom: 55px;
    }

    .nav-label {
        display: none;
    }

    .hero-title {
        font-size: 42px;
        letter-spacing: -2px;
    }

    .main-card {
        padding: 22px;
    }

    .result-value {
        font-size: 43px;
    }

    .model-info {
        grid-template-columns: 1fr;
    }

    .model-item + .model-item {
        border-left: none;
        border-top: 1px solid #E2E2DE;
        padding-left: 0;
    }

    .footer {
        flex-direction: column;
        gap: 6px;
    }

}

</style>
""", unsafe_allow_html=True)

# ============================================================
# NAVBAR
# ============================================================

st.markdown("""
<div class="navbar">

    <div class="brand">
        <div class="brand-icon">⚡</div>
        Billwise
    </div>

    <div class="nav-label">
        ML-powered electricity estimation
    </div>

</div>
""", unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="eyebrow">
        <div class="status-dot"></div>
        POLYNOMIAL REGRESSION
    </div>

    <h1 class="hero-title">
        Electricity bill,<br>
        <span>made predictable.</span>
    </h1>

    <p class="hero-description">
        Estimate your monthly electricity bill from
        AC consumption using a trained machine learning model.
    </p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN CARD
# ============================================================

st.markdown("""
<div class="main-card">

    <div class="card-top">

        <div class="card-label">
            AC CONSUMPTION
        </div>

        <div class="card-step">
            01
        </div>

    </div>

    <div class="input-heading">
        How much did your AC consume?
    </div>

    <div class="input-description">
        Enter your AC electricity consumption in units.
    </div>

</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT
# ============================================================

ac_units = st.number_input(
    "AC Units",
    min_value=0.0,
    value=72.0,
    step=1.0,
    format="%.0f",
    label_visibility="collapsed"
)

# ============================================================
# BUTTON
# ============================================================

predict = st.button(
    "Calculate estimated bill  →"
)

# ============================================================
# CLOSE CARD / RESULT
# ============================================================

if predict:

    input_data = pd.DataFrame({
        "AC_Units": [ac_units]
    })

    input_poly = poly.transform(input_data)

    prediction = model.predict(input_poly)

    bill = prediction[0]

    st.markdown(f"""
    <div class="main-card">

        <div class="card-top">

            <div class="card-label">
                ESTIMATED BILL
            </div>

            <div class="card-step">
                02
            </div>

        </div>

        <div class="result">

            <div class="result-label">
                MONTHLY ELECTRICITY BILL
            </div>

            <div class="result-value">
                <span class="currency">₹</span>{bill:,.2f}
            </div>

            <div class="result-subtitle">
                Estimated from {ac_units:.0f} AC units of consumption
            </div>

            <div class="details">

                <div class="detail">

                    <div class="detail-label">
                        Consumption
                    </div>

                    <div class="detail-value">
                        {ac_units:.0f} units
                    </div>

                </div>

                <div class="detail">

                    <div class="detail-label">
                        Model
                    </div>

                    <div class="detail-value">
                        Polynomial
                    </div>

                </div>

                <div class="detail">

                    <div class="detail-label">
                        Degree
                    </div>

                    <div class="detail-value">
                        2
                    </div>

                </div>

            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class="empty-result">

        <div class="empty-icon">
            ◌
        </div>

        <div class="empty-title">
            Your estimate will appear here
        </div>

        <div class="empty-description">
            Enter your AC consumption and calculate your estimated bill.
        </div>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MODEL INFO
# ============================================================

st.markdown("""
<div class="model-info">

    <div class="model-item">

        <div class="model-label">
            Algorithm
        </div>

        <div class="model-value">
            Polynomial Regression
        </div>

    </div>

    <div class="model-item">

        <div class="model-label">
            Degree
        </div>

        <div class="model-value">
            2
        </div>

    </div>

    <div class="model-item">

        <div class="model-label">
            Input
        </div>

        <div class="model-value">
            AC Units
        </div>

    </div>

</div>

<div class="footer">

    <div>
        Billwise · Electric Bill Predictor
    </div>

    <div>
        Built with Python · Scikit-learn · Streamlit
    </div>

</div>
""", unsafe_allow_html=True)
