import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Billwise — Electric Bill Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

pipeline = joblib.load("polynomial_regression_ac_model.pkl")

poly = pipeline["poly"]
model = pipeline["model"]

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ------------------------------
   GLOBAL
------------------------------ */

.stApp {
    background: #F7F7F5;
    color: #111111;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1180px;
    padding: 0rem 2rem 4rem 2rem;
}

/* Hide Streamlit chrome */

header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* ------------------------------
   NAVBAR
------------------------------ */

.navbar {
    height: 76px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #E5E5E1;
    margin-bottom: 90px;
}

.logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.logo-icon {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: #111111;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.nav-right {
    color: #777777;
    font-size: 13px;
}

/* ------------------------------
   HERO
------------------------------ */

.hero {
    max-width: 800px;
    margin-bottom: 65px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border: 1px solid #DEDED9;
    border-radius: 999px;
    background: #FFFFFF;
    color: #666666;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 22px;
}

.eyebrow-dot {
    width: 6px;
    height: 6px;
    background: #4CAF50;
    border-radius: 50%;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 64px;
    line-height: 1.02;
    letter-spacing: -3.5px;
    font-weight: 600;
    margin: 0;
    color: #111111;
}

.hero h1 span {
    color: #8A8A84;
}

.hero p {
    max-width: 560px;
    margin-top: 22px;
    color: #707070;
    font-size: 17px;
    line-height: 1.6;
}

/* ------------------------------
   MAIN GRID
------------------------------ */

.main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
}

/* ------------------------------
   CARDS
------------------------------ */

.card {
    background: #FFFFFF;
    border: 1px solid #E4E4DF;
    border-radius: 24px;
    padding: 30px;
    min-height: 360px;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 45px;
}

.card-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #888888;
}

.card-number {
    width: 30px;
    height: 30px;
    border: 1px solid #E3E3DE;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: #777777;
}

/* ------------------------------
   INPUT AREA
------------------------------ */

.input-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.input-description {
    color: #888888;
    font-size: 14px;
    margin-bottom: 25px;
}

/* Streamlit input */

div[data-baseweb="input"] {
    background: #F7F7F5 !important;
    border: 1px solid #DCDCD7 !important;
    border-radius: 14px !important;
    height: 70px !important;
}

div[data-baseweb="input"]:focus-within {
    border: 1px solid #111111 !important;
    box-shadow: none !important;
}

input {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #111111 !important;
}

/* Hide label */

div[data-testid="stNumberInput"] label {
    display: none;
}

/* ------------------------------
   BUTTON
------------------------------ */

.stButton {
    margin-top: 20px;
}

.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 14px;
    border: none;
    background: #111111;
    color: white;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #2A2A2A;
    transform: translateY(-1px);
}

/* ------------------------------
   RESULT CARD
------------------------------ */

.result-card {
    position: relative;
    overflow: hidden;
    background: #111111;
    color: #FFFFFF;
    border-radius: 24px;
    padding: 30px;
    min-height: 360px;
}

.result-glow {
    position: absolute;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: #3C3C3C;
    filter: blur(70px);
    right: -100px;
    top: -100px;
    opacity: 0.7;
}

.result-content {
    position: relative;
    z-index: 2;
}

.result-card .card-label {
    color: #777777;
}

.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    color: #A0A0A0;
    margin-top: 45px;
}

.result-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 56px;
    font-weight: 600;
    letter-spacing: -3px;
    margin-top: 8px;
}

.result-period {
    color: #777777;
    font-size: 13px;
}

.result-line {
    height: 1px;
    background: #2B2B2B;
    margin: 45px 0 20px 0;
}

.result-footnote {
    color: #777777;
    font-size: 12px;
}

/* ------------------------------
   EMPTY RESULT
------------------------------ */

.empty-result {
    margin-top: 70px;
    color: #666666;
    font-size: 14px;
    line-height: 1.6;
}

.empty-icon {
    font-size: 30px;
    margin-bottom: 18px;
}

/* ------------------------------
   MODEL INFO
------------------------------ */

.model-info {
    display: grid;
    grid-template-columns: 1fr 1fr;
    margin-top: 18px;
    border-top: 1px solid #E4E4DF;
    border-bottom: 1px solid #E4E4DF;
}

.info-item {
    padding: 20px 0;
}

.info-item + .info-item {
    border-left: 1px solid #E4E4DF;
    padding-left: 25px;
}

.info-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: #999999;
}

.info-value {
    margin-top: 6px;
    font-size: 14px;
    font-weight: 600;
}

/* ------------------------------
   FOOTER
------------------------------ */

.footer {
    margin-top: 35px;
    display: flex;
    justify-content: space-between;
    color: #999999;
    font-size: 11px;
}

/* ------------------------------
   RESPONSIVE
------------------------------ */

@media (max-width: 800px) {

    .hero h1 {
        font-size: 45px;
        letter-spacing: -2px;
    }

    .main-grid {
        grid-template-columns: 1fr;
    }

    .navbar {
        margin-bottom: 55px;
    }

    .card,
    .result-card {
        min-height: 320px;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# NAVBAR
# --------------------------------------------------

st.markdown("""
<div class="navbar">

    <div class="logo">
        <div class="logo-icon">⚡</div>
        Billwise
    </div>

    <div class="nav-right">
        ML-powered electricity estimation
    </div>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown("""
<div class="hero">

    <div class="eyebrow">
        <div class="eyebrow-dot"></div>
        POLYNOMIAL REGRESSION
    </div>

    <h1>
        Electricity bill,<br>
        <span>made predictable.</span>
    </h1>

    <p>
        Estimate your monthly electricity bill based on
        AC electricity consumption using a trained
        polynomial regression model.
    </p>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MAIN CARDS
# --------------------------------------------------

col1, col2 = st.columns(2, gap="medium")


# --------------------------------------------------
# INPUT CARD
# --------------------------------------------------

with col1:

    st.markdown("""
    <div class="card">

        <div class="card-header">
            <div class="card-label">
                INPUT
            </div>

            <div class="card-number">
                01
            </div>
        </div>

        <div class="input-title">
            AC consumption
        </div>

        <div class="input-description">
            Enter the electricity consumed by your AC.
        </div>

    """, unsafe_allow_html=True)

    ac_units = st.number_input(
        "AC Units",
        min_value=0.0,
        value=72.0,
        step=1.0,
        label_visibility="collapsed"
    )

    predict = st.button(
        "Calculate estimated bill  →"
    )

    st.markdown("""
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# RESULT CARD
# --------------------------------------------------

with col2:

    if predict:

        input_data = pd.DataFrame({
            "AC_Units": [ac_units]
        })

        input_poly = poly.transform(input_data)

        prediction = model.predict(input_poly)

        bill = prediction[0]

        st.markdown(f"""
        <div class="result-card">

            <div class="result-glow"></div>

            <div class="result-content">

                <div class="card-header">
                    <div class="card-label">
                        ESTIMATION
                    </div>

                    <div class="card-number">
                        02
                    </div>
                </div>

                <div class="result-title">
                    Estimated monthly bill
                </div>

                <div class="result-value">
                    ₹{bill:,.2f}
                </div>

                <div class="result-period">
                    Based on {ac_units:.0f} AC units
                </div>

                <div class="result-line"></div>

                <div class="result-footnote">
                    Generated using Polynomial Regression · Degree 2
                </div>

            </div>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card">

            <div class="result-glow"></div>

            <div class="result-content">

                <div class="card-header">
                    <div class="card-label">
                        ESTIMATION
                    </div>

                    <div class="card-number">
                        02
                    </div>
                </div>

                <div class="empty-result">

                    <div class="empty-icon">
                        ◌
                    </div>

                    <strong style="color:#FFFFFF;">
                        Your estimate will appear here.
                    </strong>

                    <br><br>

                    Enter your AC consumption and
                    calculate an estimated monthly bill.

                </div>

            </div>

        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.markdown("""
<div class="model-info">

    <div class="info-item">
        <div class="info-label">
            Model
        </div>
        <div class="info-value">
            Polynomial Regression
        </div>
    </div>

    <div class="info-item">
        <div class="info-label">
            Polynomial degree
        </div>
        <div class="info-value">
            2
        </div>
    </div>

</div>

<div class="footer">

    <div>
        Billwise · Electric Bill Predictor
    </div>

    <div>
        Built with Python + Scikit-learn + Streamlit
    </div>

</div>
""", unsafe_allow_html=True)import streamlit as st
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
