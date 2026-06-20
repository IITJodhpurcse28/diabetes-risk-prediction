# Diabetes Risk Screening App
# Requirements: streamlit, scikit-learn, numpy, plotly
#   pip install streamlit scikit-learn numpy plotly
# Run with: streamlit run app.py

import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.app-header h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    color: #134E4A;
    margin-bottom: 0.1rem;
}

.app-header p {
    color: #64748B;
    font-size: 0.95rem;
    margin-top: 0;
}

.eyebrow {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0F766E;
    background: rgba(15, 118, 110, 0.08);
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 10px;
}

.result-card {
    border-radius: 16px;
    padding: 28px 26px;
    margin-top: 6px;
}

.result-card.alert {
    background: rgba(220, 71, 49, 0.08);
    border: 1px solid rgba(220, 71, 49, 0.25);
}

.result-card.safe {
    background: rgba(22, 163, 74, 0.08);
    border: 1px solid rgba(22, 163, 74, 0.25);
}

.result-card h2 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.5rem;
    margin: 6px 0 2px 0;
    color: #1E293B;
}

.result-card .score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    color: #64748B;
}

.empty-state {
    border: 1px dashed #CBD5E1;
    border-radius: 16px;
    padding: 48px 28px;
    text-align: center;
    color: #64748B;
    font-size: 1.05rem;
}

div.stButton > button, button[kind="primaryFormSubmit"], button[kind="formSubmit"] {
    background: #0F766E;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
}

div.stButton > button:hover, button[kind="primaryFormSubmit"]:hover, button[kind="formSubmit"]:hover {
    background: #134E4A;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.markdown('<span class="eyebrow">ML risk screening</span>', unsafe_allow_html=True)
st.markdown('<h1>Diabetes Risk Screening</h1>', unsafe_allow_html=True)
st.markdown(
    '<p>Enter the measurements below to estimate diabetes risk using a trained scikit-learn model.</p>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)
st.write("")

# ---------- Load model & scaler ----------
@st.cache_resource
def load_artifacts():
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Couldn't find `diabetes_model.pkl` or `scaler.pkl`. "
        "Place both files in the same folder as this app and refresh."
    )
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### About this tool")
    st.write(
        "This app uses a model trained on the Pima Indians Diabetes dataset to "
        "estimate the likelihood of diabetes from eight clinical measurements."
    )
    st.markdown("---")
    st.markdown("### Typical reference ranges")
    st.table({
        "Measurement": ["Glucose (mg/dL)", "Blood Pressure (mm Hg)", "BMI", "Age"],
        "Typical range": ["70 – 140", "60 – 90", "18.5 – 24.9", "—"],
    })
    st.caption("Ranges are general references, not diagnostic thresholds.")
    st.markdown("---")
    st.caption("⚠️ Educational project only — not medical advice.")

# ---------- Main layout ----------
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown("#### Patient measurements")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            preg = st.slider("Pregnancies", 0, 17, 1, help="Number of times pregnant")
            bp = st.slider("Blood Pressure (mm Hg)", 0, 130, 70, help="Diastolic blood pressure")
            insulin = st.slider("Insulin (mu U/mL)", 0, 850, 80, help="2-hour serum insulin level")
            dpf = st.slider(
                "Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01,
                help="A score reflecting diabetes likelihood based on family history",
            )
        with c2:
            glucose = st.slider("Glucose (mg/dL)", 0, 200, 120, help="Plasma glucose concentration")
            skin = st.slider("Skin Thickness (mm)", 0, 100, 20, help="Triceps skinfold thickness")
            bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1, help="Body Mass Index (kg/m²)")
            age = st.slider("Age", 1, 120, 30)

        submitted = st.form_submit_button("Predict risk", use_container_width=True)

    if submitted:
        features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled)[0][1]
        else:
            probability = float(prediction)

        st.session_state["result"] = {
            "prediction": int(prediction),
            "probability": float(probability),
            "inputs": {
                "Pregnancies": preg,
                "Glucose": glucose,
                "Blood Pressure": bp,
                "Skin Thickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "Diabetes Pedigree Function": dpf,
                "Age": age,
            },
        }

with right:
    st.markdown("#### Result")
    result = st.session_state.get("result")

    if not result:
        st.markdown(
            '<div class="empty-state">🩺<br><br>Fill in the measurements and click '
            '<b>Predict risk</b> to see a result here.</div>',
            unsafe_allow_html=True,
        )
    else:
        is_diabetic = result["prediction"] == 1
        css_class = "alert" if is_diabetic else "safe"
        label = "Likely Diabetic" if is_diabetic else "Likely Not Diabetic"
        icon = "⚠️" if is_diabetic else "✅"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div style="font-size: 2rem;">{icon}</div>
            <h2>{label}</h2>
            <div class="score">Model confidence: {result['probability']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        gauge_color = "#DC4731" if is_diabetic else "#16A34A"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability"] * 100,
            number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": gauge_color},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(22, 163, 74, 0.12)"},
                    {"range": [50, 100], "color": "rgba(220, 71, 49, 0.12)"},
                ],
            },
        ))
        fig.update_layout(
            height=240,
            margin=dict(t=10, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View entered measurements"):
            st.table(result["inputs"])

st.markdown("---")
st.caption(
    "This tool provides an automated estimate based on a machine learning model and is "
    "not a medical diagnosis. Please consult a healthcare professional for medical advice."
)