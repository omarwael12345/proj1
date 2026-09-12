import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Price Range Predictor",
    page_icon="📱",
    layout="centered"
)

# ============================================================
# LOAD MODEL + SCALER + FEATURE ORDER
# Put svc_model.pkl, scaler.pkl, feature_columns.pkl
# in the SAME FOLDER as this app.py file
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("svc_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# ============================================================
# STYLING
# ============================================================
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #f0f2f6;
        text-align: center;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📱 Price Range Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the phone specs to estimate the probability of each price range</div>', unsafe_allow_html=True)

# ============================================================
# INPUT FORM — matches your real dataset columns
# ============================================================
st.subheader("Device Specifications")

col1, col2 = st.columns(2)

with col1:
    battery_power = st.number_input("Battery Power (mAh)", 500, 2000, 1200)
    clock_speed = st.number_input("Clock Speed (GHz)", 0.5, 3.0, 1.5)
    fc = st.number_input("Front Camera (MP)", 0, 20, 5)
    int_memory = st.number_input("Internal Memory (GB)", 2, 64, 32)
    m_dep = st.number_input("Mobile Depth (cm)", 0.1, 1.0, 0.5)
    mobile_wt = st.number_input("Weight (g)", 80, 200, 140)
    n_cores = st.number_input("Number of Cores", 1, 8, 4)
    pc = st.number_input("Primary Camera (MP)", 0, 20, 10)
    px_height = st.number_input("Pixel Height", 0, 2000, 800)
    px_width = st.number_input("Pixel Width", 0, 2000, 1200)

with col2:
    ram = st.number_input("RAM (MB)", 250, 4000, 2000)
    sc_h = st.number_input("Screen Height (cm)", 5, 20, 12)
    sc_w = st.number_input("Screen Width (cm)", 0, 18, 7)
    talk_time = st.number_input("Talk Time (hrs)", 2, 20, 10)
    blue = st.selectbox("Bluetooth", [0, 1], format_func=lambda x: "Yes" if x else "No")
    dual_sim = st.selectbox("Dual SIM", [0, 1], format_func=lambda x: "Yes" if x else "No")
    four_g = st.selectbox("4G", [0, 1], format_func=lambda x: "Yes" if x else "No")
    three_g = st.selectbox("3G", [0, 1], format_func=lambda x: "Yes" if x else "No")
    touch_screen = st.selectbox("Touch Screen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    wifi = st.selectbox("WiFi", [0, 1], format_func=lambda x: "Yes" if x else "No")

predict_btn = st.button("🔮 Predict Probability", use_container_width=True)

# ============================================================
# PREDICTION
# ============================================================
if predict_btn:
    # Engineered features — must match training exactly
    px_area = px_width * px_height
    memory_core_interaction = int_memory * n_cores

    # Build a dict of all raw inputs
    raw_input = {
        "battery_power": battery_power,
        "blue": blue,
        "clock_speed": clock_speed,
        "dual_sim": dual_sim,
        "fc": fc,
        "four_g": four_g,
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": three_g,
        "touch_screen": touch_screen,
        "wifi": wifi,
        "px_area": px_area,
        "memory_core_interaction": memory_core_interaction,
    }

    # Arrange columns in the EXACT order the model was trained on
    input_df = pd.DataFrame([raw_input])[feature_columns]

    # Apply the SAME fitted scaler used in training (transform, not fit)
    X_input = scaler.transform(input_df)

    # Predict class + probabilities
    pred_class = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    price_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}

    st.markdown("---")
    st.markdown(f"""
        <div class="result-box">
            <h3>Predicted Price Range: <b>{price_labels.get(pred_class, pred_class)}</b></h3>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Probability by Class")
    prob_df = pd.DataFrame({
        "Price Range": [price_labels.get(c, c) for c in model.classes_],
        "Probability": probabilities
    })

    for _, row in prob_df.iterrows():
        st.write(f"**{row['Price Range']}**")
        st.progress(float(row["Probability"]))
        st.caption(f"{row['Probability']:.1%}")

    st.bar_chart(prob_df.set_index("Price Range"))
