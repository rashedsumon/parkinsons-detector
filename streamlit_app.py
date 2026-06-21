import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from model import train_and_save_model

# Page configuration
st.set_page_config(page_title="Parkinson's Movement Assessment App", layout="wide")

st.title("🧠 Parkinson's Disease Movement Screening App")
st.write("This app uses AI to check for physical signs of Parkinson's disease by analyzing body movement metrics.")

MODEL_FILE = "parkinsons_model.pkl"

# --- CACHING & BACKGROUND PREPARATION ---
@st.cache_resource
def prepare_ai_model():
    """Trains the model if not present, and loads it into memory."""
    if not os.path.exists(MODEL_FILE):
        with st.spinner("First-time setup: Fetching dataset and preparing the AI engine... Please wait."):
            train_and_save_model(MODEL_FILE)
    return joblib.load(MODEL_FILE)

try:
    model = prepare_ai_model()
    st.success("🤖 AI Engine Active & Ready for Use!")
except Exception as e:
    st.error(f"Error starting the application: {e}")
    st.stop()

# --- USER INTERFACE / SLIDERS ---
st.markdown("---")
st.subheader("📋 Enter Patient Movement Data")
st.write("Adjust the sliders below to mimic the patient's physical movement patterns captured from camera tracking.")

# Splitting inputs into 3 friendly columns
col1, col2, col3 = st.columns(3)

with col1:
    gait_speed = st.slider("Walking Speed Stability (Lower is more steady)", 0.0, 2.5, 0.4)
    arm_swing_l = st.slider("Left Arm Swing (Movement angle in degrees)", 5.0, 45.0, 25.0)

with col2:
    tremor_freq = st.slider("Hand Tremor Speed (Hz / Shakes per second)", 0.0, 12.0, 1.2)
    arm_swing_r = st.slider("Right Arm Swing (Movement angle in degrees)", 5.0, 45.0, 22.0)

with col3:
    stride_len = st.slider("Step Length Balance (Higher is more consistent)", 0.1, 1.5, 0.8)
    posture_flexion = st.slider("Forward Spine Lean (Trunk angle in degrees)", 0.0, 30.0, 4.5)

# Trigger Assessment Button
if st.button("Run Diagnostic Assessment", type="primary"):
    # Reconstructing features to match mock/real dataset shape (102 inputs)
    input_features = np.zeros(102)
    
    # Mapping our sliders to the first few feature positions for model input
    input_features[0] = gait_speed
    input_features[1] = tremor_freq
    input_features[2] = stride_len
    input_features[3] = abs(arm_swing_l - arm_swing_r) # Calculates difference between left/right arm swings
    input_features[4] = posture_flexion
    
    # Reshape data to tell the model it's evaluating 1 single patient sample
    prediction_data = input_features.reshape(1, -1)
    
    # Run Inference
    prediction = model.predict(prediction_data)[0]
    probabilities = model.predict_proba(prediction_data)[0]
    
    st.markdown("---")
    st.subheader("📊 Screening Results")
    
    # Convert probabilities to easy-to-read percentages
    chance_healthy = round(probabilities[0] * 100)
    chance_parkinsons = round(probabilities[1] * 100)
    
    if prediction == 1:
        st.error("⚠️ **Signs of Parkinson's Detected**")
        
        st.write(
            "**What this means:** The movement patterns entered look similar to patterns "
            "often seen in individuals with Parkinson's disease. The system flagged anomalies "
            "such as elevated hand tremor frequencies or highly uneven arm movements."
        )
        st.info("💡 *Reminder: This is a screening helper tool, not a formal diagnosis. Please consult a medical professional for official evaluations.*")
    else:
        st.success("✅ **Normal Movement Patterns**")
       
        st.write(
            "**What this means:** The movement patterns look healthy, steady, and balanced. "
            "Walking speeds, step consistency, and arm swings all align within typical healthy baselines."
        )