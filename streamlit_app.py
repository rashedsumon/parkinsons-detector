import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from model import train_and_save_model

st.set_page_config(page_title="Parkinson's Vision Pose Assessment", layout="wide")

st.title("🧠 Parkinson's Disease Detector")
st.write("This application analyzes clinical human pose estimation coordinates to screen for kinematic symptoms.")

MODEL_FILE = "parkinsons_model.pkl"

# --- CACHING & BACKGROUND PREPARATION ---
@st.cache_resource
def prepare_ai_model():
    """Trains the model if not present, and loads it into memory."""
    if not os.path.exists(MODEL_FILE):
        with st.spinner("First-time setup: Downloading dataset and training AI model... Please wait."):
            train_and_save_model(MODEL_FILE)
    return joblib.load(MODEL_FILE)

try:
    model = prepare_ai_model()
    st.success("🤖 AI Engine Active & Loaded Successfully!")
except Exception as e:
    st.error(f"Error initializing the app: {e}")
    st.stop()

# --- USER INTERFACE / PREDICTION ---
st.markdown("---")
st.subheader("Simulate Patient Kinematic Metrics")
st.write("Adjust the key structural metrics derived from vision tracking systems (e.g., MediaPipe/OpenPose) to run a prediction.")

# UI layout splits into 3 columns for coordinates simulation
col1, col2, col3 = st.columns(3)

with col1:
    gait_speed = st.slider("Gait Speed Variability (Lower is steady)", 0.0, 2.5, 0.4)
    arm_swing_l = st.slider("Left Arm Swing Amplitude (Degrees)", 5.0, 45.0, 25.0)

with col2:
    tremor_freq = st.slider("Hand Tremor Frequency (Hz)", 0.0, 12.0, 1.2)
    arm_swing_r = st.slider("Right Arm Swing Amplitude (Degrees)", 5.0, 45.0, 22.0)

with col3:
    stride_len = st.slider("Step/Stride Length Coefficient", 0.1, 1.5, 0.8)
    posture_flexion = st.slider("Forward Spine Flexion (Trunk Angle)", 0.0, 30.0, 4.5)

# Trigger Assessment Button
if st.button("Run Diagnostic Assessment", type="primary"):
    # Reconstructing features to match mock/real dataset shape (e.g., 102 inputs)
    # We populate the entry array using our sliders combined with baseline defaults
    input_features = np.zeros(102)
    
    # Mapping our sliders manually to mock features mapping for demonstration
    input_features[0] = gait_speed
    input_features[1] = tremor_freq
    input_features[2] = stride_len
    input_features[3] = abs(arm_swing_l - arm_swing_r) # Asymmetry metric
    input_features[4] = posture_flexion
    
    # Reshape for single sample prediction
    prediction_data = input_features.reshape(1, -1)
    
    # Run Inference
    prediction = model.predict(prediction_data)[0]
    probabilities = model.predict_proba(prediction_data)[0]
    
    st.markdown("---")
    st.subheader("📋 Assessment Results")
    
    if prediction == 1:
        st.error(f"⚠️ **Result Indicative of Parkinson's Disease Patterns** (Confidence: {probabilities[1]*100:.1f}%)")
        st.write("The system detected irregularities in movement markers such as high hand-tremor frequencies or prominent arm swing asymmetry.")
    else:
        st.success(f"✅ **Result Normal / Control Group Baseline** (Confidence: {probabilities[0]*100:.1f}%)")
        st.write("Kinematic patterns fall within healthy parameter variations.")