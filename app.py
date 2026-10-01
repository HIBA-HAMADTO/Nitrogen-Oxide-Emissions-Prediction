# app.py
import streamlit as st
import joblib
import pandas as pd
import os
import glob
from utils import AdvancedFeatureEngineer, OutlierHandler

# ===========================
# 1️⃣ Auto-detect latest model version
# ===========================
model_folders = glob.glob("models/v1_*")
if not model_folders:
    st.error("❌ No model folders found in 'models/'. Please train and save a model first.")
    st.stop()

# Get the latest folder by modification time
MODEL_DIR = max(model_folders, key=os.path.getmtime)
model_path = os.path.join(MODEL_DIR, "best_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

# Safety check
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at {model_path}")
    st.stop()
if not os.path.exists(preprocessor_path):
    st.error(f"❌ Preprocessor file not found at {preprocessor_path}")
    st.stop()

# ✅ Make sure utils.py is imported before unpickling
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_model()

# ===========================
# 2️⃣ Streamlit UI
# ===========================
st.set_page_config(page_title="⚡ Gas Turbine NOX Predictor", layout="wide")
st.title("⚡ Gas Turbine NOX Predictor")
st.write(
    """
    Enter the features of a gas turbine to predict NOX emissions.
    This model is a **Raandom Forest Regressor** trained on a gas turbine dataset.
    """
)

st.subheader("⚙️ Gas Turbine Operating Conditions")
st.sidebar.header("Input Features")
col1, col2 = st.columns(2)

with col1:
  at = st.slider("Ambient Temperature (AT)", -5.0, 40.0, 15.0, 0.1)
  ap = st.slider("Ambient Pressure (AP)", 990.0, 1040.0, 1013.0, 0.1)
  ah = st.slider("Ambient Humidity (AH)", 30.0, 100.0, 75.0, 0.1)
  afdp = st.slider("Air Filter Diff Pressure (AFDP)", 2.0, 8.0, 4.0, 0.01)
  gtep = st.slider("Gas Turbine Exhaust Press (GTEP)", 17.0, 40.0, 25.0, 0.1)

with col2:
  tit = st.slider("Turbine Inlet Temperature (TIT)", 1000.0, 1100.0, 1080.0, 0.1)
  tat = st.slider("Turbine After Temperature (TAT)", 500.0, 580.0, 550.0, 0.1)
  tey = st.slider("Turbine Energy Yield (TEY)",100.0,180.0,135.0,0.1)
  cdp = st.slider("Compressor Discharge Press (CDP)", 10.0, 21.0, 12.0, 0.01)
  co = st.slider("Carbon Monoxide (CO)", 0.0, 10.0, 0.5, 0.01)

# ===========================
# 3️⃣ Prepare input DataFrame
# ===========================
feature_names = [
    "AT",
    "AP",
    "AH",
    "AFDP",
    "GTEP",
    "TIT",
    "TAT",
    "TEY",
    "CDP",
    "CO",
]

input_df = pd.DataFrame([[
    at, ap, ah, afdp, gtep, tit, tat, tey, cdp, co
]], columns=feature_names)

# ===========================
# 4️⃣ Preprocess & Predict
# ===========================
if st.button("🚀 Predict NOX Emissions"):
    try:
        X_input = preprocessor.transform(input_df)
        prediction = model.predict(X_input)[0]
        predicted_nox = prediction 

        st.success(f"### Predicted NOX emissions: {predicted_nox:,.0f}")
        st.write("#### Input Features") 
        st.write(input_df)
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

# ===========================
# 5️⃣ Sidebar Info
# ===========================
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
st.sidebar.write(f"**Loaded from:** `{MODEL_DIR}`")
st.sidebar.write("**Model:** Random Forest Regressor")
st.sidebar.write("**Dataset:** Gas Turbine NOX Emissions")
st.sidebar.write("**Validation R²:** 0.8974")
st.sidebar.write("**Test R²:** 0.8895 | RMSE: 3.8260 | MAE: 2.4650")