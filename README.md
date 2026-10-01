# ⚡ Gas Turbine NOx Emissions Prediction

A machine learning project for predicting **Nitrogen Oxides (NOx) emissions from gas turbine operating conditions**.

This project combines data analysis, feature engineering, data preprocessing, machine learning, and an interactive **Streamlit web application** for NOx emission prediction.

## 📌 Project Overview

NOx emissions are important indicators of combustion behavior and environmental impact in gas turbine systems.

The objective of this project is to use gas turbine operating conditions to develop a machine learning model that predicts NOx emissions.

The project workflow includes:

- Data preparation and exploration
- Feature engineering
- Outlier handling
- Feature scaling
- Random Forest regression
- Model evaluation
- Saving the trained model
- Saving the preprocessing pipeline
- Interactive prediction using Streamlit

## ⚙️ Input Features

The application uses the following gas turbine operating conditions:

- **AT** — Ambient Temperature
- **AP** — Ambient Pressure
- **AH** — Ambient Humidity
- **AFDP** — Air Filter Differential Pressure
- **GTEP** — Gas Turbine Exhaust Pressure
- **TIT** — Turbine Inlet Temperature
- **TAT** — Turbine After Temperature
- **CDP** — Compressor Discharge Pressure
- **CO** — Carbon Monoxide

### 🎯 Target Variable

**NOX — Nitrogen Oxides emissions**

## 🧠 Machine Learning Model

The project uses a **Random Forest Regressor** for NOx emission prediction.

The preprocessing pipeline includes:

1. Feature engineering
2. Outlier handling using the IQR method
3. Robust scaling
4. Random Forest regression

The trained model and preprocessing pipeline are saved and used by the Streamlit application.

## 🔬 Feature Engineering

Additional features are generated from the original gas turbine operating conditions to capture useful relationships between the input variables.

The engineered features include:

- Turbine Temperature Drop
- Air Density
- Energy Per Airflow
- Compressor Work Proxy
- Core Thermal Energy
- Thermal Severity Index
- Combustion Intensity Proxy
- Absolute Humidity Effect

## 📊 Model Performance

The trained model achieved the following evaluation results:

| Metric | Value |
|---|---:|
| Validation R² | 0.8974 |
| Test R² | 0.8895 |
| Test RMSE | 3.8260 |
| Test MAE | 2.4650 |

These metrics represent the model evaluation results obtained during the project.

## 🖥️ Streamlit Application

The project includes an interactive web application called **Gas Turbine NOX Predictor**.

The application allows users to enter gas turbine operating conditions using interactive controls and obtain a predicted NOx emission value.

The application:

1. Loads the saved Random Forest model.
2. Loads the saved preprocessing pipeline.
3. Applies the preprocessing used during model development.
4. Generates the NOx prediction.
5. Displays the predicted NOx emission and the entered input values.

## 📁 Project Structure

The main project files include:

- `app.py` — Streamlit prediction application
- `utils.py` — utility functions
- `Nitrogen_Oxide_Pred.ipynb` — machine learning development notebook
- `gas_turbine_data.csv` — gas turbine dataset
- `model_columns .pkl` — saved model column information
- `models/v1_20260930_184029/` — trained model and related files
- `.gitignore` — files and folders excluded from Git
- `.gitattributes` — Git LFS configuration

The model directory contains:

- `best_model.pkl` — trained Random Forest model
- `preprocessor.pkl` — saved preprocessing pipeline
- `model_card.json` — model information
- `requirements.json` — project requirements information

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git
- Git LFS

## 🚀 How to Run

Clone the repository from GitHub:

`git clone https://github.com/HIBA-HAMADTO/Nitrogen-Oxide-Emissions-Prediction.git`

Move into the project directory:

`cd Nitrogen-Oxide-Emissions-Prediction`

Create a virtual environment:

`python -m venv .venv`

Activate the virtual environment on Windows:

`.venv\Scripts\activate`

Install the required Python packages according to the project's saved requirements.

Run the Streamlit application:

`streamlit run app.py`

The application will then open in the browser.

## 💾 Model Files

The trained model files are stored in:

`models/v1_20260930_184029/`

The large model files are managed using **Git LFS (Git Large File Storage)**.

## 📓 Jupyter Notebook

The project includes the notebook:

`Nitrogen_Oxide_Pred.ipynb`

The notebook contains the machine learning workflow used for developing and evaluating the NOx prediction model.

## 📈 Prediction Workflow

The prediction workflow follows these main steps:

**Gas Turbine Operating Conditions → Data Preparation → Feature Engineering → Outlier Handling → Robust Scaling → Random Forest Regression → NOx Prediction**

## 🎯 Project Objective

The main objective of this project is to demonstrate the application of machine learning to gas turbine operating data for NOx emission prediction and to provide an interactive tool for generating predictions from operating conditions.

## 👩‍💻 Author

**HIBA HAMADTO**

GitHub: https://github.com/HIBA-HAMADTO/Nitrogen-Oxide-Emissions-Prediction
