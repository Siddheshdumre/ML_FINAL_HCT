# 🔬 Post-HCT Survival Predictor

A Streamlit-based machine learning application for predicting **Event-Free Survival (EFS)** time in patients undergoing Hematopoietic Cell Transplantation (HCT), using a trained LightGBM model optimized with Optuna.

---

## 📌 Project Overview

This project helps clinicians and researchers estimate the likelihood of post-transplant survival using patient and transplant-related features. The model predicts a **continuous EFS score** based on 41 clinical input parameters.

There are two interfaces:
- **🧍 Single Entry Predictor**: Manually input patient data and get an EFS prediction.
- **📄 Batch Predictor**: Upload a CSV of multiple patients and get predictions for all.

---

## ⚙️ Tech Stack

- 🧠 **LightGBM** (Regressor) optimized using **Optuna**
- 🧪 **Scikit-learn** for preprocessing (Scaling + Feature Selection)
- 🌐 **Streamlit** for the web interface
- 📊 **Matplotlib & Seaborn** for visualizations

---

## 📁 Files

- `efs_prediction_app.py`: Streamlit app for single patient prediction
- `efs_batch_app.py`: Streamlit app for CSV-based batch prediction
- `optuna_lgbm_model.pkl`: Trained LightGBM model
- `scaler.pkl`: Standard scaler used for input preprocessing
- `selector.pkl`: Feature selector (if applicable)
- `sample_data.csv`: Sample input file for batch predictions

---

## ✅ Features

- Predict **Event-Free Survival (EFS)** score for a single patient or in bulk
- Scaled and selected inputs using pre-trained pipeline
- Upload CSV and download predictions
- Visual insights for batch predictions:
  - 📉 Distribution of predicted EFS scores
  - 🟦 Box plot of EFS values
  - 📈 Feature impact via scatter plots

---

## 🏃‍♂️ How to Run

### 1. Install Requirements

```bash
pip install streamlit scikit-learn lightgbm joblib pandas numpy matplotlib seaborn
