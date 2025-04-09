import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


# Load model and scaler
model = joblib.load("optuna_lgbm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature order
all_features = [
    'sex_match_M-F', 'peptic_ulcer', 'prim_disease_hct_MDS', 'prim_disease_hct_IEA',
    'cmv_status_-/+', 'rheum_issue', 'ethnicity_Non-resident of the U.S.', 'year_hct',
    'melphalan_dose_N/A, Mel not given', 'conditioning_intensity_RIC', 'cyto_score',
    'tce_div_match_Missing', 'gvhd_proph_Cyclophosphamide alone', 'in_vivo_tcd',
    'race_group_More than one race', 'dri_score', 'sex_match_F-M', 'prim_disease_hct_AML',
    'sex_match_M-M', 'prim_disease_hct_Solid tumor', 'gvhd_proph_FK+ MMF +- others',
    'renal_issue', 'cyto_score_detail', 'comorbidity_score', 'cmv_status_+/-',
    'prim_disease_hct_ALL', 'graft_type_Peripheral blood', 'mrd_hct',
    'conditioning_intensity_Missing', 'rituximab', 'prim_disease_hct_IIS',
    'tce_div_match_Permissive mismatched', 'tbi_status_TBI +- Other, <=cGy',
    'prod_type_PB', 'tce_imm_match_Missing', 'age_at_hct', 'tce_imm_match_P/P',
    'hepatic_severe', 'karnofsky_score', 'cmv_status_-/-', 'efs_time'
]

st.set_page_config(page_title="📄 Batch EFS Predictor", layout="wide")
st.title("📄 Batch Post-HCT EFS Predictor")
st.write("Upload a CSV file with 41 input features to predict EFS scores in bulk.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check columns
    if not all(col in df.columns for col in all_features):
        st.error("❌ The uploaded CSV does not contain all required 41 features.")
        st.write("Expected columns:")
        st.code(", ".join(all_features))
    else:
        # Reorder columns
        df = df[all_features]

        # Scale
        X_scaled = scaler.transform(df)

        # Predict
        predictions = model.predict(X_scaled)
        df["Predicted_EFS"] = predictions

        st.success("✅ Predictions completed!")

        # Show preview
        st.dataframe(df.head())

        st.subheader("📈 Prediction Analysis")

        # 1. Histogram of Predicted EFS
        st.markdown("### Distribution of Predicted EFS")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Predicted_EFS"], bins=10, kde=True, ax=ax1, color="skyblue")
        ax1.set_xlabel("Predicted EFS")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        # 2. Boxplot: Karnofsky Score vs Predicted EFS
        st.markdown("### Karnofsky Score vs. Predicted EFS")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=pd.cut(df["karnofsky_score"], bins=[0, 60, 80, 100]), y="Predicted_EFS", data=df, ax=ax2)
        ax2.set_xlabel("Karnofsky Score Range")
        ax2.set_ylabel("Predicted EFS")
        st.pyplot(fig2)

        # 3. Scatter Plot: Age at HCT vs Predicted EFS
        st.markdown("### Age at HCT vs. Predicted EFS")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df, x="age_at_hct", y="Predicted_EFS", hue="sex_match_M-F", ax=ax3)
        ax3.set_xlabel("Age at HCT")
        ax3.set_ylabel("Predicted EFS")
        st.pyplot(fig3)

        # 4. Categorical Summary Table
        st.markdown("### Breakdown of Primary Disease Categories")
        disease_cols = [col for col in df.columns if col.startswith("prim_disease_hct_")]
        disease_summary = df[disease_cols].sum().sort_values(ascending=False).reset_index()
        disease_summary.columns = ["Disease", "Count"]
        st.dataframe(disease_summary)



        # Downloadable output
        csv_output = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions CSV", csv_output, "efs_predictions.csv", "text/csv")
