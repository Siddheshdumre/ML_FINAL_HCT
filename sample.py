import pandas as pd

# All 41 feature names
features = [
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

# Sample data for 5 rows
data = [
    [1, 0, 1, 0, 0, 0, 1, 2017, 0, 1, 0.95, 0, 0, 1, 0, 1.5, 0, 0, 0, 0, 1, 0, 1.2, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 60, 1, 0, 85, 0, 10],
    [0, 1, 0, 1, 1, 1, 0, 2019, 1, 0, 0.72, 1, 1, 0, 1, 2.0, 1, 1, 0, 0, 0, 1, 1.1, 2, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 55, 0, 1, 70, 1, 12],
    [1, 0, 0, 0, 1, 0, 0, 2016, 0, 1, 1.15, 0, 0, 1, 0, 1.0, 0, 0, 1, 0, 1, 0, 0.8, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 63, 1, 0, 90, 0, 15],
    [0, 1, 1, 1, 0, 1, 1, 2020, 1, 0, 0.85, 1, 0, 0, 1, 2.5, 1, 0, 0, 1, 0, 1, 1.3, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 50, 0, 1, 75, 1, 18],
    [1, 0, 0, 0, 0, 1, 0, 2018, 0, 1, 1.00, 0, 1, 1, 0, 1.8, 0, 1, 1, 0, 0, 0, 1.0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 68, 1, 0, 80, 1, 14]
]

# Create DataFrame and export
df = pd.DataFrame(data, columns=features)
df.to_csv("sample_input_efs.csv", index=False)
print("✅ File 'sample_input_efs.csv' created.")
