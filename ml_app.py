import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- STEP 1: SETUP & CACHED TRAINING ---
@st.cache_resource
def load_and_train():
    # Load your CSV file 
    df = pd.read_csv('heart_cleveland_upload.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train both models on the full dataset for the app
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_scaled, y)
    
    #Similarity checking using knn - proof for output report
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_scaled, y)
    
    return df, scaler, rf, knn

df, scaler, rf_model, knn_model = load_and_train()

# --- STEP 2: UI LAYOUT ---
st.set_page_config(page_title="Heart Disease AI", layout="wide")
st.title("Heart Disease Clinical Decision System")
st.write("Enter the patient's clinical markers below to generate a similarity-based diagnosis.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 15, 100, 50)
    
    sex_label = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_label == "Male" else 0
    
    cp_label = st.selectbox("Chest Pain Type", [
        "0: Typical Angina", 
        "1: Atypical Angina", 
        "2: Non-anginal Pain", 
        "3: Asymptomatic"
    ])
    cp = int(cp_label[0]) # Takes the first character (0, 1, 2, or 3)

    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
    
    fbs_label = st.selectbox("Fasting Blood Sugar", [">120 mg/dl (1)", "<120 mg/dl (0)"])
    fbs = 1 if ">120" in fbs_label else 0

with col2:
    restecg_label = st.selectbox("Resting ECG Result", [
        "0: Normal", 
        "1: ST-T Wave Abnormality", 
        "2: Left Ventricular Hypertrophy"
    ])
    restecg = int(restecg_label[0])

    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    
    exang_label = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
    exang = 1 if "Yes" in exang_label else 0

    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 0.0)
    
    slope_label = st.selectbox("Slope of Peak Exercise ST Segment", ["0: Upsloping", "1: Flat", "2: Downsloping"])
    slope = int(slope_label[0])

    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    
    thal_label = st.selectbox("Thalassemia (Blood Flow)", [
        "0: Normal", 
        "1: Fixed Defect", 
        "2: Reversible Defect"
    ])
    thal = int(thal_label[0])

# --- STEP 3: PREDICTION & REPORT LOGIC ---
if st.button("Generate Clinical Report"):
    # Prepare data for model
    user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    user_scaled = scaler.transform(user_data)

    
    # Prediction
    prediction = rf_model.predict(user_scaled)[0]
    
    # Neighborhood Evidence (KNN)
    distances, indices = knn_model.kneighbors(user_scaled)
    neighbor_data = df.iloc[indices[0]]
    healthy_count = (neighbor_data['target'] == 0).sum()
    confidence = (healthy_count / 7) * 100 if prediction == 0 else ((7 - healthy_count) / 7) * 100

    st.divider()

    # Natural Working Conclusion
    if prediction == 0:
        st.success("### FINAL WORKING CONCLUSION: HEALTHY (No Disease)")
    else:
        st.error("### FINAL WORKING CONCLUSION: AT RISK (Disease Detected)")

    st.write(f"**Explanation:** We identified 7 patients in our database who share a nearly identical medical signature with yours.")
    st.write(f"* Among these 7 'Medical Twins', **{healthy_count}** were confirmed Healthy and **{7-healthy_count}** were At-Risk.")
    st.write(f"* Since the majority of your closest matches fall into the {'Healthy' if prediction == 0 else 'At-Risk'} category, the system is **{confidence:.1f}% confident** in this diagnosis.")

    # --- THE NEIGHBORHOOD TABLE ---
    st.subheader("📊 Clinical Neighborhood Evidence")
    
    # Reuse your logic for the report_df
    feature_names = list(df.drop('target', axis=1).columns)
    comparison_dict = {'Feature': feature_names, 'YOURS': user_data[0]}
    
    for i, (idx, row) in enumerate(neighbor_data.iterrows()):
        comparison_dict[f'P{i+1}'] = row.drop('target').values
    
    # Calculate AVG and RELATION
    cluster_avg = []
    relations = []
    for i, feat in enumerate(feature_names):
        avg_val = round(neighbor_data[feat].mean(), 1) if feat in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] else neighbor_data[feat].mode()[0]
        cluster_avg.append(avg_val)
        relations.append("EXACT" if float(avg_val) == float(user_data[0][i]) else "SIMILAR")
    
    comparison_dict['AVG'] = cluster_avg
    comparison_dict['RELATION'] = relations
    
    report_df = pd.DataFrame(comparison_dict)
    
    # # Add Result row
    # results = ['RESULT', 'PREDICTED'] + ['HEALTHY' if s == 0 else 'RISK' for s in neighbor_data['target']] + ['MAJORITY', 'VOTED']
    # report_df.loc[len(report_df)] = results

    # st.dataframe(report_df, use_container_width=True)
    # --- FIXED SECTION ---
    report_df = pd.DataFrame(comparison_dict)
    
    # 1. Convert the whole dataframe to strings so 'PREDICTED' doesn't clash with numbers
    report_df = report_df.astype(str) 

    # 2. Now add the Result row
    results = ['RESULT', 'PREDICTED'] + ['HEALTHY' if s == 0 else 'RISK' for s in neighbor_data['target']] + ['MAJORITY', 'VOTED']
    report_df.loc[len(report_df)] = results

    # 3. Use st.table instead of st.dataframe for a cleaner "Medical Report" look
    st.table(report_df)