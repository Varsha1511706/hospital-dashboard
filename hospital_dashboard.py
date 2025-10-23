import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------- CONFIGURATION ----------------
DATA_FILE = "hospital_patients_60plus.csv"
MODEL_FILE = "risk_model.pkl"

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Hospital Patient Dashboard", page_icon="🏥", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2c3e50;
    text-align: center;
    padding: 20px;
    margin-bottom: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.12);
}
.small-card {
    background: #ffffff;
    padding: 12px;
    border-radius: 10px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}
.high-risk {
    background: linear-gradient(135deg, #ff6b6b 0%, #c0392b 100%);
    color: white;
    padding: 16px;
    border-radius: 12px;
    margin: 12px 0;
    box-shadow: 0 6px 18px rgba(255,107,107,0.18);
}
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #ffffff; }
.kpi-label { font-size: 0.95rem; color: rgba(255,255,255,0.9); margin-top:6px; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # normalize columns
        if 'LastVisit' in df.columns:
            df['LastVisit'] = pd.to_datetime(df['LastVisit'], errors='coerce')
        return df
    else:
        # sample fallback dataset
        sample_data = {
            'PatientID': [1, 2, 3, 4, 5],
            'Name': ['John Smith', 'Mary Johnson', 'Robert Brown', 'Susan Davis', 'James Wilson'],
            'Age': [72, 68, 81, 65, 75],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Condition': ['Hypertension', 'Diabetes', 'Heart Disease', 'Arthritis', 'Respiratory Issues'],
            'LastVisit': ['2024-01-15', '2024-02-20', '2024-01-30', '2024-03-05', '2024-02-10'],
            'RiskScore': [65.2, 78.5, 88.9, 59.8, 72.3]
        }
        df = pd.DataFrame(sample_data)
        df['LastVisit'] = pd.to_datetime(df['LastVisit'])
        df.to_csv(DATA_FILE, index=False)
        return df

@st.cache_resource
def load_model(_df):
    if not os.path.exists(MODEL_FILE):
        # tiny model: RiskScore ~ Age (placeholder)
        X = _df[["Age"]]
        y = _df["RiskScore"]
        model = LinearRegression()
        model.fit(X, y)
        pickle.dump(model, open(MODEL_FILE, "wb"))
        return model
    else:
        return pickle.load(open(MODEL_FILE, "rb"))

df = load_data()
model = load_model(df)

# ---------------- HEADER ----------------
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 20px; margin-bottom: 18px; box-shadow: 0 5px 20px rgba(0,0,0,0.06);'>
    <h1 style='margin: 0; color: #2c3e50; font-size: 2.4rem;'>🏥 MEDICAL DASHBOARD</h1>
    <p style='color: #6c757d; font-size: 1rem; margin: 6px 0 0 0;'>
        Senior Patient Monitoring & Risk Management — visual, aesthetic & insightful
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- KPI CARDS ----------------
st.markdown("### 📊 Key Performance Indicators")
kp1, kp2, kp3, kp4 = st.columns([1.2,1,1,1])

with kp1:
    st.markdown("<div class='metric-card'><div class='kpi-value'>{}</div><div class='kpi-label'>👥 TOTAL PATIENTS</div></div>".format(len(df)), unsafe_allow_html=True)
with kp2:
    st.markdown("<div class='metric-card'><div class='kpi-value'>{}</div><div class='kpi-label'>🚨 HIGH RISK (&gt;75)</div></div>".format(len(df[df["RiskScore"] > 75])), unsafe_allow_html=True)
with kp3:
    st.markdown("<div class='metric-card'><div class='kpi-value'>{:.1f}</div><div class='kpi-label'>📅 AVG AGE</div></div>".format(df['Age'].mean()), unsafe_allow_html=True)
with kp4:
    st.markdown("<div class='metric-card'><div class='kpi-value'>{:.1f}</div><div class='kpi-label'>⚠️ AVG RISK SCORE</div></div>".format(df['RiskScore'].mean()), unsafe_allow_html=True)

st.markdown("---")
col_stats1, col_stats2, col_stats3 = st.columns(3)
with col_stats1: st.info(f"**👵 Oldest Patient:** {df['Age'].max()} years")
with col_stats2: st.info(f"**👶 Youngest Patient:** {df['Age'].min()} years")
with col_stats3: st.info(f"**📋 Conditions:** {len(df['Condition'].unique())} types")

# ---------------- TABS ----------------
st.markdown("---")
st.markdown("## 🎯 Patient Management")
tab1, tab2, tab3 = st.tabs(["📋 Patient Records", "➕ Add New Patient", "📊 Advanced Analytics"])

# --- TAB 1: Patient Records ---
with tab1:
    st.subheader("Patient List")
    st.dataframe(df, use_container_width=True)

# --- TAB 2: Add New Patient ---
with tab2:
    st.subheader("Add New Patient")
    with st.form("add_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", 60, 100, 70)
        gender = st.selectbox("Gender", ["Male", "Female"])
        condition = st.selectbox("Condition", ["Hypertension", "Diabetes", "Heart Disease", "Arthritis"])
        last_visit = st.date_input("Last Visit")
        if st.form_submit_button("Add Patient"):
            risk = float(model.predict([[age]])[0])
            new_patient = {
                "PatientID": int(df["PatientID"].max() + 1) if "PatientID" in df.columns else len(df) + 1,
                "Name": name,
                "Age": int(age),
                "Gender": gender,
                "Condition": condition,
                "LastVisit": pd.to_datetime(last_visit),
                "RiskScore": risk
            }
            df = pd.concat([df, pd.DataFrame([new_patient])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.success("Patient added successfully ✅")
            st.experimental_rerun()

# --- TAB 3: Analytics --- #
with tab3:
    st.subheader("📊 Deep Patient Analytics")
    st.markdown("Gain a **data-driven understanding** of patient demographics, risks, and key health trends.")

    # ---------------- FILTERS ----------------
    with st.expander("🔍 Filters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            gender_filter = st.multiselect("Filter by Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
        with col2:
            condition_filter = st.multiselect("Filter by Condition", options=df["Condition"].unique(), default=df["Condition"].unique())
    filtered_df = df[df["Gender"].isin(gender_filter) & df["Condition"].isin(condition_filter)]

    if filtered_df.empty:
        st.warning("⚠️ No data available for selected filters.")
    else:
        # ---------------- INSIGHT CARDS ----------------
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("👥 Total Patients", len(filtered_df))
        with c2: st.metric("🚨 High Risk (>75)", len(filtered_df[filtered_df["RiskScore"] > 75]))
        with c3: st.metric("📅 Avg Age", f"{filtered_df['Age'].mean():.1f}")
        with c4: st.metric("📊 Avg Risk Score", f"{filtered_df['RiskScore'].mean():.1f}")

        st.markdown("---")

        # ---------------- TREND & DISTRIBUTION ----------------
        col_a, col_b = st.columns(2)
        with col_a:
            trend_data = filtered_df.groupby(filtered_df["LastVisit"].dt.to_period("M"))["RiskScore"].mean().reset_index()
            trend_data["LastVisit"] = trend_data["LastVisit"].astype(str)
            fig1 = px.line(trend_data, x="LastVisit", y="RiskScore", title="📈 Average Risk Over Time",
                           markers=True, line_shape="spline", color_discrete_sequence=["#6f42c1"])
            fig1.update_layout(xaxis_title="Month", yaxis_title="Average Risk Score")
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            fig2 = px.histogram(filtered_df, x="RiskScore", nbins=10,
                                title="🎯 Risk Score Distribution",
                                color_discrete_sequence=["#e63946"])
            fig2.update_layout(xaxis_title="Risk Score", yaxis_title="Count")
            st.plotly_chart(fig2, use_container_width=True)

        # ---------------- RISK SEGMENTATION ----------------
        st.markdown("### ⚠️ Risk Segmentation")
        bins = [0, 50, 75, 100]
        labels = ["Low (0-50)", "Medium (50-75)", "High (75-100)"]
        filtered_df["RiskCategory"] = pd.cut(filtered_df["RiskScore"], bins=bins, labels=labels)
        risk_counts = filtered_df["RiskCategory"].value_counts().reset_index()
        risk_counts.columns = ["RiskCategory", "Count"]

        fig3 = px.bar(risk_counts, x="RiskCategory", y="Count",
                      title="🧠 Patients by Risk Category",
                      color="RiskCategory",
                      color_discrete_map={
                          "Low (0-50)": "#2a9d8f",
                          "Medium (50-75)": "#f4a261",
                          "High (75-100)": "#e63946"
                      })
        st.plotly_chart(fig3, use_container_width=True)

        # ---------------- HEATMAP ----------------
        st.markdown("### 🔥 Condition vs Gender Heatmap")
        pivot = filtered_df.pivot_table(index="Condition", columns="Gender", values="RiskScore", aggfunc="mean").fillna(0)
        fig4 = px.imshow(pivot, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlBu_r",
                         title="💡 Average Risk by Condition & Gender")
        st.plotly_chart(fig4, use_container_width=True)

       # ---------------- SUMMARY INSIGHTS ----------------
st.markdown("### 🧾 Key Takeaways")
avg_risk = filtered_df['RiskScore'].mean()
high_risk_pct = (len(filtered_df[filtered_df['RiskScore'] > 75]) / len(filtered_df)) * 100

gender_avg = filtered_df.groupby('Gender')['RiskScore'].mean()

gender_risk_text = ""
if "Female" in gender_avg.index and "Male" in gender_avg.index:
    if gender_avg["Female"] > gender_avg["Male"]:
        gender_risk_text = "Female patients have a slightly **higher** average risk compared to males."
    elif gender_avg["Female"] < gender_avg["Male"]:
        gender_risk_text = "Female patients have a slightly **lower** average risk compared to males."
    else:
        gender_risk_text = "Male and Female patients have **similar average risk levels**."
else:
    gender_risk_text = "Gender-based comparison is unavailable for the current filter."

st.success(f"""
- The **average risk score** of patients is **{avg_risk:.1f}**.
- **{high_risk_pct:.1f}%** of patients are in the **high-risk category**, requiring immediate attention.
- The **most common condition** is **{filtered_df['Condition'].mode()[0]}**.
- {gender_risk_text}
""")

# ---------------- ALERTS ----------------
st.markdown("---")
st.markdown("## 🚨 Priority Alerts")
risk_threshold_global = 75
high_risk = df[df["RiskScore"] > risk_threshold_global]
if not high_risk.empty:
    for _, patient in high_risk.iterrows():
        last_visit_str = patient['LastVisit'].strftime('%Y-%m-%d') if pd.notna(patient.get('LastVisit')) else "N/A"
        st.markdown(f"""
        <div class='high-risk'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h4 style='margin: 0;'>🚑 {patient['Name']}</h4>
                    <p style='margin: 6px 0;'>Age: {patient['Age']} | Condition: {patient['Condition']}</p>
                </div>
                <div style='text-align: right;'>
                    <h3 style='margin: 0; color: #fff;'>RISK: {patient['RiskScore']:.1f}</h3>
                    <p style='margin: 0;'>Last Visit: {last_visit_str}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.success("✅ No high-risk patients at the moment!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 20px;'>
    <p>🏥 <strong>Hospital Patient Monitoring System</strong> | Built with Streamlit</p>
    <p>📞 For emergencies, contact: <strong>Medical Department</strong></p>
</div>
""", unsafe_allow_html=True)
