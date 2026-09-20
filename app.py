
# Monolithic deployment — model loaded directly in the Streamlit app.


import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from feature_engineering import engineer_features, get_feature_lists
#  Page Configuration 
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  Constants 
BASE_DIR       = Path(__file__).parent
CLF_MODEL_PATH = BASE_DIR / "models" / "model_classification.pkl"
REG_MODEL_PATH = BASE_DIR / "models" / "model_regression.pkl"

NUM_COLS = [
    "cgpa", "tenth_percentage", "twelfth_percentage", "backlogs",
    "study_hours_per_day", "attendance_percentage", "projects_completed",
    "internships_completed", "coding_skill_rating", "communication_skill_rating",
    "aptitude_skill_rating", "hackathons_participated", "certifications_count",
    "sleep_hours", "stress_level",
    "academic_score", "experience_index", "skill_avg"
]
CAT_COLS = [
    "gender", "branch", "part_time_job", "family_income_level",
    "city_tier", "internet_access", "extracurricular_involvement"
]


#  Model Loading 
@st.cache_resource
def load_models():
    """Load kedua model .pkl — di-cache agar tidak reload setiap interaksi."""
    clf_model = joblib.load(CLF_MODEL_PATH)
    reg_model = joblib.load(REG_MODEL_PATH)
    return clf_model, reg_model


#  Feature Engineering 
def engineer_features(data: dict) -> pd.DataFrame:
    """
    Tambahkan 3 fitur turunan (sama persis dengan feature_engineering.py).
    Harus konsisten dengan saat training — perubahan di sini akan
    menyebabkan prediksi tidak valid.
    """
    data["academic_score"] = (
        data["cgpa"] / 10 * 100 +
        data["tenth_percentage"] +
        data["twelfth_percentage"]
    ) / 3

    data["experience_index"] = (
        data["projects_completed"] +
        data["internships_completed"] * 2 +
        data["hackathons_participated"] +
        data["certifications_count"]
    )

    data["skill_avg"] = (
        data["coding_skill_rating"] +
        data["communication_skill_rating"] +
        data["aptitude_skill_rating"]
    ) / 3

    df = pd.DataFrame([data])[NUM_COLS + CAT_COLS]
    return df


# Prediction 
def predict(clf_model, reg_model, input_df: pd.DataFrame) -> dict:
    """Jalankan prediksi untuk kedua task."""
    placement_pred = clf_model.predict(input_df)[0]
    placement_prob = clf_model.predict_proba(input_df)[0]

    salary_pred = reg_model.predict(input_df)[0]
    salary_pred = max(0.0, salary_pred)   # clip agar tidak negatif

    label   = "Placed" if placement_pred == 1 else "Not Placed"
    conf    = placement_prob[1] if placement_pred == 1 else placement_prob[0]

    return {
        "placement_label": label,
        "placement_conf":  conf,
        "prob_placed":     placement_prob[1],
        "prob_not_placed": placement_prob[0],
        "salary_lpa":      salary_pred
    }


# Gauge Chart 
def plot_gauge(prob: float, title: str):
    """Gauge chart untuk menampilkan probabilitas."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    theta = np.linspace(np.pi, 0, 100)
    ax.plot(theta, [1] * 100, color="#e0e0e0", linewidth=15, solid_capstyle="round")

    filled = int(prob * 100)
    color  = "#2ecc71" if prob >= 0.6 else ("#f39c12" if prob >= 0.4 else "#e74c3c")
    ax.plot(theta[:filled], [1] * filled, color=color, linewidth=15, solid_capstyle="round")

    ax.set_ylim(0, 1.5)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.axis("off")
    ax.text(0, -0.2, f"{prob:.1%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=color,
            transform=ax.transData)
    ax.set_title(title, fontsize=10, pad=5, color="#555")
    return fig


# Feature Radar Chart 
def plot_radar(data: dict):
    """Radar chart profil mahasiswa berdasarkan 6 dimensi utama."""
    categories = ["CGPA\n(/10)", "Coding\nSkill", "Projects", "Internships",
                  "Hackathons", "Attendance\n(/100)"]
    raw_values = [
        data["cgpa"] / 10,
        data["coding_skill_rating"] / 5,
        data["projects_completed"] / 8,
        data["internships_completed"] / 4,
        data["hackathons_participated"] / 6,
        data["attendance_percentage"] / 100,
    ]
    values = raw_values + [raw_values[0]]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, "o-", linewidth=2, color="#3498db")
    ax.fill(angles, values, alpha=0.25, color="#3498db")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6, color="gray")
    ax.set_title("Student Profile Radar", fontsize=11, pad=15)
    return fig


#  Salary Benchmark Chart 
def plot_salary_benchmark(salary: float):
    """Bar chart bandingkan prediksi salary vs benchmark industri."""
    labels    = ["Entry Level\n(CSE/IT)", "Mid Level\n(All Branch)", "Predicted\nSalary", "Senior Level\n(Top 25%)"]
    values    = [8.0, 13.9, salary, 18.3]
    colors    = ["#bdc3c7", "#95a5a6", "#2ecc71" if salary >= 13.9 else "#e74c3c", "#7f8c8d"]

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Salary (LPA)", fontsize=9)
    ax.set_title("Salary vs Industry Benchmark", fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"₹{val:.1f}L", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 23)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


# 
# MAIN APP
#
def main():

    # Load models 
    try:
        clf_model, reg_model = load_models()
    except FileNotFoundError as e:
        st.error(f"❌ Model file tidak ditemukan: {e}")
        st.info("Pastikan kamu sudah menjalankan `python pipeline.py` terlebih dahulu.")
        st.stop()

    # Header 
    st.markdown("""
    <h1 style='text-align:center; color:#2c3e50;'>
        🎓 Student Placement Predictor
    </h1>
    <p style='text-align:center; color:#7f8c8d; font-size:16px;'>
        Prediksi peluang penempatan kerja dan estimasi gaji berdasarkan profil akademik & skill mahasiswa
    </p>
    <hr style='border: 1px solid #ecf0f1;'>
    """, unsafe_allow_html=True)

    # Sidebar — Input Form 
    with st.sidebar:
        st.markdown("## 📋 Student Profile Input")
        st.markdown("---")

        # Academic Info
        st.markdown("### 🎓 Academic Performance")
        cgpa               = st.slider("CGPA", 5.0, 10.0, 8.0, step=0.1)
        tenth_percentage   = st.slider("10th Percentage (%)", 50.0, 100.0, 75.0, step=0.5)
        twelfth_percentage = st.slider("12th Percentage (%)", 50.0, 100.0, 75.0, step=0.5)
        backlogs           = st.number_input("Number of Backlogs", 0, 10, 0)
        attendance_pct     = st.slider("Attendance (%)", 40.0, 100.0, 75.0, step=0.5)

        st.markdown("---")
        st.markdown("### 💡 Skills & Experience")
        coding_skill   = st.select_slider("Coding Skill (1-5)",     options=[1,2,3,4,5], value=3)
        comm_skill     = st.select_slider("Communication Skill (1-5)", options=[1,2,3,4,5], value=3)
        apt_skill      = st.select_slider("Aptitude Skill (1-5)",   options=[1,2,3,4,5], value=4)
        projects       = st.number_input("Projects Completed",       0, 8, 3)
        internships    = st.number_input("Internships Completed",    0, 4, 1)
        hackathons     = st.number_input("Hackathons Participated",  0, 6, 2)
        certifications = st.number_input("Certifications Count",     0, 9, 2)

        st.markdown("---")
        st.markdown("### 🌍 Personal & Lifestyle")
        gender          = st.radio("Gender", ["Male", "Female"], horizontal=True)
        branch          = st.selectbox("Branch", ["CSE", "IT", "ECE", "CE", "ME"])
        part_time_job   = st.radio("Part-time Job?", ["No", "Yes"], horizontal=True)
        income_level    = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
        city_tier       = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        internet_access = st.radio("Internet Access?", ["Yes", "No"], horizontal=True)
        extra_curr      = st.selectbox("Extracurricular Involvement", ["Low", "Medium", "High"])
        study_hours     = st.slider("Study Hours / Day",  0.0, 10.0, 4.0, step=0.5)
        sleep_hours     = st.slider("Sleep Hours / Night", 4.0,  9.0, 7.0, step=0.5)
        stress_level    = st.slider("Stress Level (1-10)", 1, 10, 5)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Now", use_container_width=True, type="primary")

    #  Main Content — Tabs 
    tab1, tab2, tab3 = st.tabs(["📊 Prediction Results", "📈 Profile Analysis", "ℹ️ Model Info"])

    # Siapkan input data
    input_data = {
        "cgpa": cgpa, "tenth_percentage": tenth_percentage,
        "twelfth_percentage": twelfth_percentage, "backlogs": int(backlogs),
        "study_hours_per_day": study_hours, "attendance_percentage": attendance_pct,
        "projects_completed": int(projects), "internships_completed": int(internships),
        "coding_skill_rating": coding_skill, "communication_skill_rating": comm_skill,
        "aptitude_skill_rating": apt_skill, "hackathons_participated": int(hackathons),
        "certifications_count": int(certifications), "sleep_hours": sleep_hours,
        "stress_level": int(stress_level), "gender": gender, "branch": branch,
        "part_time_job": part_time_job, "family_income_level": income_level,
        "city_tier": city_tier, "internet_access": internet_access,
        "extracurricular_involvement": extra_curr
    }

  
    # TAB 1 — Prediction Results

    with tab1:
        if predict_btn:
            with st.spinner("Menjalankan prediksi..."):
                input_df = engineer_features(input_data)
                results  = predict(clf_model, reg_model, input_df)

            st.markdown("### 🎯 Prediction Results")

            # ── Classification Result ──
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                label = results["placement_label"]
                color = "#2ecc71" if label == "Placed" else "#e74c3c"
                icon  = "✅" if label == "Placed" else "❌"
                st.markdown(f"""
                <div style='background:{color}22; border:2px solid {color};
                            border-radius:12px; padding:20px; text-align:center;'>
                    <h2 style='color:{color}; margin:0;'>{icon} {label}</h2>
                    <p style='color:#555; margin:4px 0 0 0;'>Placement Prediction</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                salary = results["salary_lpa"]
                st.markdown(f"""
                <div style='background:#3498db22; border:2px solid #3498db;
                            border-radius:12px; padding:20px; text-align:center;'>
                    <h2 style='color:#3498db; margin:0;'>₹ {salary:.2f} LPA</h2>
                    <p style='color:#555; margin:4px 0 0 0;'>Estimated Salary</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                conf = results["placement_conf"]
                st.markdown(f"""
                <div style='background:#9b59b622; border:2px solid #9b59b6;
                            border-radius:12px; padding:20px; text-align:center;'>
                    <h2 style='color:#9b59b6; margin:0;'>{conf:.1%}</h2>
                    <p style='color:#555; margin:4px 0 0 0;'>Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Probability Gauges ──
            st.markdown("### 📊 Placement Probability")
            g1, g2 = st.columns(2)
            with g1:
                fig_g1 = plot_gauge(results["prob_placed"], "Probability: Placed")
                st.pyplot(fig_g1, use_container_width=True)
            with g2:
                fig_g2 = plot_gauge(results["prob_not_placed"], "Probability: Not Placed")
                st.pyplot(fig_g2, use_container_width=True)

            st.markdown("---")

            # ── Salary Benchmark ──
            st.markdown("### 💰 Salary Benchmark")
            fig_sal = plot_salary_benchmark(results["salary_lpa"])
            st.pyplot(fig_sal, use_container_width=True)

            # ── Insight Message ──
            st.markdown("---")
            st.markdown("### 💬 Insight")
            if label == "Placed" and salary >= 15:
                st.success("🌟 Profil sangat kuat! CGPA, skill, dan pengalaman berada di level kompetitif. Potensi salary di atas rata-rata industri.")
            elif label == "Placed" and salary < 15:
                st.info("✅ Peluang placement baik. Untuk meningkatkan salary, pertimbangkan menambah internship atau sertifikasi di bidang teknis.")
            else:
                st.warning("⚠️ Risiko tidak placed cukup tinggi. Area yang perlu ditingkatkan: kurangi backlogs, tingkatkan coding skill rating, dan tambah pengalaman proyek.")

        else:
            # Default state sebelum predict
            st.markdown("""
            <div style='text-align:center; padding:60px; color:#95a5a6;'>
                <h3>👈 Isi profil mahasiswa di sidebar</h3>
                <p>Kemudian klik <strong>Predict Now</strong> untuk melihat hasil prediksi.</p>
            </div>
            """, unsafe_allow_html=True)

    # TAB 2 — Profile Analysis

    with tab2:
        st.markdown("### 🧠 Student Profile Analysis")
        st.markdown("Visualisasi profil berdasarkan input saat ini (real-time, tanpa perlu klik Predict).")

        col_r, col_m = st.columns([1, 1])

        with col_r:
            fig_radar = plot_radar(input_data)
            st.pyplot(fig_radar, use_container_width=True)

        with col_m:
            # Engineered features display
            acad = (cgpa/10*100 + tenth_percentage + twelfth_percentage) / 3
            exp  = projects + internships*2 + hackathons + certifications
            skl  = (coding_skill + comm_skill + apt_skill) / 3

            st.markdown("#### 📐 Engineered Features")
            st.metric("Academic Score",    f"{acad:.2f} / 100")
            st.metric("Experience Index",  f"{exp} pts",
                      help="Projects + Internships×2 + Hackathons + Certifications")
            st.metric("Skill Average",     f"{skl:.2f} / 5")

            st.markdown("---")
            st.markdown("#### 📋 Key Academic Metrics")
            m1, m2 = st.columns(2)
            m1.metric("CGPA",         f"{cgpa:.1f}")
            m1.metric("10th %",       f"{tenth_percentage:.1f}%")
            m2.metric("12th %",       f"{twelfth_percentage:.1f}%")
            m2.metric("Backlogs",     str(int(backlogs)),
                      delta=f"-{int(backlogs)} risk" if backlogs > 0 else "Clean ✓",
                      delta_color="inverse")

        st.markdown("---")
        st.markdown("#### 📊 Raw Feature Scores")
        feature_df = pd.DataFrame({
            "Feature": ["Coding Skill", "Communication", "Aptitude", "Projects",
                        "Internships", "Hackathons", "Certifications", "Study Hours",
                        "Attendance %", "Stress Level"],
            "Value": [coding_skill, comm_skill, apt_skill, int(projects),
                      int(internships), int(hackathons), int(certifications),
                      study_hours, attendance_pct, int(stress_level)],
            "Max": [5, 5, 5, 8, 4, 6, 9, 10, 100, 10]
        })
        feature_df["Score (%)"] = (feature_df["Value"] / feature_df["Max"] * 100).round(1)
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    # TAB 3 — Model Info
    
    with tab3:
        st.markdown("### ℹ️ Model Information")

        col_i1, col_i2 = st.columns(2)

        with col_i1:
            st.markdown("""
            #### 🔵 Classification Model
            | Parameter | Value |
            |---|---|
            | Algorithm | Logistic Regression |
            | Class Weight | balanced |
            | Max Iter | 1000 |
            | C (regularization) | 1.0 |
            | Test Accuracy | ~0.81 |
            | ROC-AUC | ~0.90 |
            | F1 (weighted) | ~0.84 |
            """)

        with col_i2:
            st.markdown("""
            #### 🟢 Regression Model
            | Parameter | Value |
            |---|---|
            | Algorithm | Random Forest Regressor |
            | N Estimators | 100 |
            | Max Depth | None (full) |
            | Random State | 42 |
            | MAE | ~2.70 LPA |
            | RMSE | ~4.10 LPA |
            | R² Score | ~0.58 |
            """)

        st.markdown("---")
        st.markdown("""
        #### 📌 Dataset Info
        - **Source**: Dataset A (NIM Ganjil)
        - **Total Rows**: 5,000 mahasiswa
        - **Features**: 22 raw + 3 engineered = 25 total
        - **Class Balance**: Placed 86.1% / Not Placed 13.9%
        - **Train-Test Split**: 80:20 (stratified)

        #### 📌 Feature Engineering
        | Fitur | Formula |
        |---|---|
        | `academic_score` | (CGPA×10 + 10th% + 12th%) / 3 |
        | `experience_index` | Projects + Internships×2 + Hackathons + Certs |
        | `skill_avg` | (Coding + Comm + Aptitude) / 3 |
        """)


if __name__ == "__main__":
    main()
