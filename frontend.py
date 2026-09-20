"""
Decoupled frontend — Streamlit client that calls the FastAPI backend.

Streamlit app yang bertindak sebagai CLIENT — tidak memuat model .pkl
secara langsung, melainkan mengirimkan HTTP POST request ke FastAPI backend.
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from datetime import datetime

#  Configuration 
API_BASE_URL = "http://localhost:8000"

CLF_ENDPOINT      = f"{API_BASE_URL}/predict/classification"
REG_ENDPOINT      = f"{API_BASE_URL}/predict/regression"
COMBINED_ENDPOINT = f"{API_BASE_URL}/predict/combined"
HEALTH_ENDPOINT   = f"{API_BASE_URL}/health"

TIMEOUT_SECONDS = 10

#  Page Config
st.set_page_config(
    page_title="Placement Predictor — Decoupled",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)


#  Helper: API Request 
def post_to_api(endpoint: str, payload: dict) -> tuple[dict | None, str | None]:
    """
    Kirim POST request ke FastAPI endpoint.

    Returns
    -------
    (response_dict, error_message)
    Salah satu selalu None.
    """
    try:
        resp = requests.post(endpoint, json=payload, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Tidak dapat terhubung ke API server. Pastikan `uvicorn api:app --reload` sudah berjalan di port 8000."
    except requests.exceptions.Timeout:
        return None, f"❌ Request timeout setelah {TIMEOUT_SECONDS} detik."
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = resp.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, f"❌ HTTP Error {resp.status_code}: {detail}"
    except Exception as e:
        return None, f"❌ Error tidak terduga: {str(e)}"


def get_health() -> tuple[dict | None, str | None]:
    """GET health check dari API."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


#  Helper: Visualization 
def plot_probability_bar(prob_placed: float, prob_not_placed: float):
    """Horizontal stacked bar chart probabilitas."""
    fig, ax = plt.subplots(figsize=(7, 1.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.barh(0, prob_placed,     color="#2ecc71", height=0.5, label=f"Placed ({prob_placed:.1%})")
    ax.barh(0, prob_not_placed, left=prob_placed, color="#e74c3c", height=0.5,
            label=f"Not Placed ({prob_not_placed:.1%})")

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1, alpha=0.7)
    ax.legend(loc="upper right", fontsize=8, framealpha=0)
    ax.set_title("Placement Probability Distribution", fontsize=9, pad=4)
    plt.tight_layout()
    return fig


def plot_salary_gauge(salary: float, low: float, high: float):
    """Horizontal range bar untuk salary estimasi."""
    fig, ax = plt.subplots(figsize=(7, 1.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Background bar (full range 0–20 LPA)
    ax.barh(0, 20, color="#ecf0f1", height=0.4, label="Max Range (20 LPA)")
    # Range bar
    ax.barh(0, high - low, left=low, color="#3498db", height=0.4, alpha=0.5, label=f"Range: ₹{low}–{high}L")
    # Point estimate
    ax.plot(salary, 0, "D", color="#2c3e50", markersize=10, zorder=5, label=f"Estimate: ₹{salary:.2f}L")
    # Industry average line
    ax.axvline(13.9, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8, label="Industry Avg (₹13.9L)")

    ax.set_xlim(0, 22)
    ax.set_yticks([])
    ax.set_xlabel("Salary (LPA)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0)
    ax.set_title("Salary Estimation Range", fontsize=9, pad=4)
    plt.tight_layout()
    return fig


def plot_scenario_comparison(results: list[dict]):
    """Bar chart perbandingan hasil antar skenario."""
    if not results:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")

    labels    = [r["label"] for r in results]
    probs     = [r.get("probability_placed", 0) for r in results]
    salaries  = [r.get("salary_lpa", 0) for r in results]
    colors    = ["#2ecc71" if p >= 0.5 else "#e74c3c" for p in probs]

    # Placement probability
    bars1 = axes[0].bar(labels, probs, color=colors, edgecolor="white", linewidth=1)
    axes[0].set_title("Placement Probability by Scenario", fontsize=10)
    axes[0].set_ylabel("Probability Placed")
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    for bar, v in zip(bars1, probs):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.1%}",
                     ha="center", fontsize=8, fontweight="bold")

    # Salary
    bars2 = axes[1].bar(labels, salaries, color="#3498db", edgecolor="white", linewidth=1)
    axes[1].set_title("Estimated Salary by Scenario (LPA)", fontsize=10)
    axes[1].set_ylabel("Salary (LPA)")
    axes[1].set_ylim(0, 22)
    axes[1].axhline(13.9, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="Avg ₹13.9L")
    axes[1].legend(fontsize=8)
    for bar, v in zip(bars2, salaries):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.3, f"₹{v:.1f}L",
                     ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    return fig


#  Predefined Test Scenarios    
SCENARIOS = {
    "🟢 Scenario A — Profil Kuat (CSE, CGPA 9+)": {
        "cgpa": 9.2, "tenth_percentage": 90.0, "twelfth_percentage": 88.0,
        "backlogs": 0, "attendance_percentage": 92.0, "study_hours_per_day": 7.0,
        "sleep_hours": 7.5, "stress_level": 3,
        "coding_skill_rating": 5, "communication_skill_rating": 5, "aptitude_skill_rating": 5,
        "projects_completed": 7, "internships_completed": 3, "hackathons_participated": 5,
        "certifications_count": 5, "gender": "Male", "branch": "CSE",
        "part_time_job": "No", "family_income_level": "High",
        "city_tier": "Tier 1", "internet_access": "Yes",
        "extracurricular_involvement": "High"
    },
    "🔴 Scenario B — Profil Lemah (ME, Backlogs)": {
        "cgpa": 5.8, "tenth_percentage": 55.0, "twelfth_percentage": 52.0,
        "backlogs": 3, "attendance_percentage": 50.0, "study_hours_per_day": 1.5,
        "sleep_hours": 5.0, "stress_level": 9,
        "coding_skill_rating": 1, "communication_skill_rating": 2, "aptitude_skill_rating": 2,
        "projects_completed": 1, "internships_completed": 0, "hackathons_participated": 0,
        "certifications_count": 0, "gender": "Female", "branch": "ME",
        "part_time_job": "Yes", "family_income_level": "Low",
        "city_tier": "Tier 3", "internet_access": "No",
        "extracurricular_involvement": "Low"
    },
    "🟡 Scenario C — Profil IT Menengah": {
        "cgpa": 7.8, "tenth_percentage": 75.0, "twelfth_percentage": 72.0,
        "backlogs": 1, "attendance_percentage": 75.0, "study_hours_per_day": 4.0,
        "sleep_hours": 7.0, "stress_level": 6,
        "coding_skill_rating": 3, "communication_skill_rating": 3, "aptitude_skill_rating": 4,
        "projects_completed": 4, "internships_completed": 1, "hackathons_participated": 2,
        "certifications_count": 2, "gender": "Male", "branch": "IT",
        "part_time_job": "No", "family_income_level": "Medium",
        "city_tier": "Tier 2", "internet_access": "Yes",
        "extracurricular_involvement": "Medium"
    },
    "🔵 Scenario D — Profil ECE dengan Skill Tinggi": {
        "cgpa": 8.5, "tenth_percentage": 82.0, "twelfth_percentage": 80.0,
        "backlogs": 0, "attendance_percentage": 88.0, "study_hours_per_day": 5.5,
        "sleep_hours": 7.0, "stress_level": 4,
        "coding_skill_rating": 4, "communication_skill_rating": 4, "aptitude_skill_rating": 5,
        "projects_completed": 5, "internships_completed": 2, "hackathons_participated": 3,
        "certifications_count": 4, "gender": "Female", "branch": "ECE",
        "part_time_job": "No", "family_income_level": "Medium",
        "city_tier": "Tier 1", "internet_access": "Yes",
        "extracurricular_involvement": "High"
    }
}



# MAIN APP

def main():

    #  Header 
    st.markdown("""
    <h1 style='text-align:center; color:#2c3e50;'>
        🔗 Student Placement Predictor
    </h1>
    <p style='text-align:center; color:#7f8c8d; font-size:15px;'>
        <b>Decoupled Architecture</b> — Streamlit Frontend → FastAPI Backend
    </p>
    """, unsafe_allow_html=True)

    #    API Status Banner
    health, err = get_health()
    if health and health.get("models_loaded"):
        st.success(f"✅ API Connected | CLF: {health['clf_model']} | REG: {health['reg_model']} | v{health['api_version']}")
    else:
        st.error("🔴 API server tidak terhubung. Jalankan: `uvicorn api:app --reload --port 8000`")
        st.info("Frontend tetap dapat diakses — prediksi akan gagal sampai backend aktif.")

    st.markdown("---")

    #  Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔵 Klasifikasi", "🟢 Regresi", "🔄 Combined", "📊 Perbandingan Skenario"
    ])

   
    # TAB 1 — Classification
   
    with tab1:
        st.markdown("### 🔵 Scenario: Classification — POST `/predict/classification`")
        st.markdown("Pilih skenario atau isi manual untuk memprediksi **placement status**.")

        col_left, col_right = st.columns([1, 1.3])

        with col_left:
            st.markdown("#### Input")

            scenario_clf = st.selectbox(
                "Pilih Skenario Preset",
                ["Manual Input"] + list(SCENARIOS.keys()),
                key="clf_scenario"
            )

            if scenario_clf != "Manual Input":
                d = SCENARIOS[scenario_clf].copy()
                st.info(f"Data di-load dari: **{scenario_clf}**")
            else:
                d = {}

            cgpa_clf    = st.slider("CGPA", 5.0, 10.0, float(d.get("cgpa", 8.0)), 0.1, key="cgpa_clf")
            backlogs_clf = st.number_input("Backlogs", 0, 10, int(d.get("backlogs", 0)), key="bl_clf")
            coding_clf  = st.select_slider("Coding Skill", [1,2,3,4,5], int(d.get("coding_skill_rating",3)), key="cs_clf")
            projects_clf = st.number_input("Projects", 0, 8, int(d.get("projects_completed",3)), key="pj_clf")
            internships_clf = st.number_input("Internships", 0, 4, int(d.get("internships_completed",1)), key="in_clf")
            branch_clf  = st.selectbox("Branch", ["CSE","IT","ECE","CE","ME"],
                                       index=["CSE","IT","ECE","CE","ME"].index(d.get("branch","CSE")),
                                       key="br_clf")

            if st.button("🚀 Send POST /predict/classification", type="primary", key="btn_clf"):
                # Build payload dari skenario + override manual
                base = SCENARIOS.get(scenario_clf, SCENARIOS["🟢 Scenario A — Profil Kuat (CSE, CGPA 9+)"]).copy() if scenario_clf != "Manual Input" else {
                    "tenth_percentage": 75.0, "twelfth_percentage": 73.0,
                    "attendance_percentage": 80.0, "study_hours_per_day": 4.0,
                    "sleep_hours": 7.0, "stress_level": 5,
                    "communication_skill_rating": 3, "aptitude_skill_rating": 4,
                    "hackathons_participated": 2, "certifications_count": 2,
                    "gender": "Male", "part_time_job": "No",
                    "family_income_level": "Medium", "city_tier": "Tier 2",
                    "internet_access": "Yes", "extracurricular_involvement": "Medium"
                }
                base.update({
                    "cgpa": cgpa_clf, "backlogs": int(backlogs_clf),
                    "coding_skill_rating": coding_clf,
                    "projects_completed": int(projects_clf),
                    "internships_completed": int(internships_clf),
                    "branch": branch_clf
                })

                with st.spinner("Mengirim request ke FastAPI..."):
                    result, error = post_to_api(CLF_ENDPOINT, base)

                if error:
                    st.error(error)
                else:
                    st.session_state["clf_result"]  = result
                    st.session_state["clf_payload"] = base

        with col_right:
            st.markdown("#### Response")
            if "clf_result" in st.session_state:
                r = st.session_state["clf_result"]
                label = r["placement_label"]
                color = "#2ecc71" if label == "Placed" else "#e74c3c"
                icon  = "✅" if label == "Placed" else "❌"

                st.markdown(f"""
                <div style='background:{color}22; border:2px solid {color};
                            border-radius:10px; padding:18px; text-align:center; margin-bottom:12px;'>
                    <h2 style='color:{color}; margin:0;'>{icon} {label}</h2>
                    <p style='color:#555; margin:4px 0 0;'>Confidence: <b>{r["confidence"]:.1%}</b></p>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar
                fig_p = plot_probability_bar(r["probability_placed"], r["probability_not_placed"])
                st.pyplot(fig_p, use_container_width=True)

                # Raw JSON response
                with st.expander("📄 Raw JSON Response"):
                    st.json(r)

                # Request payload
                with st.expander("📤 Request Payload yang Dikirim"):
                    st.json(st.session_state["clf_payload"])

                # Endpoint info
                st.caption(f"🔗 Endpoint: `POST {CLF_ENDPOINT}`  |  🕒 {datetime.now().strftime('%H:%M:%S')}")
            else:
                st.info("Tekan tombol **Send POST** untuk melihat response di sini.")

    
    # TAB 2 — Regression
    
    with tab2:
        st.markdown("### 🟢 Scenario: Regression — POST `/predict/regression`")
        st.markdown("Pilih skenario atau isi manual untuk memprediksi **estimasi salary**.")

        col_l2, col_r2 = st.columns([1, 1.3])

        with col_l2:
            st.markdown("#### Input")

            scenario_reg = st.selectbox(
                "Pilih Skenario Preset",
                ["Manual Input"] + list(SCENARIOS.keys()),
                key="reg_scenario"
            )

            if scenario_reg != "Manual Input":
                d2 = SCENARIOS[scenario_reg].copy()
                st.info(f"Data di-load dari: **{scenario_reg}**")
            else:
                d2 = {}

            cgpa_reg    = st.slider("CGPA", 5.0, 10.0, float(d2.get("cgpa", 8.0)), 0.1, key="cgpa_reg")
            intern_reg  = st.number_input("Internships", 0, 4, int(d2.get("internships_completed",1)), key="in_reg")
            skill_reg   = st.select_slider("Coding Skill", [1,2,3,4,5], int(d2.get("coding_skill_rating",3)), key="cs_reg")
            hack_reg    = st.number_input("Hackathons", 0, 6, int(d2.get("hackathons_participated",2)), key="hk_reg")
            branch_reg  = st.selectbox("Branch", ["CSE","IT","ECE","CE","ME"],
                                       index=["CSE","IT","ECE","CE","ME"].index(d2.get("branch","CSE")),
                                       key="br_reg")

            if st.button("🚀 Send POST /predict/regression", type="primary", key="btn_reg"):
                base2 = SCENARIOS.get(scenario_reg, SCENARIOS["🟢 Scenario A — Profil Kuat (CSE, CGPA 9+)"]).copy() if scenario_reg != "Manual Input" else {
                    "tenth_percentage": 75.0, "twelfth_percentage": 73.0,
                    "backlogs": 0, "attendance_percentage": 80.0,
                    "study_hours_per_day": 4.0, "sleep_hours": 7.0, "stress_level": 5,
                    "communication_skill_rating": 3, "aptitude_skill_rating": 4,
                    "projects_completed": 3, "certifications_count": 2,
                    "gender": "Male", "part_time_job": "No",
                    "family_income_level": "Medium", "city_tier": "Tier 2",
                    "internet_access": "Yes", "extracurricular_involvement": "Medium"
                }
                base2.update({
                    "cgpa": cgpa_reg, "internships_completed": int(intern_reg),
                    "coding_skill_rating": skill_reg,
                    "hackathons_participated": int(hack_reg),
                    "branch": branch_reg
                })

                with st.spinner("Mengirim request ke FastAPI..."):
                    result2, error2 = post_to_api(REG_ENDPOINT, base2)

                if error2:
                    st.error(error2)
                else:
                    st.session_state["reg_result"]  = result2
                    st.session_state["reg_payload"] = base2

        with col_r2:
            st.markdown("#### Response")
            if "reg_result" in st.session_state:
                r2 = st.session_state["reg_result"]

                st.markdown(f"""
                <div style='background:#3498db22; border:2px solid #3498db;
                            border-radius:10px; padding:18px; text-align:center; margin-bottom:12px;'>
                    <h2 style='color:#3498db; margin:0;'>₹ {r2["salary_lpa"]:.2f} LPA</h2>
                    <p style='color:#555; margin:4px 0 0;'>Range: ₹{r2["salary_range_low"]} – ₹{r2["salary_range_high"]} LPA</p>
                </div>
                """, unsafe_allow_html=True)

                fig_s = plot_salary_gauge(r2["salary_lpa"], r2["salary_range_low"], r2["salary_range_high"])
                st.pyplot(fig_s, use_container_width=True)

                with st.expander("📄 Raw JSON Response"):
                    st.json(r2)
                with st.expander("📤 Request Payload yang Dikirim"):
                    st.json(st.session_state["reg_payload"])
                st.caption(f"🔗 Endpoint: `POST {REG_ENDPOINT}`  |  🕒 {datetime.now().strftime('%H:%M:%S')}")
            else:
                st.info("Tekan tombol **Send POST** untuk melihat response di sini.")

   
    # TAB 3 — Combined
    
    with tab3:
        st.markdown("### 🔄 Combined — POST `/predict/combined`")
        st.markdown("Kirim satu request dan dapatkan **kedua prediksi** sekaligus.")

        c1, c2, c3 = st.columns(3)
        with c1:
            scenario_comb = st.selectbox(
                "Pilih Skenario",
                list(SCENARIOS.keys()),
                key="comb_scenario"
            )
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            send_comb = st.button("🚀 Send POST /predict/combined", type="primary", key="btn_comb")

        if send_comb:
            payload_comb = SCENARIOS[scenario_comb].copy()
            with st.spinner("Mengirim combined request..."):
                result_comb, err_comb = post_to_api(COMBINED_ENDPOINT, payload_comb)

            if err_comb:
                st.error(err_comb)
            else:
                st.session_state["comb_result"]  = result_comb
                st.session_state["comb_payload"] = payload_comb

        if "comb_result" in st.session_state:
            rc = st.session_state["comb_result"]

            st.markdown("#### Results")
            m1, m2, m3 = st.columns(3)

            label = rc["placement_label"]
            color = "#2ecc71" if label == "Placed" else "#e74c3c"
            icon  = "✅" if label == "Placed" else "❌"

            m1.markdown(f"""
            <div style='background:{color}22; border:2px solid {color}; border-radius:10px;
                        padding:14px; text-align:center;'>
                <h3 style='color:{color}; margin:0;'>{icon} {label}</h3>
                <small>Placement Status</small>
            </div>
            """, unsafe_allow_html=True)

            m2.markdown(f"""
            <div style='background:#9b59b622; border:2px solid #9b59b6; border-radius:10px;
                        padding:14px; text-align:center;'>
                <h3 style='color:#9b59b6; margin:0;'>{rc["confidence"]:.1%}</h3>
                <small>Confidence Score</small>
            </div>
            """, unsafe_allow_html=True)

            m3.markdown(f"""
            <div style='background:#3498db22; border:2px solid #3498db; border-radius:10px;
                        padding:14px; text-align:center;'>
                <h3 style='color:#3498db; margin:0;'>₹{rc["salary_lpa"]:.2f} LPA</h3>
                <small>Estimated Salary</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            vc1, vc2 = st.columns(2)
            with vc1:
                fig_pb = plot_probability_bar(rc["probability_placed"], rc["probability_not_placed"])
                st.pyplot(fig_pb, use_container_width=True)
            with vc2:
                fig_sg = plot_salary_gauge(rc["salary_lpa"], rc["salary_range_low"], rc["salary_range_high"])
                st.pyplot(fig_sg, use_container_width=True)

            with st.expander("📄 Full JSON Response"):
                st.json(rc)
            st.caption(f"🔗 `POST {COMBINED_ENDPOINT}` | Models: {rc['clf_model']} + {rc['reg_model']} | 🕒 {datetime.now().strftime('%H:%M:%S')}")

        else:
            st.info("Pilih skenario dan klik **Send POST** untuk melihat combined response.")

    
    # TAB 4 — Scenario Comparison
    
    with tab4:
        st.markdown("### 📊 Perbandingan Semua Skenario")
        st.markdown("Jalankan **semua 4 skenario** sekaligus dan bandingkan hasilnya.")

        if st.button("🚀 Run All Scenarios (4x POST /predict/combined)", type="primary"):
            all_results = []
            progress = st.progress(0, text="Mengirim request...")

            for i, (name, payload) in enumerate(SCENARIOS.items()):
                progress.progress((i+1)/len(SCENARIOS), text=f"Running {name[:30]}...")
                res, err = post_to_api(COMBINED_ENDPOINT, payload)
                if res:
                    all_results.append({
                        "label":              name[:25],
                        "probability_placed": res["probability_placed"],
                        "salary_lpa":         res["salary_lpa"],
                        "placement_label":    res["placement_label"],
                        "confidence":         res["confidence"]
                    })
                else:
                    st.warning(f"Gagal untuk {name}: {err}")

            progress.empty()

            if all_results:
                st.session_state["all_results"] = all_results

        if "all_results" in st.session_state:
            results = st.session_state["all_results"]

            # Summary table
            df_summary = pd.DataFrame(results)
            df_summary.columns = ["Scenario", "P(Placed)", "Salary (LPA)", "Prediction", "Confidence"]
            df_summary["P(Placed)"]    = df_summary["P(Placed)"].apply(lambda x: f"{x:.1%}")
            df_summary["Salary (LPA)"] = df_summary["Salary (LPA)"].apply(lambda x: f"₹{x:.2f}")
            df_summary["Confidence"]   = df_summary["Confidence"].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

            # Chart comparison
            fig_comp = plot_scenario_comparison(st.session_state["all_results"])
            if fig_comp:
                st.pyplot(fig_comp, use_container_width=True)

            st.caption(f"Total: {len(results)} API calls berhasil | 🕒 {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.info("Klik tombol di atas untuk menjalankan semua skenario.")


if __name__ == "__main__":
    main()
