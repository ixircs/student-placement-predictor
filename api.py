import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Literal

#Path Configuration
BASE_DIR       = Path(__file__).parent
CLF_MODEL_PATH = BASE_DIR / "models" / "model_classification.pkl"
REG_MODEL_PATH = BASE_DIR / "models" / "model_regression.pkl"

#Feature Column Order 
NUM_COLS = [
    "cgpa", "tenth_percentage", "twelfth_percentage", "backlogs",
    "study_hours_per_day", "attendance_percentage", "projects_completed",
    "internships_completed", "coding_skill_rating", "communication_skill_rating",
    "aptitude_skill_rating", "hackathons_participated", "certifications_count",
    "sleep_hours", "stress_level",
    "academic_score", "experience_index", "skill_avg"   # engineered
]
CAT_COLS = [
    "gender", "branch", "part_time_job", "family_income_level",
    "city_tier", "internet_access", "extracurricular_involvement"
]
ALL_COLS = NUM_COLS + CAT_COLS


app = FastAPI(
    title="Student Placement Prediction API",
    description="API prediksi placement status dan estimasi salary mahasiswa menggunakan Machine Learning.\n\n",
    version="1.0.0",
    contact={
        "name": "Ryan Christopher Setiawan",
    }
)

# CORS mengizinkan Streamlit frontend mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model Loading
def load_models():
    """Load kedua model .pkl saat startup."""
    if not CLF_MODEL_PATH.exists():
        raise FileNotFoundError(f"CLF model tidak ditemukan: {CLF_MODEL_PATH}")
    if not REG_MODEL_PATH.exists():
        raise FileNotFoundError(f"REG model tidak ditemukan: {REG_MODEL_PATH}")

    clf = joblib.load(CLF_MODEL_PATH)
    reg = joblib.load(REG_MODEL_PATH)
    return clf, reg

try:
    clf_model, reg_model = load_models()
    MODELS_LOADED = True
except FileNotFoundError as e:
    print(f"[WARNING] {e}")
    print("[WARNING] Jalankan pipeline.py terlebih dahulu untuk generate model .pkl")
    MODELS_LOADED = False


#Pydantic Schemas 

class StudentInput(BaseModel):
    """Data input mahasiswa untuk prediksi placement dan salary."""

    # Academic Performance
    cgpa: float = Field(..., ge=5.0, le=10.0,
                        description="CGPA mahasiswa (5.0 – 10.0)",
                        example=8.5)
    tenth_percentage: float = Field(..., ge=50.0, le=100.0,
                                    description="Nilai ujian kelas 10 (%)",
                                    example=80.0)
    twelfth_percentage: float = Field(..., ge=50.0, le=100.0,
                                      description="Nilai ujian kelas 12 (%)",
                                      example=78.0)
    backlogs: int = Field(..., ge=0, le=10,
                          description="Jumlah mata kuliah yang tidak lulus",
                          example=0)
    attendance_percentage: float = Field(..., ge=40.0, le=100.0,
                                         description="Persentase kehadiran (%)",
                                         example=85.0)

    # Study & Lifestyle
    study_hours_per_day: float = Field(..., ge=0.0, le=10.0,
                                       description="Rata-rata jam belajar per hari",
                                       example=5.0)
    sleep_hours: float = Field(..., ge=4.0, le=9.0,
                               description="Rata-rata jam tidur per malam",
                               example=7.0)
    stress_level: int = Field(..., ge=1, le=10,
                              description="Tingkat stres (1=sangat rendah, 10=sangat tinggi)",
                              example=5)

    # Skills
    coding_skill_rating: int = Field(..., ge=1, le=5,
                                     description="Rating kemampuan coding (1–5)",
                                     example=4)
    communication_skill_rating: int = Field(..., ge=1, le=5,
                                            description="Rating kemampuan komunikasi (1–5)",
                                            example=3)
    aptitude_skill_rating: int = Field(..., ge=1, le=5,
                                       description="Rating kemampuan aptitude (1–5)",
                                       example=4)

    # Experience
    projects_completed: int = Field(..., ge=0, le=8,
                                    description="Jumlah proyek yang selesai",
                                    example=5)
    internships_completed: int = Field(..., ge=0, le=4,
                                       description="Jumlah internship yang selesai",
                                       example=2)
    hackathons_participated: int = Field(..., ge=0, le=6,
                                         description="Jumlah hackathon yang diikuti",
                                         example=3)
    certifications_count: int = Field(..., ge=0, le=9,
                                      description="Jumlah sertifikasi yang dimiliki",
                                      example=3)

    # Categorical
    gender: Literal["Male", "Female"] = Field(..., example="Male")
    branch: Literal["CSE", "IT", "ECE", "CE", "ME"] = Field(..., example="CSE")
    part_time_job: Literal["Yes", "No"] = Field(..., example="No")
    family_income_level: Literal["Low", "Medium", "High"] = Field(..., example="Medium")
    city_tier: Literal["Tier 1", "Tier 2", "Tier 3"] = Field(..., example="Tier 1")
    internet_access: Literal["Yes", "No"] = Field(..., example="Yes")
    extracurricular_involvement: Literal["Low", "Medium", "High"] = Field(..., example="High")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Mahasiswa CSE dengan profil kuat",
                    "value": {
                        "cgpa": 8.74, "tenth_percentage": 80.0,
                        "twelfth_percentage": 78.0, "backlogs": 0,
                        "attendance_percentage": 85.0, "study_hours_per_day": 5.0,
                        "sleep_hours": 7.0, "stress_level": 4,
                        "coding_skill_rating": 5, "communication_skill_rating": 4,
                        "aptitude_skill_rating": 5, "projects_completed": 6,
                        "internships_completed": 3, "hackathons_participated": 4,
                        "certifications_count": 4, "gender": "Male",
                        "branch": "CSE", "part_time_job": "No",
                        "family_income_level": "High", "city_tier": "Tier 1",
                        "internet_access": "Yes", "extracurricular_involvement": "High"
                    }
                }
            ]
        }
    }


class ClassificationResponse(BaseModel):
    """Response schema untuk endpoint klasifikasi."""
    placement_label:       str   = Field(..., description="Hasil prediksi: 'Placed' atau 'Not Placed'")
    probability_placed:    float = Field(..., description="Probabilitas mahasiswa Placed (0.0 – 1.0)")
    probability_not_placed: float = Field(..., description="Probabilitas mahasiswa Not Placed (0.0 – 1.0)")
    confidence:            float = Field(..., description="Confidence score prediksi terpilih")
    model:                 str   = Field(..., description="Nama algoritma yang digunakan")


class RegressionResponse(BaseModel):
    """Response schema untuk endpoint regresi."""
    salary_lpa:       float = Field(..., description="Estimasi salary dalam LPA (Lakhs Per Annum)")
    salary_range_low: float = Field(..., description="Batas bawah estimasi salary (±1 std dev residual)")
    salary_range_high: float = Field(..., description="Batas atas estimasi salary (±1 std dev residual)")
    model:            str   = Field(..., description="Nama algoritma yang digunakan")


class CombinedResponse(BaseModel):
    """Response schema untuk endpoint combined (kedua prediksi)."""
    # Classification fields
    placement_label:        str   = Field(..., description="'Placed' atau 'Not Placed'")
    probability_placed:     float = Field(..., description="Probabilitas Placed")
    probability_not_placed: float = Field(..., description="Probabilitas Not Placed")
    confidence:             float = Field(..., description="Confidence score")
    clf_model:              str   = Field(..., description="Nama model klasifikasi")
    # Regression fields
    salary_lpa:             float = Field(..., description="Estimasi salary (LPA)")
    salary_range_low:       float = Field(..., description="Batas bawah estimasi salary")
    salary_range_high:      float = Field(..., description="Batas atas estimasi salary")
    reg_model:              str   = Field(..., description="Nama model regresi")


class HealthResponse(BaseModel):
    """Response schema untuk health check."""
    status:        str
    models_loaded: bool
    clf_model:     str
    reg_model:     str
    api_version:   str


# Helper Functions 

def engineer_and_build_df(data: StudentInput) -> pd.DataFrame:
    
    d = data.model_dump()

    # Feature engineering sama seperti di feature_engineering.py
    d["academic_score"] = (
        d["cgpa"] / 10 * 100 +
        d["tenth_percentage"] +
        d["twelfth_percentage"]
    ) / 3

    d["experience_index"] = (
        d["projects_completed"] +
        d["internships_completed"] * 2 +
        d["hackathons_participated"] +
        d["certifications_count"]
    )

    d["skill_avg"] = (
        d["coding_skill_rating"] +
        d["communication_skill_rating"] +
        d["aptitude_skill_rating"]
    ) / 3

    df = pd.DataFrame([d])[ALL_COLS]
    return df


def check_models_loaded():
    """Raise HTTP 503 jika model belum dimuat."""
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model belum tersedia. Jalankan pipeline.py terlebih dahulu untuk generate model .pkl."
        )


# Endpoints

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"]
)
def health_check():
   
    return HealthResponse(
        status="ok" if MODELS_LOADED else "degraded",
        models_loaded=MODELS_LOADED,
        clf_model="LogisticRegression" if MODELS_LOADED else "not loaded",
        reg_model="RandomForestRegressor" if MODELS_LOADED else "not loaded",
        api_version="1.0.0"
    )


@app.post(
    "/predict/classification",
    response_model=ClassificationResponse,
    summary="Prediksi Placement Status",
    tags=["Prediction"]
)
def predict_classification(data: StudentInput):
    
    check_models_loaded()

    try:
        df = engineer_and_build_df(data)
        pred = clf_model.predict(df)[0]
        prob = clf_model.predict_proba(df)[0]

        label   = "Placed" if pred == 1 else "Not Placed"
        conf    = float(prob[1]) if pred == 1 else float(prob[0])

        return ClassificationResponse(
            placement_label=label,
            probability_placed=round(float(prob[1]), 4),
            probability_not_placed=round(float(prob[0]), 4),
            confidence=round(conf, 4),
            model="LogisticRegression"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post(
    "/predict/regression",
    response_model=RegressionResponse,
    summary="Prediksi Estimasi Salary",
    tags=["Prediction"]
)
def predict_regression(data: StudentInput):
    
    check_models_loaded()

    try:
        df     = engineer_and_build_df(data)
        salary = float(reg_model.predict(df)[0])
        salary = max(0.0, salary)   # clip agar tidak negatif

        # Margin ±1.5 LPA berdasarkan MAE model (~2.7 LPA) dibagi 2
        margin = 1.5
        return RegressionResponse(
            salary_lpa=round(salary, 2),
            salary_range_low=round(max(0.0, salary - margin), 2),
            salary_range_high=round(salary + margin, 2),
            model="RandomForestRegressor"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post(
    "/predict/combined",
    response_model=CombinedResponse,
    summary="Prediksi Placement + Salary (Combined)",
    tags=["Prediction"]
)
def predict_combined(data: StudentInput):
    
    check_models_loaded()

    try:
        df = engineer_and_build_df(data)

        # Classification
        pred = clf_model.predict(df)[0]
        prob = clf_model.predict_proba(df)[0]
        label = "Placed" if pred == 1 else "Not Placed"
        conf  = float(prob[1]) if pred == 1 else float(prob[0])

        # Regression
        salary = float(reg_model.predict(df)[0])
        salary = max(0.0, salary)
        margin = 1.5

        return CombinedResponse(
            # Classification
            placement_label=label,
            probability_placed=round(float(prob[1]), 4),
            probability_not_placed=round(float(prob[0]), 4),
            confidence=round(conf, 4),
            clf_model="LogisticRegression",
            # Regression
            salary_lpa=round(salary, 2),
            salary_range_low=round(max(0.0, salary - margin), 2),
            salary_range_high=round(salary + margin, 2),
            reg_model="RandomForestRegressor"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ─── Root Endpoint ────────────────────────────────────────────────────────────
@app.get("/", tags=["System"], summary="API Root")
def root():
    """Root endpoint — redirect ke dokumentasi."""
    return {
        "message": "Student Placement Prediction API",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health",
        "version": "1.0.0"
    }
