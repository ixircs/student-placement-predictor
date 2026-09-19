import pandas as pd
from typing import Tuple


# ─── Feature Column Definitions ───────────────────────────────────────────────

NUM_COLS = [
    "cgpa", "tenth_percentage", "twelfth_percentage", "backlogs",
    "study_hours_per_day", "attendance_percentage", "projects_completed",
    "internships_completed", "coding_skill_rating", "communication_skill_rating",
    "aptitude_skill_rating", "hackathons_participated", "certifications_count",
    "sleep_hours", "stress_level",
    # engineered features (ditambahkan oleh engineer_features())
    "academic_score", "experience_index", "skill_avg"
]

CAT_COLS = [
    "gender", "branch", "part_time_job", "family_income_level",
    "city_tier", "internet_access", "extracurricular_involvement"
]

ALL_COLS = NUM_COLS + CAT_COLS


# ─── Core Function ────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    df["academic_score"] = (
        df["cgpa"] / 10 * 100 +
        df["tenth_percentage"] +
        df["twelfth_percentage"]
    ) / 3

    df["experience_index"] = (
        df["projects_completed"] +
        df["internships_completed"] * 2 +
        df["hackathons_participated"] +
        df["certifications_count"]
    )

    df["skill_avg"] = (
        df["coding_skill_rating"] +
        df["communication_skill_rating"] +
        df["aptitude_skill_rating"]
    ) / 3

    return df


def get_feature_lists() -> Tuple[list, list, list]:

    return NUM_COLS, CAT_COLS, ALL_COLS


# ─── Standalone Execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_ingestion import ingest_data

    df = ingest_data()
    df_eng = engineer_features(df)

    print("\nFitur hasil engineering:")
    print(df_eng[["academic_score", "experience_index", "skill_avg"]].describe().round(2))
    print(f"\nTotal fitur untuk model: {len(ALL_COLS)} ({len(NUM_COLS)} num + {len(CAT_COLS)} cat)")
