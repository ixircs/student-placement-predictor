"""
data_ingestion.py

Modul untuk load dan memvalidasi Dataset A (NIM Ganjil).
Dataset A memiliki fitur dan target dalam file terpisah:
  - A.csv        : feature file
  - A_targets.csv: target file (placement_status, salary_lpa)
"""

import pandas as pd
from pathlib import Path

# Path Configuration 
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
INGESTED_DIR = BASE_DIR / "ingested"

FEATURE_FILE = DATA_DIR / "A.csv"
TARGET_FILE  = DATA_DIR / "A_targets.csv"
OUTPUT_FILE  = INGESTED_DIR / "A_merged.csv"

# Kolom yang wajib ada di masing-masing file
REQUIRED_FEATURE_COLS = [
    "Student_ID", "gender", "branch", "cgpa", "tenth_percentage",
    "twelfth_percentage", "backlogs", "study_hours_per_day",
    "attendance_percentage", "projects_completed", "internships_completed",
    "coding_skill_rating", "communication_skill_rating", "aptitude_skill_rating",
    "hackathons_participated", "certifications_count", "sleep_hours",
    "stress_level", "part_time_job", "family_income_level", "city_tier",
    "internet_access", "extracurricular_involvement"
]
REQUIRED_TARGET_COLS = ["Student_ID", "placement_status", "salary_lpa"]


# Core Functions 

def load_raw_data(
    feature_path: str | Path = FEATURE_FILE,
    target_path:  str | Path = TARGET_FILE
) -> pd.DataFrame:
    
    feature_path = Path(feature_path)
    target_path  = Path(target_path)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file tidak ditemukan: {feature_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file tidak ditemukan: {target_path}")

    features = pd.read_csv(feature_path)
    targets  = pd.read_csv(target_path)

    print(f"  [load] Features : {features.shape[0]} rows x {features.shape[1]} cols")
    print(f"  [load] Targets  : {targets.shape[0]} rows x {targets.shape[1]} cols")

    # Inner merge — hanya baris yang ada di kedua file
    df = features.merge(targets, on="Student_ID", how="inner")
    print(f"  [load] Merged   : {df.shape[0]} rows x {df.shape[1]} cols")

    return df


def validate_data(df: pd.DataFrame) -> bool:
   
    print("\n  [validate] Menjalankan validasi data...")

    # 1. Cek dataset tidak kosong
    assert not df.empty, "GAGAL: Dataset kosong."

    # 2. Cek kolom wajib
    all_required = set(REQUIRED_FEATURE_COLS + REQUIRED_TARGET_COLS) - {"Student_ID"}
    missing_cols = all_required - set(df.columns)
    assert not missing_cols, f"GAGAL: Kolom tidak ditemukan — {missing_cols}"

    # 3. Cek missing values pada target (tidak boleh ada)
    for col in ["placement_status", "salary_lpa"]:
        n_missing = df[col].isnull().sum()
        assert n_missing == 0, f"GAGAL: Target '{col}' memiliki {n_missing} missing values."

    # 4. Cek kelas target klasifikasi valid
    valid_classes = {"Placed", "Not Placed"}
    actual_classes = set(df["placement_status"].unique())
    assert actual_classes.issubset(valid_classes), \
        f"GAGAL: Kelas tidak dikenal di placement_status — {actual_classes - valid_classes}"

    # 5. Info missing values fitur (peringatan, bukan error)
    feat_missing = df.isnull().sum()
    feat_missing = feat_missing[feat_missing > 0]
    if not feat_missing.empty:
        print(f"  [validate] Peringatan — missing values pada fitur:")
        for col, count in feat_missing.items():
            print(f"             {col}: {count} ({count/len(df)*100:.1f}%)")
        print("             → Akan ditangani oleh SimpleImputer dalam pipeline.")

    print("  [validate] Semua validasi LULUS ✓")
    return True


def ingest_data(
    feature_path: str | Path = FEATURE_FILE,
    target_path:  str | Path = TARGET_FILE,
    output_path:  str | Path = OUTPUT_FILE
) -> pd.DataFrame:
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("STEP 1 — DATA INGESTION")
    print("=" * 55)

    df = load_raw_data(feature_path, target_path)
    validate_data(df)

    df.to_csv(output_path, index=False)
    print(f"\n  [ingest] Data disimpan → {output_path}")
    print(f"  [ingest] Shape final  : {df.shape}")
    print("=" * 55)

    return df


# Standalone Execution  
if __name__ == "__main__":
    df = ingest_data()
    print("\nSample data:")
    print(df.head(3).to_string())
