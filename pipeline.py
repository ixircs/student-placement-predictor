"""
pipeline.py
===========
Master orchestrator — jalankan file ini untuk menjalankan seluruh
pipeline training dari awal sampai model siap deploy.

Urutan eksekusi:
  Step 1 → Data Ingestion        (data_ingestion.py)
  Step 2 → Feature Engineering   (feature_engineering.py)
  Step 3 → Train-Test Split      (80:20, stratified)
  Step 4 → Training              (train.py)  — 2 model
  Step 5 → Evaluation            (evaluation.py) — 2 model
  Step 6 → Threshold Check       → approve / reject
  Step 7 → Summary Report

Cara menjalankan:
  $ python pipeline.py

Output:
  - models/model_classification.pkl
  - models/model_regression.pkl
  - mlruns/  (MLflow tracking data)
  - ingested/A_merged.csv
"""

import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# pastikan semua modul bisa diimport dari folder yang sama
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from data_ingestion     import ingest_data
from feature_engineering import engineer_features, get_feature_lists
from train              import train_classification, train_regression
from evaluation         import (
    evaluate_classification,
    evaluate_regression,
    check_model_threshold
)


# ─── Pipeline Configuration ───────────────────────────────────────────────────
TEST_SIZE   = 0.2
RANDOM_STATE = 42


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "█" * 55)
    print("  STUDENT PLACEMENT — MODEL DEPLOYMENT PIPELINE")
    print("  Dataset A | NIM Ganjil")
    print("█" * 55)

    # ── STEP 1: Data Ingestion ──────────────────────────────
    df = ingest_data()

    # ── STEP 2: Feature Engineering ────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2 — FEATURE ENGINEERING")
    print("=" * 55)
    df = engineer_features(df)
    num_cols, cat_cols, all_cols = get_feature_lists()
    print(f"  Fitur engineering ditambahkan: academic_score, experience_index, skill_avg")
    print(f"  Total fitur: {len(all_cols)} ({len(num_cols)} numerik + {len(cat_cols)} kategorik)")

    # ── STEP 3: Train-Test Split ────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3 — TRAIN-TEST SPLIT (80:20)")
    print("=" * 55)

    X     = df[all_cols]
    y_clf = (df["placement_status"] == "Placed").astype(int)
    y_reg = df["salary_lpa"]

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_clf          # stratified by class label — penting untuk imbalanced data
    )

    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Train class dist: Placed={yc_train.mean():.2%}  Not Placed={(1-yc_train.mean()):.2%}")
    print(f"  Test  class dist: Placed={yc_test.mean():.2%}  Not Placed={(1-yc_test.mean()):.2%}")

    # ── STEP 4: Training ────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 4 — MODEL TRAINING")
    print("=" * 55)

    clf_run_id = train_classification(
        X_train, X_test, yc_train, yc_test, num_cols, cat_cols
    )
    reg_run_id = train_regression(
        X_train, X_test, yr_train, yr_test, num_cols, cat_cols
    )

    # ── STEP 5: Evaluation ──────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 5 — EVALUATION")
    print("=" * 55)

    clf_metrics = evaluate_classification(clf_run_id, X_test, yc_test)
    reg_metrics = evaluate_regression(reg_run_id, X_test, yr_test)

    # ── STEP 6: Threshold Check ─────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 6 — THRESHOLD CHECK (APPROVE / REJECT)")
    print("=" * 55)

    clf_approved = check_model_threshold(clf_metrics, task="clf")
    reg_approved = check_model_threshold(reg_metrics, task="reg")

    # ── STEP 7: Summary ─────────────────────────────────────
    print("\n" + "█" * 55)
    print("  PIPELINE SUMMARY")
    print("█" * 55)
    print(f"  CLF Run ID  : {clf_run_id}")
    print(f"  REG Run ID  : {reg_run_id}")
    print()
    print(f"  Classification Metrics:")
    for k, v in clf_metrics.items():
        print(f"    {k:<15}: {v:.4f}")
    print()
    print(f"  Regression Metrics:")
    for k, v in reg_metrics.items():
        print(f"    {k:<15}: {v:.4f}")
    print()
    print(f"  CLF Model : {'✅ APPROVED' if clf_approved else '❌ REJECTED'}")
    print(f"  REG Model : {'✅ APPROVED' if reg_approved else '❌ REJECTED'}")
    print()
    print(f"  Models saved:")
    print(f"    → models/model_classification.pkl")
    print(f"    → models/model_regression.pkl")
    print()
    print("  Untuk melihat MLflow UI:")
    print("    $ mlflow ui")
    print("    Buka http://localhost:5000 di browser")
    print("█" * 55)

    return {
        "clf_run_id":    clf_run_id,
        "reg_run_id":    reg_run_id,
        "clf_metrics":   clf_metrics,
        "reg_metrics":   reg_metrics,
        "clf_approved":  clf_approved,
        "reg_approved":  reg_approved
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_pipeline()
