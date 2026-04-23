"""
evaluation.py
=============
Modul evaluasi model untuk kedua task (klasifikasi & regresi).

Fungsi utama:
  - evaluate_classification(run_id, X_test, y_test) : evaluasi + log ke MLflow
  - evaluate_regression(run_id, X_test, y_test)     : evaluasi + log ke MLflow
  - print_classification_report(y_test, y_pred)     : tampilkan report lengkap
  - check_model_threshold(metrics, task)            : approve/reject model
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# ─── Threshold Configuration ──────────────────────────────────────────────────
# Model dianggap layak deploy jika memenuhi threshold berikut
CLF_THRESHOLD = {
    "accuracy":    0.75,
    "f1_weighted": 0.75,
    "roc_auc":     0.80
}
REG_THRESHOLD = {
    "r2":  0.50,     # minimal 50% variance explained
    "mae": 5.00      # max error rata-rata 5 LPA
}


# ─── Classification Evaluation ────────────────────────────────────────────────

def evaluate_classification(
    run_id: str,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluasi model klasifikasi dari MLflow run dan log metrik tambahan.

    Parameters
    ----------
    run_id : MLflow run ID yang dihasilkan oleh train_classification()
    X_test : test feature dataframe
    y_test : test target series (binary)

    Returns
    -------
    dict : {"accuracy": float, "f1_weighted": float, "roc_auc": float}
    """
    print("\n" + "=" * 55)
    print("EVALUASI — Classification")
    print("=" * 55)

    # Load model dari MLflow
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/clf_model")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":    accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc":     roc_auc_score(y_test, y_prob)
    }

    # Log ulang ke run yang sama (reopen)
    with mlflow.start_run(run_id=run_id):
        for key, val in metrics.items():
            mlflow.log_metric(f"eval_{key}", val)

    # Print hasil
    for key, val in metrics.items():
        print(f"  {key:<15}: {val:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return metrics


# ─── Regression Evaluation ────────────────────────────────────────────────────

def evaluate_regression(
    run_id: str,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluasi model regresi dari MLflow run dan log metrik tambahan.

    Parameters
    ----------
    run_id : MLflow run ID yang dihasilkan oleh train_regression()
    X_test : test feature dataframe
    y_test : test target series (continuous)

    Returns
    -------
    dict : {"mae": float, "rmse": float, "r2": float}
    """
    print("\n" + "=" * 55)
    print("EVALUASI — Regression")
    print("=" * 55)

    # Load model dari MLflow
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/reg_model")

    y_pred = model.predict(X_test)

    metrics = {
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2":   r2_score(y_test, y_pred)
    }

    # Log ulang ke run yang sama (reopen)
    with mlflow.start_run(run_id=run_id):
        for key, val in metrics.items():
            mlflow.log_metric(f"eval_{key}", val)

    # Print hasil
    for key, val in metrics.items():
        print(f"  {key:<15}: {val:.4f}")

    # Residual summary
    residuals = y_test.values - y_pred
    print(f"\n  Residual Summary:")
    print(f"    Mean  : {residuals.mean():.4f}")
    print(f"    Std   : {residuals.std():.4f}")
    print(f"    Min   : {residuals.min():.4f}")
    print(f"    Max   : {residuals.max():.4f}")

    return metrics


# ─── Threshold Check ──────────────────────────────────────────────────────────

def check_model_threshold(metrics: dict, task: str = "clf") -> bool:
    """
    Periksa apakah model memenuhi threshold minimum untuk deployment.

    Parameters
    ----------
    metrics : dict hasil evaluate_classification() atau evaluate_regression()
    task    : "clf" untuk klasifikasi, "reg" untuk regresi

    Returns
    -------
    bool : True jika model APPROVED, False jika REJECTED
    """
    print("\n" + "-" * 55)
    thresholds = CLF_THRESHOLD if task == "clf" else REG_THRESHOLD
    approved   = True

    for metric, threshold in thresholds.items():
        if metric not in metrics:
            continue
        val = metrics[metric]
        # MAE: lower is better; semua metrik lain: higher is better
        if metric == "mae":
            passed = val <= threshold
        else:
            passed = val >= threshold
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{task.upper()}] {metric:<15}: {val:.4f}  (threshold {'≤' if metric=='mae' else '≥'} {threshold}) {status}")
        if not passed:
            approved = False

    if approved:
        print(f"\n  ✅ Model [{task.upper()}] APPROVED untuk deployment.")
    else:
        print(f"\n  ❌ Model [{task.upper()}] REJECTED — perlu perbaikan.")

    print("-" * 55)
    return approved


# ─── Standalone Execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_ingestion import ingest_data
    from feature_engineering import engineer_features, get_feature_lists
    from train import train_classification, train_regression
    from sklearn.model_selection import train_test_split

    df = ingest_data()
    df = engineer_features(df)
    num_cols, cat_cols, all_cols = get_feature_lists()

    X = df[all_cols]
    y_clf = (df["placement_status"] == "Placed").astype(int)
    y_reg = df["salary_lpa"]

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    clf_run = train_classification(X_train, X_test, yc_train, yc_test, num_cols, cat_cols)
    reg_run = train_regression(X_train, X_test, yr_train, yr_test, num_cols, cat_cols)

    clf_metrics = evaluate_classification(clf_run, X_test, yc_test)
    reg_metrics = evaluate_regression(reg_run, X_test, yr_test)

    check_model_threshold(clf_metrics, task="clf")
    check_model_threshold(reg_metrics, task="reg")
