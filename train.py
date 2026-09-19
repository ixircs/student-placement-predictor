"""
Modul training untuk dua task:
  1. Klasifikasi  : memprediksi placement_status (Placed / Not Placed)
  2. Regresi      : memprediksi salary_lpa

Setiap task menggunakan sklearn.Pipeline end-to-end yang mencakup:
  - Preprocessing (SimpleImputer + StandardScaler/OrdinalEncoder)
  - Model training
  - MLflow experiment tracking (params, metrics, artifacts)
  - Penyimpanan model .pkl
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Path Configuration 
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow tracking URI — selalu gunakan path absolut agar konsisten
# di semua OS dan working directory manapun
MLRUNS_DIR = BASE_DIR / "mlruns"
TRACKING_URI = MLRUNS_DIR.as_uri()   

CLF_MODEL_PATH = MODELS_DIR / "model_classification.pkl"
REG_MODEL_PATH = MODELS_DIR / "model_regression.pkl"

# MLflow experiment names
CLF_EXPERIMENT = "Student Placement Classification"
REG_EXPERIMENT = "Student Salary Regression"


# Preprocessor Builder

def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
   
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor


#  Classification Training 

def train_classification(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    num_cols: list,
    cat_cols: list
) -> str:
    
    params = {
        "model_name":   "LogisticRegression",
        "max_iter":     1000,
        "class_weight": "balanced",
        "C":            1.0,
        "random_state": 42
    }

    preprocessor = build_preprocessor(num_cols, cat_cols)

    clf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
            C=params["C"],
            random_state=params["random_state"]
        ))
    ])

    #  MLflow Tracking
    # Set tracking URI ke path absolut — fix untuk Windows file-based tracking
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(CLF_EXPERIMENT)

    with mlflow.start_run(run_name="LR_Classification") as run:
        print("\n  [clf] Training Logistic Regression pipeline...")

        clf_pipeline.fit(X_train, y_train)

        y_pred = clf_pipeline.predict(X_test)
        y_prob = clf_pipeline.predict_proba(X_test)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy",    acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("roc_auc",     roc_auc)

        # Log model — tanpa registered_model_name agar tidak butuh MLflow Registry DB
        mlflow.sklearn.log_model(clf_pipeline, artifact_path="clf_model")

        joblib.dump(clf_pipeline, CLF_MODEL_PATH)
        mlflow.log_artifact(str(CLF_MODEL_PATH), artifact_path="pkl")

        run_id = run.info.run_id

    print(f"  [clf] Accuracy  : {acc:.4f}")
    print(f"  [clf] F1 (w)    : {f1:.4f}")
    print(f"  [clf] ROC-AUC   : {roc_auc:.4f}")
    print(f"  [clf] Run ID    : {run_id}")
    print(f"  [clf] Model saved → {CLF_MODEL_PATH}")

    return run_id


#  Regression Training 

def train_regression(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    num_cols: list,
    cat_cols: list
) -> str:
   
    params = {
        "model_name":   "RandomForestRegressor",
        "n_estimators": 100,
        "max_depth":    None,
        "random_state": 42
    }

    preprocessor = build_preprocessor(num_cols, cat_cols)

    reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        ))
    ])

    #  MLflow Tracking 
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(REG_EXPERIMENT)

    with mlflow.start_run(run_name="RF_Regression") as run:
        print("\n  [reg] Training Random Forest Regressor pipeline...")

        reg_pipeline.fit(X_train, y_train)

        y_pred = reg_pipeline.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)

        # Log model — tanpa registered_model_name
        mlflow.sklearn.log_model(reg_pipeline, artifact_path="reg_model")

        joblib.dump(reg_pipeline, REG_MODEL_PATH)
        mlflow.log_artifact(str(REG_MODEL_PATH), artifact_path="pkl")

        run_id = run.info.run_id

    print(f"  [reg] MAE       : {mae:.4f}")
    print(f"  [reg] RMSE      : {rmse:.4f}")
    print(f"  [reg] R²        : {r2:.4f}")
    print(f"  [reg] Run ID    : {run_id}")
    print(f"  [reg] Model saved → {REG_MODEL_PATH}")

    return run_id


# Standalone Execution 
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_ingestion import ingest_data
    from feature_engineering import engineer_features, get_feature_lists
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

    clf_run_id = train_classification(X_train, X_test, yc_train, yc_test, num_cols, cat_cols)
    reg_run_id = train_regression(X_train, X_test, yr_train, yr_test, num_cols, cat_cols)

    print(f"\nDone. CLF run_id={clf_run_id} | REG run_id={reg_run_id}")
