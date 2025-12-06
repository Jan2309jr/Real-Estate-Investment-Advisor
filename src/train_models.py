# src/train_models.py
import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import json
import traceback

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# local imports
from src.preprocess import basic_cleaning, fill_missing_domain, feature_engineering, create_targets, build_preprocessor, save_object
from src.utils import load_data

MLFLOW_EXPERIMENT = "real_estate_investment_advisor"

# Try to import mlflow; if it fails, continue without mlflow logging
use_mlflow = True
try:
    import mlflow
    import mlflow.sklearn
except Exception as e:
    use_mlflow = False
    MLFLOW_IMPORT_ERROR = traceback.format_exc()
    print("Warning: mlflow import failed. Continuing without MLflow. Error:")
    print(MLFLOW_IMPORT_ERROR)

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:,1]
    except Exception:
        # fallback to zeros if no proba
        y_proba = np.zeros(len(y_test))
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba.sum() != 0 else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred))
    }

def train(save_dir="models", data_path="data/india_housing_prices.csv"):
    os.makedirs(save_dir, exist_ok=True)

    # load and preprocess
    df = load_data(data_path)
    df = basic_cleaning(df)
    df = fill_missing_domain(df)
    df = feature_engineering(df)
    df = create_targets(df)

    preprocessor, numeric_cols, cat_cols = build_preprocessor(df)
    save_object(preprocessor, os.path.join(save_dir,"preprocessor.joblib"))

    # Prepare training splits
    X = df.drop(columns=["Future_Price_5Y","Good_Investment"], errors='ignore')
    y_reg = df["Future_Price_5Y"]
    y_clf = df["Good_Investment"]
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # Classification pipeline
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    # Regression pipeline
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("reg", reg)])

    clf_pipeline.fit(X_train, y_clf_train)
    clf_eval = evaluate_classification(clf_pipeline, X_test, y_clf_test)
    joblib.dump(clf_pipeline, os.path.join(save_dir, "classifier_pipeline.joblib"))

    reg_pipeline.fit(X_train, y_reg_train)
    reg_eval = evaluate_regression(reg_pipeline, X_test, y_reg_test)
    joblib.dump(reg_pipeline, os.path.join(save_dir, "regressor_pipeline.joblib"))

    summary = {"classification": clf_eval, "regression": reg_eval}
    with open(os.path.join(save_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # If mlflow is available, try to log runs (non-fatal)
    if use_mlflow:
        try:
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            with mlflow.start_run(run_name=f"clf_rf_{datetime.now().isoformat()}"):
                mlflow.log_params({"model_type":"RandomForestClassifier", "n_estimators":200})
                mlflow.log_metrics({k: v for k,v in clf_eval.items() if isinstance(v, (int,float))})
                mlflow.sklearn.log_model(clf_pipeline, "clf_model")
            with mlflow.start_run(run_name=f"reg_rf_{datetime.now().isoformat()}"):
                mlflow.log_params({"model_type":"RandomForestRegressor", "n_estimators":200})
                mlflow.log_metrics({k: v for k,v in reg_eval.items() if isinstance(v, (int,float))})
                mlflow.sklearn.log_model(reg_pipeline, "reg_model")
        except Exception:
            print("Warning: mlflow logging failed (non-fatal). Traceback:")
            print(traceback.format_exc())

    print("Training complete. Models saved to", save_dir)
    return os.path.abspath(save_dir)

if __name__ == "__main__":
    train()
