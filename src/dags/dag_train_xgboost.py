from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_model():
    data_path = Path("/opt/airflow/output/train_100_latest.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing input: {data_path} (lance merge/drift avant)")

    df = pd.read_csv(data_path)
    if "is_fraud" not in df.columns:
        raise ValueError("Missing target column: is_fraud")

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run(run_name="XGB_after_drift_merge"):
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("roc_auc", auc)
        mlflow.xgboost.log_model(model, artifact_path="model")

        print(f"Train done. ROC-AUC={auc:.4f}")


with DAG(
    dag_id="train_apres_check_drift",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task_train = PythonOperator(
        task_id="run_train_mlflow",
        python_callable=train_model,
    )
""""""