import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TARGET_COL = "is_fraud"

BEST_PARAMS = {
    "n_estimators": 1470,
    "max_depth": 9,
    "learning_rate": 0.2613835537500282,
    "gamma": 8.466157273152673,
    "min_child_weight": 10,
    "subsample": 0.7000423131953927,
    "colsample_bytree": 0.948295020405272,
    "scale_pos_weight": 8.014294568224415,
}


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    train_csv = PROJECT_ROOT / "data" / "processed" / "train_after_preprocess.csv"
    schema_json = PROJECT_ROOT / "artifacts" / "feature_columns.json"

    # Tracking DB (OK)
    mlflow_db = PROJECT_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud_detection_xgboost")

    df = pd.read_csv(train_csv)
    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    feature_cols = schema["feature_columns"]

    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        **BEST_PARAMS,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    with mlflow.start_run() as run:
        # Params
        mlflow.log_params({**BEST_PARAMS,
                           "objective": "binary:logistic",
                           "eval_metric": "auc",
                           "tree_method": "hist",
                           "n_features": int(X.shape[1]),
                           "valid_size": 0.2,
                           "random_state": 42})

        # Train
        model.fit(X_train, y_train)

        # Eval
        y_proba = model.predict_proba(X_valid)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_valid, y_proba)
        f1 = f1_score(y_valid, y_pred)
        prec = precision_score(y_valid, y_pred, zero_division=0)
        rec = recall_score(y_valid, y_pred, zero_division=0)

        mlflow.log_metrics({
            "roc_auc": float(auc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
        })

        # Artifacts
        mlflow.log_artifact(str(schema_json), artifact_path="schema")

        #  Modèle: dump local -> log artifact (inratable)
        out_dir = PROJECT_ROOT / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_file = out_dir / "xgb_model.joblib"
        joblib.dump(model, model_file)
        mlflow.log_artifact(str(model_file), artifact_path="model")

        print("RUN_ID:", run.info.run_id)
        print("ARTIFACT_URI:", mlflow.get_artifact_uri())
        print("Training XGBoost terminé.")
        print(f"Tracking URI: {tracking_uri}")
        print(f"ROC-AUC: {auc:.6f} | F1: {f1:.6f} | Prec: {prec:.6f} | Recall: {rec:.6f}")


if __name__ == "__main__":
    main()
