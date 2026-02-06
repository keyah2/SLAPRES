from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# Dans Docker Airflow, le code est souvent dans /opt/airflow

PROJECT_DIR = "/opt/airflow"
PYTHON = "python"

with DAG(
    dag_id="dag_fraud_pipeline_local.py",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["fraud", "local"],
) as dag:

    preprocess_train = BashOperator(
        task_id="train_after_preprocess",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/preprocessing/cleanerbase.py",
    )

    preprocess_update = BashOperator(
        task_id="preprocess_update",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/preprocessing/cleaner_update.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/training/train_model.py",
    )

    predict_update = BashOperator(
        task_id="predict_update",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/inference/predict_update.py",
    )

    [preprocess_train, preprocess_update] >> train_model >> predict_update
