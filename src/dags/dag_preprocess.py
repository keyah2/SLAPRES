from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils import timezone

from src.preprocessing.cleaner_update import run_update_preprocessing

default_args = {
    "owner": "airflow",
    "retries": 0,
}

with DAG(
    dag_id="etape_2_preprocess",
    default_args=default_args,
    start_date=timezone.datetime(2026, 2, 1),
    schedule_interval=None,
    catchup=False,
    tags=["preprocess", "update"],
) as dag:

    preprocess_update = PythonOperator(
        task_id="preprocess_update_10pct",
        python_callable=run_update_preprocessing,
    )

    preprocess_update
