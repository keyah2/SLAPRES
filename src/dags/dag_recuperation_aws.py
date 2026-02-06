from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
from pathlib import Path

BUCKET_NAME = "slajedhaprojectfinal"
S3_KEY = "dataset/transactions_etl_10.csv"
DEST = "/opt/airflow/output/transactions_etl_10.csv"

def download_from_s3():
    Path("/opt/airflow/output").mkdir(parents=True, exist_ok=True)

    hook = S3Hook(aws_conn_id="aws_default")
    s3 = hook.get_conn()  # boto3 client

    # boto3 écrit dans un vrai fichier (pas de temp file Airflow)
    s3.download_file(BUCKET_NAME, S3_KEY, DEST)

    if not Path(DEST).exists():
        raise FileNotFoundError(f"Download failed, missing: {DEST}")

    print(f"OK: {DEST}")

with DAG(
    dag_id="etape_1_recuperation_aws_OK",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    task_download = PythonOperator(
        task_id="telecharger_depuis_s3",
        python_callable=download_from_s3,
    )
