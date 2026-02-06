from datetime import datetime
import time

from airflow import DAG
from airflow.operators.python import PythonOperator
# Fonction exécutée


# Définition DAG
with DAG(
    dag_id="etape_4_train_modele",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # Lancement manuel
    catchup=False,
    tags=["train_modele_log_mlflow"],
) as dag:

    sleep_task = PythonOperator(
        task_id="train_modele",
        python_callable=wait_and_print,
    )
