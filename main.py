"""
SLAPRES – Main entrypoint

Ce fichier sert de point d’entrée minimal du projet.
Le pipeline complet (ingestion, preprocessing, drift, training)
est orchestré via Airflow.

main.py permet :
- de vérifier l’environnement
- de lancer manuellement les briques principales si besoin
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"


def healthcheck() -> None:
    """Vérification simple de l’environnement."""
    print("SLAPRES – Healthcheck")
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"src/ exists         : {SRC_PATH.exists()}")
    print(f"MLFLOW_TRACKING_URI : {os.getenv('MLFLOW_TRACKING_URI')}")
    print(f"AWS credentials     : {'SET' if os.getenv('AWS_ACCESS_KEY_ID') else 'NOT SET'}")
    print("Environment OK")


if __name__ == "__main__":
    healthcheck()
