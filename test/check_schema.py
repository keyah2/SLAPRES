import json
from pathlib import Path
import pandas as pd

# Racine du projet (2 niveaux au-dessus de /test)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

schema_path = PROJECT_ROOT / "artifacts/feature_columns.json"
etl_path = PROJECT_ROOT / "data/processed/etl_after_preprocess.csv"

schema = json.loads(schema_path.read_text(encoding="utf-8"))
expected = schema["feature_columns"]

etl_cols = pd.read_csv(etl_path, nrows=0).columns.tolist()

print("Same columns & same order:", etl_cols == expected)
print("ETL cols:", len(etl_cols), "| Expected:", len(expected))
