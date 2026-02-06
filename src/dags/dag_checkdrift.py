# dags/dag_checkdrift.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils import timezone

from pathlib import Path

OUTPUT_DIR = Path("/opt/airflow/output")

PATH_90 = OUTPUT_DIR / "train_after_preprocess_latest.csv"
PATH_10 = OUTPUT_DIR / "update_after_preprocess.csv"
PATH_100_LATEST = OUTPUT_DIR / "train_100_latest.csv"


def _schema_diff(ref_cols: List[str], new_cols: List[str]) -> Tuple[List[str], List[str]]:
    ref_set, new_set = set(ref_cols), set(new_cols)
    missing = sorted(list(ref_set - new_set))
    extra = sorted(list(new_set - ref_set))
    return missing, extra


def align_to_reference(df_new: pd.DataFrame, ref_cols: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    # ajoute les colonnes manquantes
    for c in ref_cols:
        if c not in df_new.columns:
            df_new[c] = fill_value

    # supprime les colonnes en trop
    extra = [c for c in df_new.columns if c not in ref_cols]
    if extra:
        df_new = df_new.drop(columns=extra)

    # réordonne
    return df_new[ref_cols]


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = pd.to_numeric(expected, errors="coerce").dropna()
    a = pd.to_numeric(actual, errors="coerce").dropna()
    if len(e) == 0 or len(a) == 0:
        return 0.0

    try:
        quantiles = np.linspace(0, 1, bins + 1)
        cutpoints = np.unique(np.quantile(e, quantiles))
        if len(cutpoints) < 3:
            return 0.0

        e_counts = pd.cut(e, bins=cutpoints, include_lowest=True).value_counts(sort=False)
        a_counts = pd.cut(a, bins=cutpoints, include_lowest=True).value_counts(sort=False)

        e_perc = (e_counts / e_counts.sum()).replace(0, 1e-6)
        a_perc = (a_counts / a_counts.sum()).replace(0, 1e-6)

        return float(((a_perc - e_perc) * np.log(a_perc / e_perc)).sum())
    except Exception:
        return 0.0


def _select_numeric_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def detect_drift_and_decide_branch() -> str:
    MIN_ROWS_FOR_DRIFT = 5_000
    PSI_THRESHOLD = 0.20
    MAX_FEATURES_TO_CHECK = 25
    target_col = "is_fraud"

    if not PATH_90.exists():
        raise FileNotFoundError(f"Missing reference (90% preprocessed): {PATH_90}")
    if not PATH_10.exists():
        raise FileNotFoundError(f"Missing update (10% preprocessed): {PATH_10}")

    df_90 = pd.read_csv(PATH_90)
    df_10 = pd.read_csv(PATH_10)

    # align schema (au lieu de fail)
    ref_cols = list(df_90.columns)
    missing, extra = _schema_diff(ref_cols, list(df_10.columns))
    if missing or extra:
        print(f"[SCHEMA] missing_in_update(total={len(missing)}): {missing[:30]}")
        print(f"[SCHEMA] extra_in_update(total={len(extra)}): {extra[:30]}")
    df_10 = align_to_reference(df_10, ref_cols, fill_value=0.0)
    print("[SCHEMA] update aligned to reference columns")

    if len(df_10) < MIN_ROWS_FOR_DRIFT:
        print(f"[DRIFT] Not enough rows: {len(df_10)} < {MIN_ROWS_FOR_DRIFT}")
        return "skip_merge"

    num_cols = _select_numeric_features(df_90, exclude=[target_col])
    num_cols = num_cols[:MAX_FEATURES_TO_CHECK]

    psi_scores: Dict[str, float] = {c: _psi(df_90[c], df_10[c], bins=10) for c in num_cols}
    top10 = sorted(psi_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    drift_cols = [c for c, v in psi_scores.items() if v >= PSI_THRESHOLD]

    print(f"[DRIFT] checked={len(num_cols)} threshold={PSI_THRESHOLD}")
    print(f"[DRIFT] top10 PSI: {top10}")

    if drift_cols:
        print(f"[DRIFT] drift_detected cols_over_threshold: {drift_cols}")
        return "merge_if_drift"

    print("[DRIFT] no drift detected")
    return "skip_merge"


def merge_if_drift() -> None:
    df_90 = pd.read_csv(PATH_90)
    df_10 = pd.read_csv(PATH_10)

    # align encore pour être sûr
    df_10 = align_to_reference(df_10, list(df_90.columns), fill_value=0.0)

    df_100 = pd.concat([df_90, df_10], axis=0, ignore_index=True)

    ts = timezone.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dated = OUTPUT_DIR / f"train_100_{ts}.csv"

    df_100.to_csv(out_dated, index=False)
    df_100.to_csv(PATH_100_LATEST, index=False)

    print(f"[MERGE] saved dated: {out_dated}")
    print(f"[MERGE] saved latest: {PATH_100_LATEST}")
    print(f"[MERGE] shapes 90={df_90.shape} 10={df_10.shape} 100={df_100.shape}")


def log_no_drift() -> None:
    print("[MERGE] No drift => merge skipped.")


with DAG(
    dag_id="etape_3_detect_drift_and_merge",
    start_date=timezone.datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["drift", "merge"],
) as dag:

    t_detect = BranchPythonOperator(
        task_id="detect_drift_and_branch",
        python_callable=detect_drift_and_decide_branch,
    )

    t_merge = PythonOperator(
        task_id="merge_if_drift",
        python_callable=merge_if_drift,
    )

    t_skip = PythonOperator(
        task_id="skip_merge",
        python_callable=log_no_drift,
    )

    t_end = EmptyOperator(task_id="end")

    t_detect >> [t_merge, t_skip]
    t_merge >> t_end
    t_skip >> t_end
