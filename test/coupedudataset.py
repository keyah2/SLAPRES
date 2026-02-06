from __future__ import annotations

from pathlib import Path
import pandas as pd


def time_based_split(
    input_csv: str | Path,
    output_train_csv: str | Path,
    output_etl_csv: str | Path,
    timestamp_col: str = "transaction_time",
    train_ratio: float = 0.90,
) -> None:
    input_csv = Path(input_csv)
    output_train_csv = Path(output_train_csv)
    output_etl_csv = Path(output_etl_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # Load
    df = pd.read_csv(input_csv)

    # Basic checks
    if timestamp_col not in df.columns:
        raise ValueError(
            f"timestamp_col='{timestamp_col}' not found in columns: {df.columns.tolist()}"
        )

    # Parse timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=False)

    # Drop invalid timestamps
    n_before = len(df)
    df = df.dropna(subset=[timestamp_col]).copy()
    n_after = len(df)

    if n_after == 0:
        raise ValueError("All rows have invalid timestamps after parsing.")

    # Sort chronologically
    df = df.sort_values(timestamp_col, ascending=True)

    # Compute split index
    split_idx = int(n_after * train_ratio)
    split_idx = max(1, min(split_idx, n_after - 1))

    df_train = df.iloc[:split_idx].copy()
    df_etl = df.iloc[split_idx:].copy()

    # Ensure output dirs
    output_train_csv.parent.mkdir(parents=True, exist_ok=True)
    output_etl_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save
    df_train.to_csv(output_train_csv, index=False)
    df_etl.to_csv(output_etl_csv, index=False)

    # Temporal sanity check
    train_max_time = df_train[timestamp_col].max()
    etl_min_time = df_etl[timestamp_col].min()

    print("=== Split summary ===")
    print(f"Input rows (before timestamp drop): {n_before}")
    print(f"Input rows (after timestamp drop):  {n_after}")
    print(f"Train rows: {len(df_train)}")
    print(f"ETL rows:   {len(df_etl)}")
    print(f"Train max {timestamp_col}: {train_max_time}")
    print(f"ETL   min {timestamp_col}: {etl_min_time}")
    print(f"Leakage check (train_max <= etl_min): {train_max_time <= etl_min_time}")


if __name__ == "__main__":
    time_based_split(
        input_csv="../data/raw/transactions.csv",
        output_train_csv="../data/split/transactions_train_90.csv",
        output_etl_csv="../data/split/transactions_etl_10.csv",
        timestamp_col="transaction_time",
        train_ratio=0.90,
    )
