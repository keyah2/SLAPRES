from pathlib import Path
import os

import numpy as np
import pandas as pd


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["transaction_time"])

    df = df.sort_values(["user_id", "transaction_time"]).reset_index(drop=True)

    df["hour"] = df["transaction_time"].dt.hour.astype(int)
    df["dayofweek"] = df["transaction_time"].dt.dayofweek.astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    df["avg_amount_user_past"] = (
        df.groupby("user_id")["amount"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    df["avg_amount_user_past"] = df["avg_amount_user_past"].fillna(df["amount"].median())
    df["amount_diff_user_avg"] = df["amount"] - df["avg_amount_user_past"]

    df["is_new_account"] = (df["account_age_days"] < 30).astype(int)

    cvv_num = pd.to_numeric(df["cvv_result"], errors="coerce")
    df["security_mismatch_score"] = (
        (df["avs_match"] == 0).astype(int) + (cvv_num == 0).astype(int)
    ).astype(int)

    df["user_fraud_count"] = (
        df.groupby("user_id")["is_fraud"].cumsum().shift(1).fillna(0.0)
    ).astype(float)

    df["user_tx_count"] = df.groupby("user_id").cumcount().astype(int)
    denom = df["user_tx_count"].replace(0, np.nan).astype(float)
    df["user_fraud_rate"] = (df["user_fraud_count"] / denom).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    df["user_has_fraud_history"] = (df["user_fraud_count"] > 0).astype(int)

    df["country_bin_mismatch"] = (df["country"] != df["bin_country"]).astype(int)
    df["distance_amount_ratio"] = df["shipping_distance_km"] / (df["amount"] + 1.0)

    df["amount_delta_prev"] = df.groupby("user_id")["amount"].diff().fillna(0.0)

    df["channel_changed"] = (
        df["channel"] != df.groupby("user_id")["channel"].shift()
    ).astype(int)

    df["time_since_last"] = (
        df.groupby("user_id")["transaction_time"].diff().dt.total_seconds().fillna(0.0)
    )

    windows = {"24h": "tx_last_24h", "7d": "tx_last_7d", "30d": "tx_last_30d"}
    for window, col in windows.items():
        out = np.empty(len(df), dtype=float)
        for _, idx in df.groupby("user_id").groups.items():
            g = df.loc[idx].sort_values("transaction_time")
            counts = (
                g.set_index("transaction_time")["amount"]
                .rolling(window, closed="left")
                .count()
                .to_numpy()
            )
            out[g.index.to_numpy()] = counts
        df[col] = pd.to_numeric(out, errors="coerce")
        df[col] = df[col].fillna(0.0).astype(float)

    df = df.drop(columns=["transaction_id", "transaction_time"], errors="ignore")

    df = pd.get_dummies(
        df,
        columns=["country", "bin_country", "channel", "merchant_category"],
        drop_first=False
    )

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    return df


def run_train_preprocessing_to_output() -> None:
    project_root = Path(__file__).resolve().parents[1]


    input_path = project_root / "data" / "split" / "transactions_train_90.csv"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # fichier daté + latest
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path_dated = output_dir / f"train_after_preprocess_{ts}.csv"
    output_path_latest = output_dir / "train_after_preprocess_latest.csv"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path_dated}")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df_train = pd.read_csv(input_path)
    print(f"RAW shape: {df_train.shape}")

    df_train_prep = preprocess_transactions(df_train)
    print(f"PREP shape: {df_train_prep.shape}")

    df_train_prep.to_csv(output_path_dated, index=False)
    df_train_prep.to_csv(output_path_latest, index=False)

    print(f"Saved dated:  {output_path_dated}")
    print(f"Saved latest: {output_path_latest}")


if __name__ == "__main__":
    run_train_preprocessing_to_output()
