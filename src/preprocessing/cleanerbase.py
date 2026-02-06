import os
from pathlib import Path

import numpy as np
import pandas as pd


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Timestamp robuste
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["transaction_time"])

    # 2) Tri chronologique par user (indispensable pour features historiques)
    df = df.sort_values(["user_id", "transaction_time"]).reset_index(drop=True)

    # 3) Features temporelles
    df["hour"] = df["transaction_time"].dt.hour.astype(int)
    df["dayofweek"] = df["transaction_time"].dt.dayofweek.astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # 4) Historique montants (expanding mean sans futur via shift(1))
    df["avg_amount_user_past"] = (
        df.groupby("user_id")["amount"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    df["avg_amount_user_past"] = df["avg_amount_user_past"].fillna(df["amount"].median())
    df["amount_diff_user_avg"] = df["amount"] - df["avg_amount_user_past"]

    # 5) Signaux de risque
    df["is_new_account"] = (df["account_age_days"] < 30).astype(int)

    cvv_num = pd.to_numeric(df["cvv_result"], errors="coerce")
    df["security_mismatch_score"] = (
        (df["avs_match"] == 0).astype(int) + (cvv_num == 0).astype(int)
    ).astype(int)

    # 6) Historique fraude utilisateur (passé connu)
    # IMPORTANT: shift(1) => pas de fuite vers l'observation courante
    df["user_fraud_count"] = (
        df.groupby("user_id")["is_fraud"].cumsum().shift(1).fillna(0.0)
    ).astype(float)

    df["user_tx_count"] = df.groupby("user_id").cumcount().astype(int)

    denom = df["user_tx_count"].replace(0, np.nan).astype(float)
    df["user_fraud_rate"] = (df["user_fraud_count"] / denom).fillna(0.0).replace([np.inf, -np.inf], 0.0)

    df["user_has_fraud_history"] = (df["user_fraud_count"] > 0).astype(int)

    # 7) Géographie et comportement
    df["country_bin_mismatch"] = (df["country"] != df["bin_country"]).astype(int)
    df["distance_amount_ratio"] = df["shipping_distance_km"] / (df["amount"] + 1.0)

    df["amount_delta_prev"] = df.groupby("user_id")["amount"].diff().fillna(0.0)

    df["channel_changed"] = (
        df["channel"] != df.groupby("user_id")["channel"].shift()
    ).astype(int)

    df["time_since_last"] = (
        df.groupby("user_id")["transaction_time"].diff().dt.total_seconds().fillna(0.0)
    )

    # 8) Rolling windows (FIX robuste: scalaire par ligne, index aligné, pas de duplicats)
    windows = {"24h": "tx_last_24h", "7d": "tx_last_7d", "30d": "tx_last_30d"}

    # Pré-allocation (plus rapide que concat/apply)
    for window, col in windows.items():
        out = np.empty(len(df), dtype=float)

        # df est déjà trié user_id + time, mais on sécurise par groupe
        for _, idx in df.groupby("user_id").groups.items():
            g = df.loc[idx].sort_values("transaction_time")

            counts = (
                g.set_index("transaction_time")["amount"]
                .rolling(window, closed="left")
                .count()
                .to_numpy()
            )

            # Remplir aux positions correspondantes (index de df)
            out[g.index.to_numpy()] = counts

        df[col] = out
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    # 9) Drop colonnes non utilisées par le modèle
    df = df.drop(columns=["transaction_id", "transaction_time"], errors="ignore")

    # 10) One-hot encoding
    df = pd.get_dummies(
        df,
        columns=["country", "bin_country", "channel", "merchant_category"],
        drop_first=False
    )

    # 11) Bool -> int (évite True/False dans le CSV)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # 12) Sécurité: aucune colonne object inattendue (hors cas rares)
    # (On ne force pas ici, mais on repère si besoin via le print dans run)
    return df


def run_train_preprocessing():
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data/split/transactions_train_90.csv"
    output_path = project_root / "data/processed/train_after_preprocess.csv"

    print("Preprocessing TRAIN uniquement")

    df_train = pd.read_csv(input_path)
    n_before = df_train.shape[1]

    df_train_prep = preprocess_transactions(df_train)
    n_after = df_train_prep.shape[1]

    os.makedirs(output_path.parent, exist_ok=True)
    df_train_prep.to_csv(output_path, index=False)

    print(f"Colonnes avant : {n_before}")
    print(f"Colonnes après : {n_after}")
    print(f"Fichier sauvegardé : {output_path}")

    # Checks rapides
    for c in ["tx_last_24h", "tx_last_7d", "tx_last_30d"]:
        if c in df_train_prep.columns:
            print(f"{c} dtype: {df_train_prep[c].dtype}")

    obj_cols = df_train_prep.select_dtypes(include=["object"]).columns.tolist()
    print(f"Colonnes object restantes: {obj_cols}")

    print(df_train_prep.head())


if __name__ == "__main__":
    run_train_preprocessing()
