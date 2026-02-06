# src/preprocessing/cleaner_update.py

import os  # gestion des dossiers / chemins
from pathlib import Path  # chemins robustes multi-OS

import numpy as np  # calculs numériques
import pandas as pd  # manipulation de données


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:  # fonction de preprocessing (update 10%)
    df = df.copy()  # copie pour éviter effets de bord

    df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce", utc=True)  # parse timestamp
    df = df.dropna(subset=["transaction_time"])  # drop lignes sans timestamp

    df = df.sort_values(["user_id", "transaction_time"]).reset_index(drop=True)  # tri chrono par user

    df["hour"] = df["transaction_time"].dt.hour.astype(int)  # heure
    df["dayofweek"] = df["transaction_time"].dt.dayofweek.astype(int)  # jour semaine
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)  # indicateur nuit

    df["avg_amount_user_past"] = (  # moyenne historique des montants (sans fuite)
        df.groupby("user_id")["amount"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )  # compute expanding mean sur passé seulement
    df["avg_amount_user_past"] = df["avg_amount_user_past"].fillna(df["amount"].median())  # fallback initial
    df["amount_diff_user_avg"] = df["amount"] - df["avg_amount_user_past"]  # écart au comportement

    df["is_new_account"] = (df["account_age_days"] < 30).astype(int)  # compte récent
    cvv_num = pd.to_numeric(df["cvv_result"], errors="coerce")  # cvv en numérique si possible
    df["security_mismatch_score"] = (  # score mismatch sécurité
        (df["avs_match"] == 0).astype(int) + (cvv_num == 0).astype(int)
    ).astype(int)  # somme indicateurs

    df["user_fraud_count"] = (  # nb fraudes passées connues (pas l’observation courante)
        df.groupby("user_id")["is_fraud"].cumsum().shift(1).fillna(0.0)
    ).astype(float)  # float pour division
    df["user_tx_count"] = df.groupby("user_id").cumcount().astype(int)  # nb transactions passées
    denom = df["user_tx_count"].replace(0, np.nan).astype(float)  # éviter division par 0
    df["user_fraud_rate"] = (df["user_fraud_count"] / denom).fillna(0.0).replace([np.inf, -np.inf], 0.0)  # taux
    df["user_has_fraud_history"] = (df["user_fraud_count"] > 0).astype(int)  # flag historique fraude

    df["country_bin_mismatch"] = (df["country"] != df["bin_country"]).astype(int)  # mismatch pays BIN
    df["distance_amount_ratio"] = df["shipping_distance_km"] / (df["amount"] + 1.0)  # ratio distance/montant
    df["amount_delta_prev"] = df.groupby("user_id")["amount"].diff().fillna(0.0)  # delta montant précédent
    df["channel_changed"] = (  # changement de canal vs précédent
        df["channel"] != df.groupby("user_id")["channel"].shift()
    ).astype(int)  # int 0/1
    df["time_since_last"] = (  # temps depuis dernière transaction
        df.groupby("user_id")["transaction_time"].diff().dt.total_seconds().fillna(0.0)
    )  # secondes

    windows = {"24h": "tx_last_24h", "7d": "tx_last_7d", "30d": "tx_last_30d"}  # fenêtres rolling
    for window, col in windows.items():  # boucle fenêtres
        out = np.empty(len(df), dtype=float)  # buffer sortie aligné index df
        for _, idx in df.groupby("user_id").groups.items():  # indices par user
            g = df.loc[idx].sort_values("transaction_time")  # sous-table user triée
            counts = (  # compte transactions passées sur fenêtre
                g.set_index("transaction_time")["amount"]
                .rolling(window, closed="left")
                .count()
                .to_numpy()
            )  # numpy
            out[g.index.to_numpy()] = counts  # réinjection aux bons index
        df[col] = np.nan_to_num(out, nan=0.0)  # NaN -> 0

    df = df.drop(columns=["transaction_id", "transaction_time"], errors="ignore")  # drop ID/time non-features

    df = pd.get_dummies(  # one-hot encoding
        df,
        columns=["country", "bin_country", "channel", "merchant_category"],
        drop_first=False,
    )  # dummies

    bool_cols = df.select_dtypes(include=["bool"]).columns  # colonnes bool
    if len(bool_cols) > 0:  # si existe
        df[bool_cols] = df[bool_cols].astype(int)  # bool -> 0/1

    return df  # retourne dataframe preprocessé (AVEC is_fraud)


def run_update_preprocessing() -> None:
    INPUT_PATH = Path("/opt/airflow/output/transactions_etl_10.csv")
    OUTPUT_PATH = Path("/opt/airflow/output/update_after_preprocess.csv")

    print("Preprocessing UPDATE (10%)")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")

    df_update = pd.read_csv(INPUT_PATH)
    n_before = df_update.shape[1]

    df_update_prep = preprocess_transactions(df_update)
    n_after = df_update_prep.shape[1]

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    df_update_prep.to_csv(OUTPUT_PATH, index=False)

    print(f"Colonnes avant : {n_before}")
    print(f"Colonnes après : {n_after}")
    print(f"Fichier sauvegardé : {OUTPUT_PATH}")
    print(df_update_prep.head())


if __name__ == "__main__":  # entrypoint
    run_update_preprocessing()  # exécution
