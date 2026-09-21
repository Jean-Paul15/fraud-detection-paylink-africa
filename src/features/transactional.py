"""
Features transactionnelles — Instant T.
Calculées en temps réel au moment de la transaction.
Aucune information future n'est utilisée (anti-leakage).
"""
import numpy as np
import pandas as pd


def extract_transactional_features(df, top_cities=None):
    """Extrait les features disponibles au moment exact de la transaction.

    Ces features ne dépendent QUE de la transaction courante,
    jamais des transactions futures ou du contexte global.

    Parameters
    ----------
    df : DataFrame avec timestamp, amount, transaction_type, channel,
         device_id, sender_id, receiver_id, city, hour, is_night,
         sender_account_age, sender_kyc, is_new_device
    top_cities : list or None
        Liste des top villes calculee sur le TRAIN uniquement.
        Si None, calculee sur df (usage train uniquement, jamais test).

    Returns
    -------
    DataFrame avec les features transactionnelles
    """
    feats = pd.DataFrame(index=df.index)

    # Montant
    feats["amount"] = df["amount"].fillna(0)
    feats["amount_log"] = np.log1p(df["amount"].clip(lower=0))

    # Type de transaction (one-hot)
    tx_dummies = pd.get_dummies(df["transaction_type"], prefix="tx_type")
    for col in tx_dummies.columns:
        feats[col] = tx_dummies[col].astype(int)

    # Canal (one-hot)
    ch_dummies = pd.get_dummies(df["channel"], prefix="channel")
    for col in ch_dummies.columns:
        feats[col] = ch_dummies[col].astype(int)

    # Temporelles
    if "hour" in df.columns:
        feats["hour"] = df["hour"].fillna(12)
        feats["hour_sin"] = np.sin(2 * np.pi * feats["hour"] / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * feats["hour"] / 24)

    if "is_night" in df.columns:
        feats["is_night"] = df["is_night"].fillna(0).astype(int)

    if "day_of_week" in df.columns:
        feats["day_of_week"] = df["day_of_week"].fillna(3)
        feats["is_weekend"] = (df["day_of_week"].isin([5, 6])).astype(int)

    # Device
    if "is_new_device" in df.columns:
        feats["is_new_device"] = df["is_new_device"].fillna(0).astype(int)

    # Compte sender
    if "sender_account_age" in df.columns:
        feats["sender_account_age"] = df["sender_account_age"].fillna(0)
        feats["sender_account_age_log"] = np.log1p(feats["sender_account_age"])

    if "sender_kyc" in df.columns:
        kyc_dummies = pd.get_dummies(df["sender_kyc"].fillna(0).astype(int), prefix="kyc")
        for col in kyc_dummies.columns:
            feats[col] = kyc_dummies[col].astype(int)

    # Bénéficiaire
    if "is_new_receiver" in df.columns:
        feats["is_new_receiver"] = df["is_new_receiver"].fillna(0).astype(int)

    # Ville (one-hot, limité à top 15)
    # ANTI-LEAKAGE: top_cities doit être calculé sur le train et réutilisé pour le test
    if "city" in df.columns:
        if top_cities is None:
            top_cities = df["city"].value_counts().head(15).index
        for city in top_cities:
            feats[f"city_{city}"] = (df["city"] == city).astype(int)
        feats["city_other"] = (~df["city"].isin(top_cities)).astype(int)

    return feats
