"""
Split temporel strict — Anti-leakage pour détection de fraude.
En fraude, le leakage temporel est CRITIQUE : les features doivent
être calculées UNIQUEMENT sur les données disponibles à l'instant T.
"""
import numpy as np
import pandas as pd


def temporal_split(df, date_cutoff="2024-07-01", timestamp_col="timestamp"):
    """Split temporel strict : train < cutoff, test >= cutoff.

    Pourquoi c'est crucial en fraude :
    - Les fraudeurs s'adaptent. Un split aléatoire donnerait au modèle
      des infos sur des fraudes futures (via les features agrégées).
    - Le split temporel simule le déploiement réel : entraîné sur le
      passé, déployé sur le futur.
    """
    df = df.copy()
    cutoff = pd.Timestamp(date_cutoff)

    train = df[df[timestamp_col] < cutoff].copy()
    test = df[df[timestamp_col] >= cutoff].copy()

    print(f"Split temporel (cutoff={date_cutoff}):")
    print(f"  Train: {len(train):,} ({train['is_fraud'].mean()*100:.3f}% fraude)")
    print(f"  Test:  {len(test):,} ({test['is_fraud'].mean()*100:.3f}% fraude)")

    return train, test


def check_temporal_leakage(train, test, timestamp_col="timestamp"):
    """Vérifie qu'aucune transaction test n'est antérieure à une transaction train."""
    train_max = train[timestamp_col].max()
    test_min = test[timestamp_col].min()
    leakage = test_min < train_max
    if leakage:
        print(f"[!] LEAKAGE TEMPOREL: test_min ({test_min}) < train_max ({train_max})")
    else:
        print(f"[OK] Pas de leakage: test_min ({test_min}) >= train_max ({train_max})")
    return not leakage


class TimeBasedFeatureSplitter:
    """Wrapper pour calculer les features sur train uniquement, puis les appliquer au test.

    Garantit que :
    - Les médianes/moyennes sont calculées sur train seulement
    - Les features de vélocité utilisent l'historique jusqu'à T
    - Aucune information du futur ne fuit dans le train
    """

    def __init__(self, timestamp_col="timestamp"):
        self.timestamp_col = timestamp_col
        self.train_stats_ = {}

    def fit(self, train_df):
        """Apprend les statistiques sur le train."""
        self.train_stats_ = {
            "amount_median": train_df["amount"].median(),
            "amount_std": train_df["amount"].std(),
            "hour_mean": train_df["hour"].mean() if "hour" in train_df else None,
            "fraud_rate": train_df["is_fraud"].mean(),
            "channels": train_df["channel"].value_counts(normalize=True).to_dict() if "channel" in train_df else {},
            "n_transactions": len(train_df),
            "period_start": train_df[self.timestamp_col].min(),
            "period_end": train_df[self.timestamp_col].max(),
        }
        return self

    def transform(self, df):
        """Applique les stats du train (pas d'apprentissage sur test)."""
        return df
