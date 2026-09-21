"""
Features comportementales — Agrégations temporelles anti-leakage.

Approche robuste : accumulation séquentielle par sender.
Pour chaque transaction à T, on agrège les transactions du même sender
strictement antérieures à T dans chaque fenêtre ]T-window, T[.

Optimisé : au lieu de rolling (qui ne supporte pas les strings sur pandas 3.0),
on calcule un cumul glissant via indexation temporelle triée.
"""
import numpy as np
import pandas as pd


def compute_behavioral_features(df, windows=None, sample_senders=None):
    """Calcule les features comportementales par sender_id.

    Parameters
    ----------
    df : DataFrame avec timestamp, sender_id, amount
         Doit être trié par sender_id puis timestamp !
    windows : list of str
        Format pandas: "1h", "6h", "24h", "7d", "30d"
    sample_senders : int or None
        Si fourni, échantillonne N senders (pour rapidité)

    Returns
    -------
    DataFrame avec les features comportementales (même index que df)
    """
    if windows is None:
        windows = ["1h", "6h", "24h"]

    df = df.sort_values(["sender_id", "timestamp"]).copy()
    original_index = df.index

    # Si trop de senders, échantillonner pour rapidité
    if sample_senders is not None:
        all_senders = df["sender_id"].unique()
        if len(all_senders) > sample_senders:
            selected = np.random.RandomState(42).choice(
                all_senders, sample_senders, replace=False
            )
            df = df[df["sender_id"].isin(selected)].copy()

    # Réinitialiser l'index pour que feats ait le bon nombre de lignes
    # Garder l'index original pour le retour
    sampled_index = df.index
    df = df.reset_index(drop=True)
    feats = pd.DataFrame(index=range(len(df)))

    # Pour chaque fenêtre, on compte les transactions antérieures
    for window in windows:
        window_td = pd.Timedelta(window)
        w_label = window.replace("d", "d").replace("h", "h")

        # Pour chaque sender : compter et sommer sur la fenêtre glissante
        counts = []
        sums = []
        means = []
        stds = []

        for sender_id, group in df.groupby("sender_id", sort=False):
            times = group["timestamp"].values
            amounts = group["amount"].values if "amount" in group.columns else np.ones(len(group))

            count_arr = np.zeros(len(group))
            sum_arr = np.zeros(len(group))
            mean_arr = np.zeros(len(group))
            std_arr = np.zeros(len(group))

            # Pour chaque transaction, compter les transactions dans la fenêtre
            for i in range(len(group)):
                # Trouver le début de la fenêtre par recherche binaire
                cutoff = times[i] - window_td
                # Recherche du premier index >= cutoff
                j = np.searchsorted(times[:i], cutoff, side="left")
                n_in_window = i - j

                if n_in_window > 0:
                    count_arr[i] = n_in_window
                    sum_arr[i] = amounts[j:i].sum()
                    mean_arr[i] = amounts[j:i].mean()
                    std_arr[i] = amounts[j:i].std() if n_in_window > 1 else 0

            counts.append(count_arr)
            sums.append(sum_arr)
            means.append(mean_arr)
            stds.append(std_arr)

        feats[f"sender_tx_count_{w_label}"] = np.concatenate(counts) if counts else 0
        feats[f"sender_amount_sum_{w_label}"] = np.concatenate(sums) if sums else 0
        feats[f"sender_amount_mean_{w_label}"] = np.concatenate(means) if means else 0
        feats[f"sender_amount_std_{w_label}"] = np.concatenate(stds) if stds else 0

    # Remapper sur l'index original
    feats.index = sampled_index
    return feats


def compute_velocity_features(df):
    """Features de vélocité — détection d'accélération soudaine.

    Approche robuste : différences temporelles entre transactions consécutives.
    """
    df = df.sort_values(["sender_id", "timestamp"]).copy()
    feats = pd.DataFrame(index=df.index)

    # Temps depuis la dernière transaction
    time_diffs = df.groupby("sender_id", sort=False)["timestamp"].diff()
    feats["time_since_last_tx_sec"] = time_diffs.dt.total_seconds().fillna(86400 * 30).clip(lower=0)
    feats["time_since_last_tx_log"] = np.log1p(feats["time_since_last_tx_sec"])

    # Changement de device (si la colonne existe)
    if "device_id" in df.columns:
        feats["device_changed"] = (
            df.groupby("sender_id", sort=False)["device_id"]
            .transform(lambda x: (x != x.shift(1)).astype(int))
            .fillna(0).astype(int)
        )

    # Changement de ville
    if "city" in df.columns:
        feats["city_changed"] = (
            df.groupby("sender_id", sort=False)["city"]
            .transform(lambda x: (x != x.shift(1)).astype(int))
            .fillna(0).astype(int)
        )

    return feats


def compute_ratio_features(df, behavioral_feats):
    """Features de ratio — écart par rapport au comportement habituel."""
    feats = pd.DataFrame(index=df.index)

    if "amount" in df.columns:
        for w in ["1h", "6h", "24h"]:
            mean_col = f"sender_amount_mean_{w}"
            if mean_col in behavioral_feats.columns:
                hist_mean = behavioral_feats[mean_col].clip(lower=1)
                feats[f"amount_ratio_{w}"] = (df["amount"].values / hist_mean.values).clip(0, 100)
                feats[f"amount_ratio_{w}"] = feats[f"amount_ratio_{w}"].fillna(1.0)

    return feats
