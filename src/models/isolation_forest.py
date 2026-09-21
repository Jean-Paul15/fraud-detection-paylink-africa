"""
Modele 1 — Isolation Forest pour détection d'anomalies.

Principe : Les fraudes sont des anomalies — des transactions qui
ne ressemblent pas aux transactions légitimes. Isolation Forest
isole ces points aberrants en construisant des arbres aléatoires.

Avantages pour la fraude :
- Non-supervisé : fonctionne sans labels (détection de nouvelles fraudes)
- Rapide : O(n) en entraînement, < 1ms en inférence
- Robuste au déséquilibre : ne suppose pas de distribution de classe
- Interprétable : anomaly_score -> probabilité d'être frauduleux
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from ..config import RANDOM_STATE


def train_isolation_forest(X_train, contamination="auto", n_estimators=200,
                           max_samples=0.5, max_features=0.8):
    """Entraîne un Isolation Forest.

    Parameters
    ----------
    X_train : array
        Données d'entraînement (légitimes + fraudes)
    contamination : float or "auto"
        Proportion estimée d'anomalies. "auto" utilise le taux de fraude réel.
    n_estimators : int
        Nombre d'arbres (plus = plus stable, mais plus lent)
    max_samples : float
        Fraction des données utilisée par arbre (0.5 = chaque arbre voit 50%)
    max_features : float
        Fraction des features utilisée par arbre

    Returns
    -------
    IsolationForest fitted
    """
    if contamination == "auto":
        # Estimation conservative du taux d'anomalie
        contamination = 0.01  # 1% — plus large que le vrai taux (0.15%)

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=False,
    )

    model.fit(X_train)
    return model


def anomaly_score_to_proba(anomaly_scores):
    """Convertit les scores d'anomalie IF en pseudo-probabilités [0, 1].

    Isolation Forest retourne des scores dans [-1, 1] où :
    - score < 0 = anomalie probable
    - score > 0 = normal probable

    On utilise une transformation sigmoïde calibrée.
    """
    # Normaliser entre 0 et 1
    scores = np.array(anomaly_scores)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    # Inverser : score élevé = probabilité de fraude élevée
    return 1 - scores_norm


def tune_isolation_forest(X_train, y_train, X_val, y_val,
                          contaminations=None, n_estimators_list=None):
    """Recherche des meilleurs hyperparamètres pour Isolation Forest.

    Évalue avec AUC-PR (plus informative que ROC-AUC pour classes rares).

    Returns
    -------
    dict avec best_params, best_model, best_auc_pr, results_df
    """
    if contaminations is None:
        contaminations = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    if n_estimators_list is None:
        n_estimators_list = [100, 200, 300]

    results = []
    best_auc_pr = 0
    best_model = None
    best_params = None

    for cont in contaminations:
        for n_est in n_estimators_list:
            model = train_isolation_forest(
                X_train,
                contamination=cont,
                n_estimators=n_est,
                max_samples=0.5,
                max_features=0.8,
            )

            # Score sur validation
            val_scores = model.decision_function(X_val)
            val_proba = anomaly_score_to_proba(val_scores)

            auc_pr = average_precision_score(y_val, val_proba)
            auc_roc = roc_auc_score(y_val, val_proba)

            results.append({
                "contamination": cont,
                "n_estimators": n_est,
                "auc_pr": auc_pr,
                "auc_roc": auc_roc,
            })

            if auc_pr > best_auc_pr:
                best_auc_pr = auc_pr
                best_model = model
                best_params = {"contamination": cont, "n_estimators": n_est}

    return {
        "best_params": best_params,
        "best_model": best_model,
        "best_auc_pr": best_auc_pr,
        "results_df": pd.DataFrame(results),
    }


def plot_if_scores(model, X, y, n_samples=5000):
    """Visualise la distribution des scores d'anomalie par classe."""
    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), min(n_samples, len(X)), replace=False
    )
    scores = model.decision_function(X[idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores[y.iloc[idx] == 0], bins=50, alpha=0.6, label="Légitime", color="#27ae60")
    ax.hist(scores[y.iloc[idx] == 1], bins=50, alpha=0.6, label="Fraude", color="#e74c3c")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Seuil IF (0)")
    ax.set_xlabel("Score d'anomalie (decision_function)")
    ax.set_ylabel("Nombre de transactions")
    ax.set_title("Isolation Forest — Distribution des scores par classe")
    ax.legend()
    return fig
