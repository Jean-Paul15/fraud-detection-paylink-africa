"""
Ensemble — Combinaison des modèles IF + AE + LGB.

Stratégie : Les 3 modèles capturent des aspects différents de la fraude.
- Isolation Forest : anomalies statistiques (non supervisé)
- Autoencoder : anomalies de reconstruction (deep learning)
- LightGBM : patterns supervisés (gradient boosting)

Le stacking combine leurs scores pour maximiser la détection.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..config import RANDOM_STATE


def stack_models(model_scores, y_true, meta_model=None, cv_folds=5):
    """Stacking avec méta-modèle (LogReg par défaut).

    Chaque modèle de base produit un score de fraude [0, 1].
    Le méta-modèle apprend à les pondérer de façon optimale.

    Parameters
    ----------
    model_scores : dict {nom: array de scores}
    y_true : array
    meta_model : classifieur or None
    cv_folds : int — folds pour out-of-fold stacking

    Returns
    -------
    meta_model fitted, DataFrame d'importance des modèles de base
    """
    if meta_model is None:
        meta_model = LogisticRegression(C=1.0, class_weight="balanced",
                                        random_state=RANDOM_STATE, max_iter=1000)

    # Construire la matrice de meta-features
    X_meta = np.column_stack([scores for scores in model_scores.values()])

    # Out-of-fold stacking pour éviter le surapprentissage
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros((len(y_true), len(model_scores)))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta, y_true)):
        X_tr, y_tr = X_meta[train_idx], y_true.iloc[train_idx] if hasattr(y_true, "iloc") else y_true[train_idx]
        X_val = X_meta[val_idx]

        meta_model.fit(X_tr, y_tr)
        oof_preds[val_idx] = meta_model.predict_proba(X_val)[:, 1]

    # Importance des modèles de base (coefficients LogReg)
    importance = pd.DataFrame({
        "model": list(model_scores.keys()),
        "coefficient": meta_model.coef_[0],
        "abs_coef": np.abs(meta_model.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    return meta_model, importance, oof_preds


def weighted_average_ensemble(model_scores, weights=None):
    """Ensemble par moyenne pondérée (simple, rapide, interprétable).

    Si pas de poids fournis, utilise la moyenne simple.
    """
    if weights is None:
        weights = {name: 1.0 for name in model_scores}

    scores = np.zeros(len(list(model_scores.values())[0]))
    total_weight = sum(weights.values())

    for name, score in model_scores.items():
        scores += score * weights[name]

    return scores / total_weight


def ensemble_predict(model_scores, ensemble_type="stacking", meta_model=None,
                     weights=None, threshold=0.5):
    """Prédiction de l'ensemble.

    Parameters
    ----------
    model_scores : dict
    ensemble_type : str — "stacking" ou "weighted"
    meta_model : fitted model (pour stacking)
    weights : dict (pour weighted)
    threshold : float

    Returns
    -------
    array de probabilités de fraude
    """
    if ensemble_type == "stacking" and meta_model is not None:
        X_meta = np.column_stack([scores for scores in model_scores.values()])
        return meta_model.predict_proba(X_meta)[:, 1]
    else:
        return weighted_average_ensemble(model_scores, weights)
