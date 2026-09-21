"""
Métriques spécifiques à la détection de fraude.
Avec 0.15% de fraude, l'accuracy et l'AUC-ROC sont trompeuses.
On utilise AUC-PR, Recall@FPR, Lift, et métriques coût-bénéfice.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve, f1_score, confusion_matrix
)


def auc_pr(y_true, y_score):
    """Area Under Precision-Recall Curve.
    Plus informative que ROC-AUC quand la classe positive est rare.
    """
    return average_precision_score(y_true, y_score)


def recall_at_fpr(y_true, y_score, target_fpr=0.01):
    """Recall quand on fixe le FPR à target_fpr (ex: 1%).

    Répond à : "Si on accepte X% de faux positifs, combien de fraudes attrape-t-on ?"
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(fpr - target_fpr))
    return float(tpr[idx])


def precision_at_recall(y_true, y_score, target_recall=0.80):
    """Precision quand on fixe le Recall à target_recall (ex: 80%).

    Répond à : "Si on veut attraper X% des fraudes, quelle proportion d'alertes sont vraies ?"
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    idx = np.argmin(np.abs(recall - target_recall))
    return float(precision[idx])


def lift_score(y_true, y_score, top_pct=0.01):
    """Lift dans le top X% : combien de fois mieux que le hasard.

    Ex: lift=50 signifie que le top 1% contient 50× plus de fraudes que la moyenne.
    """
    n = len(y_true)
    k = int(n * top_pct)
    if k == 0:
        return 1.0

    top_idx = np.argsort(y_score)[-k:]
    fraud_rate_top = y_true.iloc[top_idx].mean() if hasattr(y_true, "iloc") else y_true[top_idx].mean()
    fraud_rate_all = y_true.mean()

    if fraud_rate_all == 0:
        return np.inf

    return float(fraud_rate_top / fraud_rate_all)


def all_fraud_metrics(y_true, y_score):
    """Toutes les métriques fraude en un appel.

    Returns
    -------
    dict avec AUC-ROC, AUC-PR, Recall@FPR=1%, Precision@Recall=80%,
    F1, Lift@1%, Lift@5%
    """
    return {
        "auc_roc": roc_auc_score(y_true, y_score),
        "auc_pr": auc_pr(y_true, y_score),
        "recall_at_fpr_1pct": recall_at_fpr(y_true, y_score, target_fpr=0.01),
        "recall_at_fpr_2pct": recall_at_fpr(y_true, y_score, target_fpr=0.02),
        "precision_at_recall_80": precision_at_recall(y_true, y_score, target_recall=0.80),
        "f1": f1_score(y_true, y_score > 0.5),
        "lift_1pct": lift_score(y_true, y_score, top_pct=0.01),
        "lift_5pct": lift_score(y_true, y_score, top_pct=0.05),
    }


def cost_benefit_analysis(y_true, y_score, thresholds=None,
                          cost_fp=5000, cost_fn=500000,
                          investigation_budget=1000000):
    """Analyse coût-bénéfice pour trouver le seuil optimal.

    Parameters
    ----------
    y_true : array
    y_score : array
    thresholds : array or None (générés automatiquement)
    cost_fp : float — coût d'investigation d'un faux positif (FCFA)
    cost_fn : float — perte moyenne par fraude non détectée (FCFA)
    investigation_budget : float — budget max d'investigation (FCFA)

    Returns
    -------
    DataFrame avec colonnes: threshold, n_alerts, fp, fn, tp, tn,
    cost_fp_total, cost_fn_total, cost_total, frauds_caught, roi
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    results = []
    n_samples = len(y_true)

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cost_fp_total = fp * cost_fp
        cost_fn_total = fn * cost_fn
        cost_total = cost_fp_total + cost_fn_total

        # ROI = (pertes évitées - coûts) / coûts
        baseline_loss = (y_true.sum()) * cost_fn  # Pertes sans modèle
        losses_avoided = baseline_loss - cost_fn_total
        roi = (losses_avoided - cost_fp_total) / (cost_fp_total + 1)

        within_budget = cost_fp_total <= investigation_budget

        results.append({
            "threshold": thresh,
            "n_alerts": int(fp + tp),
            "alert_rate_pct": (fp + tp) / n_samples * 100,
            "fp": int(fp), "fn": int(fn),
            "tp": int(tp), "tn": int(tn),
            "cost_fp_total": cost_fp_total,
            "cost_fn_total": cost_fn_total,
            "cost_total": cost_total,
            "frauds_caught_pct": tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "roi": roi,
            "within_budget": within_budget,
        })

    return pd.DataFrame(results)


def find_optimal_threshold(cb_df, min_recall=0.75, max_fpr=0.015):
    """Trouve le seuil optimal : maximise le recall sous contrainte de FPR max.

    Parameters
    ----------
    cb_df : DataFrame issu de cost_benefit_analysis
    min_recall : float — recall minimum exigé
    max_fpr : float — FPR maximum toléré

    Returns
    -------
    dict avec le seuil optimal et ses métriques
    """
    # Filtrer les seuils qui respectent les contraintes
    valid = cb_df[
        (cb_df["frauds_caught_pct"] >= min_recall * 100) &
        (cb_df["alert_rate_pct"] <= max_fpr * 100 + cb_df["frauds_caught_pct"].mean() / 100)
    ].copy()

    if len(valid) == 0:
        # Relâcher les contraintes
        valid = cb_df[cb_df["frauds_caught_pct"] >= 50].copy()

    if len(valid) == 0:
        valid = cb_df.copy()

    # Meilleur = coût total minimum
    best = valid.loc[valid["cost_total"].idxmin()]
    return best.to_dict()
