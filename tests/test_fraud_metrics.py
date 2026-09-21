import numpy as np
import pandas as pd

from src.evaluation.fraud_metrics import (
    all_fraud_metrics,
    cost_benefit_analysis,
    find_optimal_threshold,
    lift_score,
    recall_at_fpr,
)


def test_lift_score_top_k_contient_uniquement_les_fraudes():
    y_true = pd.Series([0] * 90 + [1] * 10)
    y_score = np.concatenate([np.linspace(0, 0.5, 90), np.linspace(0.6, 1.0, 10)])
    assert lift_score(y_true, y_score, top_pct=0.10) == 10.0


def test_recall_at_fpr_croit_avec_le_fpr_tolere():
    # roc_curve() elague les points colineaires (drop_intermediate=True par
    # defaut) : sur une separation parfaite, seul un test de monotonie est
    # fiable, un seuil de recall exact dependrait de cet elagage interne.
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 300)
    y_score = y_true * 0.6 + rng.random(300) * 0.4  # correle au label, bruite

    recall_tolerant = recall_at_fpr(y_true, y_score, target_fpr=0.20)
    recall_strict = recall_at_fpr(y_true, y_score, target_fpr=0.01)

    assert 0.0 <= recall_strict <= recall_tolerant <= 1.0


def test_all_fraud_metrics_contient_les_cles_attendues():
    rng = np.random.default_rng(0)
    y_true = pd.Series([0] * 95 + [1] * 5)
    y_score = rng.random(100)
    metriques = all_fraud_metrics(y_true, y_score)
    assert set(metriques) == {
        "auc_roc", "auc_pr", "recall_at_fpr_1pct", "recall_at_fpr_2pct",
        "precision_at_recall_80", "f1", "lift_1pct", "lift_5pct",
    }


def test_cost_benefit_analysis_cout_nul_a_seuil_optimal_sur_classification_parfaite():
    y_true = pd.Series([0] * 95 + [1] * 5)
    y_score = y_true.astype(float)  # score = label exact : classification parfaite

    cb = cost_benefit_analysis(y_true, y_score, thresholds=np.array([0.5]))
    ligne = cb.iloc[0]

    assert ligne["fp"] == 0
    assert ligne["fn"] == 0
    assert ligne["tp"] == 5
    assert ligne["cost_total"] == 0


def test_cost_benefit_analysis_seuil_plus_haut_reduit_le_nombre_d_alertes():
    rng = np.random.default_rng(1)
    y_true = pd.Series(rng.integers(0, 2, 200))
    y_score = rng.random(200)

    cb = cost_benefit_analysis(y_true, y_score, thresholds=np.array([0.1, 0.9]))
    n_alertes_seuil_bas = cb.loc[cb["threshold"] == 0.1, "n_alerts"].iloc[0]
    n_alertes_seuil_haut = cb.loc[cb["threshold"] == 0.9, "n_alerts"].iloc[0]

    assert n_alertes_seuil_haut <= n_alertes_seuil_bas


def test_find_optimal_threshold_retourne_un_seuil_valide():
    rng = np.random.default_rng(2)
    y_true = pd.Series(rng.integers(0, 2, 300))
    y_score = rng.random(300)

    cb = cost_benefit_analysis(y_true, y_score)
    optimal = find_optimal_threshold(cb)

    assert "threshold" in optimal
    assert 0.0 <= optimal["threshold"] <= 1.0
