import numpy as np
import pytest

from src.models.isolation_forest import anomaly_score_to_proba


def test_anomaly_score_to_proba_inverse_l_ordre():
    # Isolation Forest : score bas = anomalie probable -> proba de fraude haute
    scores = np.array([-0.5, 0.0, 0.5])
    probas = anomaly_score_to_proba(scores)
    assert probas[0] > probas[1] > probas[2]


def test_anomaly_score_to_proba_reste_dans_0_1():
    scores = np.array([-0.5, -0.2, 0.1, 0.3, 0.5])
    probas = anomaly_score_to_proba(scores)
    assert (probas >= 0).all() and (probas <= 1).all()


def test_anomaly_score_to_proba_score_minimal_donne_proba_maximale():
    scores = np.array([-0.9, 0.0, 0.9])
    probas = anomaly_score_to_proba(scores)
    # Epsilon de stabilite numerique dans la formule (cf. anomaly_score_to_proba) :
    # les bornes sont approchees, pas atteintes exactement.
    assert probas[0] == pytest.approx(1.0, abs=1e-6)
    assert probas[2] == pytest.approx(0.0, abs=1e-6)
