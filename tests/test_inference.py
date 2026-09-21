"""Teste FraudDetector avec des modeles factices legers (pas les vrais
artefacts de 287 Mo) -- couvre en particulier la compatibilite entre un
ensemble_stacking entraine sur 3 scores (sans GNN) et sur 4 (avec GNN),
le bug de fond corrige dans ce module (cf. master_pipeline.py section 11)."""

import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from inference import FraudAutoencoderInference, FraudDetector

FEATURES = [
    "amount", "sender_account_age_log", "sender_tx_count_cumulative",
    "sender_tx_count_1h", "sender_tx_count_6h", "sender_tx_count_24h",
    "sender_amount_sum_1h", "sender_amount_sum_6h",
]  # au moins 8 : FraudAutoencoderInference divise input_dim par 4 en interne

TRANSACTION = {
    "transaction_id": "TX-TEST",
    "sender_id": "ACC-CONNU",
    "amount": 50000,
    "sender_account_age_log": 4.0,
    "sender_tx_count_cumulative": 10,
    "sender_tx_count_1h": 1,
    "sender_tx_count_6h": 3,
    "sender_tx_count_24h": 5,
    "sender_amount_sum_1h": 50000,
    "sender_amount_sum_6h": 150000,
}


def _donnees_entrainement_factices(rng, n=200):
    X = rng.random((n, len(FEATURES)))
    y = rng.integers(0, 2, n)
    return X, y


def _construire_detecteur(tmp_path, rng, avec_gnn: bool):
    X, y = _donnees_entrainement_factices(rng)

    scaler_path = tmp_path / "scaler.pkl"
    joblib.dump(None, scaler_path)  # pas de scaling pour ce test (chemin quand meme fourni)

    feat_path = tmp_path / "feature_columns.pkl"
    joblib.dump(FEATURES, feat_path)

    if_model = IsolationForest(n_estimators=10, random_state=0).fit(X)
    if_path = tmp_path / "isolation_forest.pkl"
    joblib.dump(if_model, if_path)

    ae = FraudAutoencoderInference(input_dim=len(FEATURES))
    ae_path = tmp_path / "autoencoder.pt"
    torch.save(ae.state_dict(), ae_path)

    lgb_model = LogisticRegression().fit(X, y)
    lgb_path = tmp_path / "lightgbm.pkl"
    joblib.dump(lgb_model, lgb_path)

    if_scores = if_model.decision_function(X)
    ae_scores = ae.reconstruction_error(X.astype(np.float32))
    lgb_scores = lgb_model.predict_proba(X)[:, 1]

    if avec_gnn:
        gnn_scores = rng.random(len(X))
        stack_X = np.column_stack([if_scores, ae_scores, lgb_scores, gnn_scores])
        gnn_account_scores = {"ACC-CONNU": 0.9}
    else:
        stack_X = np.column_stack([if_scores, ae_scores, lgb_scores])
        gnn_account_scores = {}

    gnn_path = tmp_path / "gnn_account_scores.pkl"
    joblib.dump(gnn_account_scores, gnn_path)

    stack_model = LogisticRegression().fit(stack_X, y)
    stack_path = tmp_path / "ensemble_stacking.pkl"
    joblib.dump(stack_model, stack_path)

    return FraudDetector(
        ensemble_path=stack_path,
        lightgbm_path=lgb_path,
        isolation_forest_path=if_path,
        autoencoder_path=ae_path,
        # Chemin fourni explicitement (contenant None serialise) pour eviter
        # que le detecteur, sans chemin custom, ne retombe sur le fallback
        # par defaut et charge un vrai scaler.pkl trouve sur le poste de dev.
        scaler_path=scaler_path,
        feature_columns_path=feat_path,
        gnn_account_scores_path=gnn_path,
    )


def test_predict_fonctionne_avec_un_stacking_entraine_sans_gnn(tmp_path):
    rng = np.random.default_rng(0)
    detecteur = _construire_detecteur(tmp_path, rng, avec_gnn=False)

    assert detecteur.ensemble_model.n_features_in_ == 3

    score = detecteur.predict(TRANSACTION)
    assert 0.0 <= score <= 1.0


def test_predict_fonctionne_avec_un_stacking_entraine_avec_gnn(tmp_path):
    """Regression : avant correction, inference.py ne fournissait toujours
    que 3 scores et plantait des qu'un stacking a 4 features etait charge."""
    rng = np.random.default_rng(1)
    detecteur = _construire_detecteur(tmp_path, rng, avec_gnn=True)

    assert detecteur.ensemble_model.n_features_in_ == 4

    score = detecteur.predict(TRANSACTION)
    assert 0.0 <= score <= 1.0


def test_score_gnn_utilise_zero_pour_un_compte_inconnu(tmp_path):
    rng = np.random.default_rng(2)
    detecteur = _construire_detecteur(tmp_path, rng, avec_gnn=True)

    transaction_compte_inconnu = {**TRANSACTION, "sender_id": "ACC-JAMAIS-VU"}

    # Les deux appels ne doivent pas lever d'exception (4e colonne toujours
    # fournie, avec 0.0 en secours pour un compte absent du graphe).
    score_connu = detecteur.predict(TRANSACTION)
    score_inconnu = detecteur.predict(transaction_compte_inconnu)
    assert 0.0 <= score_connu <= 1.0
    assert 0.0 <= score_inconnu <= 1.0
