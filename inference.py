"""
Inférence production — Détection de Fraude PayLink Africa.
==========================================================
Charge les 4 modèles (IF, AE, LGB, Stacking) et fournit une interface
unifiée pour le scoring temps réel et l'explicabilité.

Latence cible : < 100ms par transaction.
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import joblib
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_inference")

# =============================================================================
# Constantes
# =============================================================================

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "outputs" / "models"

# Chemins des fichiers (racine puis outputs/models en fallback)
MODEL_PATHS = {
    "ensemble_stacking": [
        ROOT / "ensemble_stacking.pkl",
        MODELS_DIR / "ensemble_stacking.pkl",
    ],
    "lightgbm": [
        ROOT / "lightgbm.pkl",
        MODELS_DIR / "lightgbm.pkl",
    ],
    "isolation_forest": [
        ROOT / "isolation_forest.pkl",
        MODELS_DIR / "isolation_forest.pkl",
    ],
    "autoencoder": [
        ROOT / "autoencoder.pt",
        MODELS_DIR / "autoencoder.pt",
    ],
    "scaler": [
        ROOT / "scaler.pkl",
        MODELS_DIR / "scaler.pkl",
    ],
    "feature_columns": [
        ROOT / "feature_columns.pkl",
        MODELS_DIR / "feature_columns.pkl",
    ],
    "gnn_account_scores": [
        ROOT / "gnn_account_scores.pkl",
        MODELS_DIR / "gnn_account_scores.pkl",
    ],
}

# Seuils de risque
RISK_THRESHOLDS = {
    "low": 0.1,
    "medium": 0.3,
    "high": 0.6,
    "critical": 1.0,
}

# Features dont on peut estimer une valeur par défaut
FEATURE_DEFAULTS = {
    "sender_account_age_log": 4.0,       # ~55 jours
    "sender_tx_count_cumulative": 0.0,
    "sender_tx_count_1h": 0.0,
    "sender_tx_count_6h": 0.0,
    "sender_tx_count_24h": 0.0,
    "sender_amount_sum_1h": 0.0,
    "sender_amount_sum_6h": 0.0,
    "sender_amount_sum_24h": 0.0,
    "sender_amount_std_6h": 0.0,
    "sender_unique_receivers": 0.0,
    "receiver_unique_senders": 0.0,
    "time_since_last_tx_sec": 86400.0,   # 24h par défaut
    "device_changed": 0.0,
    "city_changed": 0.0,
    "receiver_is_new": 0.0,
}


# =============================================================================
# Autoencoder helper (évite l'import complet de src.models)
# =============================================================================

import torch.nn as nn


class FraudAutoencoderInference(nn.Module):
    """Autoencodeur léger pour inférence (sans dépendance src.models)."""

    def __init__(self, input_dim, hidden_dims=None, latent_dim=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        if latent_dim is None:
            latent_dim = max(input_dim // 8, 4)

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            reconstructed = self.forward(x_tensor)
            return torch.mean((x_tensor - reconstructed) ** 2, dim=1).numpy()


# =============================================================================
# Inference principale
# =============================================================================

class FraudDetector:
    """
    Détecteur de fraude en production.

    Charge les 4 modèles (IF, AE, LGB, Stacking) et fournit :
    - predict(transaction) -> score de fraude
    - predict_batch(list) -> liste de scores
    - explain(transaction) -> features les plus importantes
    - predict_full(transaction) -> rapport complet
    """

    def __init__(
        self,
        ensemble_path: Optional[str] = None,
        lightgbm_path: Optional[str] = None,
        isolation_forest_path: Optional[str] = None,
        autoencoder_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        feature_columns_path: Optional[str] = None,
        gnn_account_scores_path: Optional[str] = None,
    ):
        self.ensemble_model = None
        self.lgb_model = None
        self.if_model = None
        self.ae_model = None
        self.scaler = None
        self.feature_columns = []
        self.gnn_account_scores = {}
        self.input_dim = 0
        self._loaded = False

        # Résoudre les chemins
        paths = {
            "ensemble_stacking": ensemble_path,
            "lightgbm": lightgbm_path,
            "isolation_forest": isolation_forest_path,
            "autoencoder": autoencoder_path,
            "scaler": scaler_path,
            "feature_columns": feature_columns_path,
            "gnn_account_scores": gnn_account_scores_path,
        }

        for key, custom_path in paths.items():
            if custom_path is not None:
                MODEL_PATHS[key] = [Path(custom_path)]

        self._load_models()

    def _resolve_path(self, key: str) -> Optional[Path]:
        """Trouve le premier chemin existant pour une clé donnée."""
        for p in MODEL_PATHS.get(key, []):
            if p.exists():
                return p
        return None

    def _load_models(self):
        """Charge tous les modèles et assets."""
        try:
            # 1. Feature columns
            feat_path = self._resolve_path("feature_columns")
            if feat_path is None:
                raise FileNotFoundError("feature_columns.pkl introuvable")

            feats = joblib.load(feat_path)
            if isinstance(feats, dict):
                self.feature_columns = feats.get("features", feats.get("all", []))
            elif isinstance(feats, list):
                self.feature_columns = feats
            elif isinstance(feats, (np.ndarray,)):
                self.feature_columns = list(feats)
            self.input_dim = len(self.feature_columns)
            logger.info(f"Features chargées : {self.input_dim}")

            # 2. Scaler
            scaler_path = self._resolve_path("scaler")
            if scaler_path:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler chargé : {type(self.scaler).__name__}")
            else:
                logger.warning("Scaler non trouve -- les features seront utilisees brutes")

            # 3. Isolation Forest
            if_path = self._resolve_path("isolation_forest")
            if if_path:
                self.if_model = joblib.load(if_path)
                logger.info(f"Isolation Forest chargé : {type(self.if_model).__name__}")

            # 4. Autoencoder
            ae_path = self._resolve_path("autoencoder")
            if ae_path:
                self.ae_model = FraudAutoencoderInference(self.input_dim)
                state_dict = torch.load(ae_path, map_location="cpu", weights_only=True)
                self.ae_model.load_state_dict(state_dict)
                self.ae_model.eval()
                logger.info("Autoencodeur chargé")

            # 5. LightGBM
            lgb_path = self._resolve_path("lightgbm")
            if lgb_path:
                self.lgb_model = joblib.load(lgb_path)
                lgb_type = type(self.lgb_model).__name__
                logger.info(f"LightGBM chargé : {lgb_type}")

            # 6. Ensemble Stacking (méta-modèle)
            stack_path = self._resolve_path("ensemble_stacking")
            if stack_path:
                self.ensemble_model = joblib.load(stack_path)
                logger.info(f"Stacking chargé : {type(self.ensemble_model).__name__}")

            # 7. Scores GNN par compte (le stacking attend 4 colonnes : IF/AE/LGB/GNN,
            # cf. master_pipeline.py section 11 -- meme quand le GNN etait indisponible
            # à l'entraînement, la colonne existe avec des zeros). Un compte absent du
            # graphe d'entraînement (nouveau compte) recoit 0.0, comme à l'entraînement.
            gnn_path = self._resolve_path("gnn_account_scores")
            if gnn_path:
                self.gnn_account_scores = joblib.load(gnn_path)
                logger.info(f"Scores GNN chargés : {len(self.gnn_account_scores)} comptes")

            self._loaded = True
            logger.info("Tous les modèles chargés avec succès")

        except Exception as e:
            logger.error(f"Erreur chargement modèles : {e}")
            raise

    # =====================================================================
    # Preprocessing
    # =====================================================================

    def _prepare_features(self, transaction: dict) -> np.ndarray:
        """
        Prépare un vecteur de features à partir d'un dictionnaire de transaction.

        Pour les features manquantes (comportementales, graphe), utilise des
        valeurs par défaut plausibles (0 ou médianes). En production, ces
        features doivent être fournies par un Feature Store (Redis, etc.).
        """
        feature_vector = {}

        for col in self.feature_columns:
            if col in transaction:
                feature_vector[col] = float(transaction[col])
            elif col in FEATURE_DEFAULTS:
                feature_vector[col] = FEATURE_DEFAULTS[col]
            else:
                # Feature one-hot ou graphe : 0 par défaut
                feature_vector[col] = 0.0

        # Ordre fixe selon feature_columns
        arr = np.array([[feature_vector.get(c, 0.0) for c in self.feature_columns]],
                       dtype=np.float32)
        return arr

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Applique le scaler si disponible."""
        if self.scaler is not None:
            try:
                return self.scaler.transform(X)
            except Exception:
                logger.warning("Erreur scaling -- utilisation des features brutes")
        return X

    def _get_risk_level(self, score: float) -> str:
        """Classe le risque selon les seuils."""
        for level, threshold in RISK_THRESHOLDS.items():
            if score < threshold:
                return level
        return "critical"

    # =====================================================================
    # Prediction
    # =====================================================================

    def predict(self, transaction: dict) -> float:
        """
        Score une transaction unique.

        Parameters
        ----------
        transaction : dict
            Transaction avec au minimum les colonnes de base.
            Les features manquantes sont complétées automatiquement.

        Returns
        -------
        float : Score de fraude [0, 1]
        """
        if not self._loaded:
            raise RuntimeError("Modèles non chargés")

        X_raw = self._prepare_features(transaction)
        X = self._preprocess(X_raw)

        # Ensemble Stacking : combine les scores des modèles de base (IF, AE,
        # LGB, et GNN si le méta-modèle chargé a été entraîné avec)
        if self.ensemble_model is not None and hasattr(self.ensemble_model, 'predict_proba'):
            scores = []

            if self.if_model is not None:
                if_score_raw = self.if_model.decision_function(X)[0]
                if_score = 1.0 - (if_score_raw - (-0.5)) / (0.5 - (-0.5) + 1e-9)
                if_score = np.clip(if_score, 0, 1)
                scores.append(if_score)
            else:
                scores.append(0.5)

            if self.ae_model is not None:
                errors = self.ae_model.reconstruction_error(X)
                ae_score = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
                scores.append(float(ae_score[0]))
            else:
                scores.append(0.5)

            if self.lgb_model is not None:
                if hasattr(self.lgb_model, 'predict_proba'):
                    lgb_score = float(self.lgb_model.predict_proba(X)[0, 1])
                    scores.append(lgb_score)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)

            # Le méta-modèle détermine lui-même s'il attend une 4e colonne GNN
            # (n_features_in_ vaut 3 pour un stacking entraîné avant l'ajout du
            # GNN au pipeline, 4 sinon) -- évite un plantage si un ancien
            # ensemble_stacking.pkl a 3 colonnes est chargé.
            n_features_attendues = getattr(self.ensemble_model, "n_features_in_", len(scores))
            if n_features_attendues > len(scores):
                sender_id = transaction.get("sender_id")
                scores.append(self.gnn_account_scores.get(sender_id, 0.0))

            meta_X = np.column_stack(scores)
            proba = float(self.ensemble_model.predict_proba(meta_X)[0, 1])
            return proba

        # Fallback : LightGBM seul
        if self.lgb_model is not None and hasattr(self.lgb_model, 'predict_proba'):
            return float(self.lgb_model.predict_proba(X)[0, 1])

        # Fallback ultime
        logger.warning("Aucun modèle disponible pour la prédiction")
        return 0.0

    def predict_batch(self, transactions: list) -> list:
        """
        Score un lot de transactions.

        Parameters
        ----------
        transactions : list[dict]
            Liste de transactions (max 10000 recommandees).

        Returns
        -------
        list[float] : Scores de fraude
        """
        return [self.predict(tx) for tx in transactions]

    def predict_full(self, transaction: dict) -> dict:
        """
        Rapport complet : score + risque + explication.

        Returns
        -------
        dict avec fraud_probability, risk_level, is_suspicious,
             top_features, model_version
        """
        score = self.predict(transaction)
        return {
            "transaction_id": transaction.get("transaction_id", "unknown"),
            "fraud_probability": round(score, 6),
            "risk_level": self._get_risk_level(score),
            "is_suspicious": score >= 0.5,
            "model_version": "1.0.0",
            "top_features": self.explain(transaction),
            "timestamp": datetime.now().isoformat(),
        }

    def predict_with_bandit(self, transaction_dict, bandit=None):
        """Score une transaction avec seuil adaptatif via bandit contextuel.

        Combine le score de fraude du modèle ensembliste avec un seuil
        dynamique selectionne par Thompson Sampling.

        Parameters
        ----------
        transaction_dict : dict
            Transaction a evaluer.
        bandit : ThompsonSamplingBandit, optional
            Instance du bandit contextuel. Si None, utilise un seuil fixe de 0.5.

        Returns
        -------
        dict
            Rapport complet avec 'threshold_used' et 'is_suspicious_bandit'.
        """
        result = self.predict_full(transaction_dict)
        score = result["fraud_probability"]

        if bandit is not None:
            threshold, _ = bandit.select_threshold()
            result["threshold_used"] = threshold
        else:
            threshold = 0.5
            result["threshold_used"] = threshold

        result["is_suspicious_bandit"] = score >= threshold
        return result

    # =====================================================================
    # Explicabilite
    # =====================================================================

    def explain(self, transaction: dict, top_n: int = 10) -> list:
        """
        Explique les features les plus importantes pour une transaction.

        Utilise les feature importances du LightGBM ponderees par les
        valeurs normalisees de la transaction.

        Parameters
        ----------
        transaction : dict
        top_n : int

        Returns
        -------
        list[dict] : Top features avec leur contribution estimée
        """
        score = self.predict(transaction)
        X = self._preprocess(self._prepare_features(transaction))[0]

        # Utiliser les feature importances LGB si disponibles
        if self.lgb_model is not None and hasattr(self.lgb_model, 'feature_importances_'):
            importances = self.lgb_model.feature_importances_
            # Contribution approximee : importance * valeur normalisee
            contributions = importances * np.abs(X)

            top_idx = np.argsort(contributions)[::-1][:top_n]
            result = []
            for idx in top_idx:
                result.append({
                    "feature": self.feature_columns[idx],
                    "value": round(float(X[idx]), 4),
                    "importance": round(float(importances[idx]), 6),
                    "contribution": round(float(contributions[idx]), 6),
                })
            return result

        # Fallback : features avec les plus grandes valeurs absolues
        top_idx = np.argsort(np.abs(X))[::-1][:top_n]
        return [
            {
                "feature": self.feature_columns[idx],
                "value": round(float(X[idx]), 4),
                "importance": 0.0,
                "contribution": round(float(np.abs(X[idx])), 6),
            }
            for idx in top_idx
        ]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_info(self) -> dict:
        return {
            "n_features": self.input_dim,
            "feature_columns": self.feature_columns,
            "models": {
                "ensemble": type(self.ensemble_model).__name__ if self.ensemble_model else None,
                "lightgbm": type(self.lgb_model).__name__ if self.lgb_model else None,
                "isolation_forest": type(self.if_model).__name__ if self.if_model else None,
                "autoencoder": "FraudAutoencoder" if self.ae_model else None,
                "gnn_accounts_connus": len(self.gnn_account_scores),
            },
        }


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST INFERENCE — PayLink Africa Fraud Detector")
    print("=" * 60)

    # Initialiser le détecteur
    detector = FraudDetector()
    print(f"\nModèles chargés : {detector.is_loaded}")
    print(f"Features : {detector.model_info['n_features']}")
    print(f"Modèles : {json.dumps(detector.model_info['models'], indent=2)}")

    # Transactions de test (fraude vs légitime)
    test_transactions = [
        {
            "transaction_id": "TX-TEST-001",
            "timestamp": "2024-01-15T14:30:00",
            "amount": 150000,
            "amount_log": 11.918,
            "hour": 14,
            "hour_sin": -0.5,
            "hour_cos": -0.866,
            "day_of_month": 15,
            "day_of_week": 0,
            "is_night": 0,
            "is_weekend": 0,
            "sender_id": "ACC-TEST-001",
            "receiver_id": "ACC-TEST-002",
            "transaction_type": "transfer",
            "channel": "mobile_app",
            "device_id": "DEV-001",
            "is_new_device": 0,
            "sender_account_age": 180,
            "sender_account_age_log": 5.198,
            "sender_kyc": 2,
            "sender_tx_count_cumulative": 45,
            "sender_tx_count_1h": 2,
            "sender_tx_count_6h": 8,
            "sender_tx_count_24h": 15,
            "sender_amount_sum_1h": 250000,
            "sender_amount_sum_6h": 1200000,
            "sender_amount_sum_24h": 3500000,
            "sender_amount_std_6h": 50000,
            "receiver_is_new": 0,
            "time_since_last_tx_sec": 3600,
            "device_changed": 0,
            "city_changed": 0,
            "tx_type_transfer": 1,
            "tx_type_payment": 0,
            "tx_type_withdrawal": 0,
            "tx_type_deposit": 0,
            "tx_type_bill_pay": 0,
            "channel_mobile_app": 1,
            "channel_agent": 0,
            "channel_api": 0,
            "channel_ussd": 0,
            "channel_web": 0,
            "city_DAK": 1,
            "city_DLA": 0,
            "city_ABJ": 0,
            "city_LOM": 0,
            "city_OUA": 0,
            "city_BKO": 0,
            "city_LBV": 0,
            "city_COT": 0,
            "city_OTHER": 0,
            "kyc_1": 0,
            "kyc_2": 1,
            "kyc_3": 0,
            "sender_g_total_sent": 1500000,
            "sender_g_total_received": 500000,
            "receiver_g_total_sent": 800000,
            "receiver_g_total_received": 2000000,
            "sender_g_in_degree": 25,
            "receiver_g_in_degree": 32,
            "sender_g_out_degree": 15,
            "receiver_g_out_degree": 18,
            "sender_g_flow_ratio": 0.33,
            "receiver_g_flow_ratio": 2.5,
            "sender_g_is_fraud_account": 0,
            "receiver_g_is_fraud_account": 0,
        },
        {
            "transaction_id": "TX-TEST-002",
            "timestamp": "2024-01-15T03:15:00",
            "amount": 480000,
            "amount_log": 13.081,
            "hour": 3,
            "hour_sin": 0.707,
            "hour_cos": -0.707,
            "day_of_month": 15,
            "day_of_week": 0,
            "is_night": 1,
            "is_weekend": 0,
            "sender_id": "ACC-TEST-003",
            "receiver_id": "ACC-TEST-004",
            "sender_account_age": 5,
            "sender_account_age_log": 1.792,
            "sender_kyc": 1,
            "sender_tx_count_cumulative": 2,
            "sender_tx_count_1h": 1,
            "sender_tx_count_6h": 1,
            "sender_tx_count_24h": 2,
            "sender_amount_sum_1h": 480000,
            "sender_amount_sum_6h": 480000,
            "sender_amount_sum_24h": 500000,
            "sender_amount_std_6h": 0,
            "receiver_is_new": 1,
            "time_since_last_tx_sec": 1800,
            "device_changed": 1,
            "city_changed": 1,
        },
    ]

    print("\n" + "-" * 60)
    for i, tx in enumerate(test_transactions):
        print(f"\nTransaction {i + 1} : {tx['transaction_id']}")
        result = detector.predict_full(tx)
        print(f"  Score de fraude    : {result['fraud_probability']:.6f}")
        print(f"  Niveau de risque   : {result['risk_level']}")
        print(f"  Suspecte           : {result['is_suspicious']}")
        print(f"  Top 5 features     :")
        for feat in result["top_features"][:5]:
            print(f"    {feat['feature']:35s} | val={feat['value']:8.4f} | imp={feat['importance']:.6f} | contrib={feat['contribution']:.6f}")

    # Test batch
    print(f"\n{'=' * 60}")
    print("Test batch (2 transactions) :")
    scores = detector.predict_batch(test_transactions)
    for i, s in enumerate(scores):
        print(f"  {test_transactions[i]['transaction_id']}: {s:.6f}")

    print(f"\n{'=' * 60}")
    print("TESTS TERMINES AVEC SUCCES")
    print(f"{'=' * 60}")
