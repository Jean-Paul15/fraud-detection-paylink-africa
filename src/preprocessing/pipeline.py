"""
Pipeline de preprocessing pour détection de fraude.
Gère le déséquilibre extrême (0.15%) et le split temporel strict.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from ..config import RANDOM_STATE


def build_fraud_preprocessor(num_features, cat_features):
    """Pipeline optimisé pour la détection de fraude.

    Spécificités fraude :
    - RobustScaler : les montants ont des outliers (fraudes = valeurs extrêmes),
      StandardScaler serait trop influencé par ces outliers
    - Imputation median : les valeurs manquantes sont rares mais possibles
      (ex: device_id manquant sur USSD)
    - Pas de winsorization : les valeurs extrêmes SONT le signal (fraudes)
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MANQUANT")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ], remainder="drop")


def build_fraud_pipeline(model, num_features, cat_features):
    """Pipeline complet : preprocessing + modèle.

    Parameters
    ----------
    model : classifieur sklearn-compatible
    num_features : list
    cat_features : list
    """
    preprocessor = build_fraud_preprocessor(num_features, cat_features)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
