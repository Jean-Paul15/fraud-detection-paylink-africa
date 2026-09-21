"""
Module d'ingénierie des fonctionnalités (Feature Engineering) et création de graphe.
"""
import numpy as np
import pandas as pd
from typing import Tuple

def calculate_temporal_features(transactions: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features temporelles et agrégations."""
    print("Calcul des features temporelles...")
    
    df = transactions.copy()
    df = df.sort_values(['sender_id', 'timestamp'])
    
    # Features basées sur le temps
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # Features d'argent
    df['amount_log'] = np.log1p(df['amount'])
    
    # Age du compte au moment de la transaction
    acc_reg = accounts.set_index('account_id')['registration_date']
    df['sender_account_age'] = (
        df['timestamp'] - df['sender_id'].map(acc_reg)
    ).dt.days
    df['sender_account_age'] = df['sender_account_age'].clip(lower=0)
    
    # Niveau KYC
    acc_kyc = accounts.set_index('account_id')['kyc_level']
    df['sender_kyc'] = df['sender_id'].map(acc_kyc)
    
    print("  Calcul des agrégations glissantes...")
    
    # Compte cumulatif des transactions par expéditeur
    sender_tx_count = df.groupby('sender_id').cumcount()
    df['sender_tx_count_cumulative'] = sender_tx_count
    
    # Diversité des destinataires (nouveau destinataire ?)
    df['receiver_is_new'] = (
        ~df.duplicated(subset=['sender_id', 'receiver_id'], keep='first')
    ).astype(int)
    
    return df

def build_graph_data(transactions: pd.DataFrame, accounts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prépare les données pour une structure de graphe (GNN)."""
    print("Construction des données de graphe...")
    
    # Agrégation des transactions pour former des arêtes
    edges = transactions.groupby(['sender_id', 'receiver_id']).agg({
        'amount': ['sum', 'mean', 'count'],
        'is_fraud': 'max',
        'timestamp': ['min', 'max']
    }).reset_index()
    
    edges.columns = [
        'source', 'target', 'total_amount', 'avg_amount', 'tx_count', 
        'has_fraud', 'first_tx', 'last_tx'
    ]
    
    # Features des noeuds (comptes)
    node_features = accounts[[
        'account_id', 'kyc_level', 'account_type', 'device_count', 
        'is_risky', 'account_age_days'
    ]].copy()
    
    # Ajout des features basées sur les transactions
    out_degree = transactions.groupby('sender_id').size().reset_index(name='out_degree')
    in_degree = transactions.groupby('receiver_id').size().reset_index(name='in_degree')
    
    out_amount = transactions.groupby('sender_id')['amount'].sum().reset_index(name='total_sent')
    in_amount = transactions.groupby('receiver_id')['amount'].sum().reset_index(name='total_received')
    
    node_features = node_features.merge(out_degree, left_on='account_id', right_on='sender_id', how='left')
    node_features = node_features.merge(in_degree, left_on='account_id', right_on='receiver_id', how='left')
    node_features = node_features.merge(out_amount, left_on='account_id', right_on='sender_id', how='left')
    node_features = node_features.merge(in_amount, left_on='account_id', right_on='receiver_id', how='left')
    
    # Nettoyage
    node_features = node_features.drop(
        columns=['sender_id_x', 'sender_id_y', 'receiver_id_x', 'receiver_id_y'], 
        errors='ignore'
    )
    node_features = node_features.fillna(0)
    
    # Ratio de flux (sortant/entrant)
    node_features['flow_ratio'] = (
        (node_features['total_sent'] + 1) / (node_features['total_received'] + 1)
    )
    
    # Label de fraude (Vrai Négatif / Vrai Positif pour les noeuds)
    node_features['is_fraud_account'] = accounts.set_index('account_id').loc[
        node_features['account_id'], 'is_fraudster'
    ].values
    
    return node_features, edges
