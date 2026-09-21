"""
Module de génération des transactions financières.
"""
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from config import CONFIG

def generate_transaction_time(n: int, start: str, end: str) -> pd.Series:
    """Génère des timestamps de transaction réalistes."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    total_seconds = int((end_dt - start_dt).total_seconds())
    
    # Génération aléatoire uniforme dans le temps
    random_seconds = np.random.randint(0, total_seconds, n)
    timestamps = start_dt + pd.to_timedelta(random_seconds, unit='s')
    
    return timestamps

def generate_legitimate_transactions(accounts: pd.DataFrame, n: int) -> pd.DataFrame:
    """Génère le flux de transactions légitimes."""
    print(f"Génération de {n:,} transactions légitimes...")
    
    account_ids = accounts['account_id'].values
    
    # Expéditeurs (pondérés par niveau d'activité)
    sender_weights = np.random.exponential(1, len(accounts))
    sender_weights /= sender_weights.sum()
    senders = np.random.choice(account_ids, n, p=sender_weights)
    
    # Destinataires (uniforme pour simplifier, mais exclusion auto-envoi)
    receivers = np.random.choice(account_ids, n)
    
    # Correction: Sender != Receiver
    same_mask = senders == receivers
    while same_mask.any():
        receivers[same_mask] = np.random.choice(account_ids, same_mask.sum())
        same_mask = senders == receivers
    
    # Types de transaction
    tx_types = np.random.choice(
        ['transfer', 'payment', 'withdrawal', 'deposit', 'bill_pay'],
        n, p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )
    
    # Montants (Lo-gnormale, en FCFA)
    base_amounts = np.exp(np.random.normal(9, 1.5, n))
    amounts = np.clip(base_amounts, 100, 5000000).astype(int)
    
    # Canaux
    channels = np.random.choice(
        ['mobile_app', 'ussd', 'agent', 'web', 'api'],
        n, p=[0.45, 0.25, 0.15, 0.10, 0.05]
    )
    
    # Timestamps
    timestamps = generate_transaction_time(
        n, CONFIG['date_start'], CONFIG['date_end']
    )
    
    # Device IDs (Pseudo-anonymisés)
    device_ids = [
        hashlib.md5(
            f"device_{senders[i]}_{np.random.randint(1,5)}".encode()
        ).hexdigest()[:12] 
        for i in range(n)
    ]
    
    # Localisation (Villes)
    cities = np.random.choice(
        ['DAK', 'ABJ', 'BKO', 'OUA', 'LOM', 'COT', 'DLA', 'LBV', 'OTHER'],
        n, p=[0.20, 0.18, 0.10, 0.08, 0.08, 0.08, 0.06, 0.04, 0.18]
    )
    
    transactions = pd.DataFrame({
        'transaction_id': [f'TX{str(i).zfill(12)}' for i in range(n)],
        'timestamp': timestamps,
        'sender_id': senders,
        'receiver_id': receivers,
        'amount': amounts,
        'transaction_type': tx_types,
        'channel': channels,
        'device_id': device_ids,
        'city': cities,
        'is_fraud': 0,
        'fraud_type': 'none',
    })
    
    return transactions.sort_values('timestamp').reset_index(drop=True)

def inject_fraud_patterns(
    transactions: pd.DataFrame, 
    accounts: pd.DataFrame, 
    fraud_rate: float
) -> pd.DataFrame:
    """Injecte des patterns de fraude complexes."""
    print(f"Injection de fraudes (taux cible: {fraud_rate*100:.2f}%)...")
    
    n_frauds = int(len(transactions) * fraud_rate)
    fraud_indices = np.random.choice(
        len(transactions), n_frauds, replace=False
    )
    
    # Récupération des IDs compromis
    fraudster_ids = accounts[accounts['is_fraudster'] == 1]['account_id'].values
    mule_ids = accounts[accounts['is_mule'] == 1]['account_id'].values
    
    # Distribution des types de fraude
    fraud_types = np.random.choice(
        ['account_takeover', 'stolen_card', 'mule_network', 'synthetic_id', 'internal'],
        n_frauds, p=[0.35, 0.28, 0.20, 0.12, 0.05]
    )
    
    df = transactions.copy()
    
    # Boucle d'injection
    for i, idx in enumerate(fraud_indices):
        fraud_type = fraud_types[i]
        
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = fraud_type
        
        if fraud_type == 'account_takeover':
            df.at[idx, 'amount'] = int(np.random.uniform(100000, 2000000))
            hour = df.at[idx, 'timestamp'].hour
            if 6 <= hour <= 22:
                # Décalage vers des heures suspectes (nuit)
                df.at[idx, 'timestamp'] += pd.Timedelta(
                    hours=np.random.randint(1, 5)
                )
            # Nouvel appareil inconnu
            df.at[idx, 'device_id'] = hashlib.md5(
                f"new_device_{np.random.randint(10000)}".encode()
            ).hexdigest()[:12]
            
        elif fraud_type == 'stolen_card':
            df.at[idx, 'amount'] = int(np.random.uniform(5000, 50000))
            df.at[idx, 'transaction_type'] = 'payment'
            
        elif fraud_type == 'mule_network':
            if len(mule_ids) > 0:
                df.at[idx, 'receiver_id'] = np.random.choice(mule_ids)
            df.at[idx, 'amount'] = int(np.random.uniform(50000, 500000))
            
        elif fraud_type == 'synthetic_id':
            if len(fraudster_ids) > 0:
                df.at[idx, 'sender_id'] = np.random.choice(fraudster_ids)
            df.at[idx, 'amount'] = int(np.random.uniform(200000, 1000000))
            
        elif fraud_type == 'internal':
            df.at[idx, 'channel'] = 'agent'
            df.at[idx, 'amount'] = int(
                round(np.random.uniform(100000, 500000), -4)
            )
            
    print(f"  {n_frauds:,} transactions frauduleuses injectées.")
    return df
