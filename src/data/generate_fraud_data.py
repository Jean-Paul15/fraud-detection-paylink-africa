#!/usr/bin/env python3
"""
=============================================================================
PAYLINK AFRICA - FRAUD DETECTION DATA GENERATOR
=============================================================================
Point d'entrée principal pour la génération de données.
Usage: python generate_fraud_data.py
=============================================================================
"""
import warnings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG
from accounts import generate_accounts
from transactions import generate_legitimate_transactions, inject_fraud_patterns
from features import calculate_temporal_features, build_graph_data
from utils import save_datasets

# Configuration
warnings.filterwarnings('ignore')

def generate_full_dataset():
    """Orchestre la génération complète du dataset."""
    print("=" * 70)
    print("PAYLINK AFRICA - FRAUD DETECTION DATA GENERATOR")
    print("=" * 70)
    
    # 1. Génération des comptes
    accounts = generate_accounts(CONFIG['n_accounts'])
    
    # 2. Génération des transactions légitimes
    transactions = generate_legitimate_transactions(
        accounts, 
        CONFIG['n_transactions']
    )
    
    # 3. Injection des patterns de fraude
    transactions = inject_fraud_patterns(
        transactions, 
        accounts, 
        CONFIG['fraud_rate']
    )
    
    # 4. Feature Engineering
    transactions = calculate_temporal_features(transactions, accounts)
    
    # 5. Construction du Graphe
    node_features, edges = build_graph_data(transactions, accounts)
    
    return accounts, transactions, node_features, edges

if __name__ == "__main__":
    try:
        # Exécution du pipeline
        res = generate_full_dataset()
        accounts, transactions, node_features, edges = res
        
        # Sauvegarde
        save_datasets(accounts, transactions, node_features, edges)
        
        print("\n" + "=" * 70)
        print("GÉNÉRATION TERMINÉE AVEC SUCCÈS")
        print("=" * 70)
        
        fraud_stats = transactions[
            transactions['is_fraud']==1
        ]['fraud_type'].value_counts().to_dict()
        
        print(f"Total transactions: {len(transactions):,}")
        print(f"Taux de fraude: {transactions['is_fraud'].mean()*100:.3f}%")
        print(f"Types de fraude: {fraud_stats}")
        
    except Exception as e:
        print(f"\nERREUR CRITIQUE: {str(e)}")
        raise
