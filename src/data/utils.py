"""
Utilitaires pour la sauvegarde et la gestion des fichiers.
"""
import os
import pandas as pd

def save_datasets(
    accounts: pd.DataFrame, 
    transactions: pd.DataFrame, 
    node_features: pd.DataFrame, 
    edges: pd.DataFrame
) -> None:
    """Sauvegarde l'ensemble des datasets générés."""
    output_dir = 'data_fraud'
    
    os.makedirs(f"{output_dir}/raw", exist_ok=True)
    os.makedirs(f"{output_dir}/processed", exist_ok=True)
    os.makedirs(f"{output_dir}/graph", exist_ok=True)
    
    # Validation basique
    if transactions.empty or accounts.empty:
        print("ATTENTION: Datasets vides générés!")
        return
        
    # Save raw data
    print("Sauvegarde des données brutes...")
    accounts.to_csv(f"{output_dir}/raw/accounts.csv", index=False)
    transactions.to_csv(f"{output_dir}/raw/transactions.csv", index=False)
    
    # Save processed (with features)
    # Split temporally (Train/Test split)
    cutoff = '2024-07-01'
    # Conversion forcée pour éviter les erreurs de type string vs datetime
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    train = transactions[transactions['timestamp'] < cutoff]
    test = transactions[transactions['timestamp'] >= cutoff]
    
    train.to_csv(f"{output_dir}/processed/train.csv", index=False)
    test.to_csv(f"{output_dir}/processed/test.csv", index=False)
    
    # Save graph data
    node_features.to_csv(f"{output_dir}/graph/nodes.csv", index=False)
    edges.to_csv(f"{output_dir}/graph/edges.csv", index=False)
    
    print(f"\nFichiers sauvegardés dans {output_dir}/")
    print(f"  - accounts.csv: {len(accounts):,} lignes")
    print(f"  - transactions.csv: {len(transactions):,} lignes")
    print(f"  - train.csv: {len(train):,} lignes ({train['is_fraud'].mean()*100:.3f}% fraude)")
    print(f"  - test.csv: {len(test):,} lignes ({test['is_fraud'].mean()*100:.3f}% fraude)")
    print(f"  - nodes.csv: {len(node_features):,} lignes")
    print(f"  - edges.csv: {len(edges):,} lignes")
