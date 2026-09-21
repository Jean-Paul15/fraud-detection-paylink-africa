"""
Configuration globale pour la génération de données de fraude.
"""
from typing import List, Dict, Union

CONFIG: Dict[str, Union[int, float, str, List[str], List[float]]] = {
    'n_accounts': 100000,
    'n_transactions': 2000000,
    'fraud_rate': 0.0015,  # 0.15%
    'date_start': '2023-01-01',
    'date_end': '2024-12-31',
    'countries': ['SEN', 'CIV', 'MLI', 'BFA', 'TGO', 'BEN', 'CMR', 'GAB'],
    'country_weights': [0.25, 0.22, 0.12, 0.10, 0.10, 0.08, 0.08, 0.05],
}
