"""
Module de génération des comptes clients.
"""
import numpy as np
import pandas as pd
from config import CONFIG

def generate_accounts(n: int) -> pd.DataFrame:
    """Génère des profils de comptes clients de manière synthétique."""
    print(f"Génération de {n:,} comptes...")
    
    # Identifiants de compte uniques
    account_ids = [f'PL{str(i).zfill(8)}' for i in range(n)]
    
    # Dates d'inscription (les comptes plus anciens sont souvent plus fiables)
    reg_start = pd.to_datetime('2018-01-01')
    reg_end = pd.to_datetime('2024-06-01')
    days_range = (reg_end - reg_start).days
    
    # Simulation de la croissance de la base utilisateurs (exponentielle)
    random_days = np.random.exponential(days_range/3, n).astype(int)
    reg_dates = reg_start + pd.to_timedelta(random_days, unit='D')
    
    # CORRECTION CRITIQUE: Conversion en Series pour utiliser .clip()
    # DatetimeIndex n'a pas nécessairement de méthode .clip() fiable selon la version
    reg_dates = pd.Series(reg_dates).clip(upper=reg_end).values
    
    # Attribution des pays selon les poids configurés
    countries = np.random.choice(
        CONFIG['countries'], 
        n, 
        p=CONFIG['country_weights']
    )
    
    # Niveau KYC (Know Your Customer) : 1=Basique, 2=Standard, 3=Complet
    kyc_level = np.random.choice([1, 2, 3], n, p=[0.30, 0.50, 0.20])
    
    # Type de compte
    account_type = np.random.choice(
        ['individual', 'merchant', 'agent'], 
        n, 
        p=[0.75, 0.20, 0.05]
    )
    
    # Nombre d'appareils associés
    device_count = np.random.poisson(1.5, n)
    device_count = np.clip(device_count, 1, 5)
    
    # Indicateurs de risque (Vérité Terrain pour l'entraînement)
    is_risky = np.random.random(n) < 0.02
    is_fraudster = np.random.random(n) < 0.005  # 0.5% d'utilisateurs frauduleux
    is_mule = np.random.random(n) < 0.003       # 0.3% de mules
    
    # Création du DataFrame
    accounts = pd.DataFrame({
        'account_id': account_ids,
        'registration_date': reg_dates,
        'country': countries,
        'kyc_level': kyc_level,
        'account_type': account_type,
        'device_count': device_count,
        'is_risky': is_risky.astype(int),
        'is_fraudster': is_fraudster.astype(int),
        'is_mule': is_mule.astype(int),
    })
    
    # Calcul de l'ancienneté du compte à la fin de la période
    end_date_ts = pd.to_datetime(CONFIG['date_end'])
    accounts['account_age_days'] = (
        end_date_ts - accounts['registration_date']
    ).dt.days
    
    return accounts
