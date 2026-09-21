"""
Configuration globale pour le projet de détection de fraude.
"""
RANDOM_STATE = 42
TARGET = "is_fraud"
DATE_CUTOFF = "2024-07-01"
FRAUD_RATE = 0.0015
LATENCY_TARGET_MS = 100
WINDOWS = ["1h", "6h", "24h", "7d", "30d"]
