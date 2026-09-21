"""
API de détection de fraude en temps réel — PayLink Africa.
FastAPI, latence cible < 100ms, throughput 50K tx/min.
"""
import sys
import time
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

# === Imports ===
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import FraudDetector

# === Initialisation du détecteur au demarrage ===
detector = None

try:
    # Chemins resolus automatiquement par FraudDetector (racine puis
    # outputs/models/, cf. inference.py) : pas de chemin fige ici, pour
    # rester coherent que les modeles viennent d'un ancien run ou du
    # pipeline actuel.
    detector = FraudDetector()
    logger.info(f"FraudDetector chargé : {detector.model_info['n_features']} features")
    logger.info(f"Modèles : {detector.model_info['models']}")
except Exception as e:
    logger.error(f"Erreur chargement FraudDetector : {e}")


# === Modèles de données ===
class TransactionInput(BaseModel):
    """Transaction à scorer en temps réel."""
    transaction_id: str
    timestamp: str  # ISO 8601
    sender_id: str
    receiver_id: str
    amount: float
    transaction_type: str  # transfer, payment, withdrawal, deposit
    channel: str  # mobile_app, ussd, agent, web, api
    device_id: str = ""
    city: str = ""
    is_new_device: int = 0
    sender_account_age: int = 0
    sender_kyc: int = 1


class FraudScore(BaseModel):
    """Résultat du scoring de fraude."""
    transaction_id: str
    fraud_probability: float
    is_suspicious: bool
    risk_level: str  # low, medium, high, critical
    latency_ms: float
    model_version: str


class BatchInput(BaseModel):
    """Scoring par lot (jusqu'à 1000 transactions)."""
    transactions: list[TransactionInput]


# === App ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fraud Detection API — PayLink Africa")
    yield

app = FastAPI(title="Fraud Detection API — PayLink Africa", version="1.0.0",
              lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if detector is not None and detector.is_loaded else "degrade",
        "model": detector.model_info if detector is not None else None,
        "timestamp": time.time(),
    }


@app.post("/score", response_model=FraudScore)
async def score_transaction(tx: TransactionInput):
    """Score une transaction en temps réel."""
    if detector is None or not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Détecteur non chargé")

    t0 = time.time()

    try:
        # Utiliser predict_full du FraudDetector
        result = detector.predict_full(tx.model_dump())

        latency = (time.time() - t0) * 1000

        # Classification du risque basée sur les seuils du détecteur
        score = result["fraud_probability"]
        if score > 0.6:
            risk = "critical"
        elif score > 0.3:
            risk = "high"
        elif score > 0.1:
            risk = "medium"
        else:
            risk = "low"

        return FraudScore(
            transaction_id=tx.transaction_id,
            fraud_probability=round(score, 6),
            is_suspicious=score >= 0.5,
            risk_level=risk,
            latency_ms=round(latency, 2),
            model_version="Ensemble Stacking v1",
        )
    except Exception as e:
        logger.error(f"Erreur scoring {tx.transaction_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch")
async def score_batch(batch: BatchInput):
    """Score un lot de transactions (max 1000)."""
    if detector is None or not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Détecteur non chargé")

    if len(batch.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 transactions par lot")

    t0 = time.time()
    results = []

    transactions = [tx.model_dump() for tx in batch.transactions]

    for tx_dict in transactions:
        try:
            result = detector.predict_full(tx_dict)
            score = result["fraud_probability"]
            results.append({
                "transaction_id": tx_dict.get("transaction_id", "unknown"),
                "fraud_probability": round(score, 6),
                "is_suspicious": score >= 0.5,
                "risk_level": result["risk_level"],
            })
        except Exception as e:
            logger.error(f"Erreur batch scoring {tx_dict.get('transaction_id', '?')}: {e}")
            results.append({
                "transaction_id": tx_dict.get("transaction_id", "unknown"),
                "error": "Scoring failed",
            })

    latency = (time.time() - t0) * 1000
    return {
        "n_scored": len(results),
        "latency_ms": round(latency, 2),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
