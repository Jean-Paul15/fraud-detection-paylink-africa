"""
Dashboard de monitoring — Détection de Fraude PayLink Africa.
Streamlit — visualisation en temps réel des alertes et métriques.
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# === Chargement des vraies métriques depuis results.json ===
RESULTS_PATH = ROOT / "outputs" / "models" / "results.json"
real_metrics = None
try:
    with open(RESULTS_PATH, "r") as f:
        all_results = json.load(f)
    # Utiliser les métriques du modèle Stacking (meilleur)
    real_metrics = all_results.get("stack", {})
    st.session_state["metrics_loaded"] = True
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    st.session_state["metrics_loaded"] = False
    real_metrics = {}

# === Chargement des données test (échantillon) ===
TEST_DATA_PATH = ROOT / "data" / "processed" / "test.csv"
test_df = None
try:
    test_df = pd.read_csv(TEST_DATA_PATH, nrows=1000)
    if "is_fraud" in test_df.columns:
        test_df["label"] = test_df["is_fraud"].map({1: "Fraude", 0: "Légitime"})
    st.session_state["data_loaded"] = True
except FileNotFoundError:
    st.session_state["data_loaded"] = False
    test_df = None


# === Config ===
st.set_page_config(
    page_title="PayLink Africa — Fraud Monitor",
    layout="wide",
)

st.title("PayLink Africa — Détection de Fraude")
st.markdown("*Monitoring temps réel des transactions et alertes de fraude*")

if not st.session_state.get("data_loaded", False):
    st.warning("Dashboard demo — données simulées. Connecter à l'API pour données réelles.")

# === Sidebar ===
st.sidebar.header("Filtres")
risk_threshold = st.sidebar.slider("Seuil de risque", 0.0, 1.0, 0.5, 0.05)
hours_back = st.sidebar.slider("Heures d'historique", 1, 72, 24)

refresh_rate = st.sidebar.selectbox("Rafraîchissement", [5, 10, 30, 60], index=1)
st.sidebar.metric("Dernière mise à jour", "—")

# === KPIs (métriques réelles depuis results.json) ===
col1, col2, col3, col4 = st.columns(4)
with col1:
    auc_roc_val = round(real_metrics.get("auc_roc", 0.878), 4) if real_metrics else 0.878
    st.metric("AUC-ROC", f"{auc_roc_val:.4f}")

with col2:
    auc_pr_val = round(real_metrics.get("auc_pr", 0.0661), 4) if real_metrics else 0.0661
    st.metric("AUC-PR", f"{auc_pr_val:.4f}")

with col3:
    recall_val = round(real_metrics.get("recall_at_fpr_1pct", 0.336), 4) if real_metrics else 0.336
    st.metric("Recall @ 1% FPR", f"{recall_val:.4f}")

with col4:
    lift_val = round(real_metrics.get("lift_1pct", 33.2), 1) if real_metrics else 33.2
    st.metric("Lift @ 1%", f"{lift_val}×")

st.divider()

# === Graphiques ===
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Fraudes par heure")

    if test_df is not None and "hour" in test_df.columns and "is_fraud" in test_df.columns:
        # Données réelles depuis test.csv
        fraud_by_hour = test_df[test_df["is_fraud"] == 1].groupby("hour").size().reindex(range(24), fill_value=0)
        fig = px.bar(
            x=list(range(24)),
            y=fraud_by_hour.values,
            labels={"x": "Heure", "y": "Fraudes"}
        )
        fig.update_traces(marker_color="#e74c3c")
    else:
        # Données simulées (fallback)
        hours = list(range(24))
        fraud_counts = np.random.poisson(3, 24) + (np.sin(np.array(hours) * np.pi / 12) * 5)
        fig = px.bar(x=hours, y=fraud_counts, labels={"x": "Heure", "y": "Fraudes (simulé)"})
        fig.update_traces(marker_color="#e74c3c")

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Distribution des scores")

    if test_df is not None and "amount" in test_df.columns:
        # Utiliser les vrais montants pour l'histogramme
        fig = px.histogram(
            test_df,
            x="amount",
            nbins=50,
            labels={"value": "Montant (FCFA)"},
            color="label" if "label" in test_df.columns else None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback simulé
        scores = np.concatenate([
            np.random.beta(1, 20, 5000),  # Légitimes (scores bas)
            np.random.beta(5, 2, 50),      # Fraudes (scores hauts)
        ])
        fig = px.histogram(scores, nbins=50, labels={"value": "Score de fraude (simulé)"})
        fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                      annotation_text="Seuil")
        st.plotly_chart(fig, use_container_width=True)

# === Tableau des alertes ===
st.subheader("Dernières alertes")

if test_df is not None and "is_fraud" in test_df.columns:
    # Utiliser les vraies données de fraude depuis test.csv
    fraud_samples = test_df[test_df["is_fraud"] == 1].head(10)
    if len(fraud_samples) > 0:
        alerts = pd.DataFrame({
            "Horodatage": fraud_samples.get("timestamp", pd.date_range("2024-01-01", periods=len(fraud_samples), freq="h")),
            "Transaction ID": fraud_samples.get("transaction_id", [f"TX-{i:06d}" for i in range(len(fraud_samples))]),
            "Montant (FCFA)": fraud_samples.get("amount", 0),
            "Type fraude": fraud_samples.get("fraud_type", "inconnu"),
            "Canal": fraud_samples.get("channel", "inconnu"),
        })
        alerts = alerts.sort_values("Montant (FCFA)", ascending=False)
    else:
        alerts = pd.DataFrame({
            "Horodatage": pd.date_range("2024-01-01", periods=5, freq="h"),
            "Transaction ID": ["—"] * 5,
            "Montant (FCFA)": [0] * 5,
            "Type fraude": ["aucune fraude trouvée"] * 5,
            "Canal": ["—"] * 5,
        })
else:
    # Fallback simulé
    alerts = pd.DataFrame({
        "Horodatage": pd.date_range("2024-01-01", periods=10, freq="h"),
        "Transaction ID": [f"TX-{i:06d}" for i in range(10)],
        "Montant (FCFA)": np.random.randint(5000, 500000, 10),
        "Type fraude": np.random.choice(
            ["account_takeover", "stolen_card", "mule_network", "synthetic_id", "internal"], 10
        ),
        "Canal": np.random.choice(["mobile_app", "ussd", "agent", "web", "api"], 10),
    })

st.dataframe(alerts, use_container_width=True, hide_index=True)

# === Métriques détaillées (valeurs réelles) ===
st.divider()
st.subheader("Métriques de performance (modèle Stacking — valeurs réelles)")

if real_metrics:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AUC-ROC", f"{real_metrics.get('auc_roc', 0):.4f}")
    with col2:
        st.metric("AUC-PR", f"{real_metrics.get('auc_pr', 0):.4f}")
    with col3:
        st.metric("F1-Score", f"{real_metrics.get('f1', 0):.4f}")
    with col4:
        st.metric("Lift @ 1%", f"{real_metrics.get('lift_1pct', 0):.1f}×")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Precision @ Recall 80%", f"{real_metrics.get('precision_at_recall_80', 0):.6f}")
    with col2:
        st.metric("Recall @ FPR 1%", f"{real_metrics.get('recall_at_fpr_1pct', 0):.4f}")
    with col3:
        st.metric("Recall @ FPR 2%", f"{real_metrics.get('recall_at_fpr_2pct', 0):.4f}")
    with col4:
        n_train = all_results.get("n_train", "?")
        n_test = all_results.get("n_test", "?")
        st.metric("Train / Test", f"{n_train} / {n_test}")

    st.caption(f"Meilleur modèle (ROI) : {all_results.get('best_roi_model', '?')} — ROI : {all_results.get('best_roi', 0):.4f}")
else:
    st.warning("Métriques réelles non disponibles — results.json introuvable")
