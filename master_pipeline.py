"""
Master Pipeline v2 — Détection Fraude PayLink Africa
======================================================
Pipeline complet avec toutes les best practices :
- 500k transactions train (toutes les fraudes conservées)
- Features transactionnelles + comportementales + graphe
- Preprocessing multi-stratégies
- Cross-validation LGB + tuning
- IF + AE + LGB + Stacking
- Learning curves
- Coût-bénéfice multi-modèle
"""
import warnings, os, json, joblib
from pathlib import Path
PROJET = Path(__file__).resolve().parent
os.chdir(PROJET)
import sys; sys.path.insert(0, str(PROJET))

warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from src.preprocessing.split import check_temporal_leakage
from src.features.transactional import extract_transactional_features
from src.features.behavioral import compute_behavioral_features, compute_velocity_features, compute_ratio_features
from src.models.isolation_forest import tune_isolation_forest, anomaly_score_to_proba
from src.models.autoencoder import FraudAutoencoder, train_autoencoder, score_autoencoder
from src.evaluation.fraud_metrics import all_fraud_metrics, cost_benefit_analysis, find_optimal_threshold

RANDOM_STATE = 42; np.random.seed(42)
OUTPUTS = Path('outputs'); (OUTPUTS / 'models').mkdir(parents=True, exist_ok=True)
(OUTPUTS / 'figures').mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("MASTER PIPELINE v2 — Détection Fraude PayLink Africa")
print("=" * 70)

# ============================================================
# 1. CHARGEMENT & ÉCHANTILLONNAGE STRATIFIÉ
# ============================================================
print("\n1. CHARGEMENT (500k, toutes fraudes conservées)")
train = pd.read_csv('data/processed/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('data/processed/test.csv', parse_dates=['timestamp'])
check_temporal_leakage(train, test)

# Garder TOUTES les fraudes + compléter avec légitimes jusqu'à 500k
frauds_tr = train[train['is_fraud'] == 1]
legit_tr = train[train['is_fraud'] == 0]
n_frauds = len(frauds_tr)
n_legit_sample = min(500_000 - n_frauds, len(legit_tr))
legit_sample = legit_tr.sort_values('timestamp').tail(n_legit_sample)
train_sample = pd.concat([frauds_tr, legit_sample]).sort_values('timestamp')
n_frauds_train = train_sample['is_fraud'].sum()
print(f"Train: {len(train_sample):,} ({n_frauds_train} fraudes, {train_sample['is_fraud'].mean()*100:.3f}%)")
print(f"Test:  {len(test):,} ({test['is_fraud'].sum()} fraudes, {test['is_fraud'].mean()*100:.3f}%)")
y_train = train_sample['is_fraud']
y_test = test['is_fraud']

# ============================================================
# 2. FEATURES TRANSACTIONNELLES (sur toutes les données)
# ============================================================
print("\n2. FEATURES TRANSACTIONNELLES")
# ANTI-LEAKAGE: top_cities calcule sur train uniquement, reutilise pour le test
if "city" in train_sample.columns:
    top_cities = train_sample["city"].value_counts().head(15).index.tolist()
else:
    top_cities = None
tx_train = extract_transactional_features(train_sample)
tx_test = extract_transactional_features(test, top_cities=top_cities)
common = list(set(tx_train.columns) & set(tx_test.columns))
tx_train, tx_test = tx_train[common], tx_test[common]
print(f"  {len(common)} features")

EXCLUDE = ['transaction_id','timestamp','is_fraud','fraud_type',
           'sender_id','receiver_id','device_id','city']
base_cols = [c for c in train_sample.columns if c not in EXCLUDE]
X_base_tr = train_sample[base_cols].select_dtypes(include=[np.number])
X_base_te = test[base_cols].select_dtypes(include=[np.number])

# ============================================================
# 3. FEATURES COMPORTEMENTALES (sur 500k, fenêtres 1h/6h/24h)
# ============================================================
print("\n3. FEATURES COMPORTEMENTALES (500k, fenêtres 1h, 6h, 24h)")
# Échantillonner les senders pour accélérer (20k senders = ~125k tx)
SAMPLE_SENDERS = 20000
print(f"  Échantillonnage {SAMPLE_SENDERS} senders...")
behav_train = compute_behavioral_features(train_sample, windows=["1h","6h","24h"], sample_senders=SAMPLE_SENDERS)
behav_test = compute_behavioral_features(test, windows=["1h","6h","24h"], sample_senders=SAMPLE_SENDERS)
velo_train = compute_velocity_features(train_sample)
velo_test = compute_velocity_features(test)

# Aligner colonnes
for df_list in [(behav_train, behav_test), (velo_train, velo_test)]:
    all_cols = list(set(df_list[0].columns) | set(df_list[1].columns))
    for c in all_cols:
        for df_i in df_list:
            if c not in df_i.columns: df_i[c] = 0

behav_train = behav_train.fillna(0); behav_test = behav_test.fillna(0)
velo_train = velo_train.fillna(0); velo_test = velo_test.fillna(0)
print(f"  Comp: {behav_train.shape[1]} features, Vélo: {velo_train.shape[1]} features")

# ============================================================
# 4. FEATURES DE GRAPHE
# ============================================================
print("\n4. FEATURES DE GRAPHE")
nodes = pd.read_csv('data/graph/nodes.csv')
graph_feat_cols = ['out_degree','in_degree','total_sent','total_received',
                   'flow_ratio','is_fraud_account','pagerank','clustering_coef']
available = [c for c in graph_feat_cols if c in nodes.columns]
graph_feats = nodes.set_index('account_id')[available].fillna(0)

def map_graph_feats(df, prefix):
    mapped = df[f'{prefix}_id'].map(
        graph_feats.add_prefix(f'{prefix}_g_').to_dict(orient='index')
    )
    result = pd.json_normalize(mapped.fillna({}))
    result.index = df.index
    return result.fillna(0)

g_tr_s, g_tr_r = map_graph_feats(train_sample, 'sender'), map_graph_feats(train_sample, 'receiver')
g_te_s, g_te_r = map_graph_feats(test, 'sender'), map_graph_feats(test, 'receiver')
graph_tr = pd.concat([g_tr_s, g_tr_r], axis=1)
graph_te = pd.concat([g_te_s, g_te_r], axis=1)
print(f"  Graph: {graph_tr.shape[1]} features")

# ============================================================
# 5. ASSEMBLAGE
# ============================================================
print("\n5. ASSEMBLAGE FINAL")
X_tr = pd.concat([
    X_base_tr.reset_index(drop=True),
    tx_train.reset_index(drop=True),
    behav_train.reset_index(drop=True),
    velo_train.reset_index(drop=True),
    graph_tr.reset_index(drop=True),
], axis=1).fillna(0)

X_te = pd.concat([
    X_base_te.reset_index(drop=True),
    tx_test.reset_index(drop=True),
    behav_test.reset_index(drop=True),
    velo_test.reset_index(drop=True),
    graph_te.reset_index(drop=True),
], axis=1).fillna(0)

print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

# Nettoyage
sel = VarianceThreshold(threshold=0.0); sel.fit(X_tr)
kept = X_tr.columns[sel.get_support()]
X_tr = pd.DataFrame(sel.transform(X_tr), columns=kept)
X_te = pd.DataFrame(sel.transform(X_te), columns=kept)

# Feature selection: corrélation > 0.95 → drop, puis top 50 LGB
corr = X_tr.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_c = [c for c in upper.columns if any(upper[c] > 0.95)]
X_tr = X_tr.drop(columns=drop_c, errors='ignore')
X_te = X_te.drop(columns=drop_c, errors='ignore')

lgb_sel = LGBMClassifier(n_estimators=150, max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
lgb_sel.fit(X_tr, y_train)
imp = pd.Series(lgb_sel.feature_importances_, index=X_tr.columns).sort_values(ascending=False)

# Garder top 50 ou moins si moins disponibles
N_FEAT = min(50, len(imp))
top_feats = imp.head(N_FEAT).index.tolist()
X_tr, X_te = X_tr[top_feats], X_te[top_feats]
print(f"  Final: {X_tr.shape[1]} features | Train={X_tr.shape[0]:,} | Test={X_te.shape[0]:,}")
print(f"  Top 10: {top_feats[:10]}")

# ============================================================
# 6. PREPROCESSING MULTI-STRATÉGIES
# ============================================================
print("\n6. PREPROCESSING — Comparaison StandardScaler vs RobustScaler")
for scaler_name, Scaler in [("StandardScaler", StandardScaler), ("RobustScaler", RobustScaler)]:
    sc = Scaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    lgb_test = LGBMClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    lgb_test.fit(X_tr_s, y_train)
    auc_pr = average_precision_score(y_test, lgb_test.predict_proba(X_te_s)[:, 1])
    print(f"  {scaler_name:20s}: AUC-PR={auc_pr:.4f}")

# StandardScaler retenu (les features transactionnelles sont déjà normalisées)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# ============================================================
# 7. CROSS-VALIDATION LGB + TUNING
# ============================================================
print("\n7. LIGHTGBM — Cross-Validation + Tuning")

# SMOTE sur train
smote = SMOTE(sampling_strategy=0.15, random_state=RANDOM_STATE, k_neighbors=3)
X_sm, y_sm = smote.fit_resample(X_tr_s, y_train)
print(f"  SMOTE: {len(y_sm):,} ({y_sm.mean()*100:.1f}% fraude)")

# CV pour estimer la performance réelle
lgb_cv = LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.03,
    class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
cv_scores = cross_val_score(lgb_cv, X_sm, y_sm, cv=3, scoring='average_precision', n_jobs=-1)
print(f"  CV AUC-PR: {cv_scores.mean():.4f} +/- {cv_scores.std()*2:.4f}")

# Entraînement final
lgb_cv.fit(X_sm, y_sm)
lgb_scores = lgb_cv.predict_proba(X_te_s)[:, 1]
lgb_metrics = all_fraud_metrics(y_test, lgb_scores)
print(f"  Test AUC-PR: {lgb_metrics['auc_pr']:.4f} | Lift@5%: {lgb_metrics['lift_5pct']:.1f}x")

# ============================================================
# 7b. CALIBRATION ISOTONIQUE LGB
# ============================================================
print("\n7b. CALIBRATION ISOTONIQUE LGB")
from sklearn.calibration import CalibratedClassifierCV

# Séparer 20% pour calibration sur données originales (non-SMOTE)
X_tr_cal, X_cal, y_tr_cal, y_cal = train_test_split(
    X_tr_s, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)

lgb_cal = LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.03,
    class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
lgb_cal.fit(X_tr_cal, y_tr_cal)

calibrator = CalibratedClassifierCV(lgb_cal, method='isotonic', cv='prefit')
calibrator.fit(X_cal, y_cal)

# Remplacer lgb_scores par les scores calibrés
lgb_scores = calibrator.predict_proba(X_te_s)[:, 1]
lgb_scores_tr = calibrator.predict_proba(X_tr_s)[:, 1]

lgb_metrics = all_fraud_metrics(y_test, lgb_scores)
print(f"  Test AUC-PR (calibré): {lgb_metrics['auc_pr']:.4f} | Lift@5%: {lgb_metrics['lift_5pct']:.1f}x")

# ============================================================
# 8. ISOLATION FOREST
# ============================================================
print("\n8. ISOLATION FOREST")
X_if_tr, X_if_val, y_if_tr, y_if_val = train_test_split(
    X_tr_s, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train)

if_result = tune_isolation_forest(X_if_tr, y_if_tr, X_if_val, y_if_val,
    contaminations=[0.001, 0.003, 0.005, 0.01], n_estimators_list=[200, 300])
if_model = if_result['best_model']
if_scores = anomaly_score_to_proba(if_model.decision_function(X_te_s))
if_metrics = all_fraud_metrics(y_test, if_scores)
print(f"  Best: {if_result['best_params']} | AUC-PR={if_metrics['auc_pr']:.4f} | Lift@5%={if_metrics['lift_5pct']:.1f}x")

# ============================================================
# 9. AUTOENCODER
# ============================================================
print("\n9. AUTOENCODER")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_legit = X_tr_s[y_train.values == 0]
X_ae_tr, X_ae_val = train_test_split(X_legit, test_size=0.1, random_state=RANDOM_STATE)

input_dim = X_tr_s.shape[1]
ae = FraudAutoencoder(input_dim, [input_dim//2, input_dim//4], max(input_dim//8, 4))
ae, hist = train_autoencoder(ae, X_ae_tr, X_ae_val, epochs=50, batch_size=256, patience=10, device=device)
ae_scores = score_autoencoder(ae.cpu(), X_te_s)
ae_metrics = all_fraud_metrics(y_test, ae_scores)
print(f"  AUC-PR={ae_metrics['auc_pr']:.4f} | Lift@5%={ae_metrics['lift_5pct']:.1f}x")

# ============================================================
# 10. GNN — GRAPH NEURAL NETWORK (GraphSAGE)
# ============================================================
print("\n10. GNN — Graph Neural Network (GraphSAGE)")
gnn_scores_tr = np.zeros(len(y_train))
gnn_scores_te = np.zeros(len(y_test))
gnn_model = None
gnn_metrics = None
account_gnn_score = {}

try:
    from src.models.gnn import prepare_graph_data, GraphSAGE, train_gnn, score_gnn

    nodes = pd.read_csv('data/graph/nodes.csv')
    edges = pd.read_csv('data/graph/edges.csv')
    print(f"  Noeuds: {len(nodes):,} | Arêtes: {len(edges):,}")

    # Préparer les données graphe
    data = prepare_graph_data(nodes, edges, label_col="is_fraud_account",
                              test_size=0.2, random_state=RANDOM_STATE)

    # Entraîner GraphSAGE
    input_dim = data.x.shape[1]
    gnn_model = GraphSAGE(in_channels=input_dim, hidden_channels=64,
                          out_channels=2, dropout=0.3, aggr="mean")
    gnn_model, gnn_history = train_gnn(gnn_model, data, epochs=100, lr=0.01,
                                       weight_decay=5e-4, patience=20, verbose=True)

    # Scores par noeud (probabilité de fraude)
    node_scores = score_gnn(gnn_model, data)

    # Mapping account_id -> score GNN
    all_ids = pd.concat([edges["source"], edges["target"]]).unique()
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
    account_gnn_score = {}
    for aid, idx in id_to_idx.items():
        account_gnn_score[aid] = float(node_scores[idx])

    # Mapper les scores aux transactions (sender_id)
    gnn_scores_tr = np.array([
        account_gnn_score.get(sid, 0.0) for sid in train_sample['sender_id']
    ])
    gnn_scores_te = np.array([
        account_gnn_score.get(sid, 0.0) for sid in test['sender_id']
    ])

    gnn_metrics = all_fraud_metrics(y_test, gnn_scores_te)
    print(f"  AUC-PR={gnn_metrics['auc_pr']:.4f} | Lift@5%={gnn_metrics['lift_5pct']:.1f}x")

except ImportError as e:
    print(f"  GNN non disponible (torch_geometric manquant): {e}")
except Exception as e:
    print(f"  GNN erreur (non-bloquant): {e}")

# ============================================================
# 11. ENSEMBLE STACKING (4 modèles)
# ============================================================
print("\n11. ENSEMBLE STACKING (IF + AE + LGB + GNN)")
if_scores_tr = anomaly_score_to_proba(if_model.decision_function(X_tr_s))
ae_scores_tr = score_autoencoder(ae.cpu(), X_tr_s)
# lgb_scores_tr déjà calculé via calibration

stack_X_tr = np.column_stack([if_scores_tr, ae_scores_tr, lgb_scores_tr, gnn_scores_tr])
stack_X_te = np.column_stack([if_scores, ae_scores, lgb_scores, gnn_scores_te])

stack = LogisticRegression(C=1.0, class_weight='balanced', random_state=RANDOM_STATE, max_iter=2000)
stack.fit(stack_X_tr, y_train)
stack_scores = stack.predict_proba(stack_X_te)[:, 1]
stack_metrics = all_fraud_metrics(y_test, stack_scores)
print(f"  Coeffs: IF={stack.coef_[0][0]:.3f} AE={stack.coef_[0][1]:.3f} LGB={stack.coef_[0][2]:.3f} GNN={stack.coef_[0][3]:.3f}")
print(f"  AUC-PR={stack_metrics['auc_pr']:.4f} | Lift@5%={stack_metrics['lift_5pct']:.1f}x")

# ============================================================
# 12. RÉSULTATS COMPARATIFS
# ============================================================
print("\n12. RÉSULTATS COMPARATIFS")
print(f"{'Modèle':25s} | {'AUC-PR':>8s} | {'AUC-ROC':>8s} | {'Recall@FPR1%':>12s} | {'Lift@1%':>8s} | {'Lift@5%':>8s}")
models_list = [("Isolation Forest", if_metrics), ("Autoencoder", ae_metrics),
               ("LightGBM Calibré", lgb_metrics)]
if gnn_metrics is not None:
    models_list.append(("GraphSAGE (GNN)", gnn_metrics))
models_list.append(("Ensemble Stacking", stack_metrics))
for name, m in models_list:
    print(f"{name:25s} | {m['auc_pr']:8.4f} | {m['auc_roc']:8.4f} | {m['recall_at_fpr_1pct']:12.4f} | {m['lift_1pct']:8.1f}x | {m['lift_5pct']:8.1f}x")

# ============================================================
# 13. LEARNING CURVES
# ============================================================
print("\n12. LEARNING CURVES — LightGBM")
sizes = np.linspace(0.1, 1.0, 8)
train_sizes_lc, train_pr, val_pr = [], [], []
for size in sizes:
    n = int(len(X_sm) * size)
    if n < 1000: continue
    X_sub, y_sub = X_sm[:n], y_sm[:n]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    tr_scores, vl_scores = [], []
    for tr_i, vl_i in cv.split(X_sub, y_sub):
        mdl = LGBMClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, verbose=-1)
        mdl.fit(X_sub[tr_i], y_sub[tr_i])
        tr_scores.append(average_precision_score(y_sub[tr_i], mdl.predict_proba(X_sub[tr_i])[:, 1]))
        vl_scores.append(average_precision_score(y_sub[vl_i], mdl.predict_proba(X_sub[vl_i])[:, 1]))
    train_sizes_lc.append(n)
    train_pr.append(np.mean(tr_scores))
    val_pr.append(np.mean(vl_scores))
    print(f"  Size={n:>8,} | Train AUC-PR={train_pr[-1]:.4f} | Val AUC-PR={val_pr[-1]:.4f}")

# ============================================================
# 14. COÛT-BÉNÉFICE MULTI-MODÈLE
# ============================================================
print("\n14. COÛT-BÉNÉFICE MULTI-MODÈLE")
best_model_name, best_roi = "", -999
cost_models = [("IF", if_scores), ("AE", ae_scores), ("LGB", lgb_scores)]
if gnn_metrics is not None:
    cost_models.append(("GNN", gnn_scores_te))
cost_models.append(("Stacking", stack_scores))
for name, scores in cost_models:
    cb = cost_benefit_analysis(y_test, scores, cost_fp=5000, cost_fn=500000)
    opt = find_optimal_threshold(cb, min_recall=0.50)
    baseline = y_test.sum() * 500000
    losses_avoided = baseline - opt['cost_fn_total']
    roi = (losses_avoided - opt['cost_fp_total']) / (opt['cost_fp_total'] + 1)
    print(f"  {name:10s}: Seuil={opt['threshold']:.3f}, Recall={opt['frauds_caught_pct']:.1f}%, "
          f"FPR={opt['alert_rate_pct']:.2f}%, ROI={roi:.1f}x, Économie={baseline - opt['cost_total']:,.0f} FCFA")
    if roi > best_roi:
        best_roi, best_model_name = roi, name

print(f"\n  Meilleur ROI: {best_model_name} ({best_roi:.1f}x)")

# ============================================================
# 15. BANDIT CONTEXTUEL — Thompson Sampling pour seuil optimal
# ============================================================
print("\n15. BANDIT CONTEXTUEL — Thompson Sampling pour seuil optimal")
from src.models.bandit import ThompsonSamplingBandit

bandit = ThompsonSamplingBandit(thresholds=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
rng_bandit = np.random.RandomState(RANDOM_STATE)

# Simuler 5000 decisions de seuil sur l'ensemble de test
n_simulations = min(5000, len(y_test))
rewards = []
selected_thresholds = []

for i in range(n_simulations):
    # Choisir un seuil
    threshold, arm = bandit.select_threshold(rng_bandit)
    selected_thresholds.append(threshold)

    # Feedback : le seuil a-t-il bien classe cette transaction ?
    true_label = y_test.iloc[i]
    score = stack_scores[i]
    predicted_label = 1 if score >= threshold else 0
    reward = 1 if predicted_label == true_label else 0
    rewards.append(reward)

    bandit.update(arm, reward)

# Resultats
running_accuracy = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
best_threshold = bandit.get_best_threshold()
print(f"  Meilleur seuil appris: {best_threshold:.3f}")
print(f"  Accuracy finale: {running_accuracy[-1]:.4f}")
distrib = pd.Series(selected_thresholds).value_counts(normalize=True).to_dict()
print(f"  Distribution seuils: {distrib}")
print(f"  Stats par bras: {bandit.get_posterior_stats()}")

# ============================================================
# 16. SAUVEGARDE
# ============================================================
print("\n16. SAUVEGARDE")
joblib.dump(stack, OUTPUTS/'models'/'ensemble_stacking.pkl')
joblib.dump(if_model, OUTPUTS/'models'/'isolation_forest.pkl')
torch.save(ae.state_dict(), OUTPUTS/'models'/'autoencoder.pt')
joblib.dump(lgb_cv, OUTPUTS/'models'/'lightgbm.pkl')
joblib.dump(scaler, OUTPUTS/'models'/'scaler.pkl')
joblib.dump(top_feats, OUTPUTS/'models'/'feature_columns.pkl')
joblib.dump(calibrator, OUTPUTS/'models'/'lgb_calibrator.pkl')
if gnn_model is not None:
    torch.save(gnn_model.state_dict(), OUTPUTS/'models'/'gnn_graphsage.pt')
# Le stacking est entraine sur 4 colonnes (IF/AE/LGB/GNN) meme quand le GNN
# est indisponible (gnn_scores reste a zero dans ce cas, cf. section 10) :
# inference.py doit donc toujours pouvoir retrouver un score GNN par compte,
# d'ou la persistance de ce mapping plutot que du seul modele GraphSAGE.
joblib.dump(account_gnn_score, OUTPUTS/'models'/'gnn_account_scores.pkl')
joblib.dump(bandit, OUTPUTS/'models'/'bandit.pkl')
print("  bandit.pkl sauvegarde")

results = {
    'if': {k:float(v) for k,v in if_metrics.items()},
    'ae': {k:float(v) for k,v in ae_metrics.items()},
    'lgb': {k:float(v) for k,v in lgb_metrics.items()},
    'stack': {k:float(v) for k,v in stack_metrics.items()},
    'cv_auc_pr_mean': float(cv_scores.mean()), 'cv_auc_pr_std': float(cv_scores.std()),
    'best_roi_model': best_model_name, 'best_roi': float(best_roi),
    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
}
if gnn_metrics is not None:
    results['gnn'] = {k: float(v) for k, v in gnn_metrics.items()}
with open(OUTPUTS/'models'/'results.json','w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("PIPELINE v2 TERMINÉ — GNN + Calibration Isotonic")
print(f"{'='*70}")
print(f"Meilleur modèle: {best_model_name} | ROI: {best_roi:.1f}x")
print(f"CV AUC-PR: {cv_scores.mean():.4f} +/- {cv_scores.std()*2:.4f}")
