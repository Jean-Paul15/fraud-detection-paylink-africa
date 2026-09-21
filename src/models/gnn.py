"""
Modèle 3 — Graph Neural Network (GNN) pour détection de réseaux de fraude.

Principe : Les fraudeurs n'opèrent pas isolément. Un GNN capture
les relations entre comptes (transferts d'argent) pour identifier
les réseaux de mules et les patterns de blanchiment.

Architecture : GraphSAGE (Graph Sample and Aggregate)
- Couche 1 : Agrège les voisins à 1 saut
- Couche 2 : Agrège les voisins à 2 sauts
- Classification binaire par noeud (fraude/légitime)

Utilise PyTorch Geometric.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


def prepare_graph_data(nodes_df, edges_df, label_col="is_fraud_account",
                       test_size=0.2, random_state=42):
    """Prépare les données pour PyTorch Geometric.

    Parameters
    ----------
    nodes_df : DataFrame avec account_id, features, label
    edges_df : DataFrame avec source, target
    test_size : float
    random_state : int

    Returns
    -------
    Data (PyTorch Geometric), train_mask, test_mask
    """
    # Mapping account_id -> index numérique
    all_ids = pd.concat([edges_df["source"], edges_df["target"]]).unique()
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}

    # Features des noeuds
    feature_cols = [c for c in nodes_df.columns
                    if c not in ["account_id", label_col, "account_type", "is_risky"]]
    # Ne garder que les colonnes numériques
    feature_cols = [c for c in feature_cols
                    if nodes_df[c].dtype in ["int64", "float64", "int32", "float32"]]

    node_features = nodes_df.set_index("account_id").loc[
        [a for a in all_ids if a in nodes_df["account_id"].values]
    ]
    x = torch.FloatTensor(node_features[feature_cols].fillna(0).values)

    # Labels
    y = torch.LongTensor(
        nodes_df.set_index("account_id").loc[
            [a for a in all_ids if a in nodes_df["account_id"].values],
            label_col
        ].fillna(0).values
    )

    # Arêtes (edge_index)
    edge_sources = [id_to_idx[s] for s in edges_df["source"] if s in id_to_idx]
    edge_targets = [id_to_idx[t] for t in edges_df["target"] if t in id_to_idx]
    edge_index = torch.LongTensor([edge_sources, edge_targets])

    # Split train/test
    n_nodes = x.shape[0]
    indices = np.random.RandomState(random_state).permutation(n_nodes)
    split_idx = int(n_nodes * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask)

    return data


class GraphSAGE(nn.Module):
    """GraphSAGE pour classification de noeuds.

    Architecture : 2 couches SAGEConv + classification
    """

    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 dropout=0.3, aggr="mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """Graph Attention Network — alternative à GraphSAGE.

    Utilise l'attention pour pondérer l'importance des voisins.
    """

    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def train_gnn(model, data, epochs=100, lr=0.01, weight_decay=5e-4,
              patience=20, verbose=True):
    """Entraîne un GNN pour classification de noeuds.

    Gère le déséquilibre via weighted loss (poids inversement
    proportionnels à la fréquence de classe).

    Returns
    -------
    model entraîné, dict historique
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    # Poids pour classes déséquilibrées
    n_fraud = data.y[data.train_mask].sum().item()
    n_legit = data.train_mask.sum().item() - n_fraud
    class_weight = torch.FloatTensor([1.0, n_legit / max(n_fraud, 1)]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss(weight=class_weight)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Éval
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
            val_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc.item())
        history["val_acc"].append(val_acc.item())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping à l'époque {epoch+1} (best val_acc={best_val_acc:.4f})")
            break

    return model, history


def score_gnn(model, data):
    """Score de fraude par noeud (probabilité de la classe 1)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        out = model(data)
        proba = torch.exp(out[:, 1]).cpu().numpy()  # Classe 1 = fraude

    return proba
