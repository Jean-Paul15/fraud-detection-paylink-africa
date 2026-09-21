"""
Features de graphe pour GNN — Réseaux de transactions.

Calcule les métriques de réseau (degré, PageRank, clustering, centralité)
pour détecter les réseaux de mules et les fraudes organisées.

Les features de graphe sont calculées en batch (mises à jour périodiques)
et utilisées en temps réel via lookup.
"""
import numpy as np
import pandas as pd
import networkx as nx


def build_transaction_graph(edges_df, nodes_df):
    """Construit un graphe NetworkX orienté à partir des transactions.

    Noeuds = comptes (sender_id et receiver_id)
    Arêtes = transactions agrégées entre paires

    Parameters
    ----------
    edges_df : DataFrame avec source, target, total_amount, avg_amount, tx_count
    nodes_df : DataFrame avec account_id, is_fraud_account, kyc_level, account_type

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()

    # Ajouter les noeuds
    for _, node in nodes_df.iterrows():
        G.add_node(
            node["account_id"],
            is_fraud=node.get("is_fraud_account", 0),
            kyc=node.get("kyc_level", 1),
            account_type=node.get("account_type", "individual"),
        )

    # Ajouter les arêtes (sample si trop grand)
    edges_sample = edges_df
    if len(edges_df) > 500000:
        edges_sample = edges_df.sample(500000, random_state=42)

    for _, edge in edges_sample.iterrows():
        G.add_edge(
            edge["source"], edge["target"],
            weight=edge.get("total_amount", 0),
            count=edge.get("tx_count", 1),
        )

    return G


def compute_graph_features(G, nodes_df):
    """Calcule les features de graphe pour chaque noeud.

    Parameters
    ----------
    G : nx.DiGraph
    nodes_df : DataFrame avec account_id

    Returns
    -------
    DataFrame avec features de graphe par compte
    """
    n_nodes = G.number_of_nodes()

    print(f"  Calcul des features de graphe ({n_nodes:,} noeuds)...")

    # Degré
    out_degree = dict(G.out_degree(weight="count"))
    in_degree = dict(G.in_degree(weight="count"))
    total_degree = {n: out_degree.get(n, 0) + in_degree.get(n, 0) for n in G.nodes()}

    # PageRank (importance dans le réseau)
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=50)

    # Clustering coefficient (densité du voisinage)
    # Pour les graphes dirigés, on utilise le graphe non-dirigé
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)

    # Betweenness centrality (hub de transit) — coûteux, sur échantillon
    if n_nodes > 50000:
        sample_nodes = np.random.RandomState(42).choice(
            list(G.nodes()), min(50000, n_nodes), replace=False
        )
        G_sample = G.subgraph(sample_nodes)
        betweenness = nx.betweenness_centrality(G_sample, k=min(1000, len(sample_nodes)))
    else:
        betweenness = nx.betweenness_centrality(G, k=min(1000, n_nodes))

    # Compiler
    feats = pd.DataFrame(index=nodes_df["account_id"])
    feats["out_degree"] = feats.index.map(out_degree).fillna(0)
    feats["in_degree"] = feats.index.map(in_degree).fillna(0)
    feats["total_degree"] = feats.index.map(total_degree).fillna(0)
    feats["pagerank"] = feats.index.map(pagerank).fillna(0)
    feats["clustering_coef"] = feats.index.map(clustering).fillna(0)
    feats["betweenness"] = feats.index.map(betweenness).fillna(0)

    # Ratios
    feats["flow_ratio"] = np.where(
        feats["in_degree"] > 0,
        feats["out_degree"] / feats["in_degree"],
        0
    )
    feats["degree_log"] = np.log1p(feats["total_degree"])
    feats["pagerank_log"] = np.log1p(feats["pagerank"] * 10000)

    return feats


def compute_local_graph_features(transactions_df, accounts_df, lookback_days=30):
    """Features de graphe locales pour le scoring temps réel.

    Au lieu de recalculer le graphe entier (trop lent), on calcule
    des features de voisinage immédiat pour chaque transaction :
    - Combien de transactions entre ce sender et ce receiver ?
    - Le receiver est-il un hub (beaucoup de transactions entrantes) ?

    Ces features légères peuvent être calculées quasi temps réel
    via un cache Redis ou une table pré-agrégée.
    """
    feats = pd.DataFrame(index=transactions_df.index)

    # Paires sender-receiver : nombre de transactions historiques
    pair_counts = transactions_df.groupby(["sender_id", "receiver_id"]).size()
    feats["pair_tx_count"] = transactions_df.set_index(
        ["sender_id", "receiver_id"]
    ).index.map(pair_counts).fillna(0)

    # Receiver : nombre de senders uniques (hub entrant)
    receiver_senders = transactions_df.groupby("receiver_id")["sender_id"].nunique()
    feats["receiver_unique_senders"] = transactions_df["receiver_id"].map(
        receiver_senders
    ).fillna(0)

    # Sender : nombre de receivers uniques (diversification)
    sender_receivers = transactions_df.groupby("sender_id")["receiver_id"].nunique()
    feats["sender_unique_receivers"] = transactions_df["sender_id"].map(
        sender_receivers
    ).fillna(0)

    return feats
