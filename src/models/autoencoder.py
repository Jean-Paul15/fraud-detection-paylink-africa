"""
Modèle 2 — Autoencoder pour détection d'anomalies par reconstruction.

Principe : Un autoencodeur apprend à compresser puis reconstruire les
transactions légitimes. Les fraudes, étant différentes, ont une erreur
de reconstruction plus élevée -> score d'anomalie.

Avantages pour la fraude :
- Deep learning : capture des patterns non-linéaires complexes
- Non-supervisé : entraîné uniquement sur les données légitimes
- L'erreur de reconstruction est directement interprétable
- S'adapte : peut être fine-tuné sur de nouvelles données
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


class FraudAutoencoder(nn.Module):
    """Autoencodeur pour détection de fraude.

    Architecture : encodeur -> goulot d'étranglement -> décodeur
    Le goulot force l'apprentissage d'une représentation compacte
    des transactions légitimes.
    """

    def __init__(self, input_dim, hidden_dims=None, latent_dim=None, dropout=0.2):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        if latent_dim is None:
            latent_dim = max(input_dim // 8, 4)

        # Encodeur
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Décodeur
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruction_error(self, x):
        """Calcule l'erreur de reconstruction (MSE par échantillon)."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            reconstructed = self.forward(x_tensor)
            error = torch.mean((x_tensor - reconstructed) ** 2, dim=1)
        return error.numpy()


def train_autoencoder(model, X_train, X_val=None, epochs=50, batch_size=256,
                      learning_rate=1e-3, patience=10, device="cpu"):
    """Entraîne l'autoencodeur sur les données légitimes uniquement.

    Parameters
    ----------
    model : FraudAutoencoder
    X_train : array — transactions légitimes (is_fraud=0)
    X_val : array or None
    epochs : int
    batch_size : int
    learning_rate : float
    patience : int — early stopping
    device : str

    Returns
    -------
    model entraîné, dict historique de perte
    """
    model = model.to(device)

    # Dataset
    X_train_tensor = torch.FloatTensor(X_train)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, X_train_tensor),
        batch_size=batch_size, shuffle=True
    )

    if X_val is not None:
        X_val_tensor = torch.FloatTensor(X_val)
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, X_val_tensor),
            batch_size=batch_size, shuffle=False
        )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        if X_val is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    reconstructed = model(batch_x)
                    loss = criterion(reconstructed, batch_x)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping à l'époque {epoch+1}")
                model.load_state_dict(best_state)
                break

    model.eval()
    return model, history


def score_autoencoder(model, X, device="cpu"):
    """Score de fraude basé sur l'erreur de reconstruction.

    Plus l'erreur est élevée, plus la transaction est anormale -> frauduleuse.
    """
    errors = model.reconstruction_error(X)
    # Normaliser en pseudo-probabilité [0, 1]
    proba = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
    return proba
