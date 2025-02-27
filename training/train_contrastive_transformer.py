import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from models.autoencoder import Autoencoder
from models.contrastive_transformer import ContrastiveTransformer
from utils.config_loader import load_config
from models.storage import save_model, load_model
from torch.utils.tensorboard import SummaryWriter
import os
import modin as mpd
import pandas as pd

def train_autoencoder(X):
    """
    Function to train the Autoencoder using PyTorch.
    """
    config = load_config("configs/autoencoder.yaml")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    autoencoder = Autoencoder(input_dim=X.shape[1], encoding_dim=config["model"]["encoding_dim"]).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=config["model"]["learning_rate"])

    # ✅ Use PyTorch's nn.HuberLoss() instead of string reference
    loss_function = config["model"]["loss_function"]
    if loss_function == "huber":
        criterion = nn.HuberLoss()
    elif loss_function == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    num_batches = len(X) // batch_size

    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0

        for i in range(num_batches):
            batch = X_tensor[i * batch_size: (i + 1) * batch_size]
            optimizer.zero_grad()
            _, reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / num_batches:.4f}")

    torch.save(autoencoder.state_dict(), config["model"]["save_model_path"])
    print("✅ Autoencoder training complete!")


def select_optimal_gmm_components(X_encoded, max_clusters=18):
    """
    Uses BIC (Bayesian Information Criterion) to determine the optimal number of clusters for GMM.
    """
    bic_scores = []
    best_n_components = 2  # Start with at least 2 clusters

    for n in range(2, max_clusters + 1):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', reg_covar=1e-5, random_state=42)
            gmm.fit(X_encoded)
            bic_scores.append(gmm.bic(X_encoded))  # Compute BIC score
        except:
            break  # Stop if GMM fails due to singular covariance

    best_n_components = np.argmin(bic_scores) + 2  # Add 2 because range starts from 2
    print(f"✅ Optimal GMM components selected using BIC: {best_n_components}")
    return best_n_components

def generate_contrastive_pairs(X_encoded, num_neighbors=5):
    """
    Dynamically determines the best number of clusters and generates positive/negative contrastive pairs.
    Uses Bayesian Information Criterion (BIC) for GMM selection.
    """

    X_encoded = np.array(X_encoded)

    num_clusters = select_optimal_gmm_components(X_encoded)
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', reg_covar=1e-5, random_state=42)
    soft_labels = gmm.fit_predict(X_encoded)

    knn = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean')
    knn.fit(X_encoded)
    _, neighbor_indices = knn.kneighbors(X_encoded)

    positive_pairs = np.zeros_like(X_encoded)
    negative_pairs = np.zeros_like(X_encoded)

    for i in range(len(X_encoded)):
        same_cluster_neighbors = [idx for idx in neighbor_indices[i] if soft_labels[idx] == soft_labels[i]]
        if len(same_cluster_neighbors) > 1:
            positive_pairs[i] = X_encoded[np.random.choice(same_cluster_neighbors)]
        else:
            positive_pairs[i] = X_encoded[i]

        different_cluster_indices = np.where(soft_labels != soft_labels[i])[0]
        if len(different_cluster_indices) > 0:
            negative_pairs[i] = X_encoded[np.random.choice(different_cluster_indices)]
        else:
            negative_pairs[i] = X_encoded[i]

    return positive_pairs, negative_pairs



def train_contrastive_transformer(X_encoded=None, use_saved_autoencoder=True):
    """
    Function to train the Contrastive Transformer separately using PyTorch with MPS acceleration.
    """
    config = load_config("configs/contrastive_learning.yaml")
    autoencoder_config = load_config("configs/autoencoder.yaml")
    log_dir = "logs/contrastive_transformer"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ✅ Detect Apple M2 GPU (MPS Support)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    if use_saved_autoencoder:
        print("🔄 Loading latest saved Autoencoder model...")
        autoencoder = Autoencoder(
            input_dim=autoencoder_config["model"]["input_dim"],
            encoding_dim=autoencoder_config["model"]["encoding_dim"]
        )
        autoencoder.load_state_dict(torch.load(autoencoder_config["model"]["load_model_path"], map_location=device))
        autoencoder.to(device)
        autoencoder.eval()

        print(f"🔥 Debug: X_encoded shape BEFORE Autoencoder transformation: {X_encoded.shape}")  # Should be (N, 45)

     
        X_encoded = X_encoded.to_numpy()

        with torch.no_grad():
            X_encoded = autoencoder.encoder(torch.tensor(X_encoded, dtype=torch.float32).to(device)).cpu().numpy()


        print(f"✅ Debug: X_encoded shape AFTER Autoencoder transformation: {X_encoded.shape}")  # Should be (N, 10)


    positive_pairs, negative_pairs = generate_contrastive_pairs(X_encoded)

    # ✅ Fix: Correct conversion
    X_encoded = torch.tensor(X_encoded, dtype=torch.float32).to(device)
    positive_pairs = torch.tensor(positive_pairs, dtype=torch.float32).to(device)
    negative_pairs = torch.tensor(negative_pairs, dtype=torch.float32).to(device)

    # ✅ Fix: Ensure `input_dim` matches Autoencoder output
    transformer = ContrastiveTransformer(
        input_dim=autoencoder_config["model"]["encoding_dim"],  # ✅ Ensure this matches Autoencoder
        d_model=config["model"].get("d_model", 128),
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        ff_dim=config["model"].get("ff_dim", 256)
    ).to(device)

    transformer.train()
    optimizer = optim.Adam(transformer.parameters(), lr=config["model"]["learning_rate"])

    # ✅ Fix: Use contrastive loss function instead of CrossEntropyLoss
    def contrastive_loss(z):
        pos_sim, neg_sim = z[:, 0], z[:, 1]
        margin = 0.5
        loss = torch.mean(torch.relu(neg_sim - pos_sim + margin))  # Contrastive loss
        return loss

    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # ✅ Fix: Ensure correct input format
        z = transformer(X_encoded, positive_pairs, negative_pairs)

        # ✅ Fix: Ensure the loss function receives correctly shaped inputs
        loss = contrastive_loss(z)  # ✅ Use contrastive loss

        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/Train', loss.item(), epoch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(transformer.state_dict(), config["model"]["save_model_path"])
    writer.close()
    print("✅ Contrastive Transformer training complete!")


