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

    assert not X.isnull().any().any(), "Input data contains NaN values!"
    assert len(X) > 0, "Input DataFrame is empty!"

    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)  # ✅ FIXED

    autoencoder = Autoencoder(input_dim=X.shape[1], encoding_dim=config["model"]["encoding_dim"]).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=config["model"]["learning_rate"])

    loss_function = config["model"]["loss_function"]
    if loss_function == "huber":
        criterion = nn.HuberLoss()
    elif loss_function == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

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
