import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.storage import load_model
from utils.config_loader import load_config
from models.autoencoder import Autoencoder
from models.contrastive_transformer import ContrastiveTransformer 

def predict_market_regime(input_data):
    """
    Loads trained models and predicts market regimes using PyTorch.
    """
    # Load configurations
    autoencoder_config = load_config("configs/autoencoder.yaml")
    contrastive_config = load_config("configs/contrastive_learning.yaml")  

    # ✅ Standardize input data (same as during training)
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    # ✅ Convert input data to a PyTorch tensor
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    # ✅ Load Pre-trained Autoencoder & Extract Features
    autoencoder = load_model(
        autoencoder_config["model"]["load_model_path"],
        Autoencoder,  # ✅ Pass model class
        input_dim=autoencoder_config["model"]["input_dim"],
        encoding_dim=autoencoder_config["model"]["encoding_dim"]
    ).to(device)

    autoencoder.eval()  # Set to evaluation mode

    with torch.no_grad():
        latent_features = autoencoder.encoder(input_tensor)  # ✅ Extract bottleneck features

    # ✅ Load Pre-trained Transformer
    transformer = load_model(
    contrastive_config["model"]["load_model_path"],
    ContrastiveTransformer,
    input_dim=autoencoder_config["model"]["encoding_dim"],  # ✅ Match autoencoder output
    d_model=contrastive_config["model"].get("d_model", 128),
    num_heads=contrastive_config["model"]["num_heads"],
    num_layers=contrastive_config["model"]["num_layers"],
    ff_dim=contrastive_config["model"].get("ff_dim", 256)
).to(device)

    transformer.eval()

    # ✅ Predict Market Regime
    with torch.no_grad():
        predictions = transformer.inference(latent_features)

    # ✅ Convert softmax output to discrete labels
    regime_labels = torch.argmax(F.softmax(predictions, dim=1), dim=1).cpu().numpy()

    return regime_labels
