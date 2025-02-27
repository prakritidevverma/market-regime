import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from models.storage import load_model

def predict_market_regime(input_data):
    """
    Loads trained models and predicts market regimes.
    """
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    # Load Pre-trained Autoencoder & Extract Features
    autoencoder = load_model("autoencoder.h5")
    encoder = tf.keras.Model(autoencoder.input, autoencoder.get_layer("bottleneck").output)
    latent_features = encoder.predict(input_data)

    # Load Pre-trained Transformer
    transformer = load_model("transformer.h5")

    # Predict Regime
    predictions = transformer.predict(latent_features)
    regime_labels = np.argmax(predictions, axis=1)  # Convert softmax output to discrete labels

    return regime_labels