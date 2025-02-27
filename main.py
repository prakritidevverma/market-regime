import argparse
from training.train_contrastive_transformer import train_contrastive_transformer
from training.train_autoencoder import train_autoencoder
from inference.inference import predict_market_regime
from datasets.clickhouse_loader import load_all_data, load_large_cap, load_mid_cap, load_small_cap
from datasets.preprocessing import preprocess_data
from utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="Market Regime Pipeline")
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="Mode: train or inference")
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    print(config)

    if args.mode == "train":
        print("🚀 Training Models...")
        df = load_large_cap(config['dataset']['start_date'], config['dataset']['end_date'], config['dataset']['size'])
        df = preprocess_data(df)
        train_autoencoder(df)
        print("✅ Training Complete! AutoEncoder Model saved in `saved_models/`.")
        train_contrastive_transformer(df, use_saved_autoencoder=True)
        print("✅ Training Complete! Contrastive Model saved in `saved_models/`.")

    elif args.mode == "inference":
        print("🚀 Running Inference...")
        df = load_mid_cap(config['dataset']['start_date'], config['dataset']['end_date'], config['dataset']['size'])
        df = preprocess_data(df)
        test_data = df.iloc[:100]  # Example input
        labels = predict_market_regime(test_data)
        print("Market Regime Predictions:", labels)

if __name__ == "__main__":
    main()
