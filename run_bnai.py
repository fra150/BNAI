import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now we can import from src
from src.utils import config, logger
from src.model.training import train_bnai
from src.data.io import load_bnai_data
from src.model.bnai_model import BNAIHyperNetwork

def parse_args():
    parser = argparse.ArgumentParser(description="BNAI Network Training")
    parser.add_argument('--config', type=str, default='src/utils/config.yaml', help='Path to the configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--data_path', type=str, default='data/bnai_data.json', help='Path to BNAI data (JSON file)')
    parser.add_argument('--save_path', type=str, default='models/bnai_clone.pth', help='Path to save the trained model')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = config.load_config(args.config)
    log = logger.get_logger()

    log.info("Starting BNAI Network training...")

    # Load BNAI data
    train_data, validation_data = load_bnai_data(args.data_path)

    # Calculate output_dim based on the length of parameters defined in config
    output_dim = len(cfg.get('bnai_params', []))
    if output_dim == 0:
        # Fallback to the model config if bnai_params is not defined
        output_dim = cfg['model'].get('output_dim', 21)

    # Initialize the BNAI HyperNetwork
    hypernet = BNAIHyperNetwork(latent_dim=cfg['model']['latent_dim'], output_dim=output_dim)

    # Train the model
    train_bnai(hypernet, train_data, validation_data, args.epochs, args.save_path, cfg)

    log.info(f"Training complete. Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()