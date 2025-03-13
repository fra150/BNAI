import argparse
from .utils import config, logger
from .model.training import train_bnai
from .data.io import load_bnai_data
from .model.bnai_model import BNAIHyperNetwork

def parse_args():
    parser = argparse.ArgumentParser(description="BNAI Network Training")
    parser.add_argument('--config', type=str, default='src/utils/config.yaml', help='Path to the configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--data_path', type=str, required=True, help='Path to BNAI data (JSON file)')
    parser.add_argument('--save_path', type=str, default='models/bnai_clone.pth', help='Path to save the trained model')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = config.load_config(args.config)
    log = logger.get_logger()

    log.info("Starting BNAI Network training...")

    # Carica i dati BNAI (dataset creato offline, da modelli reali o sintetici)
    train_data, validation_data = load_bnai_data(args.data_path)

    # Calcola output_dim in base alla lunghezza dei parametri definiti in config
    output_dim = len(cfg['bnai_params'])

    # Inizializza la BNAI HyperNetwork
    hypernet = BNAIHyperNetwork(latent_dim=cfg['model']['latent_dim'], output_dim=output_dim)

    # Addestra il modello, passando la configurazione
    train_bnai(hypernet, train_data, validation_data, args.epochs, args.save_path, cfg)

    log.info(f"Training complete. Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()