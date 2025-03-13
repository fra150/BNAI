import argparse
import torch
from transformers import AutoTokenizer
from src.utils import config, logger
from src.model.bnai_code_model import BNAICodeModel
from src.model.code_training import train_bnai_code
from src.data.code_data_loader import load_train_val_test_data

def parse_args():
    parser = argparse.ArgumentParser(description="BNAI Code Model Training")
    parser.add_argument('--config', type=str, default='src/utils/config.yaml', help='Path to the configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data (CSV or JSONL.GZ)')
    parser.add_argument('--val_data', type=str, help='Path to validation data (optional)')
    parser.add_argument('--test_data', type=str, help='Path to test data (optional)')
    parser.add_argument('--save_path', type=str, default='models/bnai_code_model.pth', help='Path to save the trained model')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Vocabulary size for tokenizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = config.load_config(args.config)
    log = logger.get_logger(name="BNAI_Code_Model")

    log.info("Starting BNAI Code Model training...")
    
    # Initialize tokenizer (using a simple tokenizer for now)
    # In a production environment, you would use a pre-trained tokenizer
    # or train one on your specific dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load datasets
    log.info(f"Loading data from {args.train_data}")
    train_loader, val_loader, test_loader = load_train_val_test_data(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        batch_size=args.batch_size,
        tokenizer=tokenizer
    )
    
    # Initialize the BNAI Code Model
    vocab_size = args.vocab_size
    embedding_dim = cfg.get('model', {}).get('embedding_dim', 256)
    hidden_dim = cfg.get('model', {}).get('hidden_dim', 512)
    num_layers = cfg.get('model', {}).get('num_layers', 2)
    dropout = cfg.get('model', {}).get('dropout', 0.1)
    
    log.info(f"Initializing BNAI Code Model with vocab size: {vocab_size}")
    model = BNAICodeModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Train the model
    log.info(f"Starting training for {args.epochs} epochs")
    train_bnai_code(
        model=model,
        train_data=train_loader,
        validation_data=val_loader,
        epochs=args.epochs,
        save_path=args.save_path,
        config=cfg
    )
    
    log.info(f"Training complete. Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()