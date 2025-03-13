import argparse
import os
from transformers import AutoTokenizer
from src.utils import config, logger
from src.model.bnai_code_model import BNAICodeModel
from src.model.code_training import train_bnai_code
from src.data.code_data_loader import load_train_val_test_data

def parse_args():
    parser = argparse.ArgumentParser(description="BNAI Code1 Model Training")
    parser.add_argument('--config', type=str, default='src/utils/config.yaml', help='Path to the configuration file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='models/bnai_code1_model.pth', help='Path to save the trained model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = config.load_config(args.config)
    log = logger.get_logger(name="BNAI_Code1_Model")

    log.info("Starting BNAI Code1 Model training...")
    
    # Define paths to the processed dataset files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, 'src', 'processed')
    
    # Verifica se la directory processed esiste
    if not os.path.exists(processed_dir):
        # Se non esiste, prova il percorso diretto alla directory processed
        processed_dir = os.path.join(os.path.dirname(base_dir), 'src', 'processed')
        if not os.path.exists(processed_dir):
            log.error(f"Directory processed non trovata: {processed_dir}")
            return
    
    log.info(f"Utilizzo dei dataset nella directory: {processed_dir}")
    
    train_data_path = os.path.join(processed_dir, 'bnaicode_dataset_train.jsonl.gz')
    val_data_path = os.path.join(processed_dir, 'bnaicode_dataset_val.jsonl.gz')
    test_data_path = os.path.join(processed_dir, 'bnaicode_dataset_test.jsonl.gz')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Set padding token to be the same as the eos token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    log.info(f"Loading data from {train_data_path}")
    train_loader, val_loader, test_loader = load_train_val_test_data(
        train_path=train_data_path,
        val_path=val_data_path,
        test_path=test_data_path,
        batch_size=args.batch_size,
        tokenizer=tokenizer
    )
    
    # Initialize the BNAI Code Model
    # Use the actual tokenizer vocabulary size instead of a fixed value
    vocab_size = len(tokenizer)
    embedding_dim = cfg.get('model', {}).get('embedding_dim', 256)
    hidden_dim = cfg.get('model', {}).get('hidden_dim', 512)
    num_layers = cfg.get('model', {}).get('num_layers', 2)
    dropout = cfg.get('model', {}).get('dropout', 0.1)
    
    log.info(f"Initializing BNAI Code1 Model with vocab size: {vocab_size}")
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
