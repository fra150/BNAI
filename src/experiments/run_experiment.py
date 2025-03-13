# src/experiments/run_experiment.py
import os
import sys

# Aggiungi la directory principale al path di Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from src.data.io import load_bnai_data
from src.data.convert_stanford_to_bnai import convert_stanford_to_bnai  # Aggiunta questa importazione
from src.model.bnai_model import BNAIHyperNetwork
from src.model.training import train_bnai
from src.utils.config import load_config
from src.utils.logger import get_logger
from torchvision.models import resnet18, resnet50

def run_experiment(config_path='src/utils/config.yaml'):
    """
    Pipeline principale per l'addestramento del modello BNAI.
    """
    # Carica configurazione e inizializza logger
    cfg = load_config(config_path)
    log = get_logger()

    # Definisci i percorsi corretti per il dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    bnai_dataset_file = os.path.join(base_dir, 'data', 'synthetic_bnai_data.json')
    stanford_dir = os.path.join(base_dir, 'src', 'stanfordSentimentTreebank')

    # Verifica e caricamento dati
    if not os.path.exists(bnai_dataset_file):
        log.info("Creazione dataset BNAI...")
        convert_stanford_to_bnai(
            stanford_dir=stanford_dir,
            output_file=bnai_dataset_file
        )
    else:
        log.info("Dataset BNAI già esistente.")

    # Caricamento dati
    log.info("Caricamento dati BNAI...")
    train_data, val_data = load_bnai_data(bnai_dataset_file, batch_size=cfg['training']['batch_size'])

    # Inizializzazione modello
    log.info("Creating BNAI HyperNetwork model...")
    input_dim = next(iter(train_data))[0].shape[1]  # Get actual input dimension
    output_dim = input_dim  # Set output dimension to match input
    hypernet = BNAIHyperNetwork(latent_dim=cfg['model']['latent_dim'], output_dim=output_dim)

    # Training phase
    log.info("Starting training...")
    train_bnai(hypernet, train_data, val_data,
              epochs=cfg['training']['epochs'],
              save_path=cfg['data']['save_path'],
              config=cfg)

    log.info("Experiment completed.")

if __name__ == "__main__":
    run_experiment()