# src/data/synthetic_data.py
import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm

# Modifica l'importazione per usare il percorso relativo corretto
from src.utils.config import load_config

def generate_synthetic_bnai_data(num_samples: int, latent_dim: int, bnai_dim: int, config_path="src/utils/config.yaml") -> list:
    cfg = load_config(config_path)
    data = []
    for _ in range(num_samples):
        latent_vector = torch.randn(latent_dim).tolist()
        bnai_profile = []
        for param in cfg['bnai_params']:
            min_val = param['min_val']
            max_val = param['max_val']
            if param['type'] == 'float':
                value = random.uniform(min_val, max_val)
            elif param['type'] == 'int':
                value = random.randint(min_val, max_val)
            else:
                raise ValueError("Unsupported parameter type.")
            bnai_profile.append(value)
        data.append({
            'latent_vector': latent_vector,
            'bnai_profile': bnai_profile
        })
    return data

if __name__ == '__main__':
    num_samples = 10000
    latent_dim = 100
    cfg = load_config("src/utils/config.yaml")
    bnai_dim = len(cfg['bnai_params'])
    synthetic_data = generate_synthetic_bnai_data(num_samples, latent_dim, bnai_dim)
    from data.io import save_bnai_data
    save_bnai_data(synthetic_data, 'src/data/synthetic_bnai_data.json')
    print("Dati sintetici salvati in: src/data/synthetic_bnai_data.json")