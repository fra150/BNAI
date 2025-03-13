import torch
import numpy as np

def normalize_bnai_data(data, method='min-max'):
    """Normalizza i dati BNAI utilizzando il metodo specificato"""
    if len(data) == 0:
        return data
        
    # Converti i profili BNAI in un tensore
    bnai_profiles = torch.tensor([list(profile.values()) for profile in data])
    
    if method == 'min-max':
        # Calcola min e max lungo la dimensione 0 (tra i profili)
        min_val = bnai_profiles.min(dim=0)[0]
        max_val = bnai_profiles.max(dim=0)[0]
        # Evita la divisione per zero
        denominator = (max_val - min_val)
        denominator[denominator == 0] = 1.0
        normalized_profiles = (bnai_profiles - min_val) / denominator
    elif method == 'z-score':
        mean = bnai_profiles.mean(dim=0)
        std = bnai_profiles.std(dim=0)
        # Evita la divisione per zero
        std[std == 0] = 1.0
        normalized_profiles = (bnai_profiles - mean) / std
    else:
        raise ValueError(f"Metodo di normalizzazione '{method}' non supportato")
    
    return normalized_profiles

def preprocess_bnai_data(data):
    """Preprocessa i dati BNAI"""
    if not data:
        return []
        
    # Normalizza i dati
    normalized_data = normalize_bnai_data(data, method='min-max')
    
    return normalized_data