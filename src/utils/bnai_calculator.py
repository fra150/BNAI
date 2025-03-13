# src/utils/bnai_calculator.py
import torch
import numpy as np
import math
# Modifica l'importazione per usare il percorso completo
from src.data.preprocess import normalize_bnai_data

def calculate_bnai_parameters(model, model_info=None, config=None):
    """
    Calculates BNAI parameters for an AI model.
    
    Args:
        model: AI Model (nn.Module instance).
        model_info: Additional model information (optional).
        config: Configuration dictionary.
        
    Returns:
        Dictionary containing BNAI parameters.
    """
    params = {}
    # Calculation examples (placeholder; to be implemented with real logic)
    params['A'] = 1.5          # Adaptability (to be implemented)
    params['E_g'] = 1.0        # Generational Evolution
    num_params = sum(p.numel() for p in model.parameters())
    params['G'] = np.log10(num_params)  # Now includes G_c as temporal derivative (ΔG/Δt)
    params['H'] = 1.4          # Learning Entropy
    params['I'] = 1.1          # Interconnection
    params['L'] = 0.89         # Learning Level
    params['P'] = 56.9         # Computational Power
    params['Q'] = 0.92         # Response Accuracy
    params['R'] = 0.96         # Robustness
    params['U'] = 1.0          # Autonomy
    params['V'] = 0.034        # Response Speed
    params['O'] = 0.93         # Computational Efficiency
    
    # New ethical and security parameters
    params['Ethics'] = calculate_ethics_score(model, model_info)
    params['Security'] = calculate_security_score(model, model_info)
    
    # Penalty parameters
    params['B'] = 0.03         # Bias
    params['C'] = calculate_model_complexity(model)  # Now includes T (tensor dimensionality)
    params['E'] = 0.021        # Learning Error
    params['M'] = 50           # Memory Capacity
    params['t'] = 1            # Evolutionary Time

    # Advanced coefficients
    params['R_TL'] = 1.0       # Transfer Learning Coefficient
    params['R_ML'] = 1.0       # Meta-Learning Coefficient
    params['lambda_reg'] = calculate_regularization_term(model, config)

    return params

def calculate_regularization_term(model, config):
    reg_type = config.get('regularization_type', 'l2')
    alpha = config.get('regularization_alpha', 0.01)
    if reg_type == 'l2':
        l2_reg = torch.tensor(0., requires_grad=False)
        for p in model.parameters():
            l2_reg += torch.norm(p, p=2) ** 2
        lambda_reg = alpha * torch.sqrt(l2_reg)
    elif reg_type == 'l1':
        l1_reg = torch.tensor(0., requires_grad=False)
        for p in model.parameters():
            l1_reg += torch.norm(p, p=1)
        lambda_reg = alpha * l1_reg
    else:
        raise ValueError("Unsupported regularization type. Choose 'l1' or 'l2'.")
    return lambda_reg.item()

def calculate_tensor_dimensionality(model):
    num_layers = 0
    total_dim = 0
    for p in model.parameters():
        num_layers += 1
        total_dim += p.ndim
    return total_dim / num_layers if num_layers > 0 else 0

def calculate_model_complexity(model):
    num_operations = 0
    for p in model.parameters():
        num_operations += p.numel()
    return num_operations

# Funzioni per calcolare i nuovi parametri etici e di sicurezza
def calculate_ethics_score(model, model_info=None):
    """
    Calculates the model's ethical score based on fairness, privacy, and transparency.
    
    Args:
        model: AI Model (nn.Module instance).
        model_info: Additional model information (optional).
        
    Returns:
        Normalized ethical score.
    """
    # Placeholder - to be implemented with real logic
    fairness = 0.85      # Model fairness measure
    privacy = 0.90       # Privacy compliance
    transparency = 0.80  # Model transparency
    
    # Weights for each component (to be defined through expert consensus)
    alpha_fair = 0.4
    beta_priv = 0.3
    gamma_trasp = 0.3
    
    # Composition formula
    ethics_score = alpha_fair * fairness + beta_priv * privacy + gamma_trasp * transparency
    return ethics_score

def calculate_security_score(model, model_info=None):
    """
    Calculates the model's security score based on attack resistance and OOD detection.
    
    Args:
        model: AI Model (nn.Module instance).
        model_info: Additional model information (optional).
        
    Returns:
        Normalized security score.
    """
    # Placeholder - to be implemented with real logic
    adversarial_resistance = 0.75  # Resistance to adversarial attacks
    ood_detection = 0.80          # Out-of-distribution input detection
    prompt_injection_resistance = 0.70  # Resistance to prompt injection
    
    # Component weights
    w_adv = 0.4
    w_ood = 0.4
    w_prompt = 0.2
    
    # Composition formula
    security_score = w_adv * adversarial_resistance + w_ood * ood_detection + w_prompt * prompt_injection_resistance
    return security_score

# Placeholder per altre funzioni di calcolo
def calculate_adaptability(model, data, metric_fn):
    pass

def calculate_evolution(model, previous_version, metric_fn):
    pass

def calculate_learning_entropy(model, train_data):
    pass

def calculate_interconnection(model, pre_trained_model):
    pass

def get_learning_level(model):
    pass

def measure_flops(model, input_size):
    pass

def evaluate_model(model, val_loader, metric_fn, device):
    pass

def calculate_robustness(model, test_loader, attack_fn, device):
    pass

def measure_autonomy(model):
    pass

def measure_inference_time(model, test_loader, device):
    pass

def calculate_bias(model, data, metric_fn):
    pass

def calculate_transfer_learning_coefficient(model, pre_trained_model):
    pass

def calculate_meta_learning_coefficient(model, meta_dataset):
    pass

if __name__ == '__main__':
    import torch.nn as nn
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 5)
        def forward(self, x):
            return self.linear2(self.linear1(x))
    model = DummyModel()
    cfg = {'regularization_type': 'l2', 'regularization_alpha': 0.01}
    bnai_params = calculate_bnai_parameters(model, config=cfg)
    print(bnai_params)