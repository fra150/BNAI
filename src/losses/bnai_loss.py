# src/losses/bnai_loss.py
import torch
import torch.nn as nn
from src.utils.config import load_config

class BNaiLoss(nn.Module):
    def __init__(self, config_path='src/utils/config.yaml'):
        super().__init__()
        self.config = load_config(config_path)
        self.weights = self.config['weights']['parameter_weights']
        
    def forward(self, predictions, targets):
        """
        Custom BNAI loss function combining weighted parameter errors
        and regularization terms
        """
        # Calculate weighted errors for each parameter
        weighted_errors = torch.zeros_like(predictions)
        for i, (param, weight) in enumerate(self.weights.items()):
            weighted_errors[:,i] = weight * (predictions[:,i] - targets[:,i])**2
            
        # Sum all weighted errors
        loss = torch.sum(weighted_errors)
        
        # Add regularization terms from config
        loss += self.config['weights']['alpha'] * torch.norm(predictions, p=2)
        loss += self.config['weights']['beta'] * torch.norm(predictions, p=1)
        
        return loss