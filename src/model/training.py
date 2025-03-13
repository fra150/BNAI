import torch
import torch.nn as nn
from torch.optim import Adam
from ..utils.logger import get_logger

class BNaiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        # Calculate MSE loss for BNAI parameters
        loss = torch.mean((predictions - targets) ** 2)
        return loss

def train_bnai(model, train_data, validation_data, epochs, save_path, config):
    """
    Train the BNAI HyperNetwork model
    """
    log = get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = BNaiLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for features, targets in train_data:
            optimizer.zero_grad()
            
            # Move data to device
            features = features.to(device)
            targets = targets.to(device)
            
            # Generate latent vectors
            batch_size = features.size(0)
            z = torch.randn(batch_size, config['model']['latent_dim']).to(device)
            
            # Forward pass
            outputs = model(z)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        log.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
        # Validation step
        if validation_data:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, targets in validation_data:
                    features = features.to(device)
                    targets = targets.to(device)
                    batch_size = features.size(0)
                    z = torch.randn(batch_size, config['model']['latent_dim']).to(device)
                    outputs = model(z)
                    val_loss += criterion(outputs, targets).item()
            
            avg_val_loss = val_loss / len(validation_data)
            log.info(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)