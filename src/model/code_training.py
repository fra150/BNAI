import torch
import torch.nn as nn
from torch.optim import Adam
from src.utils.logger import get_logger
from src.model.bnai_code_model import BNAICodeLoss

def train_bnai_code(model, train_data, validation_data, epochs, save_path, config):
    """
    Train the BNAI Code Model
    """
    log = get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=config.get('training', {}).get('learning_rate', 0.001))
    criterion = BNAICodeLoss(pad_token_id=0)  # Assuming 0 is the pad token ID
    
    log.info(f"Training BNAI Code Model on {device} for {epochs} epochs")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_data:
            optimizer.zero_grad()
            
            # Move data to device
            problem_ids = batch['problem_ids'].to(device)
            problem_mask = batch['problem_mask'].to(device)
            solution_ids = batch['solution_ids'].to(device)
            
            # Forward pass
            outputs = model(problem_ids, problem_mask, solution_ids)
            
            # Calculate loss (ignore padding tokens)
            loss = criterion(outputs, solution_ids)
            
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
                for batch in validation_data:
                    problem_ids = batch['problem_ids'].to(device)
                    problem_mask = batch['problem_mask'].to(device)
                    solution_ids = batch['solution_ids'].to(device)
                    
                    outputs = model(problem_ids, problem_mask, solution_ids)
                    val_loss += criterion(outputs, solution_ids).item()
            
            avg_val_loss = val_loss / len(validation_data)
            log.info(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    log.info(f"Model saved to {save_path}")
    
    return model