import torch
import torch.nn as nn

class BNAICodeModel(nn.Module):
    """
    BNAI model for code generation/understanding tasks.
    This model processes problem descriptions and generates solution code.
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.1):
        super(BNAICodeModel, self).__init__()
        
        # Embedding layer for tokenized input
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder (processes problem description)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Decoder (generates solution code)
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim * 2,  # * 2 because encoder is bidirectional
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, problem_ids, problem_mask, target_ids=None, target_mask=None, teacher_forcing_ratio=0.5):
        """
        Forward pass through the model
        Args:
            problem_ids: Tensor of shape [batch_size, seq_len] containing tokenized problem descriptions
            problem_mask: Tensor of shape [batch_size, seq_len] containing attention mask for problem
            target_ids: Tensor of shape [batch_size, seq_len] containing tokenized target solution code
            target_mask: Tensor of shape [batch_size, seq_len] containing attention mask for target
            teacher_forcing_ratio: Probability of using teacher forcing during training
        """
        batch_size = problem_ids.size(0)
        
        # Embed problem tokens
        problem_embedded = self.dropout(self.embedding(problem_ids))
        
        # Encode problem
        encoder_outputs, (hidden, cell) = self.encoder(problem_embedded)
        
        # If we're not given target_ids, we're in inference mode
        if target_ids is None:
            return self._generate(encoder_outputs, hidden, cell, batch_size)
        
        # Prepare decoder input (either using teacher forcing or previous predictions)
        target_embedded = self.dropout(self.embedding(target_ids))
        
        # Reshape encoder hidden state for decoder
        # Combine forward and backward directions
        hidden_reshaped = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
        hidden_reshaped = torch.cat([hidden_reshaped[:, 0], hidden_reshaped[:, 1]], dim=2)
        
        cell_reshaped = cell.view(self.encoder.num_layers, 2, batch_size, -1)
        cell_reshaped = torch.cat([cell_reshaped[:, 0], cell_reshaped[:, 1]], dim=2)
        
        # Decoder forward pass
        decoder_outputs, _ = self.decoder(target_embedded, (hidden_reshaped, cell_reshaped))
        
        # Project to vocabulary size
        outputs = self.output_projection(decoder_outputs)
        
        return outputs
    
    def _generate(self, encoder_outputs, hidden, cell, batch_size, max_length=100):
        """
        Generate solution code during inference
        """
        # Implementation for inference-time generation
        # This would use beam search or greedy decoding
        # Simplified version for now
        device = next(self.parameters()).device
        
        # Start with SOS token (assuming token ID 1)
        current_token = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        generated_tokens = []
        
        # Reshape encoder hidden state for decoder
        hidden_reshaped = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
        hidden_reshaped = torch.cat([hidden_reshaped[:, 0], hidden_reshaped[:, 1]], dim=2)
        
        cell_reshaped = cell.view(self.encoder.num_layers, 2, batch_size, -1)
        cell_reshaped = torch.cat([cell_reshaped[:, 0], cell_reshaped[:, 1]], dim=2)
        
        for _ in range(max_length):
            # Embed current token
            token_embedded = self.embedding(current_token)
            
            # Decoder step
            output, (hidden_reshaped, cell_reshaped) = self.decoder(token_embedded, (hidden_reshaped, cell_reshaped))
            
            # Project to vocabulary
            prediction = self.output_projection(output.squeeze(1))
            
            # Get next token (greedy)
            current_token = prediction.argmax(1).unsqueeze(1)
            
            # Add to generated tokens
            generated_tokens.append(current_token)
            
            # Check if we've generated EOS tokens for all sequences
            # (assuming EOS token ID 2)
            if (current_token == 2).all():
                break
        
        # Concatenate all tokens
        return torch.cat(generated_tokens, dim=1)


class BNAICodeLoss(nn.Module):
    """
    Loss function for BNAI code model
    """
    def __init__(self, pad_token_id=0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        
    def forward(self, predictions, targets):
        """
        Calculate loss between predictions and targets
        Args:
            predictions: Tensor of shape [batch_size, seq_len, vocab_size]
            targets: Tensor of shape [batch_size, seq_len]
        """
        # Reshape predictions for CrossEntropyLoss
        batch_size = predictions.size(0)
        seq_len = predictions.size(1)
        vocab_size = predictions.size(2)
        
        predictions = predictions.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        return self.criterion(predictions, targets)