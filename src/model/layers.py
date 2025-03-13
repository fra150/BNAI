import torch
import torch.nn as nn

class BNaiInjectionLayer(nn.Module):
    """
    Neural network layer that injects additional information into the main data flow.
    
    Architecture:
    - Takes input tensor x and a bnai vector
    - Projects bnai vector through a linear layer
    - Adds projected vector to input x
    """
    def __init__(self, input_dim, injection_dim):
        """
        Args:
            input_dim (int): Dimension of the bnai vector to be injected
            injection_dim (int): Target dimension for projection
        """
        super(BNaiInjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, injection_dim)

    def forward(self, x, bnai_vector):
        """
        Forward pass combining input with projected injection vector.
        
        Args:
            x (torch.Tensor): Main input tensor
            bnai_vector (torch.Tensor): Vector to be injected
            
        Returns:
            torch.Tensor: x + projected injection
        """
        injection = self.linear(bnai_vector)
        return x + injection

class BNaiProjectionLayer(nn.Module):
    """
    Simple projection layer that transforms input dimensions through linear mapping.
    
    Architecture:
    - Single linear transformation from input_dim to output_dim
    """
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Desired output dimension
        """
        super(BNaiProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass applying linear projection.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Projected tensor
        """
        return self.linear(x)