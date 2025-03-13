import torch
import torch.nn as nn

class BNAIHyperNetwork(nn.Module):
    """
    HyperNetwork that generates BNAI profile (digital DNA) from a latent vector.
    Architecture: 3-layer MLP with increasing hidden dimensions for better feature extraction.
    """
    def __init__(self, latent_dim, output_dim):
        """
        Args:
            latent_dim: Dimension of input latent vector
            output_dim: Dimension of output BNAI profile
        """
        super(BNAIHyperNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Progressive expansion of network dimensions (latent_dim -> 256 -> 512 -> output_dim)
        # for better feature learning capacity
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        """
        Forward pass: transforms latent vector to BNAI profile
        Args:
            z: Input latent vector
        """
        return self.network(z)

if __name__ == '__main__':
    num_bnai_params = 21  # Adjust based on config
    hypernet = BNAIHyperNetwork(latent_dim=100, output_dim=num_bnai_params)
    z = torch.randn(1, 100)  # Generate random latent vector
    bnai_profile = hypernet(z)
    print("BNAI Profile Shape:", bnai_profile.shape)
    print("BNAI Profile:", bnai_profile)