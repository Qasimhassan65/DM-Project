"""
Complete model integrating Transformer, Contrastive Loss, and GAN components.
"""

import torch
import torch.nn as nn
from transformer_model import TransformerAutoencoder
from gan_model import Generator, Discriminator
from geometric_masking import AdaptiveMasking


class AnomalyDetectionModel(nn.Module):
    """Complete anomaly detection model with all components."""
    
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.1, window_size=100,
                 latent_dim=100, use_gan=True):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.window_size = window_size
        self.use_gan = use_gan
        
        # Transformer autoencoder
        self.transformer_ae = TransformerAutoencoder(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            window_size=window_size
        )
        
        # GAN components
        if use_gan:
            self.generator = Generator(
                latent_dim=latent_dim,
                hidden_dim=256,
                output_dim=d_model,
                seq_len=window_size
            )
            self.discriminator = Discriminator(
                input_dim=d_model,
                hidden_dim=256,
                seq_len=window_size
            )
        
        # Geometric masking
        self.masking = AdaptiveMasking(mask_ratio=0.15)
    
    def forward(self, x, use_mask=True):
        """
        Args:
            x: (batch_size, seq_len, n_features) input time series
            use_mask: whether to apply geometric masking
        Returns:
            reconstructed: (batch_size, seq_len, n_features) reconstructed sequence
            encoded: (seq_len, batch_size, d_model) encoded representation
            mask: (batch_size, seq_len, n_features) applied mask (if used)
        """
        mask = None
        if use_mask and self.training:
            x_masked, mask = self.masking(x)
        else:
            x_masked = x
        
        # Encode and decode
        reconstructed, encoded = self.transformer_ae(x_masked, mask)
        
        return reconstructed, encoded, mask
    
    def encode(self, x):
        """Extract representation."""
        return self.transformer_ae.get_representation(x)
    
    def generate(self, batch_size, device):
        """Generate fake representations using GAN generator."""
        if not self.use_gan:
            raise ValueError("GAN not enabled")
        
        z = torch.randn(batch_size, self.generator.latent_dim).to(device)
        fake_repr = self.generator(z)
        return fake_repr
    
    def discriminate(self, encoded):
        """Discriminate between real and fake representations."""
        if not self.use_gan:
            raise ValueError("GAN not enabled")
        
        # Transpose encoded: (seq_len, batch_size, d_model) -> (batch_size, seq_len, d_model)
        encoded = encoded.transpose(0, 1)
        return self.discriminator(encoded)
    
    def compute_reconstruction_error(self, x, reconstructed):
        """Compute reconstruction error for anomaly detection."""
        # Mean Squared Error per sample
        mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return mse

