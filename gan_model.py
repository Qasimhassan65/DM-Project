"""
GAN (Generative Adversarial Network) implementation to enhance the model's
ability to learn realistic normal patterns and handle contamination in training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Generator network for GAN."""
    
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=128, seq_len=100):
        """
        Args:
            latent_dim: dimension of latent noise vector
            hidden_dim: dimension of hidden layers
            output_dim: dimension of output representation
            seq_len: sequence length
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc4 = nn.Linear(hidden_dim * 4, seq_len * output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, z):
        """
        Args:
            z: (batch_size, latent_dim) random noise
        Returns:
            generated: (batch_size, seq_len, output_dim) generated representation
        """
        x = F.relu(self.bn1(self.fc1(z)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.tanh(self.fc4(x))
        
        # Reshape to sequence
        batch_size = z.size(0)
        x = x.view(batch_size, self.seq_len, self.output_dim)
        
        return x


class Discriminator(nn.Module):
    """Discriminator network for GAN."""
    
    def __init__(self, input_dim=128, hidden_dim=256, seq_len=100):
        """
        Args:
            input_dim: dimension of input representation
            hidden_dim: dimension of hidden layers
            seq_len: sequence length
        """
        super().__init__()
        
        self.seq_len = seq_len
        
        # Convolutional layers for sequence processing
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classification
        self.fc = nn.Linear(hidden_dim * 4, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) input representation
        Returns:
            output: (batch_size, 1) probability of being real
        """
        # (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # (batch_size, hidden_dim * 4)
        
        # Classification
        output = torch.sigmoid(self.fc(x))
        
        return output


class GANLoss(nn.Module):
    """GAN loss functions."""
    
    def __init__(self, gan_mode='vanilla'):
        """
        Args:
            gan_mode: 'vanilla' or 'lsgan' (Least Squares GAN)
        """
        super().__init__()
        self.gan_mode = gan_mode
    
    def __call__(self, prediction, target_is_real):
        """
        Args:
            prediction: discriminator output
            target_is_real: whether target is real (True) or fake (False)
        Returns:
            loss: GAN loss value
        """
        if self.gan_mode == 'vanilla':
            if target_is_real:
                target = torch.ones_like(prediction)
            else:
                target = torch.zeros_like(prediction)
            loss = F.binary_cross_entropy(prediction, target)
        elif self.gan_mode == 'lsgan':
            if target_is_real:
                target = torch.ones_like(prediction)
            else:
                target = torch.zeros_like(prediction)
            loss = F.mse_loss(prediction, target)
        else:
            raise ValueError(f"Unknown GAN mode: {self.gan_mode}")
        
        return loss

