"""
Transformer architecture for feature extraction and sequence reconstruction
from multivariate time series data.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for feature extraction."""
    
    def __init__(self, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, input_dim=None):
        super().__init__()
        
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Input projection (will be set on first forward pass if input_dim not provided)
        if input_dim is not None:
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            self.input_projection = None
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            encoded: (seq_len, batch_size, d_model)
        """
        # x: (batch_size, seq_len, n_features) -> (seq_len, batch_size, n_features)
        x = x.transpose(0, 1)
        
        # Project to d_model
        if x.size(-1) != self.d_model:
            if self.input_projection is None:
                self.input_projection = nn.Linear(x.size(-1), self.d_model).to(x.device)
            x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        return encoded


class TransformerDecoder(nn.Module):
    """Transformer decoder for sequence reconstruction."""
    
    def __init__(self, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, 
                 dropout=0.1, output_dim=None):
        super().__init__()
        
        self.d_model = d_model
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.output_dim = output_dim
        
        if output_dim is not None:
            self.output_projection = nn.Linear(d_model, output_dim)
    
    def forward(self, memory, tgt=None):
        """
        Args:
            memory: (seq_len, batch_size, d_model) from encoder
            tgt: (seq_len, batch_size, d_model) target sequence (optional)
        Returns:
            decoded: (seq_len, batch_size, output_dim)
        """
        if tgt is None:
            # Use memory as target (autoencoder)
            tgt = memory
        else:
            if tgt.size(-1) != self.d_model:
                if not hasattr(self, 'tgt_projection'):
                    self.tgt_projection = nn.Linear(tgt.size(-1), self.d_model).to(tgt.device)
                tgt = self.tgt_projection(tgt)
        
        tgt = self.pos_encoder(tgt)
        decoded = self.transformer_decoder(tgt, memory)
        
        if self.output_dim is not None:
            decoded = self.output_projection(decoded)
        
        return decoded


class TransformerAutoencoder(nn.Module):
    """Complete transformer autoencoder for time series reconstruction."""
    
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1, window_size=100):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.window_size = window_size
        
        # Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            input_dim=n_features
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=n_features
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, n_features)
            mask: (batch_size, seq_len, n_features) optional mask
        Returns:
            reconstructed: (batch_size, seq_len, n_features)
            encoded: (seq_len, batch_size, d_model)
        """
        # Apply mask if provided
        if mask is not None:
            x = x * mask
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Transpose back: (seq_len, batch_size, n_features) -> (batch_size, seq_len, n_features)
        reconstructed = decoded.transpose(0, 1)
        
        return reconstructed, encoded
    
    def encode(self, x):
        """Extract features only."""
        encoded = self.encoder(x)
        return encoded
    
    def get_representation(self, x):
        """Get learned representation (mean pooling of encoded sequence)."""
        encoded = self.encode(x)
        # (seq_len, batch_size, d_model) -> (batch_size, d_model)
        representation = encoded.mean(dim=0)
        return representation

