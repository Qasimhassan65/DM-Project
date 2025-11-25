"""
Geometric masking techniques for data augmentation in time series.
Includes various masking strategies to improve model robustness.
"""

import torch
import numpy as np
import random


class GeometricMasking:
    """Geometric masking for time series data augmentation."""
    
    def __init__(self, mask_ratio=0.15, mask_mode='random'):
        """
        Args:
            mask_ratio: proportion of features to mask
            mask_mode: 'random', 'block', 'channel', or 'temporal'
        """
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
    
    def random_mask(self, x):
        """Random masking of features."""
        batch_size, seq_len, n_features = x.shape
        mask = torch.ones_like(x)
        
        # Randomly select features to mask
        n_mask = int(seq_len * n_features * self.mask_ratio)
        flat_indices = torch.randperm(seq_len * n_features)[:n_mask]
        
        for idx in flat_indices:
            t = idx // n_features
            f = idx % n_features
            mask[:, t, f] = 0
        
        return x * mask, mask
    
    def block_mask(self, x):
        """Block masking (consecutive time steps)."""
        batch_size, seq_len, n_features = x.shape
        mask = torch.ones_like(x)
        
        block_length = int(seq_len * self.mask_ratio)
        if block_length == 0:
            block_length = 1
        
        # Random start position
        start_pos = random.randint(0, max(1, seq_len - block_length))
        mask[:, start_pos:start_pos + block_length, :] = 0
        
        return x * mask, mask
    
    def channel_mask(self, x):
        """Mask entire channels (features)."""
        batch_size, seq_len, n_features = x.shape
        mask = torch.ones_like(x)
        
        n_channels_to_mask = max(1, int(n_features * self.mask_ratio))
        channels_to_mask = torch.randperm(n_features)[:n_channels_to_mask]
        mask[:, :, channels_to_mask] = 0
        
        return x * mask, mask
    
    def temporal_mask(self, x):
        """Mask random time steps across all features."""
        batch_size, seq_len, n_features = x.shape
        mask = torch.ones_like(x)
        
        n_steps_to_mask = max(1, int(seq_len * self.mask_ratio))
        steps_to_mask = torch.randperm(seq_len)[:n_steps_to_mask]
        mask[:, steps_to_mask, :] = 0
        
        return x * mask, mask
    
    def __call__(self, x):
        """Apply masking based on mode."""
        if self.mask_mode == 'random':
            return self.random_mask(x)
        elif self.mask_mode == 'block':
            return self.block_mask(x)
        elif self.mask_mode == 'channel':
            return self.channel_mask(x)
        elif self.mask_mode == 'temporal':
            return self.temporal_mask(x)
        else:
            return self.random_mask(x)


class AdaptiveMasking:
    """Adaptive masking that combines multiple strategies."""
    
    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio
        self.maskers = {
            'random': GeometricMasking(mask_ratio, 'random'),
            'block': GeometricMasking(mask_ratio, 'block'),
            'channel': GeometricMasking(mask_ratio, 'channel'),
            'temporal': GeometricMasking(mask_ratio, 'temporal')
        }
    
    def __call__(self, x):
        """Randomly select a masking strategy."""
        strategy = random.choice(list(self.maskers.keys()))
        return self.maskers[strategy](x)

