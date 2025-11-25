"""
Contrastive loss implementation for enforcing distinction between
normal and anomalous patterns in learned representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for anomaly detection."""
    
    def __init__(self, temperature=0.07, margin=1.0):
        """
        Args:
            temperature: temperature parameter for softmax
            margin: margin for contrastive learning
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, anchor, positive, negative=None):
        """
        Args:
            anchor: (batch_size, d_model) anchor representations
            positive: (batch_size, d_model) positive (normal) representations
            negative: (batch_size, d_model) negative (anomalous) representations (optional)
        Returns:
            loss: contrastive loss value
        """
        # Normalize representations
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Positive similarity
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        
        if negative is not None:
            negative = F.normalize(negative, p=2, dim=1)
            # Negative similarity
            neg_sim = F.cosine_similarity(anchor, negative, dim=1)
            
            # Contrastive loss: maximize pos_sim, minimize neg_sim
            loss = -torch.log(torch.sigmoid(pos_sim / self.temperature)) + \
                   torch.log(torch.sigmoid(neg_sim / self.temperature))
        else:
            # Self-supervised: use other samples in batch as negatives
            # Compute similarity matrix
            batch_size = anchor.size(0)
            sim_matrix = torch.matmul(anchor, positive.t()) / self.temperature
            
            # Positive pairs are on diagonal
            labels = torch.arange(batch_size).to(anchor.device)
            loss = F.cross_entropy(sim_matrix, labels)
        
        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss for contrastive learning."""
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: (batch_size, d_model) anchor representations
            positive: (batch_size, d_model) positive representations
            negative: (batch_size, d_model) negative representations
        Returns:
            loss: triplet loss value
        """
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class InfoNCE(nn.Module):
    """InfoNCE loss (Noise Contrastive Estimation)."""
    
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: temperature parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query, key, negatives=None):
        """
        Args:
            query: (batch_size, d_model) query representations
            key: (batch_size, d_model) key (positive) representations
            negatives: (n_negatives, d_model) negative representations (optional)
        Returns:
            loss: InfoNCE loss
        """
        # Normalize
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
        
        # Positive similarity
        pos_sim = torch.sum(query * key, dim=1) / self.temperature
        
        if negatives is not None:
            negatives = F.normalize(negatives, p=2, dim=1)
            # Negative similarities
            neg_sim = torch.matmul(query, negatives.t()) / self.temperature
            # Combine
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        else:
            # Use other samples in batch as negatives
            batch_size = query.size(0)
            all_keys = key
            logits = torch.matmul(query, all_keys.t()) / self.temperature
        
        # Labels: first element is positive
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(query.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss

