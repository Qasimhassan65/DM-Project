"""
Training script for the anomaly detection model.
Integrates all components: Transformer, Contrastive Loss, and GAN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from model import AnomalyDetectionModel
from contrastive_loss import ContrastiveLoss, InfoNCE
from gan_model import GANLoss
from data_loader import create_dataloaders, create_synthetic_data


class Trainer:
    """Trainer class for anomaly detection model."""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizers
        self.optimizer_ae = optim.Adam(
            self.model.transformer_ae.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        if config['use_gan']:
            self.optimizer_g = optim.Adam(
                self.model.generator.parameters(),
                lr=config['learning_rate'] * 0.5,
                betas=(0.5, 0.999)
            )
            self.optimizer_d = optim.Adam(
                self.model.discriminator.parameters(),
                lr=config['learning_rate'] * 0.5,
                betas=(0.5, 0.999)
            )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.contrastive_loss = InfoNCE(temperature=0.07)
        self.gan_loss = GANLoss(gan_mode='vanilla')
        
        # Training history
        self.history = {
            'train_loss': [],
            'recon_loss': [],
            'contrastive_loss': [],
            'gan_g_loss': [],
            'gan_d_loss': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_contrastive_loss = 0
        total_gan_g_loss = 0
        total_gan_d_loss = 0
        
        n_batches = 0
        
        for batch_idx, (x, labels) in enumerate(tqdm(train_loader, desc="Training")):
            x = x.to(self.device)
            batch_size = x.size(0)
            
            # ========== Autoencoder Training ==========
            self.optimizer_ae.zero_grad()
            
            # Forward pass with masking
            reconstructed, encoded, mask = self.model(x, use_mask=True)
            
            # Reconstruction loss
            recon_loss = self.reconstruction_loss(reconstructed, x)
            
            # Contrastive loss (self-supervised)
            # Get representations
            repr1 = self.model.encode(x)
            # Create augmented version
            x_aug, _ = self.model.masking(x)
            repr2 = self.model.encode(x_aug)
            
            # Contrastive loss
            contrast_loss = self.contrastive_loss(repr1, repr2)
            
            # Total autoencoder loss
            ae_loss = recon_loss + self.config['contrastive_weight'] * contrast_loss
            
            ae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.transformer_ae.parameters(), max_norm=1.0)
            self.optimizer_ae.step()
            
            total_recon_loss += recon_loss.item()
            total_contrastive_loss += contrast_loss.item()
            
            # ========== GAN Training ==========
            if self.config['use_gan']:
                # Train Discriminator
                self.optimizer_d.zero_grad()
                
                # Real representations
                with torch.no_grad():
                    _, encoded_real, _ = self.model(x, use_mask=False)
                encoded_real = encoded_real.transpose(0, 1)  # (batch_size, seq_len, d_model)
                
                # Fake representations
                fake_repr = self.model.generate(batch_size, self.device)
                
                # Discriminate
                pred_real = self.model.discriminate(encoded_real.transpose(0, 1))
                pred_fake = self.model.discriminate(fake_repr.transpose(0, 1))
                
                # Discriminator loss
                d_loss_real = self.gan_loss(pred_real, True)
                d_loss_fake = self.gan_loss(pred_fake, False)
                d_loss = (d_loss_real + d_loss_fake) / 2
                
                d_loss.backward()
                self.optimizer_d.step()
                
                total_gan_d_loss += d_loss.item()
                
                # Train Generator (every N steps)
                if batch_idx % self.config['gan_g_step'] == 0:
                    self.optimizer_g.zero_grad()
                    
                    # Generate fake
                    fake_repr = self.model.generate(batch_size, self.device)
                    pred_fake = self.model.discriminate(fake_repr.transpose(0, 1))
                    
                    # Generator loss (fool discriminator)
                    g_loss = self.gan_loss(pred_fake, True)
                    
                    g_loss.backward()
                    self.optimizer_g.step()
                    
                    total_gan_g_loss += g_loss.item()
            
            # Total loss
            epoch_loss = ae_loss.item()
            if self.config['use_gan']:
                epoch_loss += d_loss.item()
            
            total_loss += epoch_loss
            n_batches += 1
            
            # Clear GPU cache periodically to prevent memory issues
            if torch.cuda.is_available() and batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Average losses
        avg_loss = total_loss / n_batches
        avg_recon = total_recon_loss / n_batches
        avg_contrastive = total_contrastive_loss / n_batches
        avg_gan_g = total_gan_g_loss / max(1, n_batches // self.config['gan_g_step'])
        avg_gan_d = total_gan_d_loss / n_batches
        
        self.history['train_loss'].append(avg_loss)
        self.history['recon_loss'].append(avg_recon)
        self.history['contrastive_loss'].append(avg_contrastive)
        self.history['gan_g_loss'].append(avg_gan_g)
        self.history['gan_d_loss'].append(avg_gan_d)
        
        return {
            'loss': avg_loss,
            'recon': avg_recon,
            'contrastive': avg_contrastive,
            'gan_g': avg_gan_g,
            'gan_d': avg_gan_d
        }
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_ae_state_dict': self.optimizer_ae.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_ae.load_state_dict(checkpoint['optimizer_ae_state_dict'])
        self.history = checkpoint['history']
        return checkpoint


def train_model(config):
    """Main training function."""
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    
    # Load data
    print("Loading data...")
    if config['use_synthetic']:
        train_data, test_data, train_labels, test_labels = create_synthetic_data(
            n_samples=config.get('n_samples', 10000),
            n_features=config['n_features'],
            anomaly_ratio=0.1
        )
    else:
        from data_loader import load_smd_data
        try:
            train_data, test_data, test_labels = load_smd_data(
                data_dir=config['data_dir'],
                machine_id=config.get('machine_id', 'machine-1-1')
            )
            train_labels = None
        except FileNotFoundError:
            print("Dataset not found. Using synthetic data...")
            train_data, test_data, train_labels, test_labels = create_synthetic_data(
                n_samples=config.get('n_samples', 10000),
                n_features=config['n_features'],
                anomaly_ratio=0.1
            )
    
    # Create dataloaders
    train_loader, test_loader, scaler = create_dataloaders(
        train_data, test_data, train_labels, test_labels,
        window_size=config['window_size'],
        batch_size=config['batch_size'],
        stride=config.get('stride', 1),
        normalize=True
    )
    
    n_features = train_data.shape[1]
    
    # Create model
    print("Creating model...")
    model = AnomalyDetectionModel(
        n_features=n_features,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        window_size=config['window_size'],
        latent_dim=config.get('latent_dim', 100),
        use_gan=config['use_gan']
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer (model will be moved to device in Trainer.__init__)
    trainer = Trainer(model, device, config)
    
    if torch.cuda.is_available():
        print(f"✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Batch size: {config['batch_size']} (optimized for GPU)")
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        losses = trainer.train_epoch(train_loader)
        
        print(f"Loss: {losses['loss']:.4f} | "
              f"Recon: {losses['recon']:.4f} | "
              f"Contrastive: {losses['contrastive']:.4f}")
        
        if config['use_gan']:
            print(f"GAN G: {losses['gan_g']:.4f} | GAN D: {losses['gan_d']:.4f}")
        
        # Save best model
        if losses['loss'] < best_loss:
            best_loss = losses['loss']
            os.makedirs('checkpoints', exist_ok=True)
            trainer.save_model('checkpoints/best_model.pt')
            print("Saved best model")
    
    # Save final model
    trainer.save_model('checkpoints/final_model.pt')
    print("\nTraining completed!")
    
    return trainer, test_loader, scaler


if __name__ == '__main__':
    # Configuration
    config = {
        'n_features': 10,
        'window_size': 100,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'contrastive_weight': 0.5,
        'use_gan': True,
        'gan_g_step': 2,
        'latent_dim': 100,
        'use_synthetic': True,  # Set to False to use real dataset
        'data_dir': 'data'
    }
    
    trainer, test_loader, scaler = train_model(config)

